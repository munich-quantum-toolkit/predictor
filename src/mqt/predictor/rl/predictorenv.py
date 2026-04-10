# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Predictor environment for the compilation using reinforcement learning."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from bqskit import Circuit
    from pytket._tket.passes import BasePass as TketBasePass
    from pytket.circuit import Node
    from qiskit.passmanager.base_tasks import Task
    from qiskit.transpiler import Target

    from mqt.predictor.reward import figure_of_merit
    from mqt.predictor.rl.actions import Action
    from mqt.predictor.rl.parsing import (
        PreProcessTKETRoutingAfterQiskitLayout,
    )


import warnings
from math import isclose
from typing import cast

import numpy as np
from bqskit.ext import bqskit_to_qiskit, qiskit_to_bqskit
from gymnasium import Env
from gymnasium.spaces import Box, Dict, Discrete
from joblib import load
from pytket.circuit import Qubit
from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
from pytket.placement import Placement
from qiskit import QuantumCircuit
from qiskit.circuit import StandardEquivalenceLibrary
from qiskit.exceptions import QiskitError
from qiskit.transpiler import CouplingMap, Layout, PassManager, TranspileLayout
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes import (
    ApplyLayout,
    BasisTranslator,
    EnlargeWithAncilla,
    FullAncillaAllocation,
    SetLayout,
)
from qiskit.transpiler.passes.layout.vf2_layout import VF2LayoutStopReason

from mqt.predictor.hellinger import get_hellinger_model_path
from mqt.predictor.reward import (
    crit_depth,
    esp_data_available,
    estimated_hellinger_distance,
    estimated_success_probability,
    expected_fidelity,
)
from mqt.predictor.rl.actions import (
    CompilationOrigin,
    DeviceDependentAction,
    PassType,
    get_actions_by_pass_type,
)
from mqt.predictor.rl.approx_reward import (
    approx_estimated_success_probability,
    approx_expected_fidelity,
    compute_device_averages_from_target,
)
from mqt.predictor.rl.helper import (
    create_feature_dict,
    get_path_training_circuits,
    get_state_sample,
)
from mqt.predictor.rl.parsing import (
    final_layout_bqskit_to_qiskit,
    final_layout_pytket_to_qiskit,
    postprocess_vf2postlayout,
    prepare_noise_data,
)
from mqt.predictor.utils import calc_supermarq_features, get_openqasm_gates_for_rl

logger = logging.getLogger("mqt-predictor")


def _layout_output_qubits(layout: TranspileLayout) -> list[Any]:
    """Return the materialized output wires tracked by a TranspileLayout."""
    output_qubits = layout._output_qubit_list  # noqa: SLF001
    assert output_qubits is not None
    return list(output_qubits)


def _layout_input_qubit_count(layout: TranspileLayout) -> int:
    """Return the number of logical input qubits tracked by a TranspileLayout."""
    input_qubit_count = layout._input_qubit_count  # noqa: SLF001
    assert input_qubit_count is not None
    return input_qubit_count


def _clear_circuit_layout(circuit: QuantumCircuit) -> QuantumCircuit:
    """Drop any circuit-attached layout metadata.

    The env keeps ``self.layout`` as the canonical source of layout state and only
    re-attaches it when exporting a circuit to external callers.
    """
    circuit._layout = None  # noqa: SLF001
    return circuit


def _contains_operation_wider_than_two_qubits(circuit: QuantumCircuit) -> bool:
    """Return whether the circuit contains any operation acting on more than two qubits."""
    return any(len(item.qubits) > 2 for item in circuit.data)


class PredictorEnv(Env):
    """Predictor environment for reinforcement learning."""

    def __init__(
        self,
        device: Target,
        mdp: str = "paper",
        reward_function: figure_of_merit = "expected_fidelity",
        path_training_circuits: Path | None = None,
        reward_scale: float = 1.0,
        no_effect_penalty: float = -0.001,
    ) -> None:
        """Initializes the PredictorEnv object.

        Arguments:
            device: The target device to be used for compilation.
            mdp: The MDP transition policy. "paper" (default) enforces a strict, linear pipeline
                (synthesis -> (layout->routing) / mapping), "flexible" allows for a cyclical approach
                where actions can be interleaved or reversed, "thesis" uses the custom action-validity
                rules defined in ``determine_valid_actions_for_state``, and "hybrid" is flexible before
                layout but keeps layout/routing locked once they have been established.
            reward_function: The figure of merit to be used for the reward function. Defaults to "expected_fidelity".
            path_training_circuits: The path to the training circuits folder. Defaults to None, which uses the default path.
            reward_scale: Scaling factor for rewards/penalties proportional to fidelity changes.
            no_effect_penalty: Step penalty applied when an action does not change the circuit (no-op).

        Raises:
            ValueError: If the reward function is "estimated_success_probability" and no calibration data is available for the device or if the reward function is "estimated_hellinger_distance" and no trained model is available for the device.
        """
        logger.info("Init env: " + reward_function)

        self.path_training_circuits = path_training_circuits or get_path_training_circuits()

        self.action_set = {}
        self.actions_synthesis_indices = []
        self.actions_layout_indices = []
        self.actions_routing_indices = []
        self.actions_mapping_indices = []
        self.actions_opt_indices = []
        self.actions_final_optimization_indices = []
        self.actions_structure_preserving_indices = []  # Actions that preserves the mapping and native gates
        self.used_actions: list[str] = []
        self.device = device

        logger.info("MDP: " + mdp)
        self.mdp = mdp

        # check for uni-directional coupling map
        coupling_set = {tuple(pair) for pair in self.device.build_coupling_map()}
        if any((b, a) not in coupling_set for (a, b) in coupling_set):
            msg = f"The connectivity of the device '{self.device.description}' is uni-directional and MQT Predictor might return a compiled circuit that assumes bi-directionality."
            warnings.warn(msg, UserWarning, stacklevel=2)

        index = 0
        action_dict = get_actions_by_pass_type()

        for elem in action_dict[PassType.SYNTHESIS]:
            self.action_set[index] = elem
            self.actions_synthesis_indices.append(index)
            index += 1
        for elem in action_dict[PassType.OPT]:
            self.action_set[index] = elem
            self.actions_opt_indices.append(index)
            if getattr(elem, "preserve_layout", False):
                self.actions_structure_preserving_indices.append(index)
            index += 1
        for elem in action_dict[PassType.LAYOUT]:
            self.action_set[index] = elem
            self.actions_layout_indices.append(index)
            index += 1
        for elem in action_dict[PassType.ROUTING]:
            self.action_set[index] = elem
            self.actions_routing_indices.append(index)
            index += 1
        for elem in action_dict[PassType.MAPPING]:
            self.action_set[index] = elem
            self.actions_mapping_indices.append(index)
            index += 1
        for elem in action_dict[PassType.FINAL_OPT]:
            self.action_set[index] = elem
            self.actions_final_optimization_indices.append(index)
            index += 1

        self.action_set[index] = action_dict[PassType.TERMINATE][0]
        self.action_terminate_index = index

        if reward_function == "estimated_success_probability" and not esp_data_available(self.device):
            msg = f"Missing calibration data for ESP calculation on {self.device.description}."
            raise ValueError(msg)
        if reward_function == "estimated_hellinger_distance":
            hellinger_model_path = get_hellinger_model_path(self.device)
            if not hellinger_model_path.is_file():
                msg = f"Missing trained model for Hellinger distance estimates on {self.device.description}."
                raise ValueError(msg)
            self.hellinger_model = load(hellinger_model_path)
        self.reward_function = reward_function
        self.action_space = Discrete(len(self.action_set.keys()))
        self.num_steps = 0
        self.layout: TranspileLayout | None = None
        self.num_qubits_uncompiled_circuit = 0

        self.has_parameterized_gates = False
        self.rng = np.random.default_rng(10)

        gate_spaces = {g: Box(low=0, high=1, shape=(1,), dtype=np.float32) for g in get_openqasm_gates_for_rl()}

        spaces = {
            "num_qubits": Discrete(self.device.num_qubits + 1),
            "depth": Discrete(1000000),
            "program_communication": Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "critical_depth": Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "entanglement_ratio": Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "parallelism": Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "liveness": Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "measure": Box(low=0, high=1, shape=(1,), dtype=np.float32),
            **gate_spaces,
        }
        self.observation_space = Dict(spaces)
        self.filename = ""
        self.max_iter = 20
        self.node_err: dict[Node, float] | None = None
        self.edge_err: dict[tuple[Node, Node], float] | None = None
        self.readout_err: dict[Node, float] | None = None
        self.reward_scale = reward_scale
        self.no_effect_penalty = no_effect_penalty
        self.prev_reward: float | None = None
        self.prev_reward_kind: str | None = None
        self.episode_count = 0
        self.current_circuit_name = "<unknown>"
        self.err_by_gate: dict[str, float] = {}
        self.dur_by_gate: dict[str, float] = {}
        self.tbar: float | None = None
        self.dev_avgs_cached = False
        self.state: QuantumCircuit = QuantumCircuit()
        self.compilation_state_flags: tuple[bool, bool, bool] | None = None

        self.error_occurred = False

    def _apply_and_update(self, action: int) -> QuantumCircuit | None:
        """Apply an action and update the canonical circuit state."""
        altered_qc = self.apply_action(action)
        if altered_qc is None:
            return None

        altered_qc = _clear_circuit_layout(altered_qc)

        self.state: QuantumCircuit = altered_qc
        self.compilation_state_flags = None

        self.num_steps += 1
        self.valid_actions = self.determine_valid_actions_for_state()
        if not self.valid_actions:
            msg = "No valid actions left."
            raise RuntimeError(msg)

        return altered_qc

    def _create_observation(self, qc: QuantumCircuit | None = None) -> dict[str, Any]:
        """Create an observation directly from the actual circuit state."""
        circuit = self.state if qc is None else qc
        return create_feature_dict(circuit)

    def export_circuit(self, qc: QuantumCircuit | None = None) -> QuantumCircuit:
        """Return a copy of a circuit with the current env layout attached."""
        circuit = self.state if qc is None else qc
        exported = circuit.copy()
        exported._layout = self.layout  # noqa: SLF001
        return exported

    def _log_step_reward(self, step_index: int, action_name: str, reward_val: float, done: bool) -> None:
        """Log the chosen action and resulting reward for the current episode step."""
        logger.info(
            "Episode %d step %d: action=%s reward=%.6f",
            self.episode_count,
            step_index,
            action_name,
            reward_val,
        )
        if done:
            logger.info(
                "Episode %d finished: circuit=%s final_reward=%.6f",
                self.episode_count,
                self.current_circuit_name,
                reward_val,
            )

    def _get_compilation_state_flags(self) -> tuple[bool, bool, bool]:
        """Return `(synthesized, laid_out, routed)` for the current circuit state."""
        if self.compilation_state_flags is not None:
            return self.compilation_state_flags

        synthesized = self.is_circuit_synthesized(self.state)
        laid_out = self.is_circuit_laid_out(self.state, self.layout) if self.layout else False
        routed = (
            self.is_circuit_routed(self.state, CouplingMap(self.device.build_coupling_map())) if laid_out else False
        )
        self.compilation_state_flags = (synthesized, laid_out, routed)
        return self.compilation_state_flags

    def get_compilation_state_flags(self) -> tuple[bool, bool, bool]:
        """Return `(synthesized, laid_out, routed)` for the current circuit state."""
        return self._get_compilation_state_flags()

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, bool, dict[Any, Any]]:
        """Run one environment step.

        This method:
            1. Evaluates the current circuit with the configured reward function
            (using either the exact or approximate metric, depending on state).
            2. Applies the selected transpiler pass (the action).
            3. Computes a shaped step reward based on the change in the figure of merit.

        Reward design:
            - For non-terminal actions, the step reward is a scaled delta between
            the new and previous reward (plus an optional step penalty).
            - For the terminate action, the episode ends and the final reward is
            the exact (calibration-aware) metric.
        """
        action_name = str(self.action_set[action].name)
        step_index = self.num_steps + 1
        self.used_actions.append(action_name)
        logger.info("Episode %d step %d: applying %s", self.episode_count, step_index, action_name)
        previous_state_flags = self._get_compilation_state_flags()

        altered_qc = self._apply_and_update(action)
        if altered_qc is None:
            self._log_step_reward(step_index, action_name, 0.0, done=True)
            return self._create_observation(), 0.0, True, False, {}

        done = action == self.action_terminate_index

        if self.reward_function == "estimated_hellinger_distance":
            reward_val = self.calculate_reward(mode="exact")[0] if done else 0.0
            self._log_step_reward(step_index, action_name, reward_val, done)
            return self._create_observation(), reward_val, done, False, {}

        # Lazy init: compute prev_reward only once per episode (or if missing)
        if self.prev_reward is None:
            self.prev_reward, self.prev_reward_kind = self.calculate_reward(mode="auto")

        if done:
            assert action in self.valid_actions, "Terminate action is not valid but was chosen."
            self.prev_reward, self.prev_reward_kind = self.calculate_reward(mode="exact")
            reward_val = self.prev_reward
        else:
            current_state_flags = self._get_compilation_state_flags()
            new_val, new_kind = self.calculate_reward(mode="auto")
            delta_reward = new_val - self.prev_reward
            reward_kind_changed = self.prev_reward_kind != new_kind
            state_changed = any(
                not before and after for before, after in zip(previous_state_flags, current_state_flags, strict=True)
            )

            if reward_kind_changed or state_changed:
                delta_reward = 0.0

            if not isclose(delta_reward, 0.0, abs_tol=1e-12):
                reward_val = self.reward_scale * delta_reward
            elif reward_kind_changed or state_changed:
                reward_val = 0.0
            else:
                reward_val = self.no_effect_penalty
            self.prev_reward, self.prev_reward_kind = new_val, new_kind

        obs = self._create_observation()
        self._log_step_reward(step_index, action_name, reward_val, done)
        return obs, reward_val, done, False, {}

    def calculate_reward(self, qc: QuantumCircuit | None = None, mode: str = "auto") -> tuple[float, str]:
        """Compute the reward for a circuit and report whether it was computed exactly or approximately.

        This environment supports two evaluation regimes for selected figures of merit:

        - **Exact**: uses the calibration-aware implementation on the full circuit/device
        (e.g., uses the device Target calibration data as-is).
        - **Approximate**: uses a transpile-based proxy:
        the circuit is transpiled to the device's basis gates and the resulting basis-gate
        counts are combined with cached **per-basis-gate** calibration statistics
        (error rates and durations) to estimate the metric. This approximation ignores
        additional mapping/routing overhead beyond what is reflected in the transpiled
        basis-gate counts.

        Args:
            qc:
                Circuit to evaluate. If ``None``, evaluates the environment's current state.
            mode:
                Selects how the method chooses between exact and approximate evaluation:

                - ``"auto"`` (default): compute the exact metric if the circuit is already
                **native and mapped** for the device; otherwise compute the approximate metric.
                - ``"exact"``: always compute the exact, calibration-aware metric.
                - ``"approx"``: always compute the approximate, transpile-based proxy.

        Returns:
            A pair ``(value, kind)`` where:

            - ``value`` is the scalar reward value (typically in ``[0, 1]`` for EF/ESP).
            - ``kind`` is ``"exact"`` or ``"approx"`` indicating which regime was used.
        """
        if qc is None:
            qc = self.state

        # Reward functions that are always computed exactly.
        if self.reward_function not in {"expected_fidelity", "estimated_success_probability"}:
            if self.reward_function == "critical_depth":
                return crit_depth(qc), "exact"
            if self.reward_function == "estimated_hellinger_distance":
                return estimated_hellinger_distance(qc, self.device, self.hellinger_model), "exact"
            # Fallback for other unknown / not-yet-implemented reward functions:
            logger.warning(
                "Reward function '%s' is not supported in PredictorEnv. Returning 0.0 as a fallback reward.",
                self.reward_function,
            )
            return 0.0, "exact"
 
        reward_layout = cast("TranspileLayout | Layout | None", getattr(qc, "_layout", None))
        if reward_layout is None:
            # use the env layout if the circuit has no attached layout 
            # (e.g., if it's an intermediate state or a newly exported copy)
            reward_layout = self.layout

        # Dual-path evaluation (exact vs. approximate) for EF / ESP.
        if mode == "exact":
            kind = "exact"
        elif mode == "approx":
            kind = "approx"
        else:  # "auto"
            only_native = self.is_circuit_synthesized(qc)
            laid_out = self.is_circuit_laid_out(qc, reward_layout) if reward_layout is not None else False
            mapped = self.is_circuit_routed(qc, CouplingMap(self.device.build_coupling_map())) if laid_out else False

            kind = "exact" if (only_native and laid_out and mapped) else "approx"

        if kind == "exact":
            exact_qc = (
                qc if reward_layout is None or getattr(qc, "_layout", None) is not None else self.export_circuit(qc)
            )
            if self.reward_function == "expected_fidelity":
                return expected_fidelity(exact_qc, self.device), "exact"

            return estimated_success_probability(exact_qc, self.device), "exact"

        # Approximate metrics use per-basis-gate averages cached from device calibration
        self._ensure_device_averages_cached()

        if self.reward_function == "expected_fidelity":
            val = approx_expected_fidelity(
                qc,
                device=self.device,
                error_rates=self.err_by_gate,
            )
            return val, "approx"

        feats = calc_supermarq_features(qc)

        val = approx_estimated_success_probability(
            qc,
            device=self.device,
            error_rates=self.err_by_gate,
            gate_durations=self.dur_by_gate,
            tbar=self.tbar,
            par_feature=float(feats.parallelism),
            liv_feature=float(feats.liveness),
            n_qubits=int(qc.num_qubits),
        )
        return val, "approx"

    def render(self) -> None:
        """Renders the current state."""
        print(self.state.draw())

    def reset(
        self,
        qc: Path | str | QuantumCircuit | None = None,
        seed: int | None = None,
        options: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Resets the environment to the given state or a random state.

        Arguments:
            qc: The quantum circuit to be compiled or the path to a qasm file containing the quantum circuit. Defaults to None.
            seed: The seed to be used for the random number generator. Defaults to None.
            options: Additional options. Defaults to None.

        Returns:
            The initial state and additional information.
        """
        super().reset(seed=seed)

        if isinstance(qc, QuantumCircuit):
            self.state = qc
            self.filename = ""
            self.current_circuit_name = qc.name or "<unnamed>"
        elif qc:
            self.state = QuantumCircuit.from_qasm_file(str(qc))
            self.filename = str(qc)
            self.current_circuit_name = Path(str(qc)).stem
        else:
            self.state, self.filename = get_state_sample(self.device.num_qubits, self.path_training_circuits, self.rng)
            self.current_circuit_name = Path(self.filename).stem

        self.state = _clear_circuit_layout(self.state)

        self.action_space = Discrete(len(self.action_set.keys()))
        self.num_steps = 0
        self.used_actions = []
        self.episode_count += 1

        self.layout = None
        self.compilation_state_flags = None

        self.valid_actions = self.determine_valid_actions_for_state()

        self.error_occurred = False

        self.prev_reward = None
        self.prev_reward_kind = None

        self.num_qubits_uncompiled_circuit = self.state.num_qubits
        self.has_parameterized_gates = len(self.state.parameters) > 0
        logger.info("Starting episode %d with circuit=%s", self.episode_count, self.current_circuit_name)

        return self._create_observation(), {}

    def action_masks(self) -> list[bool]:
        """Returns a list of valid actions for the current state."""
        action_mask = [action in self.valid_actions for action in self.action_set]

        # TKET layout/optimization actions must not run after a Qiskit layout has been set.
        # TKET routing actions are explicitly supported through preprocessing.
        if self.layout is not None:
            action_mask = [
                action_mask[i]
                and (self.action_set[i].origin != CompilationOrigin.TKET or i in self.actions_routing_indices)
                for i in range(len(action_mask))
            ]

        if _contains_operation_wider_than_two_qubits(self.state):
            # some TKET actions do not support gates wider than 2 qubits (only after synthesis)
            action_mask = [
                action_mask[i]
                and not (
                    self.action_set[i].origin == CompilationOrigin.TKET
                    and (i in self.actions_layout_indices or i in self.actions_routing_indices)
                )
                for i in range(len(action_mask))
            ]

        if self.has_parameterized_gates or self.layout is not None:
            # remove all actions that are from "origin"=="bqskit" because they are not supported for parameterized gates
            # or after layout since using BQSKit after a layout is set may result in an error
            action_mask = [
                action_mask[i] and self.action_set[i].origin != CompilationOrigin.BQSKIT
                for i in range(len(action_mask))
            ]

        # only allow VF2PostLayout if "ibm" is in the device name
        if "ibm" not in self.device.description:
            action_mask = [
                action_mask[i] and self.action_set[i].name != "VF2PostLayout" for i in range(len(action_mask))
            ]
        return action_mask

    def apply_action(self, action_index: int) -> QuantumCircuit | None:
        """Applies the given action to the current state and returns the altered state.

        Arguments:
            action_index: The index of the action to be applied, which must be in the action set.

        Returns:
            The altered quantum circuit after applying the action, or None if the action is to terminate the compilation.

        Raises:
            ValueError: If the action index is not in the action set or if the action origin is not supported.
        """
        if action_index not in self.action_set:
            msg = f"Action {action_index} not supported."
            raise ValueError(msg)

        action = self.action_set[action_index]

        if action.name == "terminate":
            return self.state
        if action.origin == CompilationOrigin.QISKIT:
            return self._apply_qiskit_action(action, action_index)
        if action.origin == CompilationOrigin.TKET:
            return self._apply_tket_action(action, action_index)
        if action.origin == CompilationOrigin.BQSKIT:
            return self._apply_bqskit_action(action, action_index)
        msg = f"Origin {action.origin} not supported."

        raise ValueError(msg)

    def fom_aware_compile(
        self, action: Action, device: Target | None, qc: QuantumCircuit, max_iteration: int = 20
    ) -> tuple[QuantumCircuit, dict[str, Any] | None]:
        """Run a stochastic pass multiple times optimizing for the given figure of merit.

        Args:
            action: The action containing the transpile pass.
            device: The compilation target device.
            qc: The input quantum circuit.
            max_iteration: Number of iterations to run the pass.

        Returns:
            A tuple of the best circuit found and its property set (if available).
        """
        assert device is not None
        best_result: QuantumCircuit | None = None
        best_property_set: dict[str, Any] | None = None
        best_fom = -1.0
        best_swap_count = float("inf")  # for fallback

        assert callable(action.transpile_pass), "Mapping action should be callable"
        pass_factory = cast("Callable[[Target], list[Task]]", action.transpile_pass)
        for i in range(max_iteration):
            passes = pass_factory(device)
            pm = PassManager(passes)
            try:
                out_circ = pm.run(qc)
                prop_set = dict(pm.property_set)

                try:
                    # For fidelity-based metrics, do a cheap "lookahead" synthesis step:
                    # routing may have introduced non-native SWAPs, so we translate the
                    # circuit into the device's native basis before evaluating the metric.
                    #
                    # Note:
                    # - BasisTranslator *only* performs basis conversion; it does not optimize.
                    # - This isolates the effect of mapping (inserted SWAPs) on fidelity
                    #   without conflating it with further optimizations.

                    synth_pass = PassManager([
                        BasisTranslator(StandardEquivalenceLibrary, target_basis=device.operation_names)
                    ])
                    synth_circ = synth_pass.run(out_circ.copy())
                    fom, _ = self.calculate_reward(synth_circ)

                    if fom > best_fom:
                        best_fom = fom
                        best_result = out_circ
                        best_property_set = prop_set

                except (QiskitError, TranspilerError, RuntimeError, ValueError, TypeError) as e:
                    logger.warning(f"[Fallback to SWAP counts] Synthesis or fidelity computation failed: {e}")
                    swap_count = out_circ.count_ops().get("swap", 0)  # ty: ignore[no-matching-overload]
                    if best_result is None or swap_count < best_swap_count:
                        best_swap_count = swap_count
                        best_result = out_circ
                        best_property_set = prop_set

            except Exception:
                logger.exception(f"[Error] Pass failed at iteration {i + 1}")
                continue

        if best_result is not None:
            return best_result, best_property_set
        logger.error("All attempts failed.")
        return qc, None

    def _apply_qiskit_action(self, action: Action, action_index: int) -> QuantumCircuit:
        pm_property_set: dict[str, Any] | None = None
        if getattr(action, "stochastic", False):  # Wrap stochastic action to optimize for the used figure of merit
            altered_qc, pm_property_set = self.fom_aware_compile(
                action,
                self.device,
                self.state,
                max_iteration=self.max_iter,
            )
        else:
            if action.name == "Opt2qBlocks_preserve" and isinstance(action, DeviceDependentAction):
                passes_ = action.transpile_pass(
                    self.device.operation_names,
                    CouplingMap(self.device.build_coupling_map()) if self.layout else None,
                )
                passes = cast("list[Task]", passes_)
                pm = PassManager(passes)
                altered_qc = pm.run(self.state)
                pm_property_set = dict(pm.property_set) if hasattr(pm, "property_set") else None
            else:
                transpile_pass_ = (
                    cast("Callable[[Target], list[Task]]", action.transpile_pass)(self.device)
                    if callable(action.transpile_pass)
                    else action.transpile_pass
                )
                transpile_pass = cast("list[Task]", transpile_pass_)
                pm = PassManager(transpile_pass)
                altered_qc = pm.run(self.state)
                pm_property_set = dict(pm.property_set) if hasattr(pm, "property_set") else None

        if action_index in (
            self.actions_layout_indices + self.actions_mapping_indices + self.actions_final_optimization_indices
        ):
            altered_qc = self._handle_qiskit_layout_postprocessing(action, pm_property_set, altered_qc)
        elif (
            action_index in self.actions_routing_indices
            and self.layout is not None
            and pm_property_set is not None
            and pm_property_set.get("final_layout") is not None
        ):
            self.layout.final_layout = pm_property_set["final_layout"]

        # BasisTranslator errors on unitary gates; decompose them immediately so
        # the circuit is always in a consistent state after a Qiskit action.
        if altered_qc.count_ops().get("unitary"):  # ty: ignore[invalid-argument-type]
            altered_qc = altered_qc.decompose(gates_to_decompose="unitary")
        elif altered_qc.count_ops().get("clifford"):  # ty: ignore[invalid-argument-type]
            altered_qc = altered_qc.decompose(gates_to_decompose="clifford")
        return altered_qc

    def _handle_qiskit_layout_postprocessing(
        self,
        action: Action,
        pm_property_set: dict[str, Any] | None,
        altered_qc: QuantumCircuit,
    ) -> QuantumCircuit:
        if not pm_property_set:
            return altered_qc
        if action.name == "VF2PostLayout":
            assert pm_property_set["VF2PostLayout_stop_reason"] is not None
            post_layout = pm_property_set.get("post_layout")
            if post_layout:
                assert self.layout is not None
                altered_qc, _ = postprocess_vf2postlayout(altered_qc, post_layout, self.layout)
        elif action.name == "VF2Layout":
            if pm_property_set["VF2Layout_stop_reason"] != VF2LayoutStopReason.SOLUTION_FOUND:
                logger.warning(
                    "VF2Layout pass did not find a solution. Reason: %s",
                    pm_property_set["VF2Layout_stop_reason"],
                )
        else:
            assert pm_property_set["layout"]

        layout = pm_property_set.get("layout")
        if layout is not None:
            orig = pm_property_set.get("original_qubit_indices")
            final = pm_property_set.get("final_layout")

            self.layout = TranspileLayout(
                initial_layout=layout,
                input_qubit_mapping=cast("dict[Any, int]", orig),
                final_layout=final,
                _output_qubit_list=altered_qc.qubits,
                _input_qubit_count=self.num_qubits_uncompiled_circuit,
            )

        if self.layout is not None and pm_property_set.get("final_layout"):
            self.layout.final_layout = pm_property_set["final_layout"]
        return altered_qc

    def _apply_tket_action(self, action: Action, action_index: int) -> QuantumCircuit:
        tket_qc = qiskit_to_tk(self.state, preserve_param_uuid=True)

        if action.name == "NoiseAwarePlacement":
            if self.node_err is None or self.edge_err is None or self.readout_err is None:
                self.node_err, self.edge_err, self.readout_err = prepare_noise_data(self.device)
            assert callable(action.transpile_pass)
            placement_pass_factory = cast(
                "Callable[[Target, Any, Any, Any], list[TketBasePass | PreProcessTKETRoutingAfterQiskitLayout]]",
                action.transpile_pass,
            )
            transpile_pass = placement_pass_factory(self.device, self.node_err, self.edge_err, self.readout_err)
        else:
            transpile_pass = (
                cast(
                    "Callable[[Target], list[TketBasePass | PreProcessTKETRoutingAfterQiskitLayout]]",
                    action.transpile_pass,
                )(self.device)
                if callable(action.transpile_pass)
                else action.transpile_pass
            )

        assert isinstance(transpile_pass, list)

        if action_index in self.actions_layout_indices:
            if not transpile_pass:
                logger.warning(
                    "Placement failed (%s): no placement pass provided. Falling back to original circuit.", action.name
                )
                return tk_to_qiskit(tket_qc, replace_implicit_swaps=True)

            p0 = transpile_pass[0]
            if not isinstance(p0, Placement):
                logger.warning(
                    "Placement failed (%s): expected Placement pass, got %s. Falling back to original circuit.",
                    action.name,
                    type(p0).__name__,
                )
                return tk_to_qiskit(tket_qc, replace_implicit_swaps=True)

            try:
                placement = p0.get_placement_map(tket_qc)
            except (RuntimeError, TypeError, ValueError) as e:
                logger.warning("Placement failed (%s): %s. Falling back to original circuit.", action.name, e)
                return tk_to_qiskit(tket_qc, replace_implicit_swaps=True)
            else:
                qc_tmp = tk_to_qiskit(tket_qc, replace_implicit_swaps=True)

                qiskit_mapping = {
                    qc_tmp.qubits[i]: placement[list(placement.keys())[i]].index[0] for i in range(len(placement))
                }
                layout = Layout(qiskit_mapping)

                pm = PassManager([
                    SetLayout(layout),
                    FullAncillaAllocation(coupling_map=CouplingMap(self.device.build_coupling_map())),
                    EnlargeWithAncilla(),
                    ApplyLayout(),
                ])
                altered_qc = pm.run(qc_tmp)

                layout2 = pm.property_set.get("layout")
                assert isinstance(layout2, Layout)

                self.layout = TranspileLayout(
                    initial_layout=layout2,
                    input_qubit_mapping=pm.property_set["original_qubit_indices"],
                    final_layout=pm.property_set["final_layout"],
                    _output_qubit_list=altered_qc.qubits,
                    _input_qubit_count=self.num_qubits_uncompiled_circuit,
                )
                return altered_qc

        else:
            passes = cast("list[TketBasePass | PreProcessTKETRoutingAfterQiskitLayout]", transpile_pass)
            for pass_ in passes:
                pass_.apply(tket_qc)

        qbs = tket_qc.qubits
        tket_qc.rename_units({qbs[i]: Qubit("q", i) for i in range(len(qbs))})
        altered_qc = tk_to_qiskit(tket_qc, replace_implicit_swaps=True)

        if action_index in self.actions_routing_indices:
            assert self.layout is not None
            self.layout.final_layout = final_layout_pytket_to_qiskit(
                tket_qc,
                _layout_output_qubits(self.layout),
            )

        return altered_qc

    def _apply_bqskit_action(self, action: Action, action_index: int) -> QuantumCircuit:
        """Applies the given BQSKit action to the current state and returns the altered state.

        Arguments:
            action: The BQSKit action to be applied.
            action_index: The index of the action in the action set.

        Returns:
            The altered quantum circuit after applying the action.

        Raises:
            ValueError: If the action index is not in the action set or if the action origin is not supported.
        """
        bqskit_qc = qiskit_to_bqskit(self.state)
        if action_index in self.actions_opt_indices:
            transpile = cast("Callable[[Circuit], Circuit]", action.transpile_pass)
            bqskit_compiled_qc = transpile(bqskit_qc)
        elif action_index in self.actions_synthesis_indices:
            factory = cast("Callable[[Target], Callable[[Circuit], Circuit]]", action.transpile_pass)
            bqskit_compiled_qc = factory(self.device)(bqskit_qc)
        elif action_index in self.actions_mapping_indices:
            factory = cast(
                "Callable[[Target], Callable[[Circuit], tuple[Circuit, tuple[int, ...], tuple[int, ...]]]]",
                action.transpile_pass,
            )
            bqskit_compiled_qc, initial, final = factory(self.device)(bqskit_qc)
            compiled_qiskit_qc = bqskit_to_qiskit(bqskit_compiled_qc)
            self.layout = final_layout_bqskit_to_qiskit(initial, final, compiled_qiskit_qc, self.state)
            return compiled_qiskit_qc
        else:
            msg = f"Unhandled BQSKit action index: {action_index}"
            raise ValueError(msg)

        return bqskit_to_qiskit(bqskit_compiled_qc)

    def is_circuit_laid_out(self, circuit: QuantumCircuit, layout: TranspileLayout | Layout) -> bool:
        """True if every logical qubit in the circuit has a physical assignment."""
        if isinstance(layout, TranspileLayout):
            final_positions = layout.final_index_layout()
            output_qubits = _layout_output_qubits(layout)
            if len(final_positions) != _layout_input_qubit_count(layout):
                return False
            if list(circuit.qubits) == output_qubits:
                return all(0 <= index < len(output_qubits) for index in final_positions)
            if layout.final_layout is not None:
                return all(0 <= index < len(circuit.qubits) for index in final_positions)
            return False

        v2p = layout.get_virtual_bits()
        return all(q in v2p for q in circuit.qubits)

    def is_circuit_synthesized(self, circuit: QuantumCircuit) -> bool:
        """Check if the circuit uses only native gates of the device.

        Verifies that every gate name in the circuit is present in
        ``device.operation_names``, equivalent to the ``GatesInBasis`` pass.

        Args:
            circuit: QuantumCircuit to check.

        Returns:
            True if all gates are native to the device.
        """
        native_names = set(self.device.operation_names)
        return all(
            instr.operation.name in native_names or instr.operation.name in ("barrier", "measure")
            for instr in circuit.data
        )

    def is_circuit_routed(self, circuit: QuantumCircuit, coupling_map: CouplingMap) -> bool:
        """Check if a circuit is fully routed to the device, including directionality.

        A circuit is considered routed if all two-qubit gates are on qubit pairs
        that exist as directed edges in the device coupling map.

        After a layout pass the circuit's qubits are already physical qubits, so
        ``circuit.find_bit(q).index`` gives the physical index directly —
        consistent with how ``reward.py`` looks up gate calibrations.

        Args:
            circuit: QuantumCircuit to check.
            coupling_map: CouplingMap of the target device.

        Returns:
            True if fully routed, False otherwise.
        """
        directed_edges = set(coupling_map.get_edges())
        for instr in circuit.data:
            if len(instr.qubits) == 2:
                q0 = circuit.find_bit(instr.qubits[0]).index
                q1 = circuit.find_bit(instr.qubits[1]).index
                if (q0, q1) not in directed_edges:
                    return False
        return True

    def determine_valid_actions_for_state(self) -> list[int]:
        """Determine valid actions based on circuit state: synthesized, mapped, routed."""
        synthesized, laid_out, routed = self._get_compilation_state_flags()

        actions = []
        # Initial state
        if not synthesized and not laid_out and not routed:
            if self.mdp == "flexible":
                actions.extend(self.actions_synthesis_indices)
                actions.extend(self.actions_mapping_indices)
                actions.extend(self.actions_layout_indices)
                actions.extend(self.actions_opt_indices)
            if self.mdp == "hybrid":
                actions.extend(self.actions_synthesis_indices)
                actions.extend(self.actions_mapping_indices)
                actions.extend(self.actions_layout_indices)
                actions.extend(self.actions_opt_indices)
            if self.mdp == "paper":
                actions.extend(self.actions_synthesis_indices)
                actions.extend(self.actions_opt_indices)
            if self.mdp == "thesis":
                actions.extend(self.actions_synthesis_indices)
                actions.extend(self.actions_opt_indices)

        if synthesized and not laid_out and not routed:
            if self.mdp == "flexible":
                actions.extend(self.actions_mapping_indices)
                actions.extend(self.actions_layout_indices)
                actions.extend(self.actions_opt_indices)
            if self.mdp == "hybrid":
                actions.extend(self.actions_mapping_indices)
                actions.extend(self.actions_layout_indices)
                actions.extend(self.actions_opt_indices)
            if self.mdp == "paper":
                actions.extend(self.actions_mapping_indices)
                actions.extend(self.actions_layout_indices)
                actions.extend(self.actions_opt_indices)
            if self.mdp == "thesis":
                actions.extend(self.actions_mapping_indices)
                actions.extend(self.actions_layout_indices)
                actions.extend(self.actions_opt_indices)

        # Not *depicted* in paper; necessary because optimization can destroy the native gate set
        if not synthesized and laid_out and not routed:
            if self.mdp == "flexible":
                actions.extend(self.actions_synthesis_indices)
                actions.extend(self.actions_routing_indices)
                actions.extend(self.actions_opt_indices)
            if self.mdp == "hybrid":
                actions.extend(self.actions_synthesis_indices)
                actions.extend(self.actions_routing_indices)
                actions.extend(self.actions_structure_preserving_indices)
            if self.mdp == "paper":
                actions.extend(self.actions_synthesis_indices)
                actions.extend(self.actions_routing_indices)
                actions.extend(self.actions_opt_indices)
            if self.mdp == "thesis":
                actions.extend(self.actions_synthesis_indices)
                actions.extend(self.actions_structure_preserving_indices)

        # Not *depicted* in paper; necessary because of layout-only passes
        if synthesized and laid_out and not routed:
            if self.mdp == "flexible":
                actions.extend(self.actions_routing_indices)
                actions.extend(self.actions_opt_indices)
            if self.mdp == "hybrid":
                actions.extend(self.actions_routing_indices)
                actions.extend(self.actions_structure_preserving_indices)
            if self.mdp == "paper":
                actions.extend(self.actions_routing_indices)
            if self.mdp == "thesis":
                actions.extend(self.actions_routing_indices)

        # Not *depicted* in paper; necessary because routing can insert non-native SWAPs
        if not synthesized and laid_out and routed:
            if self.mdp == "flexible":
                actions.extend(self.actions_synthesis_indices)
                actions.extend(self.actions_opt_indices)
            if self.mdp == "hybrid":
                actions.extend(self.actions_synthesis_indices)
                actions.extend(self.actions_structure_preserving_indices)
            if self.mdp == "paper":
                actions.extend(self.actions_synthesis_indices)
                actions.extend(self.actions_opt_indices)
            if self.mdp == "thesis":
                actions.extend(self.actions_synthesis_indices)
                actions.extend(self.actions_structure_preserving_indices)

        # Final state
        if synthesized and laid_out and routed:
            if self.mdp == "flexible":
                actions.extend([self.action_terminate_index])
                actions.extend(self.actions_opt_indices)
            if self.mdp == "hybrid":
                actions.extend([self.action_terminate_index])
                actions.extend(self.actions_structure_preserving_indices)
                actions.extend(self.actions_final_optimization_indices)
            if self.mdp == "paper":
                actions.extend([self.action_terminate_index])
                actions.extend(self.actions_opt_indices)
            if self.mdp == "thesis":
                actions.extend([self.action_terminate_index])
                actions.extend(self.actions_structure_preserving_indices)
                actions.extend(self.actions_final_optimization_indices)

        return actions

    def _ensure_device_averages_cached(self) -> None:
        """Cache per-basis-gate averages for error, duration, and a coherence scale."""
        if self.dev_avgs_cached:
            return

        err_by_gate, dur_by_gate, tbar = compute_device_averages_from_target(self.device)

        self.err_by_gate = err_by_gate
        self.dur_by_gate = dur_by_gate
        self.tbar = tbar
        self.dev_avgs_cached = True
