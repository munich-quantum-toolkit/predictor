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
    from qiskit.transpiler import Layout, Target

    from mqt.predictor.reward import figure_of_merit


import warnings
from math import isclose
from typing import cast

import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, Dict, Discrete
from joblib import load
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap, TranspileLayout
from torch_geometric.data import Data

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
    PassType,
    ensure_ai_routing_runtime_available,
    get_actions_by_pass_type,
    run_bqskit_action,
    run_qiskit_action,
    run_tket_action,
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
from mqt.predictor.utils import calc_supermarq_features, get_openqasm_gates_for_rl

logger = logging.getLogger("mqt-predictor")


FeatureValue = int | NDArray[np.float32]
FlatObservation = dict[str, FeatureValue]
EnvironmentObservation = FlatObservation | Data


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
        max_episode_steps: int | None = None,
        graph: bool = False,
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
            max_episode_steps: Optional hard cap on environment steps per episode. When reached without
                taking the terminate action, the episode ends with ``truncated=True``.
            graph: If True, observations are returned as PyG Data objects for GNN-based agents. Defaults to False.

        Raises:
            ValueError: If the reward function is "estimated_success_probability" and no calibration data is available for the device or if the reward function is "estimated_hellinger_distance" and no trained model is available for the device.
        """
        logger.info("Init env: " + reward_function)

        self.graph = graph
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

        if any(action.name in {"AIRouting", "AIRouting_opt"} for action in self.action_set.values()):
            ensure_ai_routing_runtime_available()

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
        self.reward_scale = reward_scale
        self.no_effect_penalty = no_effect_penalty
        self.max_episode_steps = max_episode_steps
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

    def _create_observation(self, qc: QuantumCircuit | None = None) -> EnvironmentObservation:
        """Create an observation directly from the actual circuit state."""
        circuit = self.state if qc is None else qc
        graph = self.graph if qc is None else False
        return create_feature_dict(circuit, graph=graph)

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

    def _episode_budget_exhausted(self) -> bool:
        """Return whether the current episode reached the configured step cap."""
        return self.max_episode_steps is not None and self.num_steps >= self.max_episode_steps

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

    def step(self, action: int) -> tuple[EnvironmentObservation, float, bool, bool, dict[Any, Any]]:
        """Run one environment step.

        This method:
            1. Evaluates the pre-step figure of merit value (using either the exact or approximate metric, depending on state).
            2. Applies the selected transpiler pass (the action).
            3. Computes a shaped step reward based on the change in the figure of merit.

        Reward design:
            - For non-terminal actions that stay within the same reward kind (``"approx"`` or ``"exact"``),
              the step reward is a scaled delta between the new and previous figure of merit values.
            - When an action changes the reward kind, the reward is neutral because the pre- and post-step
              figure of merit values are no longer directly comparable.
            - If the figure of merit does not change within the same reward kind, an (optional) small penalty
              is applied to discourage ineffective actions.
            - For the terminate action, the episode ends and the final reward is the exact (calibration-aware) figure of merit.
            - For ``estimated_hellinger_distance``, intermediate steps use sparse rewards and only the terminate action
              returns the exact figure of merit value.
        """
        info: dict[Any, Any] = {}
        truncated = False
        done = action == self.action_terminate_index
        action_name = str(self.action_set[action].name)
        step_index = self.num_steps + 1
        self.used_actions.append(action_name)
        logger.info("Episode %d step %d: applying %s", self.episode_count, step_index, action_name)

        if self.reward_function != "estimated_hellinger_distance" and self.prev_reward is None:
            self.prev_reward, self.prev_reward_kind = self.calculate_reward(mode="auto")

        # Apply the action and update the circuit state.
        self._apply_and_update(action)

        if self.reward_function == "estimated_hellinger_distance":
            reward_val = self.calculate_reward(mode="exact")[0] if done else 0.0
            if not done and self._episode_budget_exhausted():
                truncated = True
                info = {
                    "time_limit_reached": True,
                    "max_episode_steps": self.max_episode_steps,
                }
            self._log_step_reward(step_index, action_name, reward_val, done or truncated)
            return self._create_observation(), reward_val, done, truncated, info

        if done:
            # proper end of compilation gets rewarded with the exact figure of merit value
            assert action in self.valid_actions, "Terminate action is not valid but was chosen."
            self.prev_reward, self.prev_reward_kind = self.calculate_reward(mode="exact")
            reward_val = self.prev_reward
        else:
            # determine figure of merit delta wrt the previous step
            assert self.prev_reward is not None
            assert self.prev_reward_kind is not None
            new_val, new_kind = self.calculate_reward(mode="auto")
            delta_reward = new_val - self.prev_reward

            if self.prev_reward_kind != new_kind:
                # Switching estimator kind breaks direct comparability of the figure of merit values.
                reward_val = 0.0
            elif isclose(delta_reward, 0.0, abs_tol=1e-12):
                # No change in the figure of merit after applying the action -> penalty to discourage no-ops.
                reward_val = self.no_effect_penalty
            else:
                # Positive or negative change in the figure of merit compared to the previous step, scaled by the reward factor.
                reward_val = self.reward_scale * delta_reward

            # Cache the previous reward and kind for the next step.
            self.prev_reward, self.prev_reward_kind = new_val, new_kind

        if not done and self._episode_budget_exhausted():
            truncated = True
            info = {
                "time_limit_reached": True,
                "max_episode_steps": self.max_episode_steps,
            }
            logger.info(
                "Episode %d step %d: reached max_episode_steps=%d, truncating episode",
                self.episode_count,
                step_index,
                self.max_episode_steps,
            )

        obs = self._create_observation()
        self._log_step_reward(step_index, action_name, reward_val, done or truncated)
        return obs, reward_val, done, truncated, info

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
    ) -> tuple[EnvironmentObservation, dict[str, Any]]:
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

        if self.reward_function == "estimated_hellinger_distance":
            self.prev_reward = None
            self.prev_reward_kind = None
        else:
            self.prev_reward, self.prev_reward_kind = self.calculate_reward(mode="auto")

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

        if self.has_parameterized_gates:
            # remove all actions that are from "origin"=="bqskit" because they are not supported for parameterized gates
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
            altered_qc, self.layout = run_qiskit_action(
                action=action,
                action_index=action_index,
                state=self.state,
                device=self.device,
                layout=self.layout,
                num_qubits_uncompiled_circuit=self.num_qubits_uncompiled_circuit,
                max_iteration=self.max_iter,
                score_circuit=self.calculate_reward,
                actions_layout_indices=self.actions_layout_indices,
                actions_mapping_indices=self.actions_mapping_indices,
                actions_final_optimization_indices=self.actions_final_optimization_indices,
                actions_routing_indices=self.actions_routing_indices,
            )
            return altered_qc
        if action.origin == CompilationOrigin.TKET:
            altered_qc, self.layout = run_tket_action(
                action=action,
                action_index=action_index,
                state=self.state,
                device=self.device,
                layout=self.layout,
                num_qubits_uncompiled_circuit=self.num_qubits_uncompiled_circuit,
                actions_layout_indices=self.actions_layout_indices,
                actions_routing_indices=self.actions_routing_indices,
            )
            return altered_qc
        if action.origin == CompilationOrigin.BQSKIT:
            altered_qc, self.layout = run_bqskit_action(
                action=action,
                action_index=action_index,
                state=self.state,
                device=self.device,
                layout=self.layout,
                actions_opt_indices=self.actions_opt_indices,
                actions_synthesis_indices=self.actions_synthesis_indices,
                actions_layout_indices=self.actions_layout_indices,
                actions_mapping_indices=self.actions_mapping_indices,
                actions_routing_indices=self.actions_routing_indices,
            )
            return altered_qc
        msg = f"Origin {action.origin} not supported."

        raise ValueError(msg)

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
