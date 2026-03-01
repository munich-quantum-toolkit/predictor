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
from typing import TYPE_CHECKING, Any

from pytket._tket.passes import BasePass as TketBasePass  # noqa: PLC2701

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from bqskit import Circuit
    from qiskit.passmanager.base_tasks import Task
    from qiskit.transpiler import Target

    from mqt.predictor.reward import figure_of_merit
    from mqt.predictor.rl.actions import Action


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
from qiskit import QuantumCircuit
from qiskit.passmanager.flow_controllers import DoWhileController
from qiskit.transpiler import CouplingMap, PassManager, TranspileLayout
from qiskit.transpiler.passes import (
    CheckMap,
    GatesInBasis,
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
    PreProcessTKETRoutingAfterQiskitLayout,
    final_layout_bqskit_to_qiskit,
    final_layout_pytket_to_qiskit,
    postprocess_vf2postlayout,
)
from mqt.predictor.utils import calc_supermarq_features

logger = logging.getLogger("mqt-predictor")


class PredictorEnv(Env):
    """Predictor environment for reinforcement learning."""

    def __init__(
        self,
        device: Target,
        reward_function: figure_of_merit = "expected_fidelity",
        path_training_circuits: Path | None = None,
        reward_scale: float = 1.0,
        no_effect_penalty: float = -0.001,
    ) -> None:
        """Initializes the PredictorEnv object.

        Arguments:
            device: The target device to be used for compilation.
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
        self.actions_synthesis_indices: list[int] = []
        self.actions_layout_indices: list[int] = []
        self.actions_routing_indices: list[int] = []
        self.actions_mapping_indices: list[int] = []
        self.actions_opt_indices: list[int] = []
        self.actions_final_optimization_indices: list[int] = []
        self.valid_actions: list[int] = []
        self.used_actions: list[str] = []
        self.device = device

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

        spaces = {
            "num_qubits": Discrete(self.device.num_qubits + 1),
            "depth": Discrete(1000000),
            "program_communication": Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "critical_depth": Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "entanglement_ratio": Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "parallelism": Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "liveness": Box(low=0, high=1, shape=(1,), dtype=np.float32),
        }
        self.observation_space = Dict(spaces)
        self.filename = ""
        self.reward_scale = reward_scale
        self.no_effect_penalty = no_effect_penalty
        self.prev_reward: float | None = None
        self.prev_reward_kind: str | None = None
        self._err_by_gate: dict[str, float] = {}
        self._dur_by_gate: dict[str, float] = {}
        self._tbar: float | None = None
        self._dev_avgs_cached = False
        self.state: QuantumCircuit = QuantumCircuit()

        self.error_occurred = False

    def _apply_and_update(self, action: int) -> QuantumCircuit | None:
        """Apply an action, normalize the circuit, and update internal state."""
        altered_qc = self.apply_action(action)
        if altered_qc is None:
            return None

        # in case the Qiskit.QuantumCircuit has unitary or clifford or u gates in it, decompose them (because otherwise qiskit will throw an error when applying the BasisTranslator
        for gate_type in ("unitary", "clifford"):
            if altered_qc.count_ops().get(gate_type):  # ty: ignore[invalid-argument-type]
                altered_qc = altered_qc.decompose(gates_to_decompose=gate_type)

        self.state = altered_qc
        self.num_steps += 1
        self.valid_actions = self.determine_valid_actions_for_state()
        if not self.valid_actions:
            msg = "No valid actions left."
            raise RuntimeError(msg)

        return altered_qc

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
        self.used_actions.append(str(self.action_set[action].name))
        logger.info("Applying %s", self.action_set[action].name)

        altered_qc = self._apply_and_update(action)
        if altered_qc is None:
            return create_feature_dict(self.state), 0.0, True, False, {}

        done = action == self.action_terminate_index

        if self.reward_function == "estimated_hellinger_distance":
            reward_val = self.calculate_reward(mode="exact")[0] if done else 0.0
            self.state._layout = self.layout  # noqa: SLF001
            return create_feature_dict(self.state), reward_val, done, False, {}

        # Lazy init: compute prev_reward only once per episode (or if missing)
        if self.prev_reward is None:
            self.prev_reward, self.prev_reward_kind = self.calculate_reward(mode="auto")

        if done:
            self.prev_reward, self.prev_reward_kind = self.calculate_reward(mode="exact")
            reward_val = self.prev_reward
        else:
            new_val, new_kind = self.calculate_reward(mode="auto")
            delta_reward = new_val - self.prev_reward

            if self.prev_reward_kind != new_kind:
                delta_reward = 0.0

            reward_val = (
                self.reward_scale * delta_reward
                if not isclose(delta_reward, 0.0, abs_tol=1e-12)
                else self.no_effect_penalty
            )
            self.prev_reward, self.prev_reward_kind = new_val, new_kind

        self.state._layout = self.layout  # noqa: SLF001
        return create_feature_dict(self.state), reward_val, done, False, {}

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

        # Dual-path evaluation (exact vs. approximate) for EF / ESP.
        if mode == "exact":
            kind = "exact"
        elif mode == "approx":
            kind = "approx"
        else:  # "auto"
            check_nat_gates = GatesInBasis(basis_gates=self.device.operation_names)
            check_nat_gates(qc)
            only_native = bool(check_nat_gates.property_set["all_gates_in_basis"])

            check_mapping = CheckMap(coupling_map=self.device.build_coupling_map())
            check_mapping(qc)
            mapped = bool(check_mapping.property_set["is_swap_mapped"])

            kind = "exact" if (only_native and mapped) else "approx"

        if kind == "exact":
            if self.reward_function == "expected_fidelity":
                return expected_fidelity(qc, self.device), "exact"

            return estimated_success_probability(qc, self.device), "exact"

        # Approximate metrics use per-basis-gate averages cached from device calibration
        self._ensure_device_averages_cached()

        if self.reward_function == "expected_fidelity":
            val = approx_expected_fidelity(
                qc,
                device=self.device,
                error_rates=self._err_by_gate,
            )
            return val, "approx"

        feats = calc_supermarq_features(qc)

        val = approx_estimated_success_probability(
            qc,
            device=self.device,
            error_rates=self._err_by_gate,
            gate_durations=self._dur_by_gate,
            tbar=self._tbar,
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
        elif qc:
            self.state = QuantumCircuit.from_qasm_file(str(qc))
        else:
            self.state, self.filename = get_state_sample(self.device.num_qubits, self.path_training_circuits, self.rng)

        self.action_space = Discrete(len(self.action_set.keys()))
        self.num_steps = 0
        self.used_actions = []

        self.layout = None

        self.valid_actions = self.actions_opt_indices + self.actions_synthesis_indices

        self.error_occurred = False

        self.prev_reward = None
        self.prev_reward_kind = None

        self.num_qubits_uncompiled_circuit = self.state.num_qubits
        self.has_parameterized_gates = len(self.state.parameters) > 0
        return create_feature_dict(self.state), {}

    def action_masks(self) -> list[bool]:
        """Returns a list of valid actions for the current state."""
        action_mask = [action in self.valid_actions for action in self.action_set]

        # it is not clear how tket will handle the layout, so we remove all actions that are from "origin"=="tket" if a layout is set
        if self.layout is not None:
            action_mask = [
                action_mask[i] and self.action_set[i].origin != CompilationOrigin.TKET for i in range(len(action_mask))
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

    def _apply_qiskit_action(self, action: Action, action_index: int) -> QuantumCircuit:
        if action.name == "QiskitO3" and isinstance(action, DeviceDependentAction):
            factory = cast("Callable[[list[str], CouplingMap | None], list[Task]]", action.transpile_pass)
            passes = factory(
                self.device.operation_names,
                CouplingMap(self.device.build_coupling_map()) if self.layout else None,
            )
            assert action.do_while is not None
            pm = PassManager([DoWhileController(passes, do_while=action.do_while)])
        else:
            if callable(action.transpile_pass):
                factory = cast("Callable[[Target], list[Task]]", action.transpile_pass)
                passes = factory(self.device)
            else:
                passes = cast("list[Task]", action.transpile_pass)
            pm = PassManager(passes)

        altered_qc = pm.run(self.state)

        if action_index in (
            self.actions_layout_indices + self.actions_mapping_indices + self.actions_final_optimization_indices
        ):
            altered_qc = self._handle_qiskit_layout_postprocessing(action, pm, altered_qc)

        elif action_index in self.actions_routing_indices and self.layout:
            self.layout.final_layout = pm.property_set["final_layout"]

        return altered_qc

    def _handle_qiskit_layout_postprocessing(
        self, action: Action, pm: PassManager, altered_qc: QuantumCircuit
    ) -> QuantumCircuit:
        if action.name == "VF2PostLayout":
            assert pm.property_set["VF2PostLayout_stop_reason"] is not None
            post_layout = pm.property_set["post_layout"]
            if post_layout:
                assert self.layout is not None
                altered_qc, _ = postprocess_vf2postlayout(altered_qc, post_layout, self.layout)
        elif action.name == "VF2Layout":
            assert pm.property_set["VF2Layout_stop_reason"] == VF2LayoutStopReason.SOLUTION_FOUND
            assert pm.property_set["layout"]
        else:
            assert pm.property_set["layout"]

        if pm.property_set["layout"]:
            self.layout = TranspileLayout(
                initial_layout=pm.property_set["layout"],
                input_qubit_mapping=pm.property_set["original_qubit_indices"],
                final_layout=pm.property_set["final_layout"],
                _output_qubit_list=altered_qc.qubits,
                _input_qubit_count=self.num_qubits_uncompiled_circuit,
            )
        return altered_qc

    def _apply_tket_action(self, action: Action, action_index: int) -> QuantumCircuit:
        tket_qc = qiskit_to_tk(self.state, preserve_param_uuid=True)
        if callable(action.transpile_pass):
            factory = cast("Callable[[Target], list[Task]]", action.transpile_pass)
            passes = factory(self.device)
        else:
            passes = cast("list[Task]", action.transpile_pass)
        for pass_ in passes:
            assert isinstance(pass_, TketBasePass | PreProcessTKETRoutingAfterQiskitLayout)
            pass_.apply(tket_qc)

        qbs = tket_qc.qubits
        tket_qc.rename_units({qbs[i]: Qubit("q", i) for i in range(len(qbs))})
        altered_qc = tk_to_qiskit(tket_qc, replace_implicit_swaps=True)

        if action_index in self.actions_routing_indices:
            assert self.layout is not None
            self.layout.final_layout = final_layout_pytket_to_qiskit(tket_qc, altered_qc)

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

    def determine_valid_actions_for_state(self) -> list[int]:
        """Determines and returns the valid actions for the current state."""
        check_nat_gates = GatesInBasis(basis_gates=self.device.operation_names)
        check_nat_gates(self.state)
        only_nat_gates = check_nat_gates.property_set["all_gates_in_basis"]

        if not only_nat_gates:
            actions = self.actions_synthesis_indices + self.actions_opt_indices
            if self.layout is not None:
                actions += self.actions_routing_indices
            return actions

        check_mapping = CheckMap(coupling_map=self.device.build_coupling_map())
        check_mapping(self.state)
        mapped = check_mapping.property_set["is_swap_mapped"]

        if mapped and self.layout is not None:  # The circuit is correctly mapped.
            return [self.action_terminate_index, *self.actions_opt_indices]

        if self.layout is not None:  # The circuit is not yet mapped but a layout is set.
            return self.actions_routing_indices

        # No layout applied yet
        return self.actions_mapping_indices + self.actions_layout_indices + self.actions_opt_indices

    def _ensure_device_averages_cached(self) -> None:
        """Cache per-basis-gate averages for error, duration, and a coherence scale."""
        if self._dev_avgs_cached:
            return

        err_by_gate, dur_by_gate, tbar = compute_device_averages_from_target(self.device)

        self._err_by_gate = err_by_gate
        self._dur_by_gate = dur_by_gate
        self._tbar = tbar
        self._dev_avgs_cached = True
