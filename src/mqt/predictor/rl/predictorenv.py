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
import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from gymnasium.spaces import Space
    from qiskit.transpiler import Layout, Target

    from mqt.predictor.reward import figure_of_merit

import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, Dict, Discrete
from joblib import load
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap, TranspileLayout

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
    get_actions_by_pass_type,
)
from mqt.predictor.rl.actions.bqskit_actions import is_bqskit_action_available, run_bqskit_action
from mqt.predictor.rl.actions.qiskit_actions import is_qiskit_action_available, run_qiskit_action
from mqt.predictor.rl.actions.tket_actions import is_tket_action_available, run_tket_action
from mqt.predictor.rl.helper import create_feature_dict, get_path_training_circuits, get_state_sample

logger = logging.getLogger("mqt-predictor")


class PredictorEnv(Env):
    """Predictor environment for reinforcement learning."""

    def __init__(
        self,
        device: Target,
        reward_function: figure_of_merit = "expected_fidelity",
        path_training_circuits: Path | None = None,
    ) -> None:
        """Initializes the PredictorEnv object.

        Args:
            device: The target device to be used for compilation.
            reward_function: The figure of merit to be used for the reward function. Defaults to "expected_fidelity".
            path_training_circuits: The path to the training circuits folder. Defaults to None, which uses the default path.

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
        self.actions_final_optimization_indices = []  # TODO: currently not used; will be improved by addressing issue https://github.com/munich-quantum-toolkit/predictor/issues/666
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
        for elem in action_dict[PassType.LAYOUT]:
            self.action_set[index] = elem
            self.actions_layout_indices.append(index)
            index += 1
        for elem in action_dict[PassType.ROUTING]:
            self.action_set[index] = elem
            self.actions_routing_indices.append(index)
            index += 1
        for elem in action_dict[PassType.OPT]:
            self.action_set[index] = elem
            self.actions_opt_indices.append(index)
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
        self.num_qubits_uncompiled_circuit = 0

        # Canonical layout state for the current circuit. It is mirrored to
        # QuantumCircuit.layout for callers, but kept here because some action
        # implementations do not preserve layout metadata across conversions.
        self.layout: TranspileLayout | None = None

        self.has_parameterized_gates = False
        self.rng = np.random.default_rng(10)

        spaces: dict[str, Space] = {
            "num_qubits": Discrete(128),
            "depth": Discrete(1000000),
            "program_communication": Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "critical_depth": Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "entanglement_ratio": Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "parallelism": Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "liveness": Box(low=0, high=1, shape=(1,), dtype=np.float32),
        }
        self.observation_space = Dict(spaces)
        self.filename = ""

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, bool, dict[Any, Any]]:
        """Executes the given action and returns the new state, the reward, whether the episode is done, whether the episode is truncated and additional information.

        Args:
            action: The action to be executed, represented by its index in the action set.

        Returns:
            A tuple containing the new state as a feature dictionary, the reward value, whether the episode is done, whether the episode is truncated, and additional information.

        Raises:
            RuntimeError: If no valid actions are left.
        """
        try:
            self.used_actions.append(str(self.action_set[action].name))
            altered_qc = self.apply_action(action)
        except Exception as exc:  # noqa: BLE001
            # Different passes may fail for various reasons (e.g., found no routing solution).
            self.error_occurred = True
            return (
                create_feature_dict(self.state),  # features
                0,  # reward
                False,  # terminated
                True,  # truncated
                {"Truncated because of error": f"{type(exc).__name__}: {exc}"},  # info
            )

        self.state: QuantumCircuit = altered_qc
        self.num_steps += 1

        self.state._layout = self.layout  # noqa: SLF001

        self.valid_actions = self.determine_valid_actions_for_state()
        if len(self.valid_actions) == 0:
            msg = "No valid actions left."
            raise RuntimeError(msg)

        if action == self.action_terminate_index:
            reward_val = self.calculate_reward()
            done = True
        else:
            reward_val = 0
            done = False

        obs = create_feature_dict(self.state)
        return obs, reward_val, done, False, {}

    def calculate_reward(self) -> float:
        """Calculates and returns the reward for the current state."""
        if self.reward_function == "expected_fidelity":
            return expected_fidelity(self.state, self.device)
        if self.reward_function == "estimated_success_probability":
            return estimated_success_probability(self.state, self.device)
        if self.reward_function == "estimated_hellinger_distance":
            return estimated_hellinger_distance(self.state, self.device, self.hellinger_model)
        if self.reward_function == "critical_depth":
            return crit_depth(self.state)
        msg = f"No implementation for reward function {self.reward_function}."
        raise NotImplementedError(msg)

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

        Args:
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
            self.state = QuantumCircuit.from_qasm_file(qc)
        else:
            self.state, self.filename = get_state_sample(self.device.num_qubits, self.path_training_circuits, self.rng)

        self.action_space = Discrete(len(self.action_set.keys()))
        self.num_steps = 0
        self.used_actions = []

        self.layout = None

        self.valid_actions = self.actions_synthesis_indices + self.actions_opt_indices

        self.error_occurred = False

        self.num_qubits_uncompiled_circuit = self.state.num_qubits
        self.has_parameterized_gates = len(self.state.parameters) > 0
        return create_feature_dict(self.state), {}

    def action_masks(self) -> list[bool]:
        """Build the boolean action mask exposed to the RL policy.

        ``self.valid_actions`` contains the structurally valid action indices
        for the current circuit state. This method expands that sparse list to
        one boolean per registered action and applies SDK-specific availability
        filters that depend on circuit features or the selected device.
        Terminate has no SDK origin and is accepted solely from the structural
        candidate list.

        Returns:
            A dense boolean mask ordered like ``self.action_set``.
        """
        has_layout = self.layout is not None
        valid_action_indices = set(self.valid_actions)
        action_mask: list[bool] = []

        for action_index in range(len(self.action_set)):
            if action_index not in valid_action_indices:
                action_mask.append(False)
                continue
                
            action = self.action_set[action_index]    
            if action.pass_type == PassType.TERMINATE:
                action_mask.append(True)
                continue
            if action.origin == CompilationOrigin.QISKIT:
                action_mask.append(is_qiskit_action_available(action, self.device))
            elif action.origin == CompilationOrigin.TKET:
                action_mask.append(is_tket_action_available(action=action, has_layout=has_layout))
            elif action.origin == CompilationOrigin.BQSKIT:
                action_mask.append(
                    is_bqskit_action_available(
                        has_layout=has_layout,
                        has_parameterized_gates=self.has_parameterized_gates,
                    )
                )
            else:
                msg = f"Origin {action.origin} not supported."
                raise ValueError(msg)

        return action_mask

    def apply_action(self, action_index: int) -> QuantumCircuit:
        """Applies the given action to the current state and returns the altered state.

        Args:
            action_index: The index of the action to be applied, which must be in the action set.

        Returns:
            The altered quantum circuit after applying the action.

        Raises:
            ValueError: If the action index is not in the action set or if the action cannot be applied.
        """
        if action_index not in self.action_set:
            msg = f"Action {action_index} not supported."
            raise ValueError(msg)

        action = self.action_set[action_index]

        if action.pass_type == PassType.TERMINATE:
            return self.state

        if action.origin == CompilationOrigin.QISKIT:
            altered_qc, self.layout = run_qiskit_action(
                action=action,
                circuit=self.state,
                device=self.device,
                layout=self.layout,
                input_qubit_count=self.num_qubits_uncompiled_circuit,
            )
        elif action.origin == CompilationOrigin.TKET:
            altered_qc, self.layout = run_tket_action(
                action=action,
                circuit=self.state,
                device=self.device,
                layout=self.layout,
            )
        elif action.origin == CompilationOrigin.BQSKIT:
            altered_qc, self.layout = run_bqskit_action(
                action=action,
                circuit=self.state,
                device=self.device,
                layout=self.layout,
            )
        else:
            msg = f"Origin {action.origin} not supported."
            raise ValueError(msg)

        return altered_qc

    def is_circuit_laid_out(self, circuit: QuantumCircuit, layout: TranspileLayout | Layout) -> bool:
        """True if every logical qubit in the circuit has a physical assignment."""
        if isinstance(layout, TranspileLayout):
            # Use final_layout if available; otherwise fallback to initial_layout
            layout = layout.final_layout or layout.initial_layout

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
        """Select structurally valid action indices for the current circuit state.

        The circuit is classified by compilation progress: synthesized to the
        target gate set, laid out to physical qubits, and routed against the
        directed coupling map. This method only determines which pass types can
        advance that state; SDK/backend-specific availability filters are applied later
        in ``action_masks``.

        Returns:
            Action indices whose pass type can be attempted from the current state.
        """
        synthesized = self.is_circuit_synthesized(self.state)
        laid_out = self.is_circuit_laid_out(self.state, self.layout) if self.layout else False
        # Routing is only allowed after layout
        routed = (
            self.is_circuit_routed(self.state, CouplingMap(self.device.build_coupling_map())) if laid_out else False
        )

        actions = []

        # Initial state
        if not synthesized and not laid_out and not routed:
            actions.extend(self.actions_synthesis_indices)
            actions.extend(self.actions_opt_indices)

        if synthesized and not laid_out and not routed:
            actions.extend(self.actions_mapping_indices)
            actions.extend(self.actions_layout_indices)
            actions.extend(self.actions_opt_indices)

        # Not *depicted* in paper; necessary because optimization can destroy the native gate set
        if not synthesized and laid_out and not routed:
            actions.extend(self.actions_synthesis_indices)
            actions.extend(self.actions_routing_indices)
            actions.extend(self.actions_opt_indices)

        # Not *depicted* in paper; necessary because of layout-only passes
        if synthesized and laid_out and not routed:
            actions.extend(self.actions_routing_indices)

        # Not *depicted* in paper; necessary because routing can insert non-native SWAPs
        if not synthesized and laid_out and routed:
            actions.extend(self.actions_synthesis_indices)
            actions.extend(self.actions_opt_indices)

        # Final state
        if synthesized and laid_out and routed:
            actions.extend([self.action_terminate_index])
            actions.extend(self.actions_opt_indices)

        return actions
