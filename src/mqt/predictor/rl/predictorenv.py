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

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from bqskit import Circuit
    from pytket._tket.passes import BasePass as TketBasePass
    from pytket.circuit import Node
    from qiskit.passmanager.base_tasks import Task
    from qiskit.transpiler import Layout, Target

    from mqt.predictor.reward import figure_of_merit
    from mqt.predictor.rl.actions import Action
    from mqt.predictor.rl.parsing import (
        PreProcessTKETRoutingAfterQiskitLayout,
    )


import warnings
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
from qiskit.passmanager.flow_controllers import DoWhileController
from qiskit.transpiler import CouplingMap, Layout, PassManager, TranspileLayout
from qiskit.transpiler.passes import (
    ApplyLayout,
    BasisTranslator,
    CheckMap,
    EnlargeWithAncilla,
    FullAncillaAllocation,
    GatesInBasis,
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
from mqt.predictor.utils import get_openqasm_gates_without_u

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

        Arguments:
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
        self.actions_final_optimization_indices = []
        self.actions_structure_preserving_indices = []  # Actions that preserves the mapping and native gates
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

        gate_spaces = {g: Box(low=0, high=1, shape=(1,), dtype=np.float32) for g in get_openqasm_gates_without_u()}

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

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, bool, dict[Any, Any]]:
        """Executes the given action and returns the new state, the reward, whether the episode is done, whether the episode is truncated and additional information.

        Arguments:
            action: The action to be executed, represented by its index in the action set.

        Returns:
            A tuple containing the new state as a feature dictionary, the reward value, whether the episode is done, whether the episode is truncated, and additional information.

        Raises:
            RuntimeError: If no valid actions are left.
        """
        self.used_actions.append(str(self.action_set[action].name))
        altered_qc = self.apply_action(action)
        if not altered_qc:
            return (
                create_feature_dict(self.state),
                0,
                True,
                False,
                {},
            )

        self.state: QuantumCircuit = altered_qc

        self.num_steps += 1

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

        # in case the Qiskit.QuantumCircuit has unitary or u gates in it, decompose them (because otherwise qiskit will throw an error when applying the BasisTranslator
        if self.state.count_ops().get("unitary"):  # ty: ignore[invalid-argument-type]
            self.state = self.state.decompose(gates_to_decompose="unitary")
        elif self.state.count_ops().get("clifford"):  # ty: ignore[invalid-argument-type]
            self.state = self.state.decompose(gates_to_decompose="clifford")

        self.state._layout = self.layout  # noqa: SLF001

        return create_feature_dict(self.state), reward_val, done, False, {}

    def calculate_reward(self, qc: QuantumCircuit | None = None) -> float:
        """Calculates and returns the reward for either the current state or a quantum circuit (if provided)."""
        circuit = self.state if qc is None else qc
        if self.reward_function == "expected_fidelity":
            return expected_fidelity(circuit, self.device)
        if self.reward_function == "estimated_success_probability":
            return estimated_success_probability(circuit, self.device)
        if self.reward_function == "estimated_hellinger_distance":
            return estimated_hellinger_distance(circuit, self.device, self.hellinger_model)
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
            self.state = QuantumCircuit.from_qasm_file(qc)  # ty: ignore[invalid-argument-type]
        else:
            self.state, self.filename = get_state_sample(self.device.num_qubits, self.path_training_circuits, self.rng)

        self.action_space = Discrete(len(self.action_set.keys()))
        self.num_steps = 0
        self.used_actions = []

        self.layout = None

        self.valid_actions = self.actions_opt_indices + self.actions_synthesis_indices

        self.error_occurred = False

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
        for i in range(max_iteration):
            passes = cast("list[Task]", action.transpile_pass(device))
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
                    fom = self.calculate_reward(synth_circ)

                    if fom > best_fom:
                        best_fom = fom
                        best_result = out_circ
                        best_property_set = prop_set

                except Exception as e:
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
            if action.name in ["QiskitO3", "Opt2qBlocks_preserve"] and isinstance(action, DeviceDependentAction):
                passes_ = action.transpile_pass(
                    self.device.operation_names,
                    CouplingMap(self.device.build_coupling_map()) if self.layout else None,
                )
                passes = cast("list[Task]", passes_)
                if action.name == "QiskitO3":
                    assert action.do_while is not None
                    pm = PassManager([DoWhileController(passes, do_while=action.do_while)])
                else:
                    pm = PassManager(passes)
                altered_qc = pm.run(self.state)
                pm_property_set = dict(pm.property_set) if hasattr(pm, "property_set") else None
            else:
                transpile_pass_ = (
                    action.transpile_pass(self.device) if callable(action.transpile_pass) else action.transpile_pass
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
            and "final_layout" in pm_property_set
        ):
            self.layout.final_layout = pm_property_set["final_layout"]

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
            if pm_property_set["VF2Layout_stop_reason"] == VF2LayoutStopReason.SOLUTION_FOUND:
                assert pm_property_set["layout"]
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
            transpile_pass = action.transpile_pass(self.device, self.node_err, self.edge_err, self.readout_err)
        else:
            transpile_pass = (
                action.transpile_pass(self.device) if callable(action.transpile_pass) else action.transpile_pass
            )

        assert isinstance(transpile_pass, list)

        if action_index in self.actions_layout_indices:
            try:
                p0 = transpile_pass[0]
                assert isinstance(p0, Placement)
                placement = p0.get_placement_map(tket_qc)
            except Exception as e:
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
        assert callable(action.transpile_pass)
        if action_index in self.actions_opt_indices:
            transpile = cast("Callable[[Circuit], Circuit]", action.transpile_pass)
            bqskit_compiled_qc = transpile(bqskit_qc)
        elif action_index in self.actions_synthesis_indices:
            factory = cast("Callable[[Target], Callable[[Circuit], Circuit]]", action.transpile_pass)
            bqskit_compiled_qc = factory(self.device)(bqskit_qc)
        elif action_index in self.actions_mapping_indices:
            factory = cast(
                "Callable[[Target], Callable[[Circuit], tuple[Circuit, list[int], list[int]]]]",
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
        """Determine the valid actions for the current circuit state."""
        # Check if circuit uses only native gates
        check_nat_gates = GatesInBasis(basis_gates=self.device.operation_names)
        check_nat_gates(self.state)
        only_nat_gates = check_nat_gates.property_set["all_gates_in_basis"]

        # Check if circuit is mapped to the device coupling map
        check_mapping = CheckMap(coupling_map=CouplingMap(self.device.build_coupling_map()))
        check_mapping(self.state)
        mapped = check_mapping.property_set["is_swap_mapped"]

        if not only_nat_gates:
            if not mapped:
                # Non-native + unmapped → synthesis or optimization
                return self.actions_synthesis_indices + self.actions_opt_indices
            # Non-native + mapped → synthesis or structure-preserving (native gates & mapping)
            return self.actions_synthesis_indices + self.actions_structure_preserving_indices

        if mapped:
            if self.layout is not None:
                # Native + fully mapped & layout assigned → terminate or fine-tune
                return [
                    self.action_terminate_index,
                    *self.actions_structure_preserving_indices,
                    *self.actions_final_optimization_indices,
                ]
            # Mapped but no explicit layout yet → explore layout/mapping improvements
            return (
                self.actions_structure_preserving_indices + self.actions_layout_indices + self.actions_mapping_indices
            )
        if self.layout is not None:
            # Layout chosen but not mapped → must do routing
            return self.actions_routing_indices
        # Not mapped and without layout → explore layout/mapping/optimizations
        return self.actions_opt_indices + self.actions_layout_indices + self.actions_mapping_indices
