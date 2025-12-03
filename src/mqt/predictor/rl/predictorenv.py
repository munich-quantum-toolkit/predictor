# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Predictor environment for the compilation using reinforcement learning."""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Any

if sys.version_info >= (3, 11) and TYPE_CHECKING:  # pragma: no cover
    pass

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from bqskit import Circuit
    from qiskit.passmanager import PropertySet
    from qiskit.transpiler import InstructionProperties, Target

    from mqt.predictor.reward import figure_of_merit
    from mqt.predictor.rl.actions import Action


import warnings
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
    estimated_success_probability,
    expected_fidelity,
)
from mqt.predictor.rl.actions import (
    CompilationOrigin,
    DeviceDependentAction,
    PassType,
    get_actions_by_pass_type,
)
from mqt.predictor.rl.cost_model import approx_estimated_success_probability, approx_expected_fidelity
from mqt.predictor.rl.helper import (
    create_feature_dict,
    get_path_training_circuits,
    get_state_sample,
)
from mqt.predictor.rl.parsing import (
    final_layout_bqskit_to_qiskit,
    final_layout_pytket_to_qiskit,
    postprocess_vf2postlayout,
)
from mqt.predictor.utils import calc_supermarq_features

logger = logging.getLogger("mqt-predictor")


class PredictorEnv(Env):  # type: ignore[misc]
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
        self.reward_scale = 1
        self.no_effect_penalty = 0.001
        self.prev_reward: float | None = None
        self.prev_reward_kind: str | None = None
        self._p1_avg = 0.0
        self._p2_avg = 0.0
        self._tau1_avg = 0.0
        self._tau2_avg = 0.0
        self._tbar: float | None = None
        self._dev_avgs_cached = False

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, bool, dict[Any, Any]]:
        """Run one environment step.

        This method:
            1. Evaluates the current circuit with the configured reward function
            (using either the exact or approximate metric, depending on state).
            2. Applies the selected transpiler pass (the action).
            3. Normalizes the circuit (e.g., decompose high-level gates) so that
            reward computation is well-defined.
            4. Updates the internal state and valid action set.
            5. Computes a shaped step reward based on the change in figure of merit.

        Reward design:
            - For non-terminal actions, the step reward is a scaled delta between
            the new and previous reward (plus an optional step penalty).
            - For the terminate action, the episode ends and the final reward is
            the exact (calibration-aware) metric.
        """
        self.used_actions.append(str(self.action_set[action].name))

        logger.info(f"Applying {self.action_set[action].name}")

        # 1) Evaluate reward for current circuit (before applying the action)
        prev_val, prev_kind = self.calculate_reward(mode="auto")
        self.prev_reward = prev_val
        self.prev_reward_kind = prev_kind

        # 2) Apply the selected transpiler pass
        altered_qc = self.apply_action(action)

        if not altered_qc:
            return create_feature_dict(self.state), 0.0, True, False, {}

        # 3) Normalize circuit: remove high-level gates that break reward assumptions
        #    - decompose "unitary"/"clifford"
        for gate_type in ["unitary", "clifford"]:
            if altered_qc.count_ops().get(gate_type):
                altered_qc = altered_qc.decompose(gates_to_decompose=gate_type)

        # 4) Update state and valid actions
        self.state: QuantumCircuit = altered_qc
        self.num_steps += 1
        self.valid_actions = self.determine_valid_actions_for_state()
        if len(self.valid_actions) == 0:
            msg = "No valid actions left."
            raise RuntimeError(msg)

        # 5) Compute step reward and termination flag
        if action == self.action_terminate_index:
            # Terminal action: use the exact metric as final reward
            final_val, final_kind = self.calculate_reward(mode="exact")
            logger.info(f"Final reward ({final_kind}): {final_val}")
            self.prev_reward = final_val
            self.prev_reward_kind = final_kind
            done = True
            reward_val = final_val
        else:
            done = False

            # Re-evaluate reward after applying the action
            new_val, new_kind = self.calculate_reward(mode="auto")
            delta_reward = new_val - prev_val

            if prev_kind == "approx" and new_kind == "exact":
                delta_reward = 0.0  # Delta is not defined for switch from "approx" to "exact"

            if delta_reward > 0.0:
                # Positive change: reward proportional to improvement
                reward_val = self.reward_scale * delta_reward
            elif delta_reward < 0.0:
                # Negative change: proportional penalty
                reward_val = self.reward_scale * delta_reward
            else:
                # No change: small step penalty for "doing nothing"
                reward_val = self.no_effect_penalty

            self.prev_reward = new_val
            self.prev_reward_kind = new_kind

        self.state._layout = self.layout  # noqa: SLF001
        return create_feature_dict(self.state), reward_val, done, False, {}

    def calculate_reward(self, qc: QuantumCircuit | None = None, mode: str = "auto") -> tuple[float, str]:
        """Compute the current reward and indicate whether it is exact or approximate.

        Args:
            qc:
                Circuit to evaluate. If ``None``, the environment's current state
                circuit is used.
            mode:
                Controls how the function chooses between exact (calibration-based)
                and approximate (cost-model-based) metrics:

                - ``"auto"`` (default): use the exact metric if the circuit is
                already native and mapped; otherwise fall back to the approximate
                metric.
                - ``"exact"``: always compute the exact metric (no approximation).
                - ``"approx"``: always compute the approximate metric.

        Returns:
            A pair ``(value, kind)`` where:

                - ``value`` is the scalar reward value.
                - ``kind`` is either ``"exact"`` (exact, calibration-aware) or
                ``"approx"`` (cost-model-based approximation).

        Notes:
            - Dual-path behavior (exact + approximate) is currently only implemented
            for ``"expected_fidelity"`` and
            ``"estimated_success_probability"``.
            - Other reward functions are always computed exactly.
        """
        if qc is None:
            qc = self.state

        # Reward functions that are always computed exactly, regardless of `mode`.
        if self.reward_function not in {"expected_fidelity", "estimated_success_probability"}:
            if self.reward_function == "critical_depth":
                return crit_depth(qc), "exact"
            # Fallback for other unknown / not-yet-implemented reward functions:
            logger.warning(
                "Reward function '%s' is not supported in PredictorEnv. Returning 0.0 as a fallback reward.",
                self.reward_function,
            )
            return 0.0, "exact"

        # ------------------------------------------------------------------
        # From here on: dual-path rewards (exact vs approx) for EF / ESP.
        # ------------------------------------------------------------------

        # Decide which path to use (exact vs approx)
        if mode == "exact":
            kind = "exact"
        elif mode == "approx":
            kind = "approx"
        else:  # "auto"
            kind = "exact" if self._is_native_and_mapped(qc) else "approx"

        # Exact metrics use the full circuit and device calibration data
        if kind == "exact":
            if self.reward_function == "expected_fidelity":
                return expected_fidelity(qc, self.device), "exact"
            # self.reward_function == "estimated_success_probability"
            return estimated_success_probability(qc, self.device), "exact"

        # Approximate metrics use canonical gate counts and device-wide averages
        self._ensure_device_averages_cached()

        if self.reward_function == "expected_fidelity":
            val = approx_expected_fidelity(qc, self._p1_avg, self._p2_avg, device_id=self.device.description)
            return val, "approx"

        # self.reward_function == "estimated_success_probability"
        feats = calc_supermarq_features(qc)
        val = approx_estimated_success_probability(
            qc,
            p1_avg=self._p1_avg,
            p2_avg=self._p2_avg,
            tau1_avg=self._tau1_avg,
            tau2_avg=self._tau2_avg,
            tbar=self._tbar,
            par_feature=feats.parallelism,
            liv_feature=feats.liveness,
            n_qubits=qc.num_qubits,
            device_id=self.device.description,
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
    ) -> tuple[QuantumCircuit, dict[str, Any]]:
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
        pm_property_set: PropertySet | None = {}
        if getattr(action, "stochastic", False):  # Wrap stochastic action to optimize for the used figure of merit
            altered_qc, pm_property_set = self.fom_aware_compile(
                action,
                self.device,
                self.state,
                max_iteration=self.max_iter,
            )
        else:
            if action.name in ["QiskitO3", "Opt2qBlocks_preserve"] and isinstance(action, DeviceDependentAction):
                passes = action.transpile_pass(
                    self.device.operation_names,
                    CouplingMap(self.device.build_coupling_map()) if self.layout else None,
                )
                if action.name == "QiskitO3":
                    pm = PassManager([DoWhileController(passes, do_while=action.do_while)])
                else:
                    pm = PassManager(passes)
                altered_qc = pm.run(self.state)
                pm_property_set = dict(pm.property_set) if hasattr(pm, "property_set") else {}
            else:
                transpile_pass = (
                    action.transpile_pass(self.device) if callable(action.transpile_pass) else action.transpile_pass
                )
                pm = PassManager(transpile_pass)
                altered_qc = pm.run(self.state)
                pm_property_set = dict(pm.property_set) if hasattr(pm, "property_set") else {}

        if action_index in (
            self.actions_layout_indices + self.actions_mapping_indices + self.actions_final_optimization_indices
        ):
            altered_qc = self._handle_qiskit_layout_postprocessing(action, pm_property_set, altered_qc)
        elif action_index in self.actions_routing_indices and self.layout and pm_property_set is not None:
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
                altered_qc, _ = postprocess_vf2postlayout(altered_qc, post_layout, self.layout)
        elif action.name == "VF2Layout":
            if pm_property_set["VF2Layout_stop_reason"] == VF2LayoutStopReason.SOLUTION_FOUND:
                assert pm_property_set["layout"]
        else:
            assert pm_property_set["layout"]

        layout = pm_property_set.get("layout")
        if layout:
            self.layout = TranspileLayout(
                initial_layout=layout,
                input_qubit_mapping=pm_property_set.get("original_qubit_indices"),
                final_layout=pm_property_set.get("final_layout"),
                _output_qubit_list=altered_qc.qubits,
                _input_qubit_count=self.num_qubits_uncompiled_circuit,
            )

        if self.layout is not None and pm_property_set.get("final_layout"):
            self.layout.final_layout = pm_property_set["final_layout"]
        return altered_qc

    def _apply_tket_action(self, action: Action, action_index: int) -> QuantumCircuit:
        tket_qc = qiskit_to_tk(self.state, preserve_param_uuid=True)
        transpile_pass = (
            action.transpile_pass(self.device) if callable(action.transpile_pass) else action.transpile_pass
        )
        assert isinstance(transpile_pass, list)
        for p in transpile_pass:
            p.apply(tket_qc)

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
        """Cache device-wide averages for 1q/2q errors, durations, and coherence.

        Backend-dependent preprocessing step used by the approximate reward model.
        It computes and caches:

            - _p1_avg: average single-qubit gate error probability
            - _p2_avg: average two-qubit gate error probability
            - _tau1_avg: average single-qubit gate duration (seconds)
            - _tau2_avg: average two-qubit gate duration (seconds)
            - _tbar: median of min(T1, T2) over all qubits (seconds), if available

        Assumes a modern Qiskit Target (e.g. IBM backends) and raises a RuntimeError
        if the required calibration data is not available.
        """
        if getattr(self, "_dev_avgs_cached", False):
            return

        target = self.device

        # Hard requirements: these must exist for the approximate model to make sense
        try:
            num_qubits = target.num_qubits
            op_names = list(target.operation_names)
            coupling_map = target.build_coupling_map()
            qubit_props = target.qubit_properties
        except AttributeError as exc:
            msg = "Device target does not expose the required Target API for approximate reward computation."
            raise RuntimeError(msg) from exc

        dt = getattr(target, "dt", None)
        twoq_edges = coupling_map.get_edges()  # list[(i, j)]

        p1: list[float] = []
        p2: list[float] = []
        t1: list[float] = []
        t2: list[float] = []

        # Exclude non-gate operations from gate error/duration averages
        gate_blacklist = {"measure", "reset", "delay", "barrier"}

        def _get_props(name: str, qargs: tuple[int, ...]) -> InstructionProperties | None:
            """Return calibration properties for (name, qargs) or None if unavailable."""
            try:
                props_map = target[name]
            except KeyError:
                return None

            return props_map.get(qargs, None)

        # --- Aggregate error and duration statistics over all 1q/2q gates --------
        for name in op_names:
            if name in gate_blacklist:
                continue

            # Determine arity (number of qubits) of the operation
            try:
                op = target.operation_from_name(name)
                arity = op.num_qubits
            except (KeyError, AttributeError):
                # If we can't get a proper operation object, skip this op
                continue

            if arity == 1:
                # Collect single-qubit gate error/duration over all qubits
                for q in range(num_qubits):
                    props = _get_props(name, (q,))
                    if props is None:
                        continue
                    err = getattr(props, "error", None)
                    if err is not None:
                        p1.append(float(err))
                    dur = getattr(props, "duration", None)
                    if dur is not None:
                        dur_s = float(dur if dt is None else dur * dt)
                        t1.append(dur_s)

            elif arity == 2:
                # Collect two-qubit gate error/duration over all supported edges
                for i, j in twoq_edges:
                    props = _get_props(name, (i, j))
                    if props is None:
                        # Try flipped orientation for uni-directional couplings
                        props = _get_props(name, (j, i))
                    if props is None:
                        continue
                    err = getattr(props, "error", None)
                    if err is not None:
                        p2.append(float(err))
                    dur = getattr(props, "duration", None)
                    if dur is not None:
                        dur_s = float(dur if dt is None else dur * dt)
                        t2.append(dur_s)

            else:
                # Ignore gates with arity > 2; extend here if you ever need them
                continue

        if not p1 and not p2:
            msg = "No valid 1q/2q calibration data found in Target; cannot compute approximate reward."
            raise RuntimeError(msg)

        self._p1_avg = float(np.mean(p1)) if p1 else 0.0
        self._p2_avg = float(np.mean(p2)) if p2 else 0.0
        self._tau1_avg = float(np.mean(t1)) if t1 else 0.0
        self._tau2_avg = float(np.mean(t2)) if t2 else 0.0

        # --- Compute a single coherence scale tbar from T1/T2 ---------------------
        tmins: list[float] = []
        if qubit_props:
            for i in range(num_qubits):
                props = qubit_props[i]
                if props is None:
                    continue
                t1v = getattr(props, "t1", None)
                t2v = getattr(props, "t2", None)
                vals = [v for v in (t1v, t2v) if v is not None]
                if vals:
                    tmins.append(float(min(vals)))

        self._tbar = float(np.median(tmins)) if tmins else None

        self._dev_avgs_cached = True

    def _is_native_and_mapped(self, qc: QuantumCircuit) -> bool:
        check_nat_gates = GatesInBasis(basis_gates=self.device.operation_names)
        check_nat_gates(qc)
        only_nat_gates = check_nat_gates.property_set["all_gates_in_basis"]

        check_mapping = CheckMap(coupling_map=CouplingMap(self.device.build_coupling_map()))
        check_mapping(qc)
        mapped = check_mapping.property_set["is_swap_mapped"]

        return bool(only_nat_gates and mapped)
