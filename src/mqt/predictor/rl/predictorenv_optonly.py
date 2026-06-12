# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Optimization-only reinforcement-learning environment for KIT's protocol."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, Dict, Discrete
from qiskit import QuantumCircuit
from qiskit.transpiler import Target

from mqt.predictor.reward import cx_relative_reduction
from mqt.predictor.rl.actions import Action, CompilationOrigin, DeviceIndependentAction, PassType
from mqt.predictor.rl.actions.kit_actions import kit_optimization_actions
from mqt.predictor.rl.helper import create_feature_dict, logger
from mqt.predictor.rl.kit_baseline import TARGET_BASIS, build_translation_pass_manager, count_two_qubit_gates
from mqt.predictor.rl.predictorenv import PredictorEnv

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping

KIT_QISKIT_ML_REPO = Path("/Users/patrickhopf/Code/icse-paper-2026-qiskit-ml")
KIT_QASM_DIR = KIT_QISKIT_ML_REPO / "data" / "raw"

STATUS_OK = "ok"
STATUS_TRIVIAL_BASELINE = "trivial_baseline"
STATUS_PASS_MANAGER_ERROR = "pm_error"
STATUS_TRANSLATION_ERROR = "translation_error"
STATUS_BASELINE_ERROR = "baseline_error"
STATUS_CIRCUIT_TOO_LARGE = "circuit_too_large"

_TRANSLATION_PM = build_translation_pass_manager(TARGET_BASIS)


class OptOnlyPredictorEnv(PredictorEnv):
    """Opt-only environment matching KIT's all-to-all, basis-S protocol.

    The environment exposes exactly KIT's concrete optimization pass names as
    binary choices: each pass can be selected once, then the policy terminates.
    Rewards are computed as ``(reference_cx - optimized_cx) / reference_cx`` after applying the
    same fixed post-optimization translation pass manager to the original and optimized circuits.
    """

    def __init__(
        self,
        path_training_circuits: Path | str | None = None,
        baseline_cx_lookup: Mapping[str, int | float] | None = None,
        excluded_circuit_ids: Collection[str] | None = None,
        max_qubits: int = 256,
        max_steps: int | None = 100,
        pass_timeout: int | None = None,
        max_circuit_operations: int | None = 100_000,
        max_template_optimization_operations: int | None = 40_000,
    ) -> None:
        """Initialize the optimization-only RL environment.

        Args:
            path_training_circuits: Directory containing training QASM files. Defaults to
                the ICSE Qiskit-ML raw corpus checkout path.
            baseline_cx_lookup: Optional precomputed reference counts keyed by circuit stem. If omitted, the
                reference count is computed from the input circuit after the fixed basis translation.
            excluded_circuit_ids: Circuit stems to exclude from training, for example
                held-out benchmark circuits from ``test_circuits.csv``.
            max_qubits: Maximum qubit count sampled for training.
            max_steps: The maximum number of actions per episode. If None, no step limit is enforced. Defaults to 100.
            pass_timeout: The timeout in seconds for applying a single pass. If None, no timeout is enforced.
                Defaults to None.
            max_circuit_operations: The maximum number of operations allowed after applying one pass. If None,
                no operation-count limit is enforced. Defaults to 100,000.
            max_template_optimization_operations: The maximum number of operations allowed before running
                TemplateOptimization. If None, no limit is enforced. Defaults to 40,000.
        """
        Env.__init__(self)

        self.path_training_circuits = Path(path_training_circuits) if path_training_circuits else KIT_QASM_DIR
        self.baseline_cx_lookup = {
            Path(circuit_id).stem: float(baseline_cx) for circuit_id, baseline_cx in (baseline_cx_lookup or {}).items()
        }
        self.excluded_circuit_ids = {Path(circuit_id).stem for circuit_id in (excluded_circuit_ids or set())}
        self.max_qubits = max_qubits
        self.max_steps = max_steps
        self.pass_timeout = pass_timeout
        self.max_circuit_operations = max_circuit_operations
        self.max_template_optimization_operations = max_template_optimization_operations

        self.action_set: dict[int, Any] = {}
        self.actions_synthesis_indices: list[int] = []
        self.actions_layout_indices: list[int] = []
        self.actions_routing_indices: list[int] = []
        self.actions_mapping_indices: list[int] = []
        self.actions_opt_indices: list[int] = []
        self.actions_final_optimization_indices: list[int] = []
        self.used_actions: list[str] = []

        for index, action in enumerate(kit_optimization_actions()):
            self.action_set[index] = action
            self.actions_opt_indices.append(index)

        self.action_terminate_index = len(self.action_set)
        self.action_set[self.action_terminate_index] = DeviceIndependentAction(
            "terminate",
            CompilationOrigin.GENERAL,
            PassType.TERMINATE,
            transpile_pass=[],
        )

        self.action_space = Discrete(len(self.action_set))
        self.observation_space = Dict({
            "num_qubits": Discrete(max_qubits + 1),
            "depth": Discrete(1_000_000),
            "program_communication": Box(0, 1, (1,), np.float32),
            "critical_depth": Box(0, 1, (1,), np.float32),
            "entanglement_ratio": Box(0, 1, (1,), np.float32),
            "parallelism": Box(0, 1, (1,), np.float32),
            "liveness": Box(0, 1, (1,), np.float32),
        })

        self.device = _make_basis_only_target()
        self.layout = None
        self.has_parameterized_gates = False
        self.rng = np.random.default_rng(10)
        self.filename = ""
        self.num_steps = 0
        self.num_qubits_uncompiled_circuit = 0
        self.reward_function = "cx_relative_reduction"
        self.baseline_cx = 0.0
        self.error_occurred = False
        self.status = STATUS_OK
        self.error_msg = ""
        self.log_applied_passes = False
        self.valid_actions = self.determine_valid_actions_for_state()

    def reset(
        self,
        qc: Path | str | QuantumCircuit | None = None,
        seed: int | None = None,
        options: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset the environment to a supplied circuit or a random training circuit.

        Args:
            qc: Circuit object or QASM path. If omitted, a training circuit is sampled.
            seed: Optional environment seed.
            options: Unused Gymnasium reset options.

        Returns:
            The initial observation and info dictionary.
        """
        Env.reset(self, seed=seed)
        if isinstance(qc, QuantumCircuit):
            self.state = qc
            self.filename = qc.name
        elif qc:
            self.state = QuantumCircuit.from_qasm_file(str(qc))
            self.filename = str(qc)
        else:
            self.state, self.filename = _get_state_sample(
                max_qubits=self.max_qubits,
                path_training_circuits=self.path_training_circuits,
                rng=self.rng,
                excluded_circuit_ids=self.excluded_circuit_ids,
            )

        self.action_space = Discrete(len(self.action_set))
        self.num_steps = 0
        self.used_actions = []
        self.layout = None
        self.error_occurred = False
        self.status = STATUS_OK
        self.error_msg = ""
        self.num_qubits_uncompiled_circuit = self.state.num_qubits
        self.has_parameterized_gates = len(self.state.parameters) > 0

        circuit_name = _circuit_name_from_path(self.filename)
        if circuit_name in self.baseline_cx_lookup:
            self.baseline_cx = self.baseline_cx_lookup[circuit_name]
        else:
            try:
                self.baseline_cx = float(count_two_qubit_gates(_TRANSLATION_PM.run(self.state.copy())))
            except Exception as exc:  # noqa: BLE001
                self.baseline_cx = 0.0
                self.status = STATUS_BASELINE_ERROR
                self.error_msg = f"Could not compute reference CX count: {_short_exception(exc)}"

        if self.baseline_cx == 0 and self.status == STATUS_OK:
            self.status = STATUS_TRIVIAL_BASELINE

        self.valid_actions = self.determine_valid_actions_for_state()
        return create_feature_dict(self.state), {"status": self.status}

    def snapshot_training_state(self) -> dict[str, Any]:
        """Create a serializable snapshot of the environment training state."""
        return {
            "rng_state": self.rng.bit_generator.state,
            "state": self.state.copy(),
            "filename": self.filename,
            "num_steps": self.num_steps,
            "used_actions": self.used_actions,
            "valid_actions": self.valid_actions,
            "baseline_cx": self.baseline_cx,
            "layout": self.layout,
            "error_occurred": self.error_occurred,
            "status": self.status,
            "error_msg": self.error_msg,
            "num_qubits_uncompiled_circuit": self.num_qubits_uncompiled_circuit,
            "has_parameterized_gates": self.has_parameterized_gates,
        }

    def restore_training_state(self, snapshot: Mapping[str, Any]) -> None:
        """Restore a snapshot created with :meth:`snapshot_training_state`."""
        self.rng.bit_generator.state = snapshot["rng_state"]
        self.state = cast("QuantumCircuit", snapshot["state"]).copy()
        self.filename = str(snapshot["filename"])
        self.num_steps = int(snapshot["num_steps"])
        self.used_actions = list(snapshot["used_actions"])
        self.valid_actions = list(snapshot["valid_actions"])
        self.baseline_cx = float(snapshot["baseline_cx"])
        self.layout = snapshot["layout"]
        self.error_occurred = bool(snapshot["error_occurred"])
        self.status = str(snapshot["status"])
        self.error_msg = str(snapshot["error_msg"])
        self.num_qubits_uncompiled_circuit = int(snapshot["num_qubits_uncompiled_circuit"])
        self.has_parameterized_gates = bool(snapshot["has_parameterized_gates"])

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, bool, dict[Any, Any]]:
        """Apply an optimization action and return the Gymnasium step tuple.

        Args:
            action: The integer action index.

        Returns:
            Observation, reward, termination flag, truncation flag, and info.

        Raises:
            ValueError: If the action is unsupported or currently invalid.
        """
        if action not in self.action_set:
            msg = f"Action {action} not supported."
            raise ValueError(msg)
        if action not in self.valid_actions:
            msg = f"Action {action} is not valid for the current optimization-only state."
            raise ValueError(msg)

        selected_action = self.action_set[action]
        self.used_actions.append(str(selected_action.name))
        step_number = self.num_steps + 1
        started_at = time.perf_counter()
        if self.log_applied_passes:
            logger.info(
                "Applying opt-only action %s: %s on %s with %s qubits and %s operations.",
                step_number,
                _describe_action(selected_action),
                _circuit_name_from_path(self.filename) or "<in-memory>",
                self.state.num_qubits,
                len(self.state.data),
            )

        if (
            selected_action.name == "TemplateOptimization"
            and self.max_template_optimization_operations is not None
            and len(self.state.data) > self.max_template_optimization_operations
        ):
            return self._truncate_expensive_action(
                action_name=str(selected_action.name),
                step_number=step_number,
                started_at=started_at,
                operation_count=len(self.state.data),
                operation_limit=self.max_template_optimization_operations,
            )

        try:
            altered_qc = self._apply_action_with_timeout(action)
        except Exception as exc:  # noqa: BLE001
            self.error_occurred = True
            self.status = STATUS_PASS_MANAGER_ERROR
            self.error_msg = _short_exception(exc)
            self.valid_actions = [self.action_terminate_index]
            self.num_steps += 1
            if self.log_applied_passes:
                logger.info(
                    "Failed opt-only action %s: %s after %.3fs.",
                    step_number,
                    selected_action.name,
                    time.perf_counter() - started_at,
                )
            return (
                create_feature_dict(self.state),
                0.0,
                False,
                True,
                self._truncation_info(f"Error applying action '{selected_action.name}': {exc}"),
            )

        if altered_qc is None:
            self.error_occurred = True
            self.status = STATUS_PASS_MANAGER_ERROR
            self.error_msg = "Action returned no circuit."
            self.valid_actions = [self.action_terminate_index]
            self.num_steps += 1
            if self.log_applied_passes:
                logger.info(
                    "Finished opt-only action %s: %s returned no circuit after %.3fs.",
                    step_number,
                    selected_action.name,
                    time.perf_counter() - started_at,
                )
            return (
                create_feature_dict(self.state),
                0.0,
                False,
                True,
                self._truncation_info(f"Error applying action '{selected_action.name}': Action returned no circuit."),
            )

        operation_count = len(altered_qc.data)
        if self._exceeds_circuit_operation_limit(operation_count):
            del altered_qc
            return self._truncate_oversized_circuit(
                action_name=str(selected_action.name),
                step_number=step_number,
                started_at=started_at,
                operation_count=operation_count,
            )

        candidate_qc = altered_qc
        if candidate_qc.count_ops().get("unitary"):
            candidate_qc = candidate_qc.decompose(gates_to_decompose="unitary")
            operation_count = len(candidate_qc.data)
            if self._exceeds_circuit_operation_limit(operation_count):
                del altered_qc
                del candidate_qc
                return self._truncate_oversized_circuit(
                    action_name=str(selected_action.name),
                    step_number=step_number,
                    started_at=started_at,
                    operation_count=operation_count,
                )

        self.state = candidate_qc
        self.num_steps += 1
        if self.log_applied_passes:
            logger.info(
                "Finished opt-only action %s: %s after %.3fs; circuit now has %s operations.",
                self.num_steps,
                selected_action.name,
                time.perf_counter() - started_at,
                len(self.state.data),
            )

        self.state._layout = self.layout  # noqa: SLF001
        self.valid_actions = self.determine_valid_actions_for_state()

        if action == self.action_terminate_index:
            reward_val = self.calculate_reward()
            done = True
        else:
            reward_val = 0.0
            done = False

        obs = create_feature_dict(self.state)
        if not done and self.max_steps is not None and self.num_steps >= self.max_steps:
            return obs, reward_val, False, True, self._truncation_info("max_steps_exceeded")
        return obs, reward_val, done, False, self._info()

    def determine_valid_actions_for_state(self) -> list[int]:
        """Return opt actions not yet selected plus terminate."""
        used_action_names = set(self.used_actions)
        valid_actions = [
            action_index
            for action_index in self.actions_opt_indices
            if self.action_set[action_index].name not in used_action_names
        ]
        valid_actions.append(self.action_terminate_index)
        return valid_actions

    def action_masks(self) -> list[bool]:
        """Return the valid-action mask for MaskablePPO."""
        return [action in self.valid_actions for action in range(len(self.action_set))]

    def calculate_reward(self) -> float:
        """Calculate the terminal optimization-ratio reward."""
        if self.baseline_cx == 0 or self.status != STATUS_OK:
            return 0.0

        try:
            return cx_relative_reduction(self.state, self.baseline_cx, _TRANSLATION_PM)
        except Exception as exc:  # noqa: BLE001
            self.status = STATUS_TRANSLATION_ERROR
            self.error_msg = _short_exception(exc)
            return 0.0

    def _info(self) -> dict[str, str]:
        """Return diagnostic episode information."""
        return {"status": self.status, "error_msg": self.error_msg}

    def _truncation_info(self, reason: str) -> dict[str, str]:
        """Return diagnostic episode information with a truncation reason."""
        info = self._info()
        info["truncation_reason"] = reason
        return info

    def _exceeds_circuit_operation_limit(self, operation_count: int) -> bool:
        """Return whether the operation-count guard rejects a circuit."""
        return self.max_circuit_operations is not None and operation_count > self.max_circuit_operations

    def _truncate_oversized_circuit(
        self,
        *,
        action_name: str,
        step_number: int,
        started_at: float,
        operation_count: int,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[Any, Any]]:
        """Truncate an episode before storing or featurizing an oversized circuit."""
        assert self.max_circuit_operations is not None
        reason = (
            f"Action '{action_name}' produced {operation_count} operations, "
            f"exceeding limit of {self.max_circuit_operations}."
        )
        self.error_occurred = True
        self.status = STATUS_CIRCUIT_TOO_LARGE
        self.error_msg = reason
        self.valid_actions = [self.action_terminate_index]
        self.num_steps += 1
        if self.log_applied_passes:
            logger.info(
                "Truncated opt-only action %s: %s after %.3fs; %s",
                step_number,
                action_name,
                time.perf_counter() - started_at,
                reason,
            )
        return create_feature_dict(self.state), 0.0, False, True, self._truncation_info(reason)

    def _truncate_expensive_action(
        self,
        *,
        action_name: str,
        step_number: int,
        started_at: float,
        operation_count: int,
        operation_limit: int,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[Any, Any]]:
        """Truncate before a known costly pass."""
        reason = (
            f"Action '{action_name}' skipped on {operation_count} operations, exceeding limit of {operation_limit}."
        )
        self.error_occurred = True
        self.status = STATUS_CIRCUIT_TOO_LARGE
        self.error_msg = reason
        self.valid_actions = [self.action_terminate_index]
        self.num_steps += 1
        if self.log_applied_passes:
            logger.info(
                "Truncated opt-only action %s: %s after %.3fs; %s",
                step_number,
                action_name,
                time.perf_counter() - started_at,
                reason,
            )
        return create_feature_dict(self.state), 0.0, False, True, self._truncation_info(reason)


def _make_basis_only_target() -> Target:
    """Return a target placeholder for inherited Qiskit action plumbing."""
    return Target(description="alltoall_basisS")


def _describe_action(action: Action) -> str:
    """Return a concise action description for training logs."""
    if isinstance(action.transpile_pass, list):
        pass_names = ", ".join(transpile_pass.__class__.__name__ for transpile_pass in action.transpile_pass)
        if pass_names:
            return f"{action.name} [{pass_names}]"
    return str(action.name)


def _get_state_sample(
    max_qubits: int,
    path_training_circuits: Path,
    rng: np.random.Generator,
    excluded_circuit_ids: Collection[str],
) -> tuple[QuantumCircuit, str]:
    """Sample one QASM circuit from a train-only corpus.

    Args:
        max_qubits: Maximum allowed qubit count.
        path_training_circuits: Directory containing QASM files.
        rng: Random generator used for sampling.
        excluded_circuit_ids: Circuit stems to exclude from sampling.

    Returns:
        The loaded circuit and its path as a string.

    Raises:
        FileNotFoundError: If no QASM files are present after filtering.
        RuntimeError: If no file can be loaded within the qubit limit.
    """
    file_list = [
        path for path in sorted(path_training_circuits.glob("*.qasm")) if path.stem not in excluded_circuit_ids
    ]
    if not file_list:
        msg = f"No trainable *.qasm files found in {path_training_circuits}."
        raise FileNotFoundError(msg)

    for random_index in rng.permutation(len(file_list)):
        path = file_list[int(random_index)]
        parsed_qubits = _qubit_count_from_path(path)
        if parsed_qubits is not None and parsed_qubits > max_qubits:
            continue
        try:
            qc = QuantumCircuit.from_qasm_file(str(path))
        except Exception as exc:
            msg = f"Could not read QuantumCircuit from: {path}"
            raise RuntimeError(msg) from exc
        if qc.num_qubits <= max_qubits:
            return qc, str(path)

    msg = f"No trainable circuit in {path_training_circuits} has at most {max_qubits} qubits."
    raise RuntimeError(msg)


def _qubit_count_from_path(path: Path) -> int | None:
    """Parse the MQT-Bench qubit count suffix from a QASM file name."""
    try:
        return int(path.stem.rsplit("_", maxsplit=1)[1])
    except (IndexError, ValueError):
        return None


def _circuit_name_from_path(path: str) -> str:
    """Return the circuit stem for lookup keys."""
    return Path(path).stem if path else ""


def _short_exception(exc: BaseException, limit: int = 300) -> str:
    """Return a compact exception string for environment info."""
    return f"{type(exc).__name__}: {exc}"[:limit]
