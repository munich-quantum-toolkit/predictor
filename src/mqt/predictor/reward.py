# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""This module contains the functions to calculate the reward of a quantum circuit on a given device."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import numpy as np
from joblib import load
from qiskit.transpiler import InstructionDurations, PassManager, Target
from qiskit.transpiler.passes import ASAPScheduleAnalysis

from mqt.predictor.hellinger import calc_device_specific_features, get_hellinger_model_path
from mqt.predictor.utils import calc_supermarq_features

if TYPE_CHECKING:
    from qiskit import QuantumCircuit
    from qiskit.transpiler import Target
    from sklearn.ensemble import RandomForestRegressor

logger = logging.getLogger("mqt-predictor")

figure_of_merit = Literal[
    "expected_fidelity",
    "critical_depth",
    "estimated_success_probability",
    "hellinger_distance",
    "estimated_hellinger_distance",
]


def crit_depth(qc: QuantumCircuit, precision: int = 10) -> float:
    """Calculates the critical depth of a given quantum circuit."""
    supermarq_features = calc_supermarq_features(qc)
    return float(np.round(1 - supermarq_features.critical_depth, precision).item())


def expected_fidelity(qc: QuantumCircuit, device: Target, precision: int = 10) -> float:
    """Calculates the expected fidelity of a given quantum circuit on a given device.

    Arguments:
        qc: The quantum circuit to be compiled.
        device: The device to be used for compilation.
        precision: The precision of the returned value. Defaults to 10.

    Returns:
        The expected fidelity of the given quantum circuit on the given device.
    """
    res = 1.0
    for qc_instruction in qc.data:
        instruction, qargs = qc_instruction.operation, qc_instruction.qubits
        gate_type = instruction.name

        if gate_type != "barrier":
            assert len(qargs) in [1, 2]
            first_qubit_idx = qc.find_bit(qargs[0]).index

            if len(qargs) == 1:
                specific_fidelity = 1 - device[gate_type][first_qubit_idx,].error
            else:
                second_qubit_idx = qc.find_bit(qargs[1]).index
                try:
                    specific_fidelity = 1 - device[gate_type][first_qubit_idx, second_qubit_idx].error
                except KeyError:
                    msg = f"Error rate for gate {gate_type} on qubits {first_qubit_idx} and {second_qubit_idx} not found in device properties."
                    raise KeyError(msg) from None
            res *= specific_fidelity

    return float(np.round(res, precision).item())


def _get_instruction_properties(device: Target, gate: str, qubits: tuple[int, ...]) -> object:
    """Return calibration properties for an instruction or raise if unavailable."""
    try:
        properties = device[gate][qubits]
    except KeyError as exc:
        msg = f"Missing calibration data for operation {gate} on qubit(s) {qubits}."
        raise ValueError(msg) from exc

    if properties is None:
        msg = f"Missing calibration data for operation {gate} on qubit(s) {qubits}."
        raise ValueError(msg)

    return properties


def _require_gate_error(device: Target, gate: str, qubits: tuple[int, ...]) -> float:
    """Return a gate error probability or raise if unavailable."""
    properties = _get_instruction_properties(device, gate, qubits)
    error = getattr(properties, "error", None)
    if error is None:
        msg = f"Missing error data for operation {gate} on qubit(s) {qubits}."
        raise ValueError(msg)
    if not 0 <= error <= 1:
        msg = f"Invalid error data for operation {gate} on qubit(s) {qubits}."
        raise ValueError(msg)
    return float(error)


def _require_gate_duration(device: Target, gate: str, qubits: tuple[int, ...]) -> float:
    """Return a gate duration in seconds or raise if unavailable."""
    properties = _get_instruction_properties(device, gate, qubits)
    duration = getattr(properties, "duration", None)
    if duration is None:
        msg = f"Missing duration data for operation {gate} on qubit(s) {qubits}."
        raise ValueError(msg)
    if duration < 0:
        msg = f"Invalid duration data for operation {gate} on qubit(s) {qubits}."
        raise ValueError(msg)
    return float(duration)


def _require_coherence_times(device: Target, qubit: int) -> tuple[float, float]:
    """Return (T1, T2) for a qubit or raise if unavailable."""
    if device.qubit_properties is None or device.qubit_properties[qubit] is None:
        msg = f"Missing qubit properties for qubit {qubit}."
        raise ValueError(msg)

    properties = device.qubit_properties[qubit]
    t1 = getattr(properties, "t1", None)
    t2 = getattr(properties, "t2", None)
    if t1 is None or t2 is None:
        msg = f"Missing coherence data for qubit {qubit}."
        raise ValueError(msg)
    if t1 <= 0 or t2 <= 0:
        msg = f"Invalid coherence data for qubit {qubit}."
        raise ValueError(msg)
    return float(t1), float(t2)


def _build_instruction_durations_from_target(qc: QuantumCircuit, device: Target) -> InstructionDurations:
    """Build durations for the instructions present in ``qc``."""
    op_times = []
    for instr in qc.data:
        op = instr.operation
        name = op.name

        if name in {"barrier", "id"}:
            continue

        qubit_indices = [qc.find_bit(q).index for q in instr.qubits]
        duration = _require_gate_duration(device, name, tuple(qubit_indices))
        op_times.append((name, qubit_indices, duration, "s"))

    return InstructionDurations(op_times, dt=device.dt)


def _compute_asap_timing(
    qc: QuantumCircuit, device: Target
) -> tuple[float, dict[int, float], dict[int, float], dict[int, str]]:
    """Return ASAP timing data for total duration and per-qubit activity."""
    durations = _build_instruction_durations_from_target(qc, device)
    pm = PassManager([ASAPScheduleAnalysis(durations=durations)])
    pm.run(qc)

    time_unit = pm.property_set["time_unit"]
    exec_time_per_qubit = dict.fromkeys(range(device.num_qubits), 0.0)
    last_end_per_qubit = dict.fromkeys(range(device.num_qubits), 0.0)
    last_op_per_qubit = dict.fromkeys(range(device.num_qubits), "")
    circuit_duration = 0.0

    for node, start_time in pm.property_set["node_start_time"].items():
        qubit_indices = [qc.find_bit(q).index for q in node.qargs]
        duration = float(durations.get(node.name, qubit_indices, unit=time_unit))
        end_time = float(start_time) + duration
        circuit_duration = max(circuit_duration, end_time)
        for qubit in qubit_indices:
            exec_time_per_qubit[qubit] += duration
            if end_time >= last_end_per_qubit[qubit]:
                last_end_per_qubit[qubit] = end_time
                last_op_per_qubit[qubit] = node.name

    return circuit_duration, exec_time_per_qubit, last_end_per_qubit, last_op_per_qubit


def estimated_success_probability(qc: QuantumCircuit, device: Target, precision: int = 10) -> float:
    """Estimate the success probability of ``qc`` on ``device``.

    The estimate multiplies gate fidelities and an idle-time decay factor based on
    min(T1, T2). Idle windows are derived from an ASAP schedule.
    """
    circuit_duration, exec_time_per_qubit, last_end_per_qubit, last_op_per_qubit = _compute_asap_timing(qc, device)

    active_qubits = set()
    for instr in qc.data:
        if instr.operation.name in {"barrier", "id"}:
            continue
        active_qubits.update(qc.find_bit(q).index for q in instr.qubits)

    res = 1.0

    for instr in qc.data:
        op = instr.operation
        qargs = instr.qubits
        gate_type = op.name

        if gate_type in {"barrier", "id"}:
            continue

        qubit_indices = [qc.find_bit(q).index for q in qargs]
        if len(qubit_indices) not in {1, 2}:
            msg = f"Unsupported instruction arity for {gate_type!r}: {len(qubit_indices)}"
            raise ValueError(msg)
        res *= 1.0 - _require_gate_error(device, gate_type, tuple(qubit_indices))

    for qubit in active_qubits:
        t1, t2 = _require_coherence_times(device, qubit)
        live_end = last_end_per_qubit[qubit] if last_op_per_qubit[qubit] in {"measure", "reset"} else circuit_duration
        idle_time_s = max(live_end - exec_time_per_qubit[qubit], 0.0)
        res *= np.exp(-idle_time_s / min(t1, t2))

    return float(np.round(res, precision).item())


def esp_data_available(device: Target) -> bool:
    """Check if calibration data to calculate ESP is available for the device.

    Arguments:
        device: The device to be checked for calibration data.

    Returns:
        True if all required calibration data is available, False otherwise.

    Raises:
        ValueError: If any required calibration data is missing or invalid.
    """
    single_qubit_gates = set()
    two_qubit_gates = set()

    for instruction in device.instructions:
        if instruction[0].num_qubits == 1:
            single_qubit_gates.add(instruction[0].name)
        elif instruction[0].num_qubits == 2:
            two_qubit_gates.add(instruction[0].name)
    single_qubit_gates -= {"delay", "reset", "id", "barrier"}

    def message(calibration: str, operation: str, target: int | str) -> str:
        return f"{calibration} data for {operation} operation on qubit(s) {target} is required to calculate ESP for device {device.description}."

    for qubit in range(device.num_qubits):
        try:
            _require_coherence_times(device, qubit)
        except ValueError:
            logger.exception(message("T1", "idle", qubit))
            return False
        try:
            _require_gate_error(device, "measure", (qubit,))
        except ValueError:
            logger.exception(message("Error", "readout", qubit))
            return False
        try:
            _require_gate_duration(device, "measure", (qubit,))
        except ValueError:
            logger.exception(message("Duration", "readout", qubit))
            return False

        for gate in single_qubit_gates:
            try:
                _require_gate_error(device, gate, (qubit,))
            except ValueError:
                logger.exception(message("Error", gate, qubit))
                return False
            try:
                _require_gate_duration(device, gate, (qubit,))
            except ValueError:
                logger.exception(message("Duration", gate, qubit))
                return False

    for gate in two_qubit_gates:
        for edge in device.build_coupling_map():
            try:
                _require_gate_error(device, gate, (edge[0], edge[1]))
            except ValueError:
                logger.exception(message("Error", gate, edge))
                return False
            try:
                _require_gate_duration(device, gate, (edge[0], edge[1]))
            except ValueError:
                logger.exception(message("Duration", gate, edge))
                return False

    return True


def estimated_hellinger_distance(
    qc: QuantumCircuit, device: Target, model: RandomForestRegressor | None = None, precision: int = 10
) -> float:
    """Calculates the estimated Hellinger distance of a given quantum circuit on a given device.

    Arguments:
        qc: The quantum circuit to be compiled.
        device: The device to be used for compilation.
        model: The pre-trained model to use for prediction (optional). If not provided, the model will try to be loaded from files.
        precision: The precision of the returned value. Defaults to 10.

    Returns:
        The estimated Hellinger distance of the given quantum circuit on the given device.
    """
    if model is None:
        # Load pre-trained model from files
        path = get_hellinger_model_path(device)
        model = load(path)

    feature_vector = calc_device_specific_features(qc, device)

    res = model.predict([feature_vector])
    return float(np.round(res, precision).item())
