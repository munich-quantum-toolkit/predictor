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
from qiskit.transpiler import InstructionDurations, PassManager
from qiskit.transpiler.passes import ASAPScheduleAnalysis

from mqt.predictor.hellinger import calc_device_specific_features, get_hellinger_model_path
from mqt.predictor.utils import calc_supermarq_features

if TYPE_CHECKING:
    from collections.abc import Iterable

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


def estimated_success_probability(qc: QuantumCircuit, device: Target, precision: int = 10) -> float:
    """Calculates the estimated success probability of a given quantum circuit on a given device.

    It is calculated by multiplying the expected fidelity with a min(T1,T2)-dependent
    decay factor during qubit idle times. To this end, the circuit is scheduled using ASAP scheduling.

    Arguments:
        qc: The quantum circuit to be compiled.
        device: The device to be used for compilation.
        precision: The precision of the returned value. Defaults to 10.

    Returns:
        The expected success probability of the given quantum circuit on the given device.
    """
    operation_times: list[tuple[str, Iterable[int] | None, float, str]] = []
    for instr in qc.data:
        gate_type = str(instr.operation.name)
        if gate_type in {"barrier", "id"}:
            continue
        qubit_indices = [int(qc.find_bit(qubit).index) for qubit in instr.qubits]
        properties = device[gate_type].get(tuple(qubit_indices))
        if properties is None or properties.duration is None:
            msg = f"Duration for gate {gate_type} on qubits {tuple(qubit_indices)} not found in device properties."
            raise ValueError(msg)
        operation_times.append((gate_type, qubit_indices, float(properties.duration), "s"))

    durations = InstructionDurations(operation_times, dt=device.dt)
    pass_manager = PassManager([ASAPScheduleAnalysis(durations=durations)])
    pass_manager.run(qc)

    time_unit = pass_manager.property_set["time_unit"]
    execution_time_per_qubit = dict.fromkeys(range(device.num_qubits), 0.0)
    last_end_per_qubit = dict.fromkeys(range(device.num_qubits), 0.0)
    last_operation_per_qubit = dict.fromkeys(range(device.num_qubits), "")
    circuit_duration = 0.0

    for node, start_time in pass_manager.property_set["node_start_time"].items():
        qubit_indices = [qc.find_bit(qubit).index for qubit in node.qargs]
        duration = float(durations.get(node.name, qubit_indices, unit=time_unit))
        end_time = float(start_time) + duration
        circuit_duration = max(circuit_duration, end_time)
        for qubit in qubit_indices:
            execution_time_per_qubit[qubit] += duration
            if end_time >= last_end_per_qubit[qubit]:
                last_end_per_qubit[qubit] = end_time
                last_operation_per_qubit[qubit] = node.name

    res = 1.0
    active_qubits = set()
    for instr in qc.data:
        instruction = instr.operation
        qargs = instr.qubits
        gate_type = instruction.name

        if gate_type in {"barrier", "id"}:
            continue

        assert len(qargs) in (1, 2)
        qubit_indices = [qc.find_bit(qubit).index for qubit in qargs]
        active_qubits.update(qubit_indices)
        first_qubit_idx = qubit_indices[0]

        if len(qargs) == 1:
            res *= 1 - device[gate_type][first_qubit_idx,].error
        else:
            second_qubit_idx = qubit_indices[1]
            try:
                res *= 1 - device[gate_type][first_qubit_idx, second_qubit_idx].error
            except KeyError:
                msg = f"Error rate for gate {gate_type} on qubits {first_qubit_idx} and {second_qubit_idx} not found in device properties."
                raise KeyError(msg) from None

    assert device.qubit_properties is not None
    for qubit in active_qubits:
        properties = device.qubit_properties[qubit]
        assert properties is not None
        assert properties.t1 is not None
        assert properties.t2 is not None
        live_end = (
            last_end_per_qubit[qubit] if last_operation_per_qubit[qubit] in {"measure", "reset"} else circuit_duration
        )
        idle_time = max(live_end - execution_time_per_qubit[qubit], 0.0)
        res *= np.exp(-idle_time / min(properties.t1, properties.t2))
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
            if device.qubit_properties is None or not device.qubit_properties[qubit].t1 >= 0:
                msg = "No T1 qubit properties available"
                raise ValueError(msg)  # ruff:ignore[raise-within-try]
        except ValueError:
            logger.exception(message("T1", "idle", qubit))
            return False
        try:
            if device.qubit_properties is None or not device.qubit_properties[qubit].t2 >= 0:
                msg = "No T2 qubit properties available"
                raise ValueError(msg)  # ruff:ignore[raise-within-try]

        except ValueError:
            logger.exception(message("T2", "idle", qubit))
            return False
        try:
            error = device["measure"][qubit,].error
            if not (0 <= error <= 1):
                msg = "Error rate must be between 0 and 1."
                raise ValueError(msg)  # ruff:ignore[raise-within-try]
        except ValueError:
            logger.exception(message("Error", "readout", qubit))
            return False
        try:
            duration = device["measure"][qubit,].duration
            if not (duration >= 0):
                msg = "Duration must be >=0."
                raise ValueError(msg)  # ruff:ignore[raise-within-try]
        except ValueError:
            logger.exception(message("Duration", "readout", qubit))
            return False

        for gate in single_qubit_gates:
            try:
                error = device[gate][qubit,].error
                if not (0 <= error <= 1):
                    msg = "Error rate must be between 0 and 1."
                    raise ValueError(msg)  # ruff:ignore[raise-within-try]
            except ValueError:
                logger.exception(message("Error", gate, qubit))
                return False
            try:
                duration = device[gate][qubit,].duration
                if not (duration >= 0):
                    msg = "Duration must be >=0."
                    raise ValueError(msg)  # ruff:ignore[raise-within-try]
            except ValueError:
                logger.exception(message("Duration", gate, qubit))
                return False

    for gate in two_qubit_gates:
        for edge in device.build_coupling_map():
            try:
                error = device[gate][edge[0], edge[1]].error
                if not (0 <= error <= 1):
                    msg = "Error rate must be between 0 and 1."
                    raise ValueError(msg)  # ruff:ignore[raise-within-try]
            except ValueError:
                logger.exception(message("Error", gate, edge))
                return False
            try:
                duration = device[gate][edge[0], edge[1]].duration
                if not (duration >= 0):
                    msg = "Duration must be >=0."
                    raise ValueError(msg)  # ruff:ignore[raise-within-try]
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
