# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
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
from mqt.bench.utils import calc_supermarq_features

from mqt.predictor.hellinger import calc_device_specific_features, get_hellinger_model_path

if TYPE_CHECKING:
    from mqt.bench.devices import Device
    from qiskit import QuantumCircuit, QuantumRegister, Qubit
    from sklearn.ensemble import RandomForestRegressor

logger = logging.getLogger("mqt-predictor")

figure_of_merit = Literal[
    "expected_fidelity",
    "critical_depth",
    "estimated_success_probability",
    "estimated_hellinger_distance",
]


def crit_depth(qc: QuantumCircuit, precision: int = 10) -> float:
    """Calculates the critical depth of a given quantum circuit."""
    supermarq_features = calc_supermarq_features(qc)
    return float(np.round(1 - supermarq_features.critical_depth, precision).item())


def expected_fidelity(qc: QuantumCircuit, device: Device, precision: int = 10) -> float:
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
            first_qubit_idx = calc_qubit_index(qargs, qc.qregs, 0)

            if len(qargs) == 1:
                if gate_type == "measure":
                    specific_fidelity = device.get_readout_fidelity(first_qubit_idx)
                else:
                    specific_fidelity = device.get_single_qubit_gate_fidelity(gate_type, first_qubit_idx)
            else:
                second_qubit_idx = calc_qubit_index(qargs, qc.qregs, 1)
                specific_fidelity = device.get_two_qubit_gate_fidelity(gate_type, first_qubit_idx, second_qubit_idx)

            res *= specific_fidelity

    return float(np.round(res, precision).item())


def calc_qubit_index(qargs: list[Qubit], qregs: list[QuantumRegister], index: int) -> int:
    """Calculates the global qubit index for a given quantum circuit and qubit index."""
    offset = 0
    for reg in qregs:
        if qargs[index] not in reg:
            offset += reg.size
        else:
            qubit_index: int = offset + reg.index(qargs[index])
            return qubit_index
    error_msg = f"Global qubit index for local qubit {index} index not found."
    raise ValueError(error_msg)


def estimated_success_probability(qc: QuantumCircuit, device: Device, precision: int = 10) -> float:
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
    # lazy import of qiskit transpiler
    from qiskit.transpiler import InstructionDurations, Layout, PassManager, passes  # noqa: PLC0415
    from qiskit.transpiler.passes import ApplyLayout, SetLayout  # noqa: PLC0415

    # collect gate and measurement durations for active qubits
    op_times, active_qubits = [], set()
    for instruction, qargs, _cargs in qc.data:
        gate_type = instruction.name

        if gate_type == "barrier" or gate_type == "id":
            continue
        assert len(qargs) in (1, 2)
        first_qubit_idx = calc_qubit_index(qargs, qc.qregs, 0)
        active_qubits.add(first_qubit_idx)

        if len(qargs) == 1:  # single-qubit gate
            if gate_type == "measure":
                duration = device.get_readout_duration(first_qubit_idx)
            else:
                duration = device.get_single_qubit_gate_duration(gate_type, first_qubit_idx)
            op_times.append((gate_type, [first_qubit_idx], duration, "s"))
        else:  # multi-qubit gate
            second_qubit_idx = calc_qubit_index(qargs, qc.qregs, 1)
            active_qubits.add(second_qubit_idx)
            duration = device.get_two_qubit_gate_duration(gate_type, first_qubit_idx, second_qubit_idx)
            op_times.append((gate_type, [first_qubit_idx, second_qubit_idx], duration, "s"))

    # check whether the circuit was transformed by tket (i.e. changed register name)
    # qiskit ASAPScheduleAnalysis expects all qubit registers to be named 'q'
    if qc.qregs[0].name != "q":
        # create a layout that maps the (tket) 'node' registers to the (qiskit) 'q' registers
        layouts = [SetLayout(Layout({node_qubit: i for i, node_qubit in enumerate(node_reg)})) for node_reg in qc.qregs]
        # create a pass manager with the SetLayout and ApplyLayout passes
        pm = PassManager(list(layouts))
        pm.append(ApplyLayout())

        # replace the 'node' register with the 'q' register in the circuit
        qc = pm.run(qc)
        assert qc.qregs[0].name == "q"

    # associate gate and idle (delay) times for each qubit through asap scheduling
    sched_pass = passes.ASAPScheduleAnalysis(InstructionDurations(op_times))
    delay_pass = passes.PadDelay()
    pm = PassManager([sched_pass, delay_pass])
    scheduled_circ = pm.run(qc)

    res = 1.0
    for instruction, qargs, _cargs in scheduled_circ.data:
        gate_type = instruction.name

        if gate_type == "barrier" or gate_type == "id":
            continue

        assert len(qargs) in (1, 2)
        first_qubit_idx = calc_qubit_index(qargs, qc.qregs, 0)

        if len(qargs) == 1:
            if gate_type == "measure":
                res *= device.get_readout_fidelity(first_qubit_idx)
                continue

            if gate_type == "delay":
                # only consider active qubits
                if first_qubit_idx not in active_qubits:
                    continue

                res *= np.exp(
                    -instruction.duration
                    / min(device.calibration.get_t1(first_qubit_idx), device.calibration.get_t2(first_qubit_idx))
                )
                continue

            res *= device.get_single_qubit_gate_fidelity(gate_type, first_qubit_idx)
        else:
            second_qubit_idx = calc_qubit_index(qargs, qc.qregs, 1)
            res *= device.get_two_qubit_gate_fidelity(gate_type, first_qubit_idx, second_qubit_idx)

    return float(np.round(res, precision).item())


def esp_data_available(device: Device) -> bool:
    """Check if calibration data to calculate ESP is available for the device."""
    # TODO: consider removal once the `Result` class is no longer used

    def message(calibration: str, operation: str, target: int | str) -> str:
        return f"{calibration} data for {operation} operation on qubit(s) {target} is required to calculate ESP for device {device.name}."

    for qubit in range(device.num_qubits):
        try:
            device.calibration.get_t1(qubit)
        except ValueError:
            logger.exception(message("T1", "idle", qubit))
            return False
        try:
            device.calibration.get_t2(qubit)
        except ValueError:
            logger.exception(message("T2", "idle", qubit))
            return False
        try:
            device.get_readout_fidelity(qubit)
        except ValueError:
            logger.exception(message("Fidelity", "readout", qubit))
            return False
        try:
            device.get_readout_duration(qubit)
        except ValueError:
            logger.exception(message("Duration", "readout", qubit))
            return False

        for gate in device.get_single_qubit_gates():
            try:
                device.get_single_qubit_gate_fidelity(gate, qubit)
            except ValueError:
                logger.exception(message("Fidelity", gate, qubit))
                return False
            try:
                device.get_single_qubit_gate_duration(gate, qubit)
            except ValueError:
                logger.exception(message("Duration", gate, qubit))
                return False

    for gate in device.get_two_qubit_gates():
        for edge in device.coupling_map:
            try:
                device.get_two_qubit_gate_fidelity(gate, edge[0], edge[1])
            except ValueError:
                logger.exception(message("Fidelity", gate, edge))
                return False
            try:
                device.get_two_qubit_gate_duration(gate, edge[0], edge[1])
            except ValueError:
                logger.exception(message("Duration", gate, edge))
                return False

    return True


def estimated_hellinger_distance(
    qc: QuantumCircuit, device: Device, model: RandomForestRegressor | None = None, precision: int = 10
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
