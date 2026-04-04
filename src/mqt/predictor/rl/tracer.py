# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Visualization module for recording and exporting the RL compilation process."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import qiskit.qasm2 as qasm2

if TYPE_CHECKING:
    from qiskit import QuantumCircuit
    from qiskit.transpiler import InstructionProperties, Target


@dataclass
class TopologyEdge:
    """Represents a topology edge between two qubits."""

    control: int
    target: int


@dataclass
class GateCalibration:
    """Calibration data for a specific gate on a specific set of qubits."""

    qubits: list[int]
    duration: float | None
    error: float | None


@dataclass
class DeviceMetadata:
    """Metadata containing information about the target quantum device for compilation."""

    description: str
    device_qubits: int
    native_gates: list[str]
    topology: list[TopologyEdge]
    calibration_data: dict[str, list[GateCalibration]]


@dataclass
class InputCircuitMetadata:
    """Metadata containing information about the initial, uncompiled quantum circuit."""

    name: str
    num_qubits: int
    depth: int
    figure_of_merit: str
    circuit_qasm: str


@dataclass
class CompilationStep:
    """A snapshot of the circuit state and environment metrics at a single timestep.

    Attributes:
        step_index: The current step number in the reinforcement learning episode.
        action: The string representation of the compilation pass applied (e.g., 'OptimizeCliffords').
        reward: The calculated reward value for applying this specific action.
        current_depth: The depth of the quantum circuit after the action was applied.
        total_gates: The total number of gates included in the circuit.
        fom_value: The figure of merit value for this compilation pass.
        fom_kind: The kind of fom value: 'exact' or 'approx'.
        is_terminal: A flag indicating if the compilation process has concluded.
        circuit_qasm: The structural representation of the circuit in OpenQASM 2.0 format.
    """

    step_index: int
    action: str
    reward: float
    current_depth: int
    total_gates: int
    fom_value: float
    fom_kind: str
    is_terminal: bool
    circuit_qasm: str


@dataclass
class CompilationTracer:
    """Aggregates compilation steps and metadata for export to a JSON file.

    This class acts as an in-memory buffer during the reinforcement learning compilation
    process. It tracks the physical transformations of the circuit and exports the
    entire episode as a structured JSON file upon termination.

    Attributes:
        device: The target device metadata.
        input_circuit: The uncompiled circuit metadata.
        steps: An ordered list of CompilationStep snapshots.
    """

    device: DeviceMetadata
    input_circuit: InputCircuitMetadata
    steps: list[CompilationStep] = field(default_factory=list)

    @classmethod
    def from_initial_state(
        cls, device: Target, input_circuit: QuantumCircuit, circuit_name: str, figure_of_merit: str
    ) -> CompilationTracer:
        """Alternative constructor to build the tracer more conveniently from the environment's initial state."""
        device_meta = cls._extract_device_metadata(device)
        input_meta = cls._extract_circuit_metadata(input_circuit, circuit_name, figure_of_merit)
        return cls(device=device_meta, input_circuit=input_meta)

    def record_step(
        self,
        step_index: int,
        action: str,
        reward: float,
        current_qc: QuantumCircuit,
        fom_value: float,
        fom_kind: str,
        done: bool,
    ) -> None:
        """Records a single compilation action and the resulting circuit state.

        Args:
            step_index: The current step number in the environment.
            action: The name of the compilation pass that was just applied.
            reward: The calculated reward for the applied pass.
            current_qc: The current Qiskit QuantumCircuit object after the pass.
            fom_value: The figure of merit value for the compilation pass.
            fom_kind: The kind of fom value: 'exact' or 'approx'.
            done: Boolean indicating if this is the final step of the compilation.
        """
        present_ops_dict = current_qc.count_ops()
        total_gates = sum(present_ops_dict.values()) if present_ops_dict else 0

        new_step = CompilationStep(
            step_index=step_index,
            action=action,
            reward=round(reward, 6),
            current_depth=current_qc.depth(),
            total_gates=total_gates,
            fom_value=round(fom_value, 6),
            fom_kind=fom_kind,
            is_terminal=done,
            circuit_qasm=qasm2.dumps(current_qc),
        )
        self.steps.append(new_step)

    def save_to_json(self, filepath: str | Path) -> None:
        """Serializes the metadata and all recorded steps to a JSON file.

        Args:
            filepath: The destination path or filename for the output JSON file.
        """
        with Path(filepath).open("w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=4)

    @staticmethod
    def _extract_circuit_metadata(
        input_circuit: QuantumCircuit, circuit_name: str, figure_of_merit: str
    ) -> InputCircuitMetadata:
        """Internal helper to parse the initial quantum circuit."""
        return InputCircuitMetadata(
            name=circuit_name,
            num_qubits=input_circuit.num_qubits,
            depth=input_circuit.depth(),
            figure_of_merit=figure_of_merit,
            circuit_qasm=qasm2.dumps(input_circuit),
        )

    @staticmethod
    def _extract_device_metadata(device: Target) -> DeviceMetadata:
        """Internal helper to extract topology and calibration data from the device."""
        native_gates = list(device.operation_names)
        cmap = device.build_coupling_map()
        topology = [TopologyEdge(control=c, target=t) for c, t in cmap] if cmap is not None else []
        calibration_data: dict[str, list[GateCalibration]] = {}

        for gate_name in native_gates:
            gate_calibrations = []
            props: InstructionProperties
            qubit_tuples: tuple[int, ...]

            for qubit_tuples, props in device[gate_name].items():
                if qubit_tuples is None or props is None:
                    continue

                gate_calibrations.append(
                    GateCalibration(qubits=list(qubit_tuples), duration=props.duration, error=props.error)
                )

            calibration_data[gate_name] = gate_calibrations

        return DeviceMetadata(
            description=device.description,
            device_qubits=device.num_qubits,
            native_gates=native_gates,
            topology=topology,
            calibration_data=calibration_data,
        )
