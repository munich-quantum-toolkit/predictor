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
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import qiskit.qasm2 as qasm2

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray
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
class CompilationStep:
    """A snapshot of the circuit state and environment metrics at a single timestep.

    Attributes:
        step_index: The current step number in the reinforcement learning episode.
        action: The string representation of the compilation pass applied (e.g., 'OptimizeCliffords').
        reward: The calculated reward value for applying this specific action.
        current_depth: The depth of the quantum circuit after the action was applied.
        num_qubits: The number of qubits in the circuit.
        gates_per_operation: The number of gates per operation occurring in the circuit.
        total_gates: The total number of gates included in the circuit.
        expected_fidelity: The expected fidelity of the circuit.
        fidelity_kind: The kind of fidelity value: 'exact' or 'approx'.
        fom_value: The figure of merit value for this compilation pass.
        fom_kind: The kind of fom value: 'exact' or 'approx'.
        synthesized: Whether the circuit has already been synthesized.
        laid_out: Whether the circuit has already been laid out.
        routed: Whether the circuit has already been routed.
        is_terminal: A flag indicating if the compilation process has concluded.
        circuit_qasm: The structural representation of the circuit in OpenQASM 2.0 format.
    """

    step_index: int
    action: str
    reward: float
    current_depth: int
    num_qubits: int
    gates_per_operation: dict[str, int]
    total_gates: int
    expected_fidelity: float
    fidelity_kind: str
    fom_value: float
    fom_kind: str
    synthesized: bool
    laid_out: bool
    routed: bool
    is_terminal: bool
    circuit_qasm: str
    program_communication: float
    critical_depth: float
    entanglement_ratio: float
    parallelism: float
    liveness: float


@dataclass
class CompilationTracer:
    """Aggregates compilation steps and metadata for export to a JSON file.

    This class acts as an in-memory buffer during the reinforcement learning compilation
    process. It tracks the physical transformations of the circuit and exports the
    entire episode as a structured JSON file upon termination.

    Attributes:
        circuit_name: The name of the circuit being compiled.
        figure_of_merit: The chosen figure of merit for this compilation.
        mdp_policy: The MDP transition policy.
        device: The target device metadata.
        schema_version: The version of this schema. Upgrade in case of schema changes to maintain compatibility with tracer frontend.
        timestamp: A timestamp indicating start of the compilation.
        steps: An ordered list of CompilationStep snapshots.
    """

    circuit_name: str
    figure_of_merit: str
    mdp_policy: str
    device: DeviceMetadata
    schema_version: str = "1.0.0"
    timestamp: float = field(default_factory=time.time)
    steps: list[CompilationStep] = field(default_factory=list)

    @classmethod
    def from_initial_state(
        cls,
        device: Target,
        circuit_name: str,
        figure_of_merit: str,
        mdp_policy: str,
    ) -> CompilationTracer:
        """Alternative constructor to build the tracer more conveniently from the environment's initial state."""
        device_meta = cls._extract_device_metadata(device)
        return cls(
            circuit_name=circuit_name, figure_of_merit=figure_of_merit, mdp_policy=mdp_policy, device=device_meta
        )

    def record_step(
        self,
        step_index: int,
        action: str,
        reward: float,
        current_qc: QuantumCircuit,
        expected_fidelity: float,
        fidelity_kind: str,
        fom_value: float,
        fom_kind: str,
        features: dict[str, int | NDArray[np.float32]],
        synthesized: bool,
        laid_out: bool,
        routed: bool,
        done: bool,
    ) -> None:
        """Records a single compilation action and the resulting circuit state.

        Args:
            step_index: The current step number in the environment.
            action: The name of the compilation pass that was just applied.
            reward: The calculated reward for the applied pass.
            current_qc: The current Qiskit QuantumCircuit object after the pass.
            expected_fidelity: The expected fidelity of the circuit after applying the pass.
            fidelity_kind: The kind of fidelity value: 'exact' or 'approx'.
            fom_value: The figure of merit value for the compilation pass.
            fom_kind: The kind of fom value: 'exact' or 'approx'.
            features: The quantum circuit's feature vector used by the RL agent.
            synthesized: Whether the circuit has already been synthesized.
            laid_out: Whether the circuit has already been laid out.
            routed: Whether the circuit has already been routed.
            done: Boolean indicating if this is the final step of the compilation.
        """
        present_ops_dict: dict[str, int] = {
            str(gate_name): int(count)
            for gate_name, count in current_qc.count_ops().items()
            if str(gate_name) != "barrier"
        }
        total_gates = sum(present_ops_dict.values())

        new_step = CompilationStep(
            step_index=step_index,
            action=action,
            reward=round(reward, 6),
            current_depth=current_qc.depth(),
            num_qubits=current_qc.num_qubits,
            gates_per_operation=present_ops_dict,
            total_gates=total_gates,
            expected_fidelity=round(expected_fidelity, 6),
            fidelity_kind=fidelity_kind,
            fom_value=round(fom_value, 6),
            fom_kind=fom_kind,
            is_terminal=done,
            circuit_qasm=qasm2.dumps(current_qc),
            program_communication=self._extract_float(features["program_communication"]),
            critical_depth=self._extract_float(features["critical_depth"]),
            entanglement_ratio=self._extract_float(features["entanglement_ratio"]),
            parallelism=self._extract_float(features["parallelism"]),
            liveness=self._extract_float(features["liveness"]),
            synthesized=synthesized,
            laid_out=laid_out,
            routed=routed,
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

    @staticmethod
    def _extract_float(val: int | NDArray[np.float32]) -> float:
        """Safely extracts a float from a scalar or a 1D NumPy array to satisfy linter requirements."""
        if isinstance(val, int):
            return float(val)
        return float(val[0])
