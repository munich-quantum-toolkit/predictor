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
    from qiskit.transpiler import Target


@dataclass
class DeviceMetadata:
    """Metadata containing information about the target quantum device for compilation."""

    description: str
    device_qubits: int


@dataclass
class InputCircuitMetadata:
    """Metadata containing information about the initial, uncompiled quantum circuit."""

    name: str
    num_qubits: int
    depth: int
    circuit_qasm: str


@dataclass
class CompilationStep:
    """A snapshot of the circuit state and environment metrics at a single timestep.

    Attributes:
        step_index: The current step number in the reinforcement learning episode.
        action: The string representation of the compilation pass applied (e.g., 'OptimizeCliffords').
        reward: The calculated reward value for applying this specific action.
        current_depth: The depth of the quantum circuit after the action was applied.
        is_terminal: A flag indicating if the compilation process has concluded.
        circuit_qasm: The structural representation of the circuit in OpenQASM 2.0 format.
    """

    step_index: int
    action: str
    reward: float
    current_depth: int
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
    def from_initial_state(cls, device: Target, input_circuit: QuantumCircuit, circuit_name: str) -> CompilationTracer:
        """Alternative constructor to build the tracer more conveniently from the environment's initial state."""
        device_meta = DeviceMetadata(
            description=device.description,
            device_qubits=device.num_qubits,
        )
        input_meta = InputCircuitMetadata(
            name=circuit_name,
            num_qubits=input_circuit.num_qubits,
            depth=input_circuit.depth(),
            circuit_qasm=qasm2.dumps(input_circuit),
        )

        return cls(device=device_meta, input_circuit=input_meta)

    def record_step(self, step_index: int, action: str, reward: float, current_qc: QuantumCircuit, done: bool) -> None:
        """Records a single compilation action and the resulting circuit state.

        Args:
            step_index: The current step number in the environment.
            action: The name of the compilation pass that was just applied.
            reward: The calculated reward for the applied pass.
            current_qc: The current Qiskit QuantumCircuit object after the pass.
            done: Boolean indicating if this is the final step of the compilation.
        """
        new_step = CompilationStep(
            step_index=step_index,
            action=action,
            reward=round(reward, 6),
            current_depth=current_qc.depth(),
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
