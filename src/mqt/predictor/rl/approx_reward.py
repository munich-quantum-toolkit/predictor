# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Approximate reward calculations for intermediate RL compilation states."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import BasisTranslator

if TYPE_CHECKING:
    from qiskit import QuantumCircuit
    from qiskit.transpiler import InstructionProperties, Target


_EXCLUDED_OPERATIONS = {
    "barrier",
    "delay",
    "id",
    "if_else",
    "while_loop",
    "for_loop",
    "switch_case",
    "box",
    "break",
    "continue",
}


def _operation_arity(device: Target, name: str) -> int | None:
    """Return the arity of a target operation if it behaves like a gate."""
    try:
        operation = device.operation_from_name(name)
    except KeyError:
        return None

    try:
        return int(operation.num_qubits)
    except (AttributeError, TypeError, ValueError):
        return None


def _basis_gates(device: Target) -> list[str]:
    """Return gate-like target operations used for approximate rewards."""
    basis_gates = [
        name
        for name in device.operation_names
        if name not in _EXCLUDED_OPERATIONS and _operation_arity(device, name) is not None
    ]
    if "reset" in basis_gates and not device["reset"]:
        basis_gates.remove("reset")
    return sorted(basis_gates)


def _basis_gate_counts(qc: QuantumCircuit, basis_gates: list[str]) -> dict[str, int]:
    """Translate a circuit to the target basis and count its operations."""
    translated = PassManager([BasisTranslator(SessionEquivalenceLibrary, basis_gates)]).run(qc)
    counts = dict.fromkeys(basis_gates, 0)
    for instruction in translated.data:
        name = instruction.operation.name
        if name in counts:
            counts[name] += 1
    return counts


def approximate_expected_fidelity(
    qc: QuantumCircuit,
    *,
    device: Target,
    error_rates: dict[str, float],
) -> float:
    """Approximate expected fidelity using average per-gate error rates."""
    counts = _basis_gate_counts(qc, _basis_gates(device))
    fidelity = 1.0
    for gate, count in counts.items():
        fidelity *= (1.0 - error_rates.get(gate, 0.0)) ** count
    return float(np.clip(fidelity, 0.0, 1.0))


def approximate_estimated_success_probability(
    qc: QuantumCircuit,
    *,
    device: Target,
    error_rates: dict[str, float],
    gate_durations: dict[str, float],
    coherence_time: float | None,
    parallelism: float,
    liveness: float,
) -> float:
    """Approximate ESP from gate errors, duration, and circuit-level features."""
    basis_gates = _basis_gates(device)
    counts = _basis_gate_counts(qc, basis_gates)

    gate_fidelity = 1.0
    for gate, count in counts.items():
        gate_fidelity *= (1.0 - error_rates.get(gate, 0.0)) ** count

    effective_parallelism = 1.0 + (max(qc.num_qubits, 1) - 1.0) * parallelism
    total_gate_time = sum(counts[gate] * gate_durations.get(gate, 0.0) for gate in basis_gates) / effective_parallelism
    idle_fraction = max(0.0, 1.0 - liveness)
    idle_factor = (
        1.0
        if coherence_time is None or coherence_time <= 0.0
        else float(np.exp(-(total_gate_time * idle_fraction) / coherence_time))
    )
    return float(np.clip(gate_fidelity * idle_factor, 0.0, 1.0))


def average_target_calibration(
    device: Target,
) -> tuple[dict[str, float], dict[str, float], float | None]:
    """Return per-gate error/duration averages and a representative coherence time."""
    try:
        num_qubits = device.num_qubits
        coupling_map = device.build_coupling_map()
        qubit_properties = device.qubit_properties
    except AttributeError as exc:
        msg = "Device target does not expose the required API for approximate reward computation."
        raise RuntimeError(msg) from exc

    basis_gates = _basis_gates(device)
    edges = coupling_map.get_edges() if coupling_map is not None else []

    def get_properties(name: str, qubits: tuple[int, ...]) -> InstructionProperties | None:
        return device[name].get(qubits)

    error_samples: dict[str, list[float]] = {name: [] for name in basis_gates}
    duration_samples: dict[str, list[float]] = {name: [] for name in basis_gates}

    for name in basis_gates:
        arity = _operation_arity(device, name)
        qubit_tuples: list[tuple[int, ...]]
        if arity == 1:
            qubit_tuples = [(qubit,) for qubit in range(num_qubits)]
        elif arity == 2:
            qubit_tuples = [tuple(edge) for edge in edges]
        else:
            continue

        for qubits in qubit_tuples:
            properties = get_properties(name, qubits)
            if properties is None and len(qubits) == 2:
                properties = get_properties(name, (qubits[1], qubits[0]))
            if properties is None:
                continue
            if properties.error is not None:
                error_samples[name].append(float(properties.error))
            if properties.duration is not None:
                duration_samples[name].append(float(properties.duration))

    all_errors = [error for samples in error_samples.values() for error in samples]
    all_durations = [duration for samples in duration_samples.values() for duration in samples]
    if not all_errors and not all_durations:
        msg = "No valid calibration data found in target; cannot compute approximate reward."
        raise RuntimeError(msg)

    fallback_error = float(np.mean(all_errors)) if all_errors else 0.0
    fallback_duration = float(np.mean(all_durations)) if all_durations else 0.0
    error_rates = {
        name: float(np.mean(samples)) if samples else fallback_error for name, samples in error_samples.items()
    }
    gate_durations = {
        name: float(np.mean(samples)) if samples else fallback_duration for name, samples in duration_samples.items()
    }

    coherence_samples: list[float] = []
    if qubit_properties:
        for properties in qubit_properties:
            if properties is None:
                continue
            values = [value for value in (properties.t1, properties.t2) if value is not None]
            if values:
                coherence_samples.append(float(min(values)))

    coherence_time = float(np.median(coherence_samples)) if coherence_samples else None
    return error_rates, gate_durations, coherence_time
