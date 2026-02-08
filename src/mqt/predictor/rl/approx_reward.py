# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""This module provides helper functions to approximate expected fidelity and estimated success probability (ESP) by transpiling a circuit to a device's basis gate set and combining resulting gate counts with calibration-derived per-gate error rates and durations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from qiskit import transpile

if TYPE_CHECKING:
    from qiskit import QuantumCircuit
    from qiskit.transpiler import Target

BLACKLIST: set[str] = {"measure", "reset", "delay", "barrier"}  # These gates do not directly contribute to the error


def get_basis_gates_from_target(device: Target) -> list[str]:
    """Return the basis gate names from a Qiskit Target."""
    return sorted([g for g in device.operation_names if g not in BLACKLIST])


def estimate_basis_gate_counts(qc: QuantumCircuit, *, basis_gates: list[str]) -> dict[str, int]:
    """Transpile ``qc`` to ``basis_gates`` and count occurrences of each basis gate."""
    qc_t = transpile(qc, basis_gates=basis_gates, optimization_level=1, seed_transpiler=42)
    counts = dict.fromkeys(basis_gates, 0)
    for ci in qc_t.data:
        name = ci.operation.name
        if name in BLACKLIST:
            continue
        if name in counts:
            counts[name] += 1
    return counts


def approx_expected_fidelity(
    qc: QuantumCircuit,
    *,
    device: Target,
    error_rates: dict[str, float],
) -> float:
    """Approximate expected fidelity using per-basis-gate error rates.

    The circuit is first transpiled to the device basis. Then a simple product
    model is applied: Π_g (1 - p_g)^{count_g}.

    Args:
        qc: Circuit to evaluate.
        device: Target providing the basis gate set.
        error_rates: Mapping ``basis_gate -> error_probability``.

    Returns:
        Approximate fidelity in [0, 1].
    """
    basis = get_basis_gates_from_target(device)
    counts = estimate_basis_gate_counts(qc, basis_gates=basis)
    f = 1.0
    for g, c in counts.items():
        f *= (1.0 - error_rates.get(g, 0.0)) ** c
    return float(max(min(f, 1.0), 0.0))


def approx_estimated_success_probability(
    qc: QuantumCircuit,
    *,
    device: Target,
    error_rates: dict[str, float],
    gate_durations: dict[str, float],
    tbar: float | None,
    par_feature: float,
    liv_feature: float,
    n_qubits: int,
) -> float:
    """Approximate ESP using per-basis-gate error rates, durations, and coherence.

    This combines:
    (1) a gate-infidelity product term, and
    (2) an idle/decoherence penalty based on an effective circuit duration.

    Args:
        qc: Circuit to evaluate.
        device: Target providing the basis gate set.
        error_rates: Mapping ``basis_gate -> error_probability``.
        gate_durations: Mapping ``basis_gate -> duration`` (seconds).
        tbar: Representative coherence time (seconds). If None, idle penalty is skipped.
        par_feature: Parallelism feature in [0, 1].
        liv_feature: Liveness feature in [0, 1].
        n_qubits: Number of qubits in the circuit.

    Returns:
        Approximate ESP in [0, 1].
    """
    basis = get_basis_gates_from_target(device)
    counts = estimate_basis_gate_counts(qc, basis_gates=basis)

    f_gate = 1.0
    for g, c in counts.items():
        f_gate *= (1.0 - error_rates.get(g, 0.0)) ** c

    n_q = max(n_qubits, 1)
    k_eff = 1.0 + (n_q - 1.0) * float(par_feature)

    total_gate_time = sum(counts[g] * gate_durations.get(g, 0.0) for g in basis) / k_eff

    idle_fraction = max(0.0, 1.0 - float(liv_feature))
    idle_factor = 1.0 if tbar is None or tbar <= 0.0 else float(np.exp(-(total_gate_time * idle_fraction) / tbar))

    esp = f_gate * idle_factor
    return float(max(min(esp, 1.0), 0.0))
