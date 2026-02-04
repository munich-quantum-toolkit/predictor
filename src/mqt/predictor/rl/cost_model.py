# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Helper functions for approximating transformations to device-native gates.

This module provides a dynamic canonical gate cost model and approximate
fidelity/ESP estimates based on averaged 1q/2q error rates.

For each backend, a cost table of gate decompositions into the native gate set
is generated programmatically (and cached). This avoids rigid hard-coding of
costs. If a backend is unknown, a default basis (IBM Qiskit basis) is used as
a fallback with a warning, or users can extend the known device basis list.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Mapping
from typing import cast

import numpy as np

# Attempt to import Qiskit for transpilation
from qiskit import QuantumCircuit, transpile

logger = logging.getLogger(__name__)

CanonicalCostTable = dict[str, tuple[int, int]]

# Cache for generated cost tables
DEVICE_COST_CACHE: dict[str, dict[str, tuple[int, int]]] = {}

# Pre-defined native gate sets for known devices (can be extended)
KNOWN_DEVICE_BASES: dict[str, list[str]] = {
    "ibm_torino": [
        "id",
        "rz",
        "rx",
        "sx",
        "x",
        "cz",
        "rzz",
    ],  # IBM device example (native 1q: id/rz/rx/sx/x, 2q: cz, rzz)
    "ankaa_3": ["id", "rz", "rx", "iswap"],  # Rigetti Ankaa-3 (native 1q: rx, rz; native 2q: iSWAP)
    "emerald": [
        "id",
        "rz",
        "rx",
        "cz",
        "u",
    ],  # IQM Emerald (native 1q: arbitrary single-qubit rotation 'u'; native 2q: cz)
    # Additional devices can be added here...
}

# Heuristic set of known two-qubit basis gate names used to map averages to per-basis values.
# This is intentionally conservative; device-specific basis sets should be used when available.
TWO_Q_GATES: set[str] = {
    "cx",
    "cz",
    "iswap",
    "rzz",
    "rxx",
    "ryy",
    "rzx",
    "dcx",
    "ecr",
    "swap",
}


def build_error_rates_from_averages(device_id: str, p1_avg: float, p2_avg: float) -> dict[str, float]:
    """Construct a per-basis error rate mapping from averaged 1q/2q values.

    This uses a simple heuristic to decide whether a basis gate is 1q or 2q.
    """
    basis_gates = KNOWN_DEVICE_BASES.get(device_id, ["id", "rz", "sx", "x", "cx"])
    error_rates: dict[str, float] = {}
    for g in basis_gates:
        error_rates[g] = p2_avg if g in TWO_Q_GATES else p1_avg
    return error_rates


def build_gate_durations_from_averages(device_id: str, tau1_avg: float, tau2_avg: float) -> dict[str, float]:
    """Construct a per-basis gate duration mapping from averaged 1q/2q durations."""
    basis_gates = KNOWN_DEVICE_BASES.get(device_id, ["id", "rz", "sx", "x", "cx"])
    durations: dict[str, float] = {}
    for g in basis_gates:
        durations[g] = tau2_avg if g in TWO_Q_GATES else tau1_avg
    return durations


def generate_cost_table(device_id: str) -> dict[str, tuple[int, int]]:
    """Generate a canonical gate cost table for the given device_id.

    This function programmatically derives the (n_1q, n_2q) costs for common gates
    by decomposing them into the device's native gate set via Qiskit transpilation.
    If the device_id is not recognized in KNOWN_DEVICE_BASES, a generic basis
    is assumed (using IBM's basis as a fallback) and a warning is emitted.
    """
    if transpile is None or QuantumCircuit is None:
        msg = "Qiskit is required to generate cost tables dynamically."
        raise ImportError(msg)

    # Determine the basis gates for this device
    basis_gates = KNOWN_DEVICE_BASES.get(device_id)
    if basis_gates is None:
        warnings.warn(
            f"No native gate-set defined for device '{device_id}'. "
            "Generating cost table using a minimal universal basis (Qiskit default). "
            "Results may be inaccurate. Consider specifying the gate set in KNOWN_DEVICE_BASES.",
            UserWarning,
            stacklevel=2,
        )
        logger.warning(f"No basis for device '{device_id}', using minimal universal basis for cost generation.")
        # Default to minimal universal basis (Qiskit default)
        basis_gates = ["id", "rz", "sx", "x", "cx"]

    cost_table: dict[str, tuple[int, int]] = {}

    # Structured gate definitions for dynamic profiling
    gate_profiles = [
        # Single-qubit gates (no params)
        {
            "gates": ["id", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "sx", "sxdg", "u0"],
            "qubits": 1,
            "params": 0,
            "controls": 0,
        },
        # Single-qubit gates (1 param)
        {"gates": ["p", "rx", "ry", "rz", "u1", "r"], "qubits": 1, "params": 1, "controls": 0},
        # Single-qubit gates (2 params)
        {"gates": ["u2"], "qubits": 1, "params": 2, "controls": 0},
        # Single-qubit gates (3 params)
        {"gates": ["u", "u3"], "qubits": 1, "params": 3, "controls": 0},
        # Two-qubit gates (no params)
        {
            "gates": ["cx", "cy", "cz", "ch", "csx", "swap", "iswap", "dcx", "ecr"],
            "qubits": 2,
            "params": 0,
            "controls": 0,
        },
        # Two-qubit gates (1 param)
        {"gates": ["rxx", "ryy", "rzz", "rzx", "cu1", "cp"], "qubits": 2, "params": 1, "controls": 0},
        # Controlled single-qubit gates (1 param)
        {"gates": ["crx", "cry", "crz"], "qubits": 1, "params": 1, "controls": 1},
        # Controlled U3 (3 params)
        {"gates": ["cu3", "cu"], "qubits": 1, "params": 3, "controls": 1},
        # Multi-qubit gates (no params)
        {"gates": ["ccx", "cswap", "rccx", "rc3x", "c3x", "c3sqrtx", "c4x"], "qubits": 1, "params": 0, "controls": 2},
    ]

    def add_gate_to_cost_table(gate: str, qubits: int, params: int, controls: int) -> None:
        total_qubits = qubits + controls
        qc = QuantumCircuit(total_qubits)
        try:
            gate_name = "c" * controls + gate if controls > 0 else gate
            if params == 0:
                getattr(qc, gate_name)(*list(range(total_qubits)))
            else:
                param_values = list(range(1, params + 1))
                getattr(qc, gate_name)(*param_values, *list(range(total_qubits)))
        except Exception:
            return  # skip if not available
        # For reference: store the transpiled circuit size (total basis gate count)
        qc_trans = transpile(qc, basis_gates=basis_gates, optimization_level=1, seed_transpiler=42)
        cost_table[gate if controls == 0 else ("c" * controls) + gate] = (qc_trans.size(), 0)

    for profile in gate_profiles:
        gates = cast("list[str]", profile["gates"])
        qubits = int(cast("int", profile["qubits"]))
        params = int(cast("int", profile.get("params", 0)))
        controls = int(cast("int", profile.get("controls", 0)))
        for gate in gates:
            add_gate_to_cost_table(gate, qubits, params, controls)

    # Ensure 'id' is treated as no-op (if not already in table due to optimization removal)
    cost_table["id"] = (0, 0)

    return cost_table


def get_cost_table(device_id: str) -> CanonicalCostTable:
    """Return the canonical cost table for `device_id`, generating it if necessary.

    If the device is unknown (not predefined), the cost table is generated using a
    default basis and a warning is emitted to indicate a potential inaccuracy.
    The result is cached to avoid repeated computation.
    """
    if device_id not in DEVICE_COST_CACHE:
        # Generate and cache the cost table for this device
        DEVICE_COST_CACHE[device_id] = generate_cost_table(device_id)
    return DEVICE_COST_CACHE[device_id]


# --- Helper: Estimate basis gate counts in a transpiled circuit ---
def estimate_basis_gate_counts(qc: QuantumCircuit, *, basis_gates: list[str]) -> dict[str, int]:
    """Estimate the count of each basis gate in the transpiled circuit."""
    qc_trans = transpile(qc, basis_gates=basis_gates, optimization_level=1, seed_transpiler=42)
    gate_counts = dict.fromkeys(basis_gates, 0)
    for instr, _, _ in qc_trans.data:
        name = instr.name
        if name in gate_counts:
            gate_counts[name] += 1
    return gate_counts


def approx_expected_fidelity(
    qc: QuantumCircuit,
    error_rates: dict[str, float],
    *,
    device_id: str = "ibm_torino",
) -> float:
    """Estimate expected fidelity using per-basis-gate error rates.

    Args:
        qc: QuantumCircuit to analyze
        error_rates: dict mapping basis gate name to error rate (e.g., {"cx": 0.01, "rz": 0.001, ...})
        device_id: device identifier for basis gates
    Returns:
        Estimated total fidelity as a float in [0, 1]
    """
    basis_gates = KNOWN_DEVICE_BASES.get(device_id, ["id", "rz", "sx", "x", "cx"])
    gate_counts = estimate_basis_gate_counts(qc, basis_gates=basis_gates)
    fidelity = 1.0
    for gate, count in gate_counts.items():
        p = error_rates.get(gate, 0.0)
        fidelity *= (1.0 - p) ** count
    return float(max(min(fidelity, 1.0), 0.0))


def approx_estimated_success_probability(
    qc: QuantumCircuit,
    error_rates: dict[str, float],
    gate_durations: dict[str, float],
    tbar: float | None,
    par_feature: float,
    liv_feature: float,
    n_qubits: int,
    *,
    device_id: str = "ibm_torino",
) -> float:
    """Estimate the Estimated Success Probability (ESP) using per-basis-gate error rates and durations.

    Args:
        qc: QuantumCircuit to analyze
        error_rates: dict mapping basis gate name to error rate
        gate_durations: dict mapping basis gate name to average duration (in same units as tbar)
        tbar: average T1/T2 time (decoherence time)
        par_feature: parallelism feature (0=serial, 1=fully parallel)
        liv_feature: liveness feature (fraction of time qubits are active)
        n_qubits: number of qubits in the circuit
        device_id: device identifier for basis gates
    Returns:
        Estimated ESP as a float in [0, 1]
    """
    basis_gates = KNOWN_DEVICE_BASES.get(device_id, ["id", "rz", "sx", "x", "cx"])
    gate_counts = estimate_basis_gate_counts(qc, basis_gates=basis_gates)
    # Fidelity from gate operations
    f_gate = 1.0
    for gate, count in gate_counts.items():
        p = error_rates.get(gate, 0.0)
        f_gate *= (1.0 - p) ** count

    # Estimate effective circuit duration based on parallelism
    n_q = max(n_qubits, 1)
    k_eff = 1.0 + (n_q - 1.0) * float(par_feature)
    # Total gate time: sum over all basis gates
    total_gate_time = sum(gate_counts[g] * gate_durations.get(g, 0.0) for g in basis_gates) / k_eff

    # Idle time penalty factor based on liveness
    idle_fraction = max(0.0, 1.0 - float(liv_feature))
    idle_factor = 1.0 if tbar is None or tbar <= 0.0 else float(np.exp(-(total_gate_time * idle_fraction) / tbar))

    esp = f_gate * idle_factor
    return float(max(min(esp, 1.0), 0.0))
