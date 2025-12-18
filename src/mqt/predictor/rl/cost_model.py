# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Helper functions for approximating transformations to device-native gates.

This module provides a simple canonical gate cost model and approximate
fidelity/ESP estimates based on averaged 1q/2q error rates.

It ships hand-crafted canonical cost tables for a small set of backends
(currently: ``ibm_torino``, ``ankaa_3``, ``emerald``).

Support for additional devices is not automatic: for each new backend,
add a corresponding canonical cost table (and, if needed, device-specific
approximations) to ``DEVICE_CANONICAL_COSTS``.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from qiskit import QuantumCircuit

logger = logging.getLogger(__name__)

CanonicalCostTable = Mapping[str, tuple[int, int]]

# ---------------------------------------------------------------------------
# Canonical cost tables
# ---------------------------------------------------------------------------

TORINO_CANONICAL_COSTS: CanonicalCostTable = {
    # native 1q
    "rz": (1, 0),
    "rx": (1, 0),
    "sx": (1, 0),
    "x": (1, 0),
    "id": (0, 0),  # treat as no-op for fidelity; timing can be handled elsewhere
    # native 2q
    "cz": (0, 1),
    "rzz": (0, 1),
    # ------------------------------------------------------------------
    # Common 1q non-natives decomposed into {rz, rx, sx, x}
    # ------------------------------------------------------------------
    "u": (3, 0),  # generic U(θ, φ, λ) ~ 3 Euler angles
    "u3": (3, 0),
    "u2": (2, 0),
    "h": (3, 0),  # H ≈ Rz(π) • SX • Rz(π) up to phase
    "ry": (3, 0),  # Ry ≈ Rz(-π/2) • Rx(θ) • Rz(π/2)
    "s": (1, 0),  # S = Rz(π/2)
    "sdg": (1, 0),
    "t": (1, 0),  # T = Rz(π/4)
    "tdg": (1, 0),
    # ------------------------------------------------------------------
    # Common 2q gates expressed in a CZ + 1q basis (approximate)
    # ------------------------------------------------------------------
    "rxx": (4, 1),  # ~4 single-qubit rotations + 1 entangler (rzz/cz)
    # Controlled-1q rotations / phases:
    #   roughly: 1 CZ + a few single-qubit rotations on control/target.
    "crx": (4, 1),
    "cry": (4, 1),
    "crz": (4, 1),
    "cp": (4, 1),
    "cu1": (4, 1),
    "cu3": (6, 1),
    "cu": (6, 1),
    "ch": (6, 1),
    "cy": (6, 1),
    "csx": (4, 1),
    "cx": (6, 1),  # CX = H(t) • CZ • H(t) -> ~2*H + 1*CZ => ~6 singles + 1 two-qubit
    "czx": (6, 1),
    "swap": (12, 3),  # SWAP ≈ 3 CX; in a CZ basis still ~3 two-qubit gates
}

ANKAA3_CANONICAL_COSTS: CanonicalCostTable = {
    "rx": (1, 0),
    "rz": (1, 0),
    "iswap": (0, 1),
    "u": (3, 0),
    "u3": (3, 0),
    "u2": (2, 0),
    "h": (3, 0),
    "ry": (3, 0),
    "s": (1, 0),
    "sdg": (1, 0),
    "t": (1, 0),
    "tdg": (1, 0),
    "rzz": (4, 2),  # ~2 iSWAP + ~4 1q rotations
    "rxx": (4, 2),
    # controlled gates: ~ 2 iSWAP + some 1q each
    "crx": (6, 2),
    "cry": (6, 2),
    "crz": (6, 2),
    "cp": (6, 2),
    "cu1": (6, 2),
    "cu3": (8, 2),
    "cu": (8, 2),
    "ch": (8, 2),
    "cy": (8, 2),
    "csx": (6, 2),
    "swap": (12, 3),
}

EMERALD_CANONICAL_COSTS: CanonicalCostTable = {
    # native
    "rz": (1, 0),
    "rx": (1, 0),
    "r": (1, 0),
    "cz": (0, 1),
    "u": (1, 0),
    "u3": (1, 0),
    "u2": (1, 0),
    "h": (1, 0),
    "ry": (1, 0),
    "s": (1, 0),
    "sdg": (1, 0),
    "t": (1, 0),
    "tdg": (1, 0),
    "rzz": (4, 2),
    "rxx": (4, 2),
    "crx": (4, 1),
    "cry": (4, 1),
    "crz": (4, 1),
    "cp": (4, 1),
    "cu1": (4, 1),
    "cu3": (6, 1),
    "cu": (6, 1),
    "ch": (6, 1),
    "cy": (6, 1),
    "csx": (4, 1),
    "swap": (12, 3),
}

DEVICE_CANONICAL_COSTS: dict[str, CanonicalCostTable] = {
    "ibm_torino": TORINO_CANONICAL_COSTS,
    "ankaa_3": ANKAA3_CANONICAL_COSTS,
    "emerald": EMERALD_CANONICAL_COSTS,
}


def get_cost_table(device_id: str) -> CanonicalCostTable:
    """Return the canonical cost table for ``device_id``, with a safe fallback.

    If the device is unknown, a warning is emitted and the ``ibm_torino`` table
    is used as a generic fallback. This keeps the code running, but the
    approximation should be treated with care.
    """
    table = DEVICE_CANONICAL_COSTS.get(device_id)
    if table is None:
        msg = (
            f"No canonical cost table defined for device '{device_id}'. "
            "Falling back to 'ibm_torino' table; approximate metrics may "
            "be inaccurate. Consider adding a dedicated entry to "
            "DEVICE_CANONICAL_COSTS."
        )
        warnings.warn(msg, UserWarning, stacklevel=3)
        logger.warning(msg)
        table = TORINO_CANONICAL_COSTS
    return table


def canonical_cost(
    gate_name: str,
    *,
    device_id: str = "ibm_torino",
) -> tuple[int, int]:
    """Return (n_1q, n_2q) cost for ``gate_name`` on the given device.

    Note:
        Hand-crafted tables are available for a small set of backends
        (see ``DEVICE_CANONICAL_COSTS``). For additional devices, extend
        ``DEVICE_CANONICAL_COSTS`` accordingly.
    """
    table = get_cost_table(device_id)
    return table.get(gate_name, (0, 0))


def estimate_counts(
    qc: QuantumCircuit,
    *,
    cost_table: CanonicalCostTable,
) -> tuple[int, int]:
    """Estimate canonical (n_1q, n_2q) counts for a circuit.

    Uses the provided ``cost_table`` where available and a simple, conservative
    fallback otherwise (3*1q for unknown 1q gates, 1*2q + 4*1q for unknown 2q gates).
    """
    n_1q = 0
    n_2q = 0

    for circuit_instr in qc.data:
        name = circuit_instr.operation.name
        qargs = circuit_instr.qubits

        # Ignore non-unitary / timing-only ops for this count
        if name in ("barrier", "delay", "measure"):
            continue

        cost = cost_table.get(name)
        if cost is None:
            # Conservative fallback by arity (only used for gates missing in the table)
            if len(qargs) == 1:
                n_1q += 3
            elif len(qargs) == 2:
                n_2q += 1
                n_1q += 4
        else:
            n_1q += cost[0]
            n_2q += cost[1]
    return n_1q, n_2q


def approx_expected_fidelity(
    qc: QuantumCircuit,
    p1_avg: float,
    p2_avg: float,
    *,
    device_id: str = "ibm_torino",
) -> float:
    """Approximate expected fidelity from canonical gate counts.

    Args:
        qc: Circuit for which to estimate fidelity.
        p1_avg: Average single-qubit error probability across the device.
        p2_avg: Average two-qubit error probability across the device.
        device_id: Identifier of the backend (used to select the cost table).

    Returns:
        Approximate expected fidelity in [0, 1].
    """
    cost_table = get_cost_table(device_id)
    n_1q, n_2q = estimate_counts(qc, cost_table=cost_table)

    f_1q = (1.0 - p1_avg) ** max(n_1q, 0)
    f_2q = (1.0 - p2_avg) ** max(n_2q, 0)
    f = f_1q * f_2q

    # Clamp to [0, 1] for numerical robustness
    return float(max(min(f, 1.0), 0.0))


def approx_estimated_success_probability(
    qc: QuantumCircuit,
    p1_avg: float,
    p2_avg: float,
    tau1_avg: float,
    tau2_avg: float,
    tbar: float | None,
    par_feature: float,
    liv_feature: float,
    n_qubits: int,
    *,
    device_id: str = "ibm_torino",
) -> float:
    """Approximate ESP using canonical counts and simple idle-time modeling.

    The ESP is modeled as:

        ESP ≈ F_gates * exp(- T_idle / T̄)

    where F_gates is approximated from canonical 1q/2q counts and mean error
    rates, and T_idle is estimated from a crude duration model modulated by
    a parallelism and liveness feature.

    Args:
        qc: Circuit for which to estimate ESP.
        p1_avg: Average single-qubit error probability.
        p2_avg: Average two-qubit error probability.
        tau1_avg: Average single-qubit gate duration.
        tau2_avg: Average two-qubit gate duration.
        tbar: Effective characteristic decoherence time (e.g. derived from T1/T2).
        par_feature: Parallelism feature in [0, 1] (e.g. Supermarq parallelism).
        liv_feature: Liveness feature in [0, 1], where 1 ≈ always active.
        n_qubits: Number of qubits in the circuit / device.
        device_id: Identifier of the backend (used to select the cost table).

    Returns:
        Approximate ESP in [0, 1].
    """
    cost_table = get_cost_table(device_id)

    # Fidelity part from gate errors
    n_1q, n_2q = estimate_counts(qc, cost_table=cost_table)
    f_1q = (1.0 - p1_avg) ** max(n_1q, 0)
    f_2q = (1.0 - p2_avg) ** max(n_2q, 0)
    f = f_1q * f_2q

    # Effective duration via parallelism (par_feature ∈ [0, 1])
    n_q = max(n_qubits, 1)
    k_eff = 1.0 + (n_q - 1.0) * float(par_feature)  # ∈ [1, n_qubits]

    t_hat = 0.0
    t_hat = (n_1q * tau1_avg + n_2q * tau2_avg) / k_eff

    # Idle-time penalty via (1 - liveness)
    idle_frac = max(0.0, 1.0 - float(liv_feature))
    idle_factor = 1.0 if tbar is None or tbar <= 0.0 else float(np.exp(-(t_hat * idle_frac) / tbar))

    esp = f * idle_factor
    return float(max(min(esp, 1.0), 0.0))
