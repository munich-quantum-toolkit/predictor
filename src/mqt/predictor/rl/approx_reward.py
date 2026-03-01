# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""This module provides helper functions to approximate expected fidelity and estimated success probability (ESP) by transpiling a circuit to a device's basis gate set and combining resulting gate counts with calibration-derived per-gate error rates and durations."""

from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING

import numpy as np
from qiskit import transpile

if TYPE_CHECKING:
    from qiskit import QuantumCircuit
    from qiskit.transpiler import InstructionProperties, Target

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
        error_rates: Mapping ``basis_gate: g -> error_probability: p_g``.

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


def compute_device_averages_from_target(
    device: Target,
) -> tuple[dict[str, float], dict[str, float], float | None]:
    """Compute per-basis-gate averages for error, duration, and a coherence scale.

    Computes and returns:
        - error_rates: dict[basis_gate -> mean error probability]
        - gate_durations: dict[basis_gate -> mean duration (seconds)]
        - tbar: median over qubits of min(T1, T2) (seconds) if available, else None

    Raises:
        RuntimeError: if required Target API is missing or if no calibration data exists.
    """
    # ---- Hard requirements -------------------------------------------------------
    try:
        num_qubits = device.num_qubits
        op_names = list(device.operation_names)
        coupling_map = device.build_coupling_map()
        qubit_props = device.qubit_properties
    except AttributeError as exc:
        msg = "Device target does not expose the required Target API for approximate reward computation."
        raise RuntimeError(msg) from exc

    basis_ops = [name for name in op_names if name not in BLACKLIST]
    twoq_edges = coupling_map.get_edges()  # list[tuple[int, int]]

    # ---- Helpers ----------------------------------------------------------------
    def _get_props(name: str, qargs: tuple[int, ...]) -> InstructionProperties | None:
        """Return calibration properties for (name, qargs) or None if unavailable."""
        with suppress(KeyError):
            return device[name].get(qargs, None)
        return None

    def _infer_arity(name: str) -> int | None:
        """Infer operation arity from Target (best effort)."""
        # Preferred: operation_from_name
        with suppress(Exception):
            op = device.operation_from_name(name)
            return int(op.num_qubits)

        # Fallback: infer from any qargs key in device[name]
        with suppress(Exception):
            props_map = device[name]
            for qargs in props_map:
                return len(qargs)
        return None

    # ---- Accumulate raw samples --------------------------------------------------
    err_samples: dict[str, list[float]] = {name: [] for name in basis_ops}
    dur_samples: dict[str, list[float]] = {name: [] for name in basis_ops}

    arity_by_name: dict[str, int] = {}
    for name in basis_ops:
        arity = _infer_arity(name)
        if arity is not None:
            arity_by_name[name] = arity

    # ---- Aggregate error/duration per gate --------------------------------------
    for name in basis_ops:
        arity = arity_by_name.get(name)
        if arity is None:
            continue

        if arity == 1:
            for q in range(num_qubits):
                props = _get_props(name, (q,))
                if props is None:
                    continue
                err = getattr(props, "error", None)
                if err is not None:
                    err_samples[name].append(float(err))
                dur = getattr(props, "duration", None)
                if dur is not None:
                    dur_samples[name].append(float(dur))

        elif arity == 2:
            for i, j in twoq_edges:
                props = _get_props(name, (i, j))
                if props is None:
                    props = _get_props(name, (j, i))  # flipped orientation
                if props is None:
                    continue
                err = getattr(props, "error", None)
                if err is not None:
                    err_samples[name].append(float(err))
                dur = getattr(props, "duration", None)
                if dur is not None:
                    dur_samples[name].append(float(dur))

        else:
            # Ignore gates with arity > 2
            continue

    # ---- Global fallbacks --------------------------------------------------------
    all_err = [x for xs in err_samples.values() for x in xs]
    all_dur = [x for xs in dur_samples.values() for x in xs]

    if not all_err and not all_dur:
        msg = "No valid calibration data found in Target, cannot compute approximate reward."
        raise RuntimeError(msg)

    global_err = float(np.mean(all_err)) if all_err else 0.0
    global_dur = float(np.mean(all_dur)) if all_dur else 0.0

    error_rates = {name: (float(np.mean(vals)) if vals else global_err) for name, vals in err_samples.items()}
    gate_durations = {name: (float(np.mean(vals)) if vals else global_dur) for name, vals in dur_samples.items()}

    # ---- Coherence scale ---------------------------------------------------------
    tmins: list[float] = []
    if qubit_props:
        for i in range(num_qubits):
            props = qubit_props[i]
            if props is None:
                continue
            t1v = getattr(props, "t1", None)
            t2v = getattr(props, "t2", None)
            vals = [v for v in (t1v, t2v) if v is not None]
            if vals:
                tmins.append(float(min(vals)))

    tbar = float(np.median(tmins)) if tmins else None
    return error_rates, gate_durations, tbar
