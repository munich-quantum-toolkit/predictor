# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the SafeAIRouting wrapper behavior."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import ClassicalRegister, QuantumRegister
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler import Layout

from mqt.predictor.rl.actions import IS_WIN_PY313

if TYPE_CHECKING:
    from qiskit.dagcircuit import DAGCircuit

if not IS_WIN_PY313:
    from qiskit_ibm_transpiler.ai.routing import AIRouting

    from mqt.predictor.rl.actions import SafeAIRouting


pytestmark = pytest.mark.skipif(IS_WIN_PY313, reason="SafeAIRouting is disabled on Windows + Python 3.13")


@pytest.mark.skipif(IS_WIN_PY313, reason="SafeAIRouting is unavailable on this platform")
def test_safe_airouting_preserves_and_remaps_measurements(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure SafeAIRouting preserves classical structure and remaps measured qubits."""
    q0 = QuantumRegister(2, "qa")
    q1 = QuantumRegister(2, "qb")
    c0 = ClassicalRegister(2, "ca")
    c1 = ClassicalRegister(2, "cb")
    qc = QuantumCircuit(q0, q1, c0, c1)
    qc.h(q0[0])
    qc.cx(q0[0], q1[1])
    qc.cz(q1[0], q0[1])
    qc.swap(q0[1], q1[0])
    qc.barrier()
    qc.measure(q0[0], c0[0])
    qc.measure(q1[1], c1[1])
    qc.measure(q0[1], c1[0])

    permutation = [2, 0, 3, 1]

    def fake_parent_run(self: AIRouting, dag: DAGCircuit) -> DAGCircuit:
        in_qc = dag_to_circuit(dag)
        self.property_set["final_layout"] = Layout({q: permutation[i] for i, q in enumerate(in_qc.qubits)})
        return dag

    monkeypatch.setattr(AIRouting, "run", fake_parent_run, raising=True)

    p = SafeAIRouting(
        coupling_map=[[0, 1], [1, 2], [2, 3]],
        optimization_level=3,
        layout_mode="improve",
        local_mode=True,
    )
    out = dag_to_circuit(p.run(circuit_to_dag(qc)))

    assert out.num_qubits == qc.num_qubits
    assert out.num_clbits == qc.num_clbits
    assert len(out.cregs) == len(qc.cregs)

    in_pairs = [
        (qc.find_bit(item.qubits[0]).index, qc.find_bit(item.clbits[0]).index)
        for item in qc.data
        if item.operation.name == "measure"
    ]
    out_pairs = [
        (out.find_bit(item.qubits[0]).index, out.find_bit(item.clbits[0]).index)
        for item in out.data
        if item.operation.name == "measure"
    ]
    expected_pairs = [(permutation[qidx], cidx) for qidx, cidx in in_pairs]

    assert Counter(out_pairs) == Counter(expected_pairs)
