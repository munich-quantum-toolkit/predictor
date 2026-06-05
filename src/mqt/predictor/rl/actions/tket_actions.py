# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""TKET actions and execution helpers."""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, cast

from pytket import Qubit
from pytket._tket.passes import BasePass as TketBasePass  # noqa: PLC2701
from pytket.architecture import Architecture
from pytket.circuit import Node
from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
from pytket.passes import CliffordSimp, FullPeepholeOptimise, PeepholeOptimise2Q, RemoveRedundancies, RoutingPass
from pytket.placement import place_with_map
from qiskit.transpiler import Layout

from mqt.predictor.rl.actions import CompilationOrigin, DeviceDependentAction, DeviceIndependentAction, PassType

if TYPE_CHECKING:
    from collections.abc import Callable

    from pytket import Circuit
    from qiskit import QuantumCircuit
    from qiskit.passmanager.base_tasks import Task
    from qiskit.transpiler import Target, TranspileLayout

    from mqt.predictor.rl.actions import Action


class PreProcessTKETRoutingAfterQiskitLayout:
    """Pre-process TKET routing for circuits that already carry a Qiskit layout.

    Qiskit layout passes rewrite the circuit into physical-qubit order. Before
    TKET routing can operate on that circuit, it needs an equivalent trivial
    placement so the current wire order is treated as the starting placement.
    """

    def apply(self, circuit: Circuit) -> None:
        """Applies the pre-processing step to route a circuit with tket after a Qiskit Layout pass has been applied."""
        mapping = {Qubit(i): Node(i) for i in range(circuit.n_qubits)}
        place_with_map(circuit=circuit, qmap=mapping)


def tket_optimization_actions() -> list[Action]:
    """Returns the TKET optimization actions."""
    return [
        DeviceIndependentAction(
            "PeepholeOptimise2Q",
            CompilationOrigin.TKET,
            PassType.OPT,
            [PeepholeOptimise2Q()],
        ),
        DeviceIndependentAction(
            "CliffordSimp",
            CompilationOrigin.TKET,
            PassType.OPT,
            [CliffordSimp()],
        ),
        DeviceIndependentAction(
            "FullPeepholeOptimiseCX",
            CompilationOrigin.TKET,
            PassType.OPT,
            [FullPeepholeOptimise()],
        ),
        DeviceIndependentAction(
            "RemoveRedundancies",
            CompilationOrigin.TKET,
            PassType.OPT,
            [RemoveRedundancies()],
        ),
    ]


def tket_routing_action() -> Action:
    """Returns the TKET routing action."""
    return DeviceDependentAction(
        "RoutingPass",
        CompilationOrigin.TKET,
        PassType.ROUTING,
        transpile_pass=lambda device: cast(
            "list[Task]",
            [
                PreProcessTKETRoutingAfterQiskitLayout(),
                RoutingPass(Architecture(list(device.build_coupling_map()))),
            ],
        ),
    )


def final_layout_pytket_to_qiskit(pytket_circuit: Circuit, qiskit_circuit: QuantumCircuit) -> Layout:
    """Converts a final layout from pytket to qiskit."""
    pytket_layout = pytket_circuit.qubit_readout
    size_circuit = pytket_circuit.n_qubits
    qiskit_layout = {}
    qiskit_qreg = qiskit_circuit.qregs[0]

    pytket_layout = dict(sorted(pytket_layout.items(), key=operator.itemgetter(1)))

    for node, qubit_index in pytket_layout.items():
        qiskit_layout[node.index[0]] = qiskit_qreg[qubit_index]

    for i in range(size_circuit):
        if i not in set(pytket_layout.values()):
            qiskit_layout[i] = qiskit_qreg[i]

    return Layout(input_dict=qiskit_layout)


def run_tket_action(
    action: Action,
    circuit: QuantumCircuit,
    device: Target,
    layout: TranspileLayout | None,
) -> tuple[QuantumCircuit, TranspileLayout | None]:
    """Apply a TKET action and return the updated circuit and layout metadata."""
    tket_qc = qiskit_to_tk(circuit, preserve_param_uuid=True)
    if callable(action.transpile_pass):
        factory = cast("Callable[[Target], list[Task]]", action.transpile_pass)
        passes = factory(device)
    else:
        passes = cast("list[Task]", action.transpile_pass)
    for pass_ in passes:
        assert isinstance(pass_, TketBasePass | PreProcessTKETRoutingAfterQiskitLayout)
        pass_.apply(tket_qc)

    qbs = tket_qc.qubits
    tket_qc.rename_units({qbs[i]: Qubit("q", i) for i in range(len(qbs))})
    altered_qc = tk_to_qiskit(tket_qc, replace_implicit_swaps=True)

    if action.pass_type == PassType.ROUTING:
        assert layout is not None
        layout.final_layout = final_layout_pytket_to_qiskit(tket_qc, altered_qc)

    return altered_qc, layout


def is_tket_action_available(*, action: Action, has_layout: bool) -> bool:
    """Return whether a TKET action is available for the current layout state."""
    # TKET layout/optimization actions must not run after a Qiskit layout has been set
    # (it is not clear how tket will handle the layout). TKET routing actions, however, are
    #  designed to work after a Qiskit layout via PreProcessTKETRoutingAfterQiskitLayout.
    return not has_layout or action.pass_type == PassType.ROUTING
