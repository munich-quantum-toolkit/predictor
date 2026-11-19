# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""TKET actions and execution helpers."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

from pytket import Qubit
from pytket._tket.passes import BasePass as TketBasePass  # ruff:ignore[import-private-name]
from pytket.architecture import Architecture
from pytket.circuit import Node
from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
from pytket.passes import (
    CliffordSimp,
    FullPeepholeOptimise,
    KAKDecomposition,
    PeepholeOptimise2Q,
    PlacementPass,
    RemoveRedundancies,
    RoutingPass,
)
from pytket.placement import GraphPlacement, NoiseAwarePlacement, Placement, place_with_map
from pytket.predicates import CompilationUnit
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.transpiler import Layout, PassManager
from qiskit.transpiler.passes import ApplyLayout, EnlargeWithAncilla, FullAncillaAllocation, SetLayout

from mqt.predictor.rl.actions.base import CompilationOrigin, DeferredDeviceAction, DeviceIndependentAction, PassType

if TYPE_CHECKING:
    from pytket import Circuit
    from qiskit.transpiler import Target, TranspileLayout

    from mqt.predictor.rl.actions.base import Action

logger = logging.getLogger("mqt-predictor")


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


def _prepare_noise_data(device: Target) -> tuple[dict[Node, float], dict[tuple[Node, Node], float], dict[Node, float]]:
    """Extract calibration errors for TKET's noise-aware placement."""
    node_error_samples: defaultdict[Node, list[float]] = defaultdict(list)
    link_error_samples: defaultdict[tuple[Node, Node], list[float]] = defaultdict(list)
    readout_errors: dict[Node, float] = {}

    for operation_name in device.operation_names:
        if not isinstance(device.operation_from_name(operation_name), Gate):
            continue
        for qubits, properties in device[operation_name].items():
            if qubits is None or properties is None or properties.error is None:
                continue
            if len(qubits) == 1:
                node_error_samples[Node(qubits[0])].append(properties.error)
            elif len(qubits) == 2:
                link_error_samples[Node(qubits[0]), Node(qubits[1])].append(properties.error)

    if "measure" in device:
        for qubits, properties in device["measure"].items():
            if qubits is not None and len(qubits) == 1 and properties is not None and properties.error is not None:
                readout_errors[Node(qubits[0])] = properties.error

    node_errors = {node: sum(errors) / len(errors) for node, errors in node_error_samples.items()}
    link_errors = {link: sum(errors) / len(errors) for link, errors in link_error_samples.items()}
    return node_errors, link_errors, readout_errors


def _noise_aware_placement(device: Target) -> list[Placement]:
    node_errors, link_errors, readout_errors = _prepare_noise_data(device)
    return [
        NoiseAwarePlacement(
            Architecture(list(device.build_coupling_map())),
            node_errors=node_errors,
            link_errors=link_errors,
            readout_errors=readout_errors,
            timeout=5000,
            maximum_matches=5000,
        )
    ]


def _apply_placement(circuit: QuantumCircuit, placement: Placement, device: Target) -> QuantumCircuit:
    """Apply a TKET placement to the original Qiskit wires, including unregistered qubits."""
    # pytket-qiskit converts registers. A numbered copy includes loose wires and
    # gives the TKET map stable indices into the original circuit.
    numbered = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
    numbered.compose(circuit, inplace=True)
    unit = CompilationUnit(qiskit_to_tk(numbered, preserve_param_uuid=True))
    PlacementPass(placement).apply(unit)
    assigned = {
        source.index[0]: node.index[0]
        for source, node in unit.initial_map.items()
        if isinstance(source, Qubit) and node.reg_name == "node"
    }
    used = set(assigned.values())
    free = (index for index in range(device.num_qubits) if index not in used)
    physical = [assigned[index] if index in assigned else next(free) for index in range(circuit.num_qubits)]
    return PassManager([SetLayout(physical), FullAncillaAllocation(device), EnlargeWithAncilla(), ApplyLayout()]).run(
        circuit
    )


def tket_optimization_actions() -> list[Action]:
    """Returns the TKET optimization actions."""
    return [
        DeviceIndependentAction(
            "PeepholeOptimise2Q",
            CompilationOrigin.TKET,
            PassType.OPT,
            [PeepholeOptimise2Q()],
            preserves_layout=False,
            preserves_routing=False,
            preserves_synthesis=False,
        ),
        DeviceIndependentAction(
            "CliffordSimp",
            CompilationOrigin.TKET,
            PassType.OPT,
            [CliffordSimp()],
            preserves_layout=False,
            preserves_routing=False,
            preserves_synthesis=False,
        ),
        DeviceIndependentAction(
            "KAKDecomposition",
            CompilationOrigin.TKET,
            PassType.OPT,
            [KAKDecomposition(allow_swaps=False)],
            preserves_layout=True,
            preserves_routing=True,
            preserves_synthesis=False,
        ),
        DeviceIndependentAction(
            "FullPeepholeOptimiseCX",
            CompilationOrigin.TKET,
            PassType.OPT,
            [FullPeepholeOptimise()],
            preserves_layout=False,
            preserves_routing=False,
            preserves_synthesis=False,
        ),
        DeviceIndependentAction(
            "RemoveRedundancies",
            CompilationOrigin.TKET,
            PassType.OPT,
            [RemoveRedundancies()],
            preserves_layout=True,
            preserves_routing=True,
            preserves_synthesis=True,
        ),
    ]


def tket_layout_actions() -> list[Action]:
    """Return the TKET layout actions."""
    return [
        DeferredDeviceAction(
            "GraphPlacement",
            CompilationOrigin.TKET,
            PassType.LAYOUT,
            transpile_pass=lambda device: [
                GraphPlacement(
                    Architecture(list(device.build_coupling_map())),
                    timeout=5000,
                    maximum_matches=5000,
                )
            ],
        ),
        DeferredDeviceAction(
            "NoiseAwarePlacement",
            CompilationOrigin.TKET,
            PassType.LAYOUT,
            transpile_pass=_noise_aware_placement,
        ),
    ]


def tket_routing_action() -> Action:
    """Returns the TKET routing action."""
    return DeferredDeviceAction(
        "RoutingPass",
        CompilationOrigin.TKET,
        PassType.ROUTING,
        transpile_pass=lambda device: [
            PreProcessTKETRoutingAfterQiskitLayout(),
            RoutingPass(Architecture(list(device.build_coupling_map()))),
        ],
    )


def final_layout_pytket_to_qiskit(compilation_unit: CompilationUnit, qiskit_circuit: QuantumCircuit) -> Layout:
    """Convert TKET's routing map into a permutation of the input circuit's wires."""
    return Layout({
        qiskit_circuit.qubits[source.index[0]]: destination.index[0]
        for source, destination in compilation_unit.final_map.items()
        if isinstance(source, Qubit)
    })


def run_tket_action(
    action: Action,
    circuit: QuantumCircuit,
    device: Target,
    layout: TranspileLayout | None,
) -> tuple[QuantumCircuit, TranspileLayout | None]:
    """Apply a TKET action and return the updated circuit and layout metadata."""
    payload = action.transpile_pass
    if action.pass_type == PassType.LAYOUT:
        try:
            placements = payload(device) if callable(payload) else payload
            if len(placements) != 1 or not isinstance(placements[0], Placement):
                msg = f"TKET layout action {action.name} must provide exactly one placement."
                raise TypeError(msg)
            altered_qc = _apply_placement(circuit, placements[0], device)
        except (RuntimeError, ValueError) as error:
            logger.warning("Placement failed (%s): %s.", action.name, error)
            return circuit, layout
        return altered_qc, altered_qc.layout

    tket_qc = qiskit_to_tk(circuit, preserve_param_uuid=True)
    passes = payload(device) if callable(payload) else payload
    compilation_unit = None
    for pass_ in passes:
        assert isinstance(pass_, TketBasePass | PreProcessTKETRoutingAfterQiskitLayout)
        if action.pass_type == PassType.ROUTING and isinstance(pass_, TketBasePass):
            if compilation_unit is None:
                compilation_unit = CompilationUnit(tket_qc)
            pass_.apply(compilation_unit)
            tket_qc = compilation_unit.circuit
        else:
            pass_.apply(tket_qc)

    qbs = tket_qc.qubits
    tket_qc.rename_units({qbs[i]: Qubit("q", i) for i in range(len(qbs))})
    altered_qc = tk_to_qiskit(tket_qc, replace_implicit_swaps=True)

    if action.pass_type == PassType.ROUTING:
        assert layout is not None
        assert compilation_unit is not None
        routing_layout = final_layout_pytket_to_qiskit(compilation_unit, circuit)
        final_layout = (
            layout.final_layout.compose(routing_layout, circuit.qubits) if layout.final_layout else routing_layout
        )
        layout.final_layout = Layout({
            altered_qc.qubits[circuit.find_bit(qubit).index]: physical
            for qubit, physical in final_layout.get_virtual_bits().items()
        })
        layout._output_qubit_list = altered_qc.qubits  # ruff: ignore[private-member-access]

    return altered_qc, layout


def is_tket_action_available(*, action: Action, has_layout: bool, has_wide_operations: bool = False) -> bool:
    """Return whether a TKET action is available for the current layout state."""
    if has_wide_operations and action.pass_type in {PassType.LAYOUT, PassType.ROUTING}:
        return False
    # TKET layout/optimization actions must not run after a Qiskit layout has been set
    # (it is not clear how tket will handle the layout). TKET routing actions, however, are
    #  designed to work after a Qiskit layout via PreProcessTKETRoutingAfterQiskitLayout.
    return not has_layout or action.pass_type == PassType.ROUTING
