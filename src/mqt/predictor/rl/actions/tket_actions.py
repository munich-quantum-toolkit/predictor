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
import operator
from collections import defaultdict
from functools import cache
from typing import TYPE_CHECKING, cast

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
    RemoveRedundancies,
    RoutingPass,
)
from pytket.placement import GraphPlacement, NoiseAwarePlacement, Placement, place_with_map
from qiskit.transpiler import CouplingMap, Layout, PassManager, TranspileLayout
from qiskit.transpiler.passes import ApplyLayout, EnlargeWithAncilla, FullAncillaAllocation, SetLayout

from mqt.predictor.rl.actions.base import CompilationOrigin, DeferredDeviceAction, DeviceIndependentAction, PassType

if TYPE_CHECKING:
    from collections.abc import Callable

    from pytket import Circuit
    from qiskit import QuantumCircuit
    from qiskit.circuit import Qubit as QiskitQubit
    from qiskit.transpiler import Target

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


@cache
def _prepare_noise_data(device: Target) -> tuple[dict[Node, float], dict[tuple[Node, Node], float], dict[Node, float]]:
    """Extract calibration errors for TKET's noise-aware placement."""
    node_error_samples: defaultdict[Node, list[float]] = defaultdict(list)
    link_error_samples: defaultdict[tuple[Node, Node], list[float]] = defaultdict(list)
    readout_errors: dict[Node, float] = {}

    for operation_name in device.operation_names:
        if operation_name == "measure":
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


def _translate_placement(
    circuit: QuantumCircuit,
    placement: dict[Qubit, Node],
    action_name: str,
    num_device_qubits: int,
) -> Layout | None:
    qiskit_qubits: dict[tuple[str, tuple[int, ...]], QiskitQubit] = {}
    for qubit in circuit.qubits:
        location = circuit.find_bit(qubit)
        if location.registers:
            register, register_index = location.registers[0]
            qiskit_qubits[register.name, (register_index,)] = qubit
        else:
            qiskit_qubits["q", (location.index,)] = qubit

    qiskit_mapping: dict[QiskitQubit, int] = {}
    unassigned_qubits: list[QiskitQubit] = []
    used_physical_indices: set[int] = set()
    for tket_qubit, target_node in placement.items():
        qiskit_qubit = qiskit_qubits.get((str(tket_qubit.reg_name), tuple(int(i) for i in tket_qubit.index)))
        if qiskit_qubit is None:
            logger.warning("Placement failed (%s): unknown logical qubit %s.", action_name, tket_qubit)
            return None

        if target_node.reg_name == "node" and target_node.index:
            physical_index = int(target_node.index[0])
            qiskit_mapping[qiskit_qubit] = physical_index
            used_physical_indices.add(physical_index)
        else:
            unassigned_qubits.append(qiskit_qubit)

    unassigned_qubits.extend(qubit for qubit in circuit.qubits if qubit not in qiskit_mapping)
    unassigned_qubits = list(dict.fromkeys(unassigned_qubits))
    remaining_indices = [index for index in range(num_device_qubits) if index not in used_physical_indices]
    if len(remaining_indices) < len(unassigned_qubits):
        logger.warning("Placement failed (%s): insufficient free physical qubits.", action_name)
        return None

    qiskit_mapping.update(zip(unassigned_qubits, remaining_indices, strict=False))
    return Layout(qiskit_mapping)


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
        factory = cast(
            "Callable[[Target], list[TketBasePass | PreProcessTKETRoutingAfterQiskitLayout | Placement]]",
            action.transpile_pass,
        )
        passes = factory(device)
    else:
        passes = cast("list[TketBasePass | PreProcessTKETRoutingAfterQiskitLayout | Placement]", action.transpile_pass)

    if action.pass_type == PassType.LAYOUT:
        if len(passes) != 1 or not isinstance(passes[0], Placement):
            msg = f"TKET layout action {action.name} must provide exactly one placement."
            raise TypeError(msg)
        try:
            placement = passes[0].get_placement_map(tket_qc)
        except (RuntimeError, TypeError, ValueError) as error:
            logger.warning("Placement failed (%s): %s.", action.name, error)
            return circuit, layout

        qiskit_layout = _translate_placement(circuit, placement, action.name, device.num_qubits)
        if qiskit_layout is None:
            return circuit, layout
        pass_manager = PassManager([
            SetLayout(qiskit_layout),
            FullAncillaAllocation(coupling_map=CouplingMap(device.build_coupling_map())),
            EnlargeWithAncilla(),
            ApplyLayout(),
        ])
        altered_qc = pass_manager.run(circuit)
        applied_layout = cast("Layout", pass_manager.property_set["layout"])
        return altered_qc, TranspileLayout(
            initial_layout=applied_layout,
            input_qubit_mapping=pass_manager.property_set["original_qubit_indices"],
            final_layout=pass_manager.property_set["final_layout"],
            _output_qubit_list=altered_qc.qubits,
            _input_qubit_count=circuit.num_qubits,
        )

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


def is_tket_action_available(*, action: Action, has_layout: bool, has_wide_operations: bool) -> bool:
    """Return whether a TKET action is available for the current layout state."""
    if has_wide_operations and action.pass_type in {PassType.LAYOUT, PassType.ROUTING}:
        return False
    # TKET layout/optimization actions must not run after a Qiskit layout has been set
    # (it is not clear how tket will handle the layout). TKET routing actions, however, are
    #  designed to work after a Qiskit layout via PreProcessTKETRoutingAfterQiskitLayout.
    return not has_layout or action.pass_type == PassType.ROUTING
