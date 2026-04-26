# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""TKET RL actions and their action-local helper logic."""

from __future__ import annotations

import logging
import operator
from functools import cache
from typing import TYPE_CHECKING, cast

from pytket import Qubit as TketQubit
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

from . import CompilationOrigin, DeviceDependentAction, DeviceIndependentAction, PassType, register_action

logger = logging.getLogger("mqt-predictor")

if TYPE_CHECKING:
    from collections.abc import Callable

    from pytket import Circuit as TketCircuit
    from pytket._tket.passes import BasePass as TketBasePass
    from qiskit import QuantumCircuit
    from qiskit.circuit import Qubit as QiskitQubit
    from qiskit.transpiler import Target

    from . import Action


def _layout_output_qubits(layout: TranspileLayout) -> list[QiskitQubit]:
    """Return the materialized output wires tracked by a ``TranspileLayout``."""
    output_qubits = layout._output_qubit_list  # noqa: SLF001
    assert output_qubits is not None
    return list(output_qubits)


class PreProcessTKETRoutingAfterQiskitLayout:
    """Reapply the physical ordering expected by TKET after a Qiskit layout pass."""

    def apply(self, circuit: TketCircuit) -> None:
        """Align TKET qubit ids with the already materialized Qiskit layout."""
        mapping = {TketQubit(i): Node(i) for i in range(circuit.n_qubits)}
        place_with_map(circuit=circuit, qmap=mapping)


@cache
def prepare_noise_data(device: Target) -> tuple[dict[Node, float], dict[tuple[Node, Node], float], dict[Node, float]]:
    """Extract the calibration data needed by ``NoiseAwarePlacement``."""
    node_err: dict[Node, float] = {}
    edge_err: dict[tuple[Node, Node], float] = {}
    readout_err: dict[Node, float] = {}

    for op_name in device.operation_names:
        inst_props = device[op_name]
        if inst_props is None:
            continue
        for qtuple, props in inst_props.items():
            if props is None or not hasattr(props, "error") or props.error is None:
                continue
            if len(qtuple) == 1:
                node_err[Node(qtuple[0])] = props.error
            elif len(qtuple) == 2:
                edge_err[Node(qtuple[0]), Node(qtuple[1])] = props.error

    if "measure" in device:
        for (q,), props in device["measure"].items():
            if props is not None and hasattr(props, "error") and props.error is not None:
                readout_err[Node(q)] = props.error

    return node_err, edge_err, readout_err


def _build_noise_aware_placement_passes(device: Target) -> list[object]:
    """Build ``NoiseAwarePlacement`` with calibration data derived from the device target."""
    node_err, edge_err, readout_err = prepare_noise_data(device)
    return [
        NoiseAwarePlacement(
            Architecture(list(device.build_coupling_map())),
            node_errors=node_err,
            link_errors=edge_err,
            readout_errors=readout_err,
            timeout=5000,
            maximum_matches=5000,
        )
    ]


def translate_tket_placement_to_qiskit_layout(
    qc: QuantumCircuit,
    placement: dict[TketQubit, Node],
    action_name: str,
    num_device_qubits: int,
) -> Layout | None:
    """Translate a TKET placement map into a full Qiskit layout."""
    qiskit_qubits_by_identity: dict[tuple[str, tuple[int, ...]], QiskitQubit] = {}
    for qiskit_qubit in qc.qubits:
        bit_location = qc.find_bit(qiskit_qubit)
        registers = bit_location.registers
        if registers:
            register, register_index = registers[0]
            identity = (register.name, (register_index,))
        else:
            identity = ("q", (bit_location.index,))
        qiskit_qubits_by_identity[identity] = qiskit_qubit

    qiskit_mapping: dict[QiskitQubit, int] = {}
    unassigned_qiskit_qubits: list[QiskitQubit] = []
    used_physical_indices: set[int] = set()

    for tket_qubit, target_node in placement.items():
        identity = (str(tket_qubit.reg_name), tuple(int(index) for index in tket_qubit.index))
        qiskit_qubit = qiskit_qubits_by_identity.get(identity)
        if qiskit_qubit is None:
            logger.warning(
                "Placement failed (%s): unknown logical qubit %s. Falling back to original circuit.",
                action_name,
                tket_qubit,
            )
            return None

        reg_name = getattr(target_node, "reg_name", None)
        node_index = getattr(target_node, "index", None)
        if reg_name == "node" and node_index:
            physical_index = int(node_index[0])
            qiskit_mapping[qiskit_qubit] = physical_index
            used_physical_indices.add(physical_index)
        else:
            unassigned_qiskit_qubits.append(qiskit_qubit)

    for qiskit_qubit in qc.qubits:
        if qiskit_qubit not in qiskit_mapping and qiskit_qubit not in unassigned_qiskit_qubits:
            unassigned_qiskit_qubits.append(qiskit_qubit)

    remaining_physical_indices = [i for i in range(num_device_qubits) if i not in used_physical_indices]
    if len(remaining_physical_indices) < len(unassigned_qiskit_qubits):
        logger.warning(
            "Placement failed (%s): only %d free physical qubits for %d unassigned logical qubits. "
            "Falling back to original circuit.",
            action_name,
            len(remaining_physical_indices),
            len(unassigned_qiskit_qubits),
        )
        return None

    qiskit_mapping.update(
        dict(
            zip(
                unassigned_qiskit_qubits,
                remaining_physical_indices[: len(unassigned_qiskit_qubits)],
                strict=True,
            )
        )
    )
    return Layout(qiskit_mapping)


def final_layout_pytket_to_qiskit(pytket_circuit: TketCircuit, output_qubits: list[QiskitQubit]) -> Layout:
    """Convert a TKET routing permutation into a Qiskit final layout."""
    pytket_layout = dict(sorted(pytket_circuit.qubit_readout.items(), key=operator.itemgetter(1)))
    qiskit_layout: dict[int, QiskitQubit] = {}
    used_output_positions: set[int] = set()

    for qubit, readout_index in pytket_layout.items():
        qiskit_layout[readout_index] = output_qubits[qubit.index[0]]
        used_output_positions.add(qubit.index[0])

    remaining_physical_positions = [i for i in range(len(output_qubits)) if i not in qiskit_layout]
    remaining_output_positions = [i for i in range(len(output_qubits)) if i not in used_output_positions]

    for physical_position, output_position in zip(
        remaining_physical_positions, remaining_output_positions, strict=True
    ):
        qiskit_layout[physical_position] = output_qubits[output_position]

    return Layout(input_dict=qiskit_layout)


def run_tket_action(
    *,
    action: Action,
    circuit: QuantumCircuit,
    device: Target,
    layout: TranspileLayout | None,
) -> tuple[QuantumCircuit, TranspileLayout | None]:
    """Apply a TKET action and update the layout bookkeeping it owns.

    Args:
        action: The TKET action to apply.
        circuit: The current quantum circuit.
        device: The target device.
        layout: The current layout (if any).

    Returns:
        Tuple of (compiled circuit, updated layout).
    """
    tket_qc = qiskit_to_tk(circuit, preserve_param_uuid=True)
    transpile_pass = (
        cast("Callable[[Target], list[TketBasePass | PreProcessTKETRoutingAfterQiskitLayout]]", action.transpile_pass)(
            device
        )
        if callable(action.transpile_pass)
        else action.transpile_pass
    )

    assert isinstance(transpile_pass, list)

    # LAYOUT actions need special handling to extract placement
    if action.pass_type == PassType.LAYOUT:
        if not transpile_pass:
            logger.warning(
                "Placement failed (%s): no placement pass provided. Falling back to original circuit.", action.name
            )
            return tk_to_qiskit(tket_qc, replace_implicit_swaps=True), layout

        placement_pass = transpile_pass[0]
        if not isinstance(placement_pass, Placement):
            logger.warning(
                "Placement failed (%s): expected Placement pass, got %s. Falling back to original circuit.",
                action.name,
                type(placement_pass).__name__,
            )
            return tk_to_qiskit(tket_qc, replace_implicit_swaps=True), layout

        try:
            placement = placement_pass.get_placement_map(tket_qc)
        except (RuntimeError, TypeError, ValueError) as exc:
            logger.warning("Placement failed (%s): %s. Falling back to original circuit.", action.name, exc)
            return tk_to_qiskit(tket_qc, replace_implicit_swaps=True), layout

        qc_tmp = tk_to_qiskit(tket_qc, replace_implicit_swaps=True)
        qiskit_layout = translate_tket_placement_to_qiskit_layout(qc_tmp, placement, action.name, device.num_qubits)
        if qiskit_layout is None:
            return tk_to_qiskit(tket_qc, replace_implicit_swaps=True), layout

        pm = PassManager([
            SetLayout(qiskit_layout),
            FullAncillaAllocation(coupling_map=CouplingMap(device.build_coupling_map())),
            EnlargeWithAncilla(),
            ApplyLayout(),
        ])
        altered_qc = pm.run(qc_tmp)

        layout2 = pm.property_set.get("layout")
        assert isinstance(layout2, Layout)
        return altered_qc, TranspileLayout(
            initial_layout=layout2,
            input_qubit_mapping=pm.property_set["original_qubit_indices"],
            final_layout=pm.property_set["final_layout"],
            _output_qubit_list=altered_qc.qubits,
            _input_qubit_count=circuit.num_qubits,
        )

    # All other action types (synthesis, routing, optimization)
    for pass_ in cast("list[TketBasePass | PreProcessTKETRoutingAfterQiskitLayout]", transpile_pass):
        pass_.apply(tket_qc)

    qbs = tket_qc.qubits
    tket_qc.rename_units({qbs[i]: TketQubit("q", i) for i in range(len(qbs))})
    altered_qc = tk_to_qiskit(tket_qc, replace_implicit_swaps=True)

    # ROUTING actions update the final layout
    if action.pass_type == PassType.ROUTING:
        assert layout is not None
        layout.final_layout = final_layout_pytket_to_qiskit(tket_qc, _layout_output_qubits(layout))

    return altered_qc, layout


register_action(
    DeviceIndependentAction(
        "PeepholeOptimise2Q",
        CompilationOrigin.TKET,
        PassType.OPT,
        [PeepholeOptimise2Q()],
        preserves_layout=False,  # can produce gates on non-adjacent qubits, breaking established routing
        preserves_routing=False,
        preserves_synthesis=False,
    )
)

register_action(
    DeviceIndependentAction(
        "CliffordSimp",
        CompilationOrigin.TKET,
        PassType.OPT,
        [CliffordSimp()],
        preserves_layout=False,  # can produce gates on non-adjacent qubits, breaking established routing
        preserves_routing=False,
        preserves_synthesis=False,
    )
)

register_action(
    DeviceIndependentAction(
        "KAKDecomposition",
        CompilationOrigin.TKET,
        PassType.OPT,
        [KAKDecomposition(allow_swaps=False)],
        preserves_layout=True,
        preserves_routing=True,
        preserves_synthesis=False,
    )
)

register_action(
    DeviceIndependentAction(
        "FullPeepholeOptimiseCX",
        CompilationOrigin.TKET,
        PassType.OPT,
        [FullPeepholeOptimise()],
        preserves_layout=False,  # can produce gates on non-adjacent qubits, breaking established routing
        preserves_routing=False,
        preserves_synthesis=False,
    )
)

register_action(
    DeviceIndependentAction(
        "RemoveRedundancies",
        CompilationOrigin.TKET,
        PassType.OPT,
        [RemoveRedundancies()],
        preserves_layout=True,
        preserves_routing=True,
        preserves_synthesis=True,
    )
)

register_action(
    DeviceDependentAction(
        "GraphPlacement",
        CompilationOrigin.TKET,
        PassType.LAYOUT,
        transpile_pass=lambda device: [
            GraphPlacement(Architecture(list(device.build_coupling_map())), timeout=5000, maximum_matches=5000)
        ],
    )
)

register_action(
    DeviceDependentAction(
        "NoiseAwarePlacement",
        CompilationOrigin.TKET,
        PassType.LAYOUT,
        transpile_pass=_build_noise_aware_placement_passes,
    )
)

register_action(
    DeviceDependentAction(
        "RoutingPass",
        CompilationOrigin.TKET,
        PassType.ROUTING,
        transpile_pass=lambda device: [
            PreProcessTKETRoutingAfterQiskitLayout(),
            RoutingPass(Architecture(list(device.build_coupling_map()))),
        ],
    )
)


__all__ = [
    "PreProcessTKETRoutingAfterQiskitLayout",
    "final_layout_pytket_to_qiskit",
    "prepare_noise_data",
    "run_tket_action",
    "translate_tket_placement_to_qiskit_layout",
]
