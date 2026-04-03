# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Helper methods necessary for parsing between circuit formats."""

from __future__ import annotations

import operator
from functools import cache
from typing import TYPE_CHECKING

from bqskit.ir import gates
from pytket import Qubit
from pytket.circuit import Node
from pytket.placement import place_with_map
from qiskit import QuantumRegister
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler import Layout, Target, TranspileLayout
from qiskit.transpiler.passes import ApplyLayout

if TYPE_CHECKING:
    from bqskit.ir import Gate
    from pytket import Circuit
    from qiskit import QuantumCircuit
    from qiskit.circuit import Qubit as QiskitQubit
    from qiskit.transpiler import Target


class PreProcessTKETRoutingAfterQiskitLayout:
    """Pre-processing step to route a circuit with TKET after a Qiskit Layout pass has been applied.

        The reason why we can apply the trivial layout here is that the circuit already got assigned a layout by qiskit.
        Implicitly, Qiskit is reordering its qubits in a sequential manner, i.e., the qubit with the lowest *physical* qubit
        first.

        Assuming, the layouted circuit is given by

                       ┌───┐           ░       ┌─┐
              q_2 -> 0 ┤ H ├──■────────░───────┤M├
                       └───┘┌─┴─┐      ░    ┌─┐└╥┘
              q_1 -> 1 ─────┤ X ├──■───░────┤M├─╫─
                            └───┘┌─┴─┐ ░ ┌─┐└╥┘ ║
              q_0 -> 2 ──────────┤ X ├─░─┤M├─╫──╫─
                                 └───┘ ░ └╥┘ ║  ║
        ancilla_0 -> 3 ───────────────────╫──╫──╫─
                                          ║  ║  ║
        ancilla_1 -> 4 ───────────────────╫──╫──╫─
                                          ║  ║  ║
               meas: 3/═══════════════════╩══╩══╩═
                                          0  1  2

        Applying the trivial layout, we get the same qubit order as in the original circuit and can be respectively
        routed. This results in:
                ┌───┐           ░       ┌─┐
           q_0: ┤ H ├──■────────░───────┤M├
                └───┘┌─┴─┐      ░    ┌─┐└╥┘
           q_1: ─────┤ X ├──■───░────┤M├─╫─
                     └───┘┌─┴─┐ ░ ┌─┐└╥┘ ║
           q_2: ──────────┤ X ├─░─┤M├─╫──╫─
                          └───┘ ░ └╥┘ ║  ║
           q_3: ───────────────────╫──╫──╫─
                                   ║  ║  ║
           q_4: ───────────────────╫──╫──╫─
                                   ║  ║  ║
        meas: 3/═══════════════════╩══╩══╩═
                                   0  1  2


        If we would not apply the trivial layout, no layout would be considered resulting, e.g., in the following circuit:
                 ┌───┐         ░    ┌─┐
       q_0: ─────┤ X ├─────■───░────┤M├───
            ┌───┐└─┬─┘   ┌─┴─┐ ░ ┌─┐└╥┘
       q_1: ┤ H ├──■───X─┤ X ├─░─┤M├─╫────
            └───┘      │ └───┘ ░ └╥┘ ║ ┌─┐
       q_2: ───────────X───────░──╫──╫─┤M├
                               ░  ║  ║ └╥┘
       q_3: ──────────────────────╫──╫──╫─
                                  ║  ║  ║
       q_4: ──────────────────────╫──╫──╫─
                                  ║  ║  ║
    meas: 3/══════════════════════╩══╩══╩═
                                  0  1  2

    """

    def apply(self, circuit: Circuit) -> None:
        """Applies the pre-processing step to route a circuit with tket after a Qiskit Layout pass has been applied."""
        mapping = {Qubit(i): Node(i) for i in range(circuit.n_qubits)}
        place_with_map(circuit=circuit, qmap=mapping)


@cache
def get_bqskit_native_gates(device: Target) -> list[Gate]:
    """Returns the native gates of the given device.

    Arguments:
        device: The device for which the native gates are returned.

    Returns:
        The native gates of the given device as BQSKit gates.

    Raises:
        ValueError: If a gate in the device is not supported in BQSKit.
    """
    gate_map = {
        # --- 1-qubit gates ---
        "id": gates.IdentityGate(),
        "x": gates.XGate(),
        "y": gates.YGate(),
        "z": gates.ZGate(),
        "h": gates.HGate(),
        "s": gates.SGate(),
        "sdg": gates.SdgGate(),
        "t": gates.TGate(),
        "tdg": gates.TdgGate(),
        "sx": gates.SXGate(),
        "rx": gates.RXGate(),
        "ry": gates.RYGate(),
        "rz": gates.RZGate(),
        "u1": gates.U1Gate(),
        "u2": gates.U2Gate(),
        "u3": gates.U3Gate(),
        # --- Controlled 1-qubit gates ---
        "cx": gates.CXGate(),
        "cy": gates.CYGate(),
        "cz": gates.CZGate(),
        "ch": gates.CHGate(),
        "crx": gates.CRXGate(),
        "cry": gates.CRYGate(),
        "crz": gates.CRZGate(),
        "cp": gates.CPGate(),
        "cu": gates.CUGate(),
        # --- 2-qubit gates ---
        "swap": gates.SwapGate(),
        "iswap": gates.ISwapGate(),
        "ecr": gates.ECRGate(),
        "rzz": gates.RZZGate(),
        "rxx": gates.RXXGate(),
        "ryy": gates.RYYGate(),
        "zz": gates.ZZGate(),
        # --- 3-qubit gates ---
        "ccx": gates.CCXGate(),
        # --- Others / approximations ---
        "reset": gates.Reset(),
    }

    native_gates = []
    # Some devices declare support for non-gate operations, which some compiler passes can not handle.
    ignored_non_gate_ops = {
        "barrier",
        "measure",
        "delay",
        "for_loop",
        "while_loop",
        "if_test",
        "if_else",
        "switch_case",
        "break",
        "continue",
        "box",
        "control",
    }

    for instr in device.operation_names:
        name = instr

        if name in ignored_non_gate_ops:
            continue

        if name not in gate_map:
            msg = f"The '{name}' gate of device '{device.description}' is not supported in BQSKIT."
            raise ValueError(msg)

        native_gates.append(gate_map[name])

    return native_gates


def final_layout_pytket_to_qiskit(
    pytket_circuit: Circuit,
    output_qubits: list[QiskitQubit],
    initial_positions: list[int],
) -> Layout:
    """Convert a pytket routing permutation into a Qiskit final layout.

    The routed pytket circuit may be compacted to only the active qubits. We therefore
    re-express the permutation on the pre-routing output wires tracked by the existing
    ``TranspileLayout`` instead of on the compact routed circuit's fresh qubits.
    """
    pytket_layout = pytket_circuit.qubit_readout
    size_circuit = len(output_qubits)
    qiskit_layout = {}
    used_output_positions = set()

    pytket_layout = dict(sorted(pytket_layout.items(), key=operator.itemgetter(1)))

    for node, qubit_index in pytket_layout.items():
        output_position = initial_positions[qubit_index]
        qiskit_layout[node.index[0]] = output_qubits[output_position]
        used_output_positions.add(output_position)

    remaining_physical_positions = [i for i in range(size_circuit) if i not in qiskit_layout]
    remaining_output_positions = [i for i in range(size_circuit) if i not in used_output_positions]

    # Layout is bijective: once TKET moves an output wire, the untouched physical positions
    # must be filled from the remaining unused output wires, not by identity on the index.
    for physical_position, output_position in zip(
        remaining_physical_positions, remaining_output_positions, strict=True
    ):
        qiskit_layout[physical_position] = output_qubits[output_position]

    return Layout(input_dict=qiskit_layout)


def final_layout_bqskit_to_qiskit(
    bqskit_initial_layout: tuple[int, ...],
    bqskit_final_layout: tuple[int, ...],
    compiled_qc: QuantumCircuit,
    initial_qc: QuantumCircuit,
) -> TranspileLayout:
    """Converts a final layout from bqskit to qiskit.

    BQSKit provides an initial layout as a list[int] where each virtual qubit is mapped to a physical qubit
    similarly, it provides a final layout as a list[int] representing where each virtual qubit is mapped to at the end
    of the circuit.
    """
    ancilla = QuantumRegister(compiled_qc.num_qubits - initial_qc.num_qubits, "ancilla")
    qiskit_initial_layout = {}
    counter_ancilla_qubit = 0
    for i in range(compiled_qc.num_qubits):
        if i in bqskit_initial_layout:
            qiskit_initial_layout[i] = initial_qc.qubits[bqskit_initial_layout.index(i)]
        else:
            qiskit_initial_layout[i] = ancilla[counter_ancilla_qubit]
            counter_ancilla_qubit += 1

    initial_qubit_mapping = {bit: index for index, bit in enumerate(initial_qc.qubits)}
    initial_qubit_mapping.update({bit: initial_qc.num_qubits + index for index, bit in enumerate(ancilla)})

    if bqskit_initial_layout == bqskit_final_layout:
        qiskit_final_layout = None
    else:
        qiskit_final_layout = {}
        used_output_wires = set()
        for initial_position, final_position in zip(bqskit_initial_layout, bqskit_final_layout, strict=False):
            qiskit_final_layout[final_position] = compiled_qc.qubits[initial_position]
            used_output_wires.add(initial_position)

        remaining_physical_positions = [i for i in range(compiled_qc.num_qubits) if i not in qiskit_final_layout]
        remaining_output_wires = [
            compiled_qc.qubits[i] for i in range(compiled_qc.num_qubits) if i not in used_output_wires
        ]

        for physical_position, output_wire in zip(remaining_physical_positions, remaining_output_wires, strict=False):
            qiskit_final_layout[physical_position] = output_wire

    return TranspileLayout(
        initial_layout=Layout(input_dict=qiskit_initial_layout),
        input_qubit_mapping=initial_qubit_mapping,
        final_layout=Layout(input_dict=qiskit_final_layout) if qiskit_final_layout else None,
        _output_qubit_list=compiled_qc.qubits,
        _input_qubit_count=initial_qc.num_qubits,
    )


def postprocess_vf2postlayout(
    qc: QuantumCircuit, post_layout: Layout, layout_before: TranspileLayout
) -> tuple[QuantumCircuit, ApplyLayout]:
    """Postprocess a quantum circuit after VF2 layout assignment.

    Args:
        qc: The quantum circuit to transform.
        post_layout: The layout computed after routing.
        layout_before: The layout before post-routing adjustment.

    Returns:
        A tuple of the transformed circuit and the ApplyLayout used.
    """
    # `ApplyLayout` requires that every virtual qubit in `layout` has a
    # corresponding entry in `original_qubit_indices`. Some layouts include
    # ancilla virtual qubits that are missing from `input_qubit_mapping`.
    original_qubit_indices = dict(layout_before.input_qubit_mapping)
    for virt, phys in layout_before.initial_layout.get_virtual_bits().items():
        if virt not in original_qubit_indices:
            original_qubit_indices[virt] = phys

    apply_layout = ApplyLayout()
    apply_layout.property_set["layout"] = layout_before.initial_layout
    apply_layout.property_set["original_qubit_indices"] = original_qubit_indices
    apply_layout.property_set["final_layout"] = layout_before.final_layout
    apply_layout.property_set["post_layout"] = post_layout

    altered_qc = apply_layout.run(circuit_to_dag(qc))
    return dag_to_circuit(altered_qc), apply_layout


def prepare_noise_data(device: Target) -> tuple[dict[Node, float], dict[tuple[Node, Node], float], dict[Node, float]]:
    """Extract node, edge, and readout errors from the device target."""
    node_err: dict[Node, float] = {}
    edge_err: dict[tuple[Node, Node], float] = {}
    readout_err: dict[Node, float] = {}

    # Collect errors from operation properties
    for op_name in device.operation_names:
        inst_props = device[op_name]
        if inst_props is None:
            continue
        for qtuple, props in inst_props.items():
            if props is None or not hasattr(props, "error") or props.error is None:
                continue
            if len(qtuple) == 1:  # single-qubit op
                q = qtuple[0]
                node_err[Node(q)] = props.error
            elif len(qtuple) == 2:  # two-qubit op
                q1, q2 = qtuple
                edge_err[Node(q1), Node(q2)] = props.error

    # Collect readout errors
    if "measure" in device:
        for (q,), props in device["measure"].items():
            if props is not None and hasattr(props, "error") and props.error is not None:
                readout_err[Node(q)] = props.error

    return node_err, edge_err, readout_err
