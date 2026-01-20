# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Integration tests for the compilation actions using further SDKs."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest
from bqskit.ext import bqskit_to_qiskit, qiskit_to_bqskit
from bqskit.ir.circuit import Circuit
from mqt.bench.targets import get_available_device_names, get_device
from pytket._tket.passes import BasePass as TketBasePass  # noqa: PLC2701
from pytket.circuit import Qubit
from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.basepasses import BasePass as QiskitBasePass
from qiskit.transpiler.layout import TranspileLayout
from qiskit.transpiler.passes import CheckMap, GatesInBasis

from mqt.predictor.rl.actions import Action, CompilationOrigin, PassType, get_actions_by_pass_type
from mqt.predictor.rl.parsing import final_layout_bqskit_to_qiskit, final_layout_pytket_to_qiskit

if TYPE_CHECKING:
    from qiskit.transpiler import Target


@pytest.fixture
def available_actions_dict() -> dict[PassType, list[Action]]:
    """Return a dictionary of available actions."""
    return get_actions_by_pass_type()


def test_bqskit_o2_action(available_actions_dict: dict[str, list[Action]]) -> None:
    """Test the BQSKitO2 action."""
    action_bqskit_o2: Action | None = None
    for action in available_actions_dict[PassType.OPT]:
        if action.name == "BQSKitO2":
            action_bqskit_o2 = action
    assert isinstance(action_bqskit_o2, Action)

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    bqskit_qc = qiskit_to_bqskit(qc)
    assert callable(action_bqskit_o2.transpile_pass)
    optimized_bqskit_qc = action_bqskit_o2.transpile_pass(bqskit_qc)
    assert isinstance(optimized_bqskit_qc, Circuit)
    optimized_qc = bqskit_to_qiskit(optimized_bqskit_qc)

    assert optimized_qc != qc


@pytest.mark.parametrize("device", [get_device(name) for name in get_available_device_names()])
def test_bqskit_synthesis_action(device: Target, available_actions_dict: dict[str, list[Action]]) -> None:
    """Test the BQSKitSynthesis action for all devices."""
    action_bqskit_synthesis_action: Action | None = None
    for action in available_actions_dict[PassType.SYNTHESIS]:
        if action.name == "BQSKitSynthesis":
            action_bqskit_synthesis_action = action
    assert isinstance(action_bqskit_synthesis_action, Action)

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    check_nat_gates = GatesInBasis(target=device)
    check_nat_gates(qc)
    assert not check_nat_gates.property_set["all_gates_in_basis"]

    assert callable(action_bqskit_synthesis_action.transpile_pass)
    transpile_pass = action_bqskit_synthesis_action.transpile_pass(device)
    assert isinstance(transpile_pass, TketBasePass)
    bqskit_qc = qiskit_to_bqskit(qc)
    if "rigetti" in device.description or "ionq" in device.description or "iqm" in device.description:
        with pytest.raises(ValueError, match=re.escape("not supported in BQSKIT")):
            bqskit_to_qiskit(transpile_pass(bqskit_qc))
        return
    native_gates_qc = bqskit_to_qiskit(transpile_pass(bqskit_qc))

    check_nat_gates = GatesInBasis(target=device)
    check_nat_gates(native_gates_qc)
    only_nat_gates = check_nat_gates.property_set["all_gates_in_basis"]
    assert only_nat_gates


def test_bqskit_mapping_action_swaps_necessary(available_actions_dict: dict[str, list[Action]]) -> None:
    """Test the BQSKitMapping action for quantum circuit that requires SWAP gates."""
    bqskit_mapping_action = None
    for action in available_actions_dict[PassType.MAPPING]:
        if action.name == "BQSKitMapping":
            bqskit_mapping_action = action

    assert bqskit_mapping_action is not None

    qc = QuantumCircuit(8)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.cx(0, 3)
    qc.cx(0, 4)
    qc.cx(0, 5)
    qc.cx(0, 6)
    qc.cx(0, 7)

    device = get_device("ibm_falcon_27")
    bqskit_qc = qiskit_to_bqskit(qc)
    assert callable(bqskit_mapping_action.transpile_pass)
    lambda_ = bqskit_mapping_action.transpile_pass(device)
    assert callable(lambda_)
    bqskit_qc_mapped, input_mapping, output_mapping = lambda_(bqskit_qc)
    mapped_qc = bqskit_to_qiskit(bqskit_qc_mapped)
    layout = final_layout_bqskit_to_qiskit(input_mapping, output_mapping, mapped_qc, qc)

    assert input_mapping != output_mapping
    assert layout.final_layout is not None
    check_mapped_circuit(initial_qc=qc, mapped_qc=mapped_qc, device=device, layout=layout)


def check_mapped_circuit(
    initial_qc: QuantumCircuit, mapped_qc: QuantumCircuit, device: Target, layout: TranspileLayout
) -> None:
    """Check if the mapped quantum circuit is correctly mapped to the device."""
    # check if the altered circuit is correctly mapped to the device
    check_mapping = CheckMap(coupling_map=device.build_coupling_map())
    check_mapping(mapped_qc)
    mapped = check_mapping.property_set["is_swap_mapped"]
    assert mapped
    assert mapped_qc != initial_qc
    assert layout is not None
    assert len(layout.initial_layout) == device.num_qubits
    if layout.final_layout is not None:
        assert len(layout.final_layout) == device.num_qubits

    # each qubit of the initial layout is part of the initial quantum circuit and the register name is correctly set
    for assigned_physical_qubit in layout.initial_layout._p2v.values():  # noqa: SLF001
        qreg = assigned_physical_qubit._register  # noqa: SLF001
        assert qreg.name in {"q", "ancilla"}

        # assigned_physical_qubit is part of the original quantum circuit
        if qreg.name == "q":
            assert qreg.size == initial_qc.num_qubits
            # each qubit is also part of the initial uncompiled quantum circuit
            assert initial_qc.find_bit(assigned_physical_qubit).registers[0][0].name == "q"
        # assigned_physical_qubit is an ancilla qubit
        else:
            assert qreg.size == device.num_qubits - initial_qc.num_qubits
    # each qubit of the final layout is part of the mapped quantum circuit and the register name is correctly set
    if layout.final_layout is not None:
        for assigned_physical_qubit in layout.final_layout._p2v.values():  # noqa: SLF001
            assert mapped_qc.find_bit(assigned_physical_qubit).registers[0][0].name == "q"
    # each virtual qubit of the original quantum circuit is part of the initial layout
    for virtual_qubit in initial_qc.qubits:
        assert virtual_qubit in layout.initial_layout._p2v.values()  # noqa: SLF001


def test_bqskit_mapping_action_no_swaps_necessary(available_actions_dict: dict[str, list[Action]]) -> None:
    """Test the BQSKitMapping action for a simple quantum circuit that does not require SWAP gates."""
    bqskit_mapping_action: Action | None = None
    for action in available_actions_dict[PassType.MAPPING]:
        if action.name == "BQSKitMapping":
            bqskit_mapping_action = action
    assert isinstance(bqskit_mapping_action, Action)

    qc_no_swap_needed = QuantumCircuit(2)
    qc_no_swap_needed.h(0)
    qc_no_swap_needed.cx(0, 1)

    device = get_device("quantinuum_h2_56")

    bqskit_qc = qiskit_to_bqskit(qc_no_swap_needed)
    assert callable(bqskit_mapping_action.transpile_pass)
    lambda_ = bqskit_mapping_action.transpile_pass(device)
    assert callable(lambda_)
    bqskit_qc_mapped, input_mapping, output_mapping = lambda_(bqskit_qc)
    mapped_qc = bqskit_to_qiskit(bqskit_qc_mapped)
    layout = final_layout_bqskit_to_qiskit(input_mapping, output_mapping, mapped_qc, qc_no_swap_needed)
    assert layout is not None
    assert input_mapping == output_mapping
    assert layout.final_layout is None

    check_mapped_circuit(qc_no_swap_needed, mapped_qc, device, layout)


def test_tket_routing(available_actions_dict: dict[str, list[Action]]) -> None:
    """Test the TKETRouting action."""
    qc = QuantumCircuit(5)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.cx(0, 3)
    qc.cx(0, 4)

    device = get_device("quantinuum_h2_56")

    layout_action = available_actions_dict[PassType.LAYOUT][0]
    assert callable(layout_action.transpile_pass)
    transpile_pass = layout_action.transpile_pass(device)
    assert isinstance(transpile_pass, QiskitBasePass)
    pm = PassManager(transpile_pass)
    layouted_qc = pm.run(qc)
    initial_layout = pm.property_set["layout"]
    input_qubit_mapping = pm.property_set["original_qubit_indices"]

    routing_action: Action | None = None
    for action in available_actions_dict[PassType.ROUTING]:
        if action.origin == CompilationOrigin.TKET:
            routing_action = action
    assert isinstance(routing_action, Action)

    tket_qc = qiskit_to_tk(layouted_qc, preserve_param_uuid=True)
    assert callable(routing_action.transpile_pass)
    passes = routing_action.transpile_pass(device)
    assert isinstance(passes, list)
    for pass_ in passes:
        assert isinstance(pass_, TketBasePass)
        pass_.apply(tket_qc)

    qbs = tket_qc.qubits
    qubit_map = {qbs[i]: Qubit("q", i) for i in range(len(qbs))}
    tket_qc.rename_units(qubit_map)

    mapped_qc = tk_to_qiskit(tket_qc)

    final_layout = final_layout_pytket_to_qiskit(tket_qc, mapped_qc)

    layout = TranspileLayout(
        initial_layout=initial_layout,
        input_qubit_mapping=input_qubit_mapping,
        final_layout=final_layout,
        _output_qubit_list=mapped_qc.qubits,
        _input_qubit_count=qc.num_qubits,
    )

    check_mapped_circuit(qc, mapped_qc, device, layout)
