# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""BQSKit actions and execution helpers."""

from __future__ import annotations

import os
from functools import cache
from typing import TYPE_CHECKING, cast

from bqskit import MachineModel
from bqskit import compile as bqskit_compile
from bqskit.ext import qiskit_to_bqskit
from bqskit.ext.qiskit.translate import OPENQASM2Language
from bqskit.ir import gates
from qiskit import qasm2
from qiskit.circuit import QuantumRegister
from qiskit.circuit.library import RGate
from qiskit.transpiler import Layout, TranspileLayout

from mqt.predictor.rl.actions import CompilationOrigin, DeviceDependentAction, PassType

if TYPE_CHECKING:
    from collections.abc import Callable

    from bqskit import Circuit
    from bqskit.ir import Gate
    from qiskit import QuantumCircuit
    from qiskit.circuit import Instruction
    from qiskit.transpiler import Target

    from mqt.predictor.rl.actions import Action


def _r_gate(theta: float, phi: float) -> Instruction:
    return RGate(theta, phi)


def _bqskit_compilation_options() -> dict[str, float | int]:
    """Returns BQSKit options tuned for local runs and CI."""
    return {
        "optimization_level": 1 if os.getenv("GITHUB_ACTIONS") == "true" else 2,
        "synthesis_epsilon": 1e-1 if os.getenv("GITHUB_ACTIONS") == "true" else 1e-8,
        "max_synthesis_size": 3,
        "seed": 10,
        "num_workers": 1 if os.getenv("GITHUB_ACTIONS") == "true" else -1,
    }


def bqskit_to_qiskit(circuit: Circuit) -> QuantumCircuit:
    """Convert a BQSKit circuit to Qiskit.

    This mirrors BQSKit's Qiskit converter and supplements it with handling
    for IQM's native ``r`` gate, which BQSKit emits as ``U1q``.
    """
    qasm = OPENQASM2Language().encode(circuit).replace("U1q(", "r(")
    return qasm2.loads(
        qasm,
        custom_instructions=(
            *qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
            qasm2.CustomInstruction(
                "r", 2, 1, cast("Callable[[tuple[int | float, ...]], Instruction]", _r_gate), builtin=True
            ),
        ),
    )


def bqskit_optimization_action() -> Action:
    """Returns the BQSKit optimization action."""
    return DeviceDependentAction(
        "BQSKitO2",
        CompilationOrigin.BQSKIT,
        PassType.OPT,
        transpile_pass=lambda circuit: bqskit_compile(circuit, **_bqskit_compilation_options()),
    )


def bqskit_mapping_action() -> Action:
    """Returns the BQSKit mapping action."""
    return DeviceDependentAction(
        "BQSKitMapping",
        CompilationOrigin.BQSKIT,
        PassType.MAPPING,
        transpile_pass=lambda device: (
            lambda bqskit_circuit: bqskit_compile(
                bqskit_circuit,
                model=MachineModel(
                    num_qudits=device.num_qubits,
                    gate_set=get_bqskit_native_gates(device),
                    coupling_graph=[(elem[0], elem[1]) for elem in device.build_coupling_map()],
                ),
                with_mapping=True,
                **_bqskit_compilation_options(),
            )
        ),
    )


def bqskit_synthesis_action() -> Action:
    """Returns the BQSKit synthesis action."""
    return DeviceDependentAction(
        "BQSKitSynthesis",
        CompilationOrigin.BQSKIT,
        PassType.SYNTHESIS,
        transpile_pass=lambda device: (
            lambda bqskit_circuit: bqskit_compile(
                bqskit_circuit,
                model=MachineModel(bqskit_circuit.num_qudits, gate_set=get_bqskit_native_gates(device)),
                **_bqskit_compilation_options(),
            )
        ),
    )


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
        "r": gates.U1qGate(),
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

    for instr in device.operation_names:
        name = instr

        if name in [
            "barrier",
            "measure",
            "delay",
            "for_loop",
            "control",
            "while_loop",
            "if_test",
            "if_else",
            "switch_case",
            "break",
            "continue",
            "box",
        ]:
            continue

        if name not in gate_map:
            msg = f"The '{name}' gate of device '{device.description}' is not supported in BQSKIT."
            raise ValueError(msg)

        native_gates.append(gate_map[name])

    return native_gates


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

    initial_qubit_mapping = {bit: index for index, bit in enumerate(compiled_qc.qubits)}

    if bqskit_initial_layout == bqskit_final_layout:
        qiskit_final_layout = None
    else:
        qiskit_final_layout = {}
        for i in range(compiled_qc.num_qubits):
            if i in bqskit_final_layout:
                qiskit_final_layout[i] = compiled_qc.qubits[bqskit_initial_layout[bqskit_final_layout.index(i)]]
            else:
                qiskit_final_layout[i] = compiled_qc.qubits[i]

    return TranspileLayout(
        initial_layout=Layout(input_dict=qiskit_initial_layout),
        input_qubit_mapping=initial_qubit_mapping,
        final_layout=Layout(input_dict=qiskit_final_layout) if qiskit_final_layout else None,
        _output_qubit_list=compiled_qc.qubits,
        _input_qubit_count=initial_qc.num_qubits,
    )


def run_bqskit_action(
    action: Action,
    circuit: QuantumCircuit,
    device: Target,
    layout: TranspileLayout | None,
) -> tuple[QuantumCircuit, TranspileLayout | None]:
    """Apply a BQSKit action and return the updated circuit and layout metadata."""
    bqskit_qc = qiskit_to_bqskit(circuit)
    if action.pass_type == PassType.OPT:
        transpile = cast("Callable[[Circuit], Circuit]", action.transpile_pass)
        bqskit_compiled_qc = transpile(bqskit_qc)
    elif action.pass_type == PassType.SYNTHESIS:
        factory = cast("Callable[[Target], Callable[[Circuit], Circuit]]", action.transpile_pass)
        bqskit_compiled_qc = factory(device)(bqskit_qc)
    elif action.pass_type == PassType.MAPPING:
        factory = cast(
            "Callable[[Target], Callable[[Circuit], tuple[Circuit, tuple[int, ...], tuple[int, ...]]]]",
            action.transpile_pass,
        )
        bqskit_compiled_qc, initial, final = factory(device)(bqskit_qc)
        compiled_qiskit_qc = bqskit_to_qiskit(bqskit_compiled_qc)
        return compiled_qiskit_qc, final_layout_bqskit_to_qiskit(initial, final, compiled_qiskit_qc, circuit)
    else:
        msg = f"Unhandled BQSKit pass type: {action.pass_type}"
        raise ValueError(msg)

    return bqskit_to_qiskit(bqskit_compiled_qc), layout
