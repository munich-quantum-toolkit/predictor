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
import re
from functools import cache
from typing import TYPE_CHECKING, TypeAlias, cast

import numpy as np
from bqskit import MachineModel
from bqskit.compiler import Compiler, Workflow
from bqskit.compiler.compile import (
    build_multi_qudit_retarget_workflow,
    build_partitioning_workflow,
    build_seqpam_mapping_optimization_workflow,
    build_single_qudit_retarget_workflow,
    get_instantiate_options,
)
from bqskit.ext import qiskit_to_bqskit
from bqskit.ext.qiskit.translate import OPENQASM2Language
from bqskit.ir import gates
from bqskit.passes import (
    ApplyPlacement,
    BlockZXZPass,
    ExtractMeasurements,
    FullBlockZXZPass,
    FullQSDPass,
    GeneralizedSabreLayoutPass,
    GeneralizedSabreRoutingPass,
    GreedyPlacementPass,
    IfThenElsePass,
    LEAPSynthesisPass,
    PassPredicate,
    QSearchSynthesisPass,
    RestoreMeasurements,
    SetModelPass,
    SetRandomSeedPass,
    StaticPlacementPass,
    TrivialPlacementPass,
    UnfoldPass,
    WalshDiagonalSynthesisPass,
)
from qiskit import qasm2
from qiskit.circuit import Instruction, QuantumRegister
from qiskit.circuit.library import RGate
from qiskit.transpiler import Layout, TranspileLayout

from mqt.predictor.rl.actions import CompilationOrigin, DeviceDependentAction, PassType

if TYPE_CHECKING:
    from collections.abc import Callable

    from bqskit import Circuit
    from bqskit.compiler.basepass import BasePass as BQSKitBasePass
    from bqskit.compiler.passdata import PassData
    from bqskit.compiler.workflow import WorkflowLike
    from bqskit.ir import Gate
    from qiskit import QuantumCircuit
    from qiskit.circuit import Qubit as QiskitQubit
    from qiskit.transpiler import Target

    from mqt.predictor.rl.actions import Action

    BQSKitMapping: TypeAlias = tuple[Circuit, tuple[int, ...], tuple[int, ...]]


_BQSKIT_OPT_LEVEL = 1 if os.getenv("GITHUB_ACTIONS") == "true" else 2
_BQSKIT_SYNTHESIS_EPSILON = 1e-1 if os.getenv("GITHUB_ACTIONS") == "true" else 1e-8
_BQSKIT_BLOCK_SIZE = 4
_BQSKIT_SEED = 10
_BQSKIT_NUM_WORKERS = 1 if os.getenv("GITHUB_ACTIONS") == "true" else -1


def _r_gate(theta: float, phi: float) -> Instruction:
    """Construct an RGate with the given parameters."""
    return RGate(theta, phi)


def bqskit_to_qiskit(circuit: Circuit) -> QuantumCircuit:
    """Convert a BQSKit circuit to Qiskit while preserving IQM's R gate."""
    qasm = OPENQASM2Language().encode(circuit)
    qasm = re.sub(r"\bU1q\(", "r(", qasm)
    return qasm2.loads(
        qasm,
        custom_instructions=(
            *qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
            qasm2.CustomInstruction(
                "r", 2, 1, cast("Callable[[tuple[int | float, ...]], Instruction]", _r_gate), builtin=True
            ),
        ),
    )


class _DiagonalUnitaryPredicate(PassPredicate):
    """Return ``True`` when the current BQSKit block unitary is diagonal."""

    def __init__(self, atol: float = 1e-9) -> None:
        self.atol = atol

    def get_truth_value(self, circuit: Circuit, data: PassData) -> bool:
        """Check whether ``circuit`` represents a diagonal unitary."""
        del data
        unitary = np.asarray(circuit.get_unitary())
        diagonal = np.diag(np.diag(unitary))
        return np.allclose(unitary, diagonal, atol=self.atol)


def _run_bqskit_workflow(
    circuit: Circuit, workflow: Workflow, request_data: bool = False
) -> Circuit | tuple[Circuit, PassData]:
    """Compile ``circuit`` with a custom BQSKit workflow."""
    compiler = Compiler(num_workers=_BQSKIT_NUM_WORKERS)
    try:
        result = compiler.compile(circuit, workflow, request_data=request_data)
    finally:
        compiler.close()

    if request_data:
        return cast("tuple[Circuit, PassData]", result)

    return cast("Circuit", result)


@cache
def get_bqskit_native_gates(device: Target) -> list[Gate]:
    """Returns the native gates of the given device.

    Args:
        device: Target whose operation names are translated to BQSKit native gates.

    Returns:
        The native gates of the given Target as a list of BQSKit Gate objects.

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


def _bqskit_partitioned_synthesis_factory(
    device: Target,
    synthesis_pass: WorkflowLike,
) -> Callable[[Circuit], Circuit]:
    """Create a block-based BQSKit synthesis callable for an RL action."""

    def _compile(circuit: Circuit) -> Circuit:
        model = MachineModel(circuit.num_qudits, gate_set=get_bqskit_native_gates(device))
        workflow = Workflow([
            SetRandomSeedPass(_BQSKIT_SEED),
            UnfoldPass(),
            ExtractMeasurements(),
            SetModelPass(model),
            build_partitioning_workflow(
                synthesis_pass,
                block_size=_BQSKIT_BLOCK_SIZE,
                replace_filter_method="less-than-respecting-fully",
            ),
            build_multi_qudit_retarget_workflow(
                _BQSKIT_OPT_LEVEL,
                _BQSKIT_SYNTHESIS_EPSILON,
                max_synthesis_size=_BQSKIT_BLOCK_SIZE,
            ),
            build_single_qudit_retarget_workflow(
                _BQSKIT_OPT_LEVEL,
                _BQSKIT_SYNTHESIS_EPSILON,
                max_synthesis_size=_BQSKIT_BLOCK_SIZE,
            ),
            RestoreMeasurements(),
        ])
        return cast("Circuit", _run_bqskit_workflow(circuit, workflow))

    return _compile


def _bqskit_mapping_factory(
    device: Target,
    *mapping_passes: BQSKitBasePass,
    apply_placement: bool,
) -> Callable[[Circuit], BQSKitMapping]:
    """Create a BQSKit mapping callable for layout or routing actions."""

    def _compile(circuit: Circuit) -> BQSKitMapping:
        model = MachineModel(
            num_qudits=device.num_qubits,
            gate_set=get_bqskit_native_gates(device),
            coupling_graph=[(edge[0], edge[1]) for edge in device.build_coupling_map()],
        )
        workflow_passes = [*mapping_passes]
        if apply_placement:
            workflow_passes.append(ApplyPlacement())
        workflow = Workflow([
            SetRandomSeedPass(_BQSKIT_SEED),
            UnfoldPass(),
            ExtractMeasurements(),
            SetModelPass(model),
            *workflow_passes,
            RestoreMeasurements(),
        ])
        compiled_circuit, data = cast("tuple[Circuit, PassData]", _run_bqskit_workflow(circuit, workflow, True))
        return compiled_circuit, tuple(data.initial_mapping), tuple(data.final_mapping)

    return _compile


def bqskit_synthesis_actions() -> list[Action]:
    """Returns the BQSKit synthesis actions."""
    return [
        DeviceDependentAction(
            "QSearchSynthesisPass",
            CompilationOrigin.BQSKIT,
            PassType.SYNTHESIS,
            transpile_pass=lambda device: _bqskit_partitioned_synthesis_factory(
                device,
                QSearchSynthesisPass(
                    success_threshold=_BQSKIT_SYNTHESIS_EPSILON,
                    instantiate_options=get_instantiate_options(_BQSKIT_OPT_LEVEL),
                ),
            ),
        ),
        DeviceDependentAction(
            "LEAPSynthesisPass",
            CompilationOrigin.BQSKIT,
            PassType.SYNTHESIS,
            transpile_pass=lambda device: _bqskit_partitioned_synthesis_factory(
                device,
                LEAPSynthesisPass(
                    success_threshold=_BQSKIT_SYNTHESIS_EPSILON,
                    min_prefix_size=[3, 4][min(_BQSKIT_OPT_LEVEL, 2) - 1],
                    instantiate_options=get_instantiate_options(_BQSKIT_OPT_LEVEL),
                ),
            ),
        ),
        DeviceDependentAction(
            "WalshDiagonalSynthesisPass",
            CompilationOrigin.BQSKIT,
            PassType.SYNTHESIS,
            transpile_pass=lambda device: _bqskit_partitioned_synthesis_factory(
                device,
                IfThenElsePass(_DiagonalUnitaryPredicate(), WalshDiagonalSynthesisPass()),
            ),
        ),
        DeviceDependentAction(
            "FullQSDPass",
            CompilationOrigin.BQSKIT,
            PassType.SYNTHESIS,
            transpile_pass=lambda device: _bqskit_partitioned_synthesis_factory(
                device,
                FullQSDPass(min_qudit_size=2, perform_scan=False),
            ),
        ),
        DeviceDependentAction(
            "BlockZXZPass",
            CompilationOrigin.BQSKIT,
            PassType.SYNTHESIS,
            transpile_pass=lambda device: _bqskit_partitioned_synthesis_factory(
                device,
                BlockZXZPass(min_qudit_size=_BQSKIT_BLOCK_SIZE - 1),
            ),
        ),
        DeviceDependentAction(
            "FullBlockZXZPass",
            CompilationOrigin.BQSKIT,
            PassType.SYNTHESIS,
            transpile_pass=lambda device: _bqskit_partitioned_synthesis_factory(device, FullBlockZXZPass()),
        ),
    ]


def bqskit_pam_mapping_action() -> Action:
    """Returns the BQSKit sequential permutation-aware mapping action."""

    def _factory(device: Target) -> Callable[[Circuit], BQSKitMapping]:
        def _compile(circuit: Circuit) -> BQSKitMapping:
            model = MachineModel(
                num_qudits=device.num_qubits,
                gate_set=get_bqskit_native_gates(device),
                coupling_graph=[(edge[0], edge[1]) for edge in device.build_coupling_map()],
            )
            workflow = Workflow([
                SetRandomSeedPass(_BQSKIT_SEED),
                UnfoldPass(),
                ExtractMeasurements(),
                SetModelPass(model),
                build_seqpam_mapping_optimization_workflow(
                    _BQSKIT_OPT_LEVEL,
                    _BQSKIT_SYNTHESIS_EPSILON,
                    block_size=_BQSKIT_BLOCK_SIZE,
                ),
                RestoreMeasurements(),
            ])
            compiled_circuit, data = cast(
                "tuple[Circuit, PassData]",
                _run_bqskit_workflow(circuit, workflow, True),
            )
            return compiled_circuit, tuple(data.initial_mapping), tuple(data.final_mapping)

        return _compile

    return DeviceDependentAction(
        "SeqPAMMapping",
        CompilationOrigin.BQSKIT,
        PassType.MAPPING,
        transpile_pass=_factory,
    )


def bqskit_layout_actions() -> list[Action]:
    """Returns the BQSKit layout actions."""
    return [
        DeviceDependentAction(
            "GreedyPlacementPass",
            CompilationOrigin.BQSKIT,
            PassType.LAYOUT,
            transpile_pass=lambda device: _bqskit_mapping_factory(device, GreedyPlacementPass(), apply_placement=True),
        ),
        DeviceDependentAction(
            "TrivialPlacementPass",
            CompilationOrigin.BQSKIT,
            PassType.LAYOUT,
            transpile_pass=lambda device: _bqskit_mapping_factory(device, TrivialPlacementPass(), apply_placement=True),
        ),
        DeviceDependentAction(
            "StaticPlacementPass",
            CompilationOrigin.BQSKIT,
            PassType.LAYOUT,
            transpile_pass=lambda device: _bqskit_mapping_factory(device, StaticPlacementPass(), apply_placement=True),
        ),
        DeviceDependentAction(
            "GeneralizedSabreLayoutPass",
            CompilationOrigin.BQSKIT,
            PassType.LAYOUT,
            transpile_pass=lambda device: _bqskit_mapping_factory(
                device,
                GreedyPlacementPass(),
                GeneralizedSabreLayoutPass(),
                apply_placement=True,
            ),
        ),
    ]


def bqskit_routing_action() -> Action:
    """Returns the BQSKit routing action."""
    return DeviceDependentAction(
        "GeneralizedSabreRoutingPass",
        CompilationOrigin.BQSKIT,
        PassType.ROUTING,
        transpile_pass=lambda device: _bqskit_mapping_factory(
            device, GeneralizedSabreRoutingPass(), apply_placement=False
        ),
    )


def bqskit_routing_actions() -> list[Action]:
    """Returns the BQSKit routing actions."""
    return [bqskit_routing_action()]


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

    Args:
        bqskit_initial_layout: Tuple mapping each BQSKit virtual qubit index to its initial physical qubit index.
        bqskit_final_layout: Tuple mapping each BQSKit virtual qubit index to its final physical qubit index.
        compiled_qc: Compiled QuantumCircuit whose qubits define the output qubit list and final layout values.
        initial_qc: Initial QuantumCircuit whose qubits define the input layout values.

    Returns:
        A TranspileLayout with a Qiskit Layout for the initial layout, an input-qubit-to-index mapping, and a final
        Layout mapping physical qubit indices to compiled QuantumCircuit qubits when BQSKit changed the layout.
    """
    ancilla = QuantumRegister(compiled_qc.num_qubits - initial_qc.num_qubits, "ancilla")
    qiskit_initial_layout: dict[int, object] = {}
    counter_ancilla_qubit = 0
    for i in range(compiled_qc.num_qubits):
        if i in bqskit_initial_layout:
            qiskit_initial_layout[i] = initial_qc.qubits[bqskit_initial_layout.index(i)]
        else:
            qiskit_initial_layout[i] = ancilla[counter_ancilla_qubit]
            counter_ancilla_qubit += 1

    initial_qubit_mapping = {bit: index for index, bit in enumerate(initial_qc.qubits)}
    initial_qubit_mapping.update({bit: initial_qc.num_qubits + index for index, bit in enumerate(ancilla)})

    qiskit_final_layout: dict[int, object] | None = None
    if bqskit_initial_layout != bqskit_final_layout:
        qiskit_final_layout = {}
        used_output_wires: set[int] = set()
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


def final_layout_bqskit_routing_to_qiskit(
    bqskit_final_layout: tuple[int, ...],
    output_qubits: list[QiskitQubit],
) -> Layout | None:
    """Convert a BQSKit routing permutation into a Qiskit final layout."""
    if bqskit_final_layout == tuple(range(len(bqskit_final_layout))):
        return None

    qiskit_final_layout: dict[int, QiskitQubit] = {}
    used_output_positions: set[int] = set()
    for input_position, final_position in enumerate(bqskit_final_layout):
        qiskit_final_layout[final_position] = output_qubits[input_position]
        used_output_positions.add(input_position)

    remaining_physical_positions = [i for i in range(len(output_qubits)) if i not in qiskit_final_layout]
    remaining_output_positions = [i for i in range(len(output_qubits)) if i not in used_output_positions]

    for physical_position, output_position in zip(
        remaining_physical_positions, remaining_output_positions, strict=True
    ):
        qiskit_final_layout[physical_position] = output_qubits[output_position]

    return Layout(input_dict=qiskit_final_layout)


def run_bqskit_action(
    *,
    action: Action,
    circuit: QuantumCircuit,
    device: Target,
    layout: TranspileLayout | None,
) -> tuple[QuantumCircuit, TranspileLayout | None]:
    """Apply a BQSKit action and update the layout bookkeeping it owns.

    Args:
        action: The BQSKit action to apply.
        circuit: The current quantum circuit.
        device: The target device.
        layout: The current layout (if any).

    Returns:
        Tuple of (compiled circuit, updated layout).
    """
    bqskit_qc = qiskit_to_bqskit(circuit)

    # OPT actions don't take a device parameter
    if action.pass_type == PassType.OPT:
        transpile = cast("Callable[[Circuit], Circuit]", action.transpile_pass)
        compiled_qc = transpile(bqskit_qc)
        return bqskit_to_qiskit(compiled_qc), layout

    # SYNTHESIS actions use device factory
    if action.pass_type == PassType.SYNTHESIS:
        factory = cast("Callable[[Target], Callable[[Circuit], Circuit]]", action.transpile_pass)
        compiled_qc = factory(device)(bqskit_qc)
        return bqskit_to_qiskit(compiled_qc), layout

    # LAYOUT and MAPPING actions establish layout
    if action.pass_type in (PassType.LAYOUT, PassType.MAPPING):
        factory = cast("Callable[[Target], Callable[[Circuit], BQSKitMapping]]", action.transpile_pass)
        compiled_qc, initial, final = factory(device)(bqskit_qc)
        compiled_qiskit_qc = bqskit_to_qiskit(compiled_qc)
        return compiled_qiskit_qc, final_layout_bqskit_to_qiskit(initial, final, compiled_qiskit_qc, circuit)

    # ROUTING actions require existing layout
    if action.pass_type == PassType.ROUTING:
        assert layout is not None, "BQSKit routing requires an existing layout."
        factory = cast("Callable[[Target], Callable[[Circuit], BQSKitMapping]]", action.transpile_pass)
        compiled_qc, _initial, final = factory(device)(bqskit_qc)
        compiled_qiskit_qc = bqskit_to_qiskit(compiled_qc)
        output_qubits = layout._output_qubit_list  # noqa: SLF001
        assert output_qubits is not None
        layout.final_layout = final_layout_bqskit_routing_to_qiskit(final, list(output_qubits))
        return compiled_qiskit_qc, layout

    msg = f"Unhandled BQSKit action pass type: {action.pass_type}"
    raise ValueError(msg)


__all__ = [
    "bqskit_layout_actions",
    "bqskit_pam_mapping_action",
    "bqskit_routing_action",
    "bqskit_routing_actions",
    "bqskit_synthesis_actions",
    "bqskit_to_qiskit",
    "final_layout_bqskit_routing_to_qiskit",
    "final_layout_bqskit_to_qiskit",
    "get_bqskit_native_gates",
    "is_bqskit_action_available",
    "run_bqskit_action",
]


def is_bqskit_action_available(*, has_parameterized_gates: bool) -> bool:
    """Return whether a BQSKit action is available for the current circuit state."""
    # BQSKit does not support parameterized gates
    return not has_parameterized_gates
