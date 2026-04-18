# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""BQSKit RL actions and their action-local helper logic."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, cast

import numpy as np
from bqskit import MachineModel
from bqskit.compiler import Compiler, Workflow
from bqskit.compiler.compile import (
    build_multi_qudit_retarget_workflow,
    build_partitioning_workflow,
    build_single_qudit_retarget_workflow,
    get_instantiate_options,
)
from bqskit.ir import gates

if TYPE_CHECKING:
    from bqskit.ir import Gate
    from qiskit import QuantumCircuit
    from qiskit.transpiler import Target
from bqskit.ext import bqskit_to_qiskit, qiskit_to_bqskit
from bqskit.passes import (
    ApplyPlacement,
    BlockZXZPass,
    ExtractMeasurements,
    FullBlockZXZPass,
    GeneralizedSabreLayoutPass,
    GeneralizedSabreRoutingPass,
    GreedyPlacementPass,
    IfThenElsePass,
    LEAPSynthesisPass,
    MGDPass,
    PassPredicate,
    QSDPass,
    QSearchSynthesisPass,
    RestoreMeasurements,
    SetModelPass,
    SetRandomSeedPass,
    StaticPlacementPass,
    TrivialPlacementPass,
    UnfoldPass,
    WalshDiagonalSynthesisPass,
)
from qiskit import QuantumRegister
from qiskit.transpiler import Layout, TranspileLayout

from . import CompilationOrigin, DeviceDependentAction, PassType, register_action

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Protocol

    from bqskit import Circuit
    from bqskit.compiler.basepass import BasePass as BQSKitBasePass
    from bqskit.compiler.passdata import PassData
    from bqskit.compiler.workflow import WorkflowLike
    from qiskit.circuit import Qubit as QiskitQubit
    from qiskit.transpiler import Target

    from . import Action, BQSKitMapping

    class BQSKitWorkflowDataLike(Protocol):
        """Minimal attribute shape used from BQSKit request-data payloads."""

        initial_mapping: Sequence[int]
        final_mapping: Sequence[int]


_BQSKIT_OPT_LEVEL = 1 if os.getenv("GITHUB_ACTIONS") == "true" else 2
_BQSKIT_SYNTHESIS_EPSILON = 1e-1 if os.getenv("GITHUB_ACTIONS") == "true" else 1e-8
_BQSKIT_MAX_SYNTHESIS_SIZE = 3
_BQSKIT_SEED = 10


@dataclass
class _BQSKitCompileData:
    """Materialized mapping metadata returned from BQSKit workflows."""

    initial_mapping: tuple[int, ...]
    final_mapping: tuple[int, ...]


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


def _layout_output_qubits(layout: TranspileLayout) -> list[QiskitQubit]:
    """Return the materialized output wires tracked by a ``TranspileLayout``."""
    output_qubits = layout._output_qubit_list  # noqa: SLF001
    assert output_qubits is not None
    return list(output_qubits)


def _run_bqskit_workflow(
    circuit: Circuit, workflow: Workflow, request_data: bool = False
) -> Circuit | tuple[Circuit, _BQSKitCompileData]:
    """Compile ``circuit`` with a custom BQSKit workflow."""
    compiler = Compiler()
    try:
        result = compiler.compile(circuit, workflow, request_data=request_data)
    finally:
        compiler.close()

    if request_data:
        compiled_circuit, data = cast("tuple[Circuit, BQSKitWorkflowDataLike]", result)
        return compiled_circuit, _BQSKitCompileData(
            initial_mapping=tuple(data.initial_mapping),
            final_mapping=tuple(data.final_mapping),
        )

    return cast("Circuit", result)


def _build_bqskit_common_prefix(model: MachineModel) -> list[BQSKitBasePass]:
    """Build the common BQSKit workflow prefix used for RL actions."""
    return [
        SetRandomSeedPass(_BQSKIT_SEED),
        UnfoldPass(),
        ExtractMeasurements(),
        SetModelPass(model),
    ]


def _build_bqskit_common_suffix() -> list[BQSKitBasePass]:
    """Build the common BQSKit workflow suffix used for RL actions."""
    return [RestoreMeasurements()]


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


def _bqskit_partitioned_synthesis_factory(
    device: Target,
    synthesis_pass: WorkflowLike,
) -> Callable[[Circuit], Circuit]:
    """Create a block-based BQSKit synthesis callable for an RL action."""

    def _compile(circuit: Circuit) -> Circuit:
        model = MachineModel(circuit.num_qudits, gate_set=get_bqskit_native_gates(device))
        workflow = Workflow([
            *_build_bqskit_common_prefix(model),
            build_partitioning_workflow(
                synthesis_pass,
                _BQSKIT_MAX_SYNTHESIS_SIZE,
                replace_filter_method="less-than-respecting-fully",
            ),
            build_multi_qudit_retarget_workflow(
                _BQSKIT_OPT_LEVEL,
                _BQSKIT_SYNTHESIS_EPSILON,
                _BQSKIT_MAX_SYNTHESIS_SIZE,
            ),
            build_single_qudit_retarget_workflow(
                _BQSKIT_OPT_LEVEL,
                _BQSKIT_SYNTHESIS_EPSILON,
                _BQSKIT_MAX_SYNTHESIS_SIZE,
            ),
            *_build_bqskit_common_suffix(),
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
            *_build_bqskit_common_prefix(model),
            *workflow_passes,
            *_build_bqskit_common_suffix(),
        ])
        compiled_circuit, data = cast(
            "tuple[Circuit, _BQSKitCompileData]", _run_bqskit_workflow(circuit, workflow, request_data=True)
        )
        return compiled_circuit, data.initial_mapping, data.final_mapping

    return _compile


def final_layout_bqskit_to_qiskit(
    bqskit_initial_layout: tuple[int, ...],
    bqskit_final_layout: tuple[int, ...],
    compiled_qc: QuantumCircuit,
    initial_qc: QuantumCircuit,
) -> TranspileLayout:
    """Convert BQSKit layout metadata into a Qiskit ``TranspileLayout``."""
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
        layout.final_layout = final_layout_bqskit_routing_to_qiskit(final, _layout_output_qubits(layout))
        return compiled_qiskit_qc, layout

    msg = f"Unhandled BQSKit action pass type: {action.pass_type}"
    raise ValueError(msg)


register_action(
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
    )
)

register_action(
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
    )
)

register_action(
    DeviceDependentAction(
        "WalshDiagonalSynthesisPass",
        CompilationOrigin.BQSKIT,
        PassType.SYNTHESIS,
        transpile_pass=lambda device: _bqskit_partitioned_synthesis_factory(
            device,
            IfThenElsePass(_DiagonalUnitaryPredicate(), WalshDiagonalSynthesisPass()),
        ),
    )
)

register_action(
    DeviceDependentAction(
        "QSDPass",
        CompilationOrigin.BQSKIT,
        PassType.SYNTHESIS,
        transpile_pass=lambda device: _bqskit_partitioned_synthesis_factory(device, QSDPass()),
    )
)

register_action(
    DeviceDependentAction(
        "MGDPass",
        CompilationOrigin.BQSKIT,
        PassType.SYNTHESIS,
        transpile_pass=lambda device: _bqskit_partitioned_synthesis_factory(device, MGDPass()),
    )
)

register_action(
    DeviceDependentAction(
        "BlockZXZPass",
        CompilationOrigin.BQSKIT,
        PassType.SYNTHESIS,
        transpile_pass=lambda device: _bqskit_partitioned_synthesis_factory(device, BlockZXZPass()),
    )
)

register_action(
    DeviceDependentAction(
        "FullBlockZXZPass",
        CompilationOrigin.BQSKIT,
        PassType.SYNTHESIS,
        transpile_pass=lambda device: _bqskit_partitioned_synthesis_factory(device, FullBlockZXZPass()),
    )
)

register_action(
    DeviceDependentAction(
        "GreedyPlacementPass",
        CompilationOrigin.BQSKIT,
        PassType.LAYOUT,
        transpile_pass=lambda device: _bqskit_mapping_factory(device, GreedyPlacementPass(), apply_placement=True),
    )
)

register_action(
    DeviceDependentAction(
        "TrivialPlacementPass",
        CompilationOrigin.BQSKIT,
        PassType.LAYOUT,
        transpile_pass=lambda device: _bqskit_mapping_factory(device, TrivialPlacementPass(), apply_placement=True),
    )
)

register_action(
    DeviceDependentAction(
        "StaticPlacementPass",
        CompilationOrigin.BQSKIT,
        PassType.LAYOUT,
        transpile_pass=lambda device: _bqskit_mapping_factory(device, StaticPlacementPass(), apply_placement=True),
    )
)

register_action(
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
    )
)

register_action(
    DeviceDependentAction(
        "GeneralizedSabreRoutingPass",
        CompilationOrigin.BQSKIT,
        PassType.ROUTING,
        transpile_pass=lambda device: _bqskit_mapping_factory(
            device, GeneralizedSabreRoutingPass(), apply_placement=False
        ),
    )
)


__all__ = [
    "final_layout_bqskit_routing_to_qiskit",
    "final_layout_bqskit_to_qiskit",
    "run_bqskit_action",
]
