# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Qiskit actions and execution helpers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

from qiskit.circuit import StandardEquivalenceLibrary
from qiskit.circuit.library import (
    CXGate,
    CYGate,
    CZGate,
    ECRGate,
    HGate,
    SdgGate,
    SGate,
    SwapGate,
    SXdgGate,
    SXGate,
    TdgGate,
    TGate,
    XGate,
    YGate,
    ZGate,
)
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.passmanager import ConditionalController
from qiskit.passmanager.flow_controllers import DoWhileController
from qiskit.transpiler import CouplingMap, PassManager, TranspileLayout
from qiskit.transpiler.passes import (
    ApplyLayout,
    BasisTranslator,
    Collect2qBlocks,
    CommutativeCancellation,
    CommutativeInverseCancellation,
    ConsolidateBlocks,
    DenseLayout,
    Depth,
    EnlargeWithAncilla,
    FixedPoint,
    FullAncillaAllocation,
    GatesInBasis,
    InverseCancellation,
    MinimumPoint,
    Optimize1qGatesDecomposition,
    OptimizeCliffords,
    RemoveDiagonalGatesBeforeMeasure,
    SabreLayout,
    Size,
    UnitarySynthesis,
    VF2Layout,
    VF2PostLayout,
)
from qiskit.transpiler.passes.layout.vf2_layout import VF2LayoutStopReason
from qiskit.transpiler.preset_passmanagers import common

from mqt.predictor.rl.actions import (
    CompilationOrigin,
    DeviceDependentAction,
    DeviceIndependentAction,
    PassType,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from qiskit import QuantumCircuit
    from qiskit.passmanager import PropertySet
    from qiskit.passmanager.base_tasks import Task
    from qiskit.transpiler import Layout, Target

    from mqt.predictor.rl.actions import (
        Action,
    )

logger = logging.getLogger("mqt-predictor")


def qiskit_optimization_actions() -> list[Action]:
    """Returns the Qiskit optimization actions."""
    return [
        DeviceIndependentAction(
            "Optimize1qGatesDecomposition",
            CompilationOrigin.QISKIT,
            PassType.OPT,
            [Optimize1qGatesDecomposition()],
        ),
        DeviceIndependentAction(
            "CommutativeCancellation",
            CompilationOrigin.QISKIT,
            PassType.OPT,
            [CommutativeCancellation()],
        ),
        DeviceIndependentAction(
            "CommutativeInverseCancellation",
            CompilationOrigin.QISKIT,
            PassType.OPT,
            [CommutativeInverseCancellation()],
        ),
        DeviceIndependentAction(
            "RemoveDiagonalGatesBeforeMeasure",
            CompilationOrigin.QISKIT,
            PassType.OPT,
            [RemoveDiagonalGatesBeforeMeasure()],
        ),
        DeviceIndependentAction(
            "InverseCancellation",
            CompilationOrigin.QISKIT,
            PassType.OPT,
            [
                InverseCancellation([
                    CXGate(),
                    ECRGate(),
                    CZGate(),
                    CYGate(),
                    XGate(),
                    YGate(),
                    ZGate(),
                    HGate(),
                    SwapGate(),
                    (TGate(), TdgGate()),
                    (SGate(), SdgGate()),
                    (SXGate(), SXdgGate()),
                ])
            ],
        ),
        DeviceIndependentAction(
            "OptimizeCliffords",
            CompilationOrigin.QISKIT,
            PassType.OPT,
            [OptimizeCliffords()],
        ),
        DeviceIndependentAction(
            "Opt2qBlocks",
            CompilationOrigin.QISKIT,
            PassType.OPT,
            [Collect2qBlocks(), ConsolidateBlocks(), UnitarySynthesis()],
        ),
    ]


def qiskit_o3_action() -> Action:
    """Returns the Qiskit level-3 optimization action."""
    return DeviceDependentAction(
        "QiskitO3",
        CompilationOrigin.QISKIT,
        PassType.OPT,
        transpile_pass=lambda native_gate, coupling_map: cast(
            "list[Task]",
            [
                Collect2qBlocks(),
                ConsolidateBlocks(basis_gates=native_gate),
                UnitarySynthesis(basis_gates=native_gate, coupling_map=coupling_map),
                Optimize1qGatesDecomposition(basis=native_gate),
                CommutativeCancellation(basis_gates=native_gate),
                GatesInBasis(native_gate),
                ConditionalController(
                    common.generate_translation_passmanager(
                        target=None, basis_gates=native_gate, coupling_map=coupling_map
                    ).to_flow_controller(),
                    condition=lambda property_set: not property_set["all_gates_in_basis"],
                ),
                Depth(recurse=True),
                FixedPoint("depth"),
                Size(recurse=True),
                FixedPoint("size"),
                MinimumPoint(["depth", "size"], "optimization_loop"),
            ],
        ),
        do_while=lambda property_set: not property_set["optimization_loop_minimum_point"],
    )


def qiskit_final_optimization_action() -> Action:
    """Returns the Qiskit final layout optimization action."""
    return DeviceDependentAction(
        "VF2PostLayout",
        CompilationOrigin.QISKIT,
        PassType.FINAL_OPT,
        transpile_pass=lambda device: [VF2PostLayout(target=device)],
    )


def qiskit_layout_actions() -> list[Action]:
    """Returns the Qiskit layout actions."""
    return [
        DeviceDependentAction(
            "DenseLayout",
            CompilationOrigin.QISKIT,
            PassType.LAYOUT,
            transpile_pass=lambda device: cast(
                "list[Task]",
                [
                    DenseLayout(coupling_map=CouplingMap(device.build_coupling_map())),
                    FullAncillaAllocation(coupling_map=CouplingMap(device.build_coupling_map())),
                    EnlargeWithAncilla(),
                    ApplyLayout(),
                ],
            ),
        ),
        DeviceDependentAction(
            "VF2Layout",
            CompilationOrigin.QISKIT,
            PassType.LAYOUT,
            transpile_pass=lambda device: cast(
                "list[Task]",
                [
                    VF2Layout(target=device),
                    ConditionalController(
                        [
                            FullAncillaAllocation(coupling_map=CouplingMap(device.build_coupling_map())),
                            EnlargeWithAncilla(),
                            ApplyLayout(),
                        ],
                        condition=lambda property_set: (
                            property_set["VF2Layout_stop_reason"] == VF2LayoutStopReason.SOLUTION_FOUND
                        ),
                    ),
                ],
            ),
        ),
    ]


def qiskit_mapping_action() -> Action:
    """Returns the Qiskit mapping action."""
    return DeviceDependentAction(
        "SabreMapping",
        CompilationOrigin.QISKIT,
        PassType.MAPPING,
        transpile_pass=lambda device: cast(
            "list[Task]", [SabreLayout(coupling_map=CouplingMap(device.build_coupling_map()), skip_routing=False)]
        ),
    )


def qiskit_synthesis_action() -> Action:
    """Returns the Qiskit synthesis action."""
    return DeviceDependentAction(
        "BasisTranslator",
        CompilationOrigin.QISKIT,
        PassType.SYNTHESIS,
        transpile_pass=lambda device: cast(
            "list[Task]", [BasisTranslator(StandardEquivalenceLibrary, target_basis=device.operation_names)]
        ),
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
    apply_layout = ApplyLayout()
    apply_layout.property_set["layout"] = layout_before.initial_layout
    apply_layout.property_set["original_qubit_indices"] = layout_before.input_qubit_mapping
    apply_layout.property_set["final_layout"] = layout_before.final_layout
    apply_layout.property_set["post_layout"] = post_layout

    altered_qc = apply_layout.run(circuit_to_dag(qc))
    return dag_to_circuit(altered_qc), apply_layout


def _qiskit_passes(action: Action, device: Target, layout: TranspileLayout | None) -> list[Task]:
    """Build the concrete Qiskit pass list for an action."""
    if action.name == "QiskitO3" and isinstance(action, DeviceDependentAction):
        factory = cast("Callable[[list[str], CouplingMap | None], list[Task]]", action.transpile_pass)
        return factory(
            device.operation_names,
            CouplingMap(device.build_coupling_map()) if layout else None,
        )
    if callable(action.transpile_pass):
        factory = cast("Callable[[Target], list[Task]]", action.transpile_pass)
        return factory(device)
    return cast("list[Task]", action.transpile_pass)


def _postprocess_layout_action(
    action: Action,
    property_set: PropertySet,
    altered_qc: QuantumCircuit,
    layout: TranspileLayout | None,
    input_qubit_count: int,
) -> tuple[QuantumCircuit, TranspileLayout | None]:
    """Update Qiskit's layout metadata after passes that can create or alter layouts."""
    if action.name == "VF2PostLayout":
        assert property_set["VF2PostLayout_stop_reason"] is not None
        post_layout = property_set["post_layout"]
        if post_layout:
            assert layout is not None
            altered_qc, _ = postprocess_vf2postlayout(altered_qc, post_layout, layout)
    elif action.name == "VF2Layout":
        if property_set["VF2Layout_stop_reason"] != VF2LayoutStopReason.SOLUTION_FOUND:
            logger.warning(
                "VF2Layout pass did not find a solution. Reason: %s",
                property_set["VF2Layout_stop_reason"],
            )
        else:
            assert property_set["layout"]
    else:
        assert property_set["layout"]

    if property_set["layout"]:
        return altered_qc, TranspileLayout(
            initial_layout=property_set["layout"],
            input_qubit_mapping=property_set["original_qubit_indices"],
            final_layout=property_set["final_layout"],
            _output_qubit_list=altered_qc.qubits,
            _input_qubit_count=input_qubit_count,
        )
    return altered_qc, layout


def run_qiskit_action(
    action: Action,
    circuit: QuantumCircuit,
    device: Target,
    layout: TranspileLayout | None,
    input_qubit_count: int,
) -> tuple[QuantumCircuit, TranspileLayout | None]:
    """Apply a Qiskit action and return the updated circuit and layout metadata."""
    passes = _qiskit_passes(action, device, layout)
    if action.name == "QiskitO3" and isinstance(action, DeviceDependentAction):
        assert action.do_while is not None
        pm = PassManager([DoWhileController(passes, do_while=action.do_while)])
    else:
        pm = PassManager(passes)

    altered_qc = pm.run(circuit)

    if action.pass_type in {PassType.LAYOUT, PassType.MAPPING, PassType.FINAL_OPT}:
        altered_qc, layout = _postprocess_layout_action(action, pm.property_set, altered_qc, layout, input_qubit_count)
    elif action.pass_type == PassType.ROUTING and layout and pm.property_set["final_layout"] is not None:
        layout.final_layout = pm.property_set["final_layout"]

    if altered_qc.count_ops().get("unitary"):
        altered_qc = altered_qc.decompose(gates_to_decompose="unitary")

    return altered_qc, layout
