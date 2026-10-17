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
from qiskit.passmanager import ConditionalController, PropertySet
from qiskit.passmanager.flow_controllers import DoWhileController
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import (
    ApplyLayout,
    BasisTranslator,
    Collect2qBlocks,
    CommutativeCancellation,
    CommutativeInverseCancellation,
    ConsolidateBlocks,
    Decompose,
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

from mqt.predictor.rl.actions.base import (
    CompilationOrigin,
    DeferredDeviceAction,
    DeviceIndependentAction,
    PassType,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from qiskit import QuantumCircuit
    from qiskit.passmanager.base_tasks import Task
    from qiskit.transpiler import Target, TranspileLayout

    from mqt.predictor.rl.actions.base import Action

logger = logging.getLogger("mqt-predictor")


def qiskit_optimization_actions() -> list[Action]:
    """Returns the Qiskit optimization actions."""
    return [
        DeviceIndependentAction(
            "Optimize1qGatesDecomposition",
            CompilationOrigin.QISKIT,
            PassType.OPT,
            [Optimize1qGatesDecomposition()],
            preserves_layout=True,
            preserves_routing=True,
            preserves_synthesis=False,
        ),
        DeviceIndependentAction(
            "CommutativeCancellation",
            CompilationOrigin.QISKIT,
            PassType.OPT,
            [CommutativeCancellation()],
            preserves_layout=True,
            preserves_routing=True,
            preserves_synthesis=True,
        ),
        DeviceIndependentAction(
            "CommutativeInverseCancellation",
            CompilationOrigin.QISKIT,
            PassType.OPT,
            [CommutativeInverseCancellation()],
            preserves_layout=True,
            preserves_routing=True,
            preserves_synthesis=True,
        ),
        DeviceIndependentAction(
            "RemoveDiagonalGatesBeforeMeasure",
            CompilationOrigin.QISKIT,
            PassType.OPT,
            [RemoveDiagonalGatesBeforeMeasure()],
            preserves_layout=True,
            preserves_routing=True,
            preserves_synthesis=True,
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
            preserves_layout=True,
            preserves_routing=True,
            preserves_synthesis=True,
        ),
        DeviceIndependentAction(
            "OptimizeCliffords",
            CompilationOrigin.QISKIT,
            PassType.OPT,
            [OptimizeCliffords()],
            preserves_layout=True,
            preserves_routing=False,
            preserves_synthesis=False,
        ),
        DeviceIndependentAction(
            "Opt2qBlocks",
            CompilationOrigin.QISKIT,
            PassType.OPT,
            [Collect2qBlocks(), ConsolidateBlocks(), UnitarySynthesis()],
            preserves_layout=True,
            preserves_routing=True,
            preserves_synthesis=False,
        ),
    ]


def qiskit_o3_action() -> Action:
    """Returns the Qiskit level-3 optimization action."""
    return DeferredDeviceAction(
        "QiskitO3",
        CompilationOrigin.QISKIT,
        PassType.OPT,
        preserves_layout=True,
        preserves_routing=True,
        preserves_synthesis=True,
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
    return DeferredDeviceAction(
        "VF2PostLayout",
        CompilationOrigin.QISKIT,
        PassType.FINAL_OPT,
        transpile_pass=lambda device: [
            VF2PostLayout(target=device),
            ConditionalController(ApplyLayout(), condition=lambda property_set: bool(property_set["post_layout"])),
        ],
    )


def qiskit_layout_actions() -> list[Action]:
    """Returns the Qiskit layout actions."""
    return [
        DeferredDeviceAction(
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
        DeferredDeviceAction(
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
    return DeferredDeviceAction(
        "QiskitSabreMapping",
        CompilationOrigin.QISKIT,
        PassType.MAPPING,
        transpile_pass=lambda device: cast(
            "list[Task]", [SabreLayout(coupling_map=CouplingMap(device.build_coupling_map()), skip_routing=False)]
        ),
    )


def qiskit_synthesis_action() -> Action:
    """Returns the Qiskit synthesis action."""
    return DeferredDeviceAction(
        "BasisTranslator",
        CompilationOrigin.QISKIT,
        PassType.SYNTHESIS,
        transpile_pass=lambda device: cast(
            "list[Task]", [BasisTranslator(StandardEquivalenceLibrary, target_basis=device.operation_names)]
        ),
    )


def run_qiskit_action(
    action: Action,
    circuit: QuantumCircuit,
    device: Target,
    layout: TranspileLayout | None,
) -> tuple[QuantumCircuit, TranspileLayout | None]:
    """Apply a Qiskit action and return the updated circuit and layout metadata."""
    # Build the concrete Qiskit pass list for given action.
    if action.name == "QiskitO3" and isinstance(action, DeferredDeviceAction):
        factory = cast("Callable[[list[str], CouplingMap | None], list[Task]]", action.transpile_pass)
        passes = factory(device.operation_names, CouplingMap(device.build_coupling_map()) if layout else None)
    elif callable(action.transpile_pass):
        factory = cast("Callable[[Target], list[Task]]", action.transpile_pass)
        passes = factory(device)
    else:
        passes = cast("list[Task]", action.transpile_pass)

    if action.name == "QiskitO3" and isinstance(action, DeferredDeviceAction):
        assert action.do_while is not None
        pm = PassManager([DoWhileController(passes, do_while=action.do_while)])
    else:
        pm = PassManager(passes)

    pm.append(Decompose(gates_to_decompose="unitary", apply_synthesis=True))
    property_set = PropertySet()
    if layout is not None:
        layout.write_into_property_set(property_set)
    altered_qc = pm.run(circuit, property_set=property_set)

    if action.name == "VF2Layout" and pm.property_set["VF2Layout_stop_reason"] != VF2LayoutStopReason.SOLUTION_FOUND:
        logger.warning("VF2Layout pass did not find a solution. Reason: %s", pm.property_set["VF2Layout_stop_reason"])

    return altered_qc, altered_qc.layout


def is_qiskit_action_available(action: Action, device: Target) -> bool:
    """Return whether a Qiskit action is available for the current device."""
    # Only allow VF2PostLayout if "ibm" is in the device name # TODO: Why?
    return action.name != "VF2PostLayout" or "ibm" in device.description
