# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""This modules provides the actions that can be used in the reinforcement learning environment."""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from bqskit import MachineModel
from bqskit import compile as bqskit_compile
from pytket.architecture import Architecture
from pytket.passes import (
    CliffordSimp,
    FullPeepholeOptimise,
    PeepholeOptimise2Q,
    RemoveRedundancies,
    RoutingPass,
)
from qiskit.circuit import StandardEquivalenceLibrary
from qiskit.circuit.library import XGate, ZGate
from qiskit.passmanager import ConditionalController
from qiskit.transpiler import (
    CouplingMap,
)
from qiskit.transpiler.passes import (
    ApplyLayout,
    BasicSwap,
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
    TrivialLayout,
    UnitarySynthesis,
    VF2Layout,
    VF2PostLayout,
)
from qiskit.transpiler.passes.layout.vf2_layout import VF2LayoutStopReason
from qiskit.transpiler.preset_passmanagers import common

from mqt.predictor.rl.parsing import PreProcessTKETRoutingAfterQiskitLayout, get_bqskit_native_gates

if TYPE_CHECKING:
    from collections.abc import Callable


class CompilationOrigin(str, Enum):
    """Enumeration of the origin of the compilation action."""

    QISKIT = "qiskit"
    TKET = "tket"
    BQSKIT = "bqskit"
    GENERAL = "general"


class PassType(str, Enum):
    """Enumeration of the type of compilation pass."""

    OPT = "optimization"
    SYNTHESIS = "synthesis"
    MAPPING = "mapping"
    LAYOUT = "layout"
    ROUTING = "routing"
    FINAL_OPT = "final_optimization"
    TERMINATE = "terminate"


@dataclass
class Action:
    """Base class for all actions in the reinforcement learning environment."""

    name: str
    origin: CompilationOrigin
    pass_type: PassType
    transpile_pass: Any  # Either a list or a callable depending on subclass


@dataclass
class StaticPassAction(Action):
    """Action that represents a static compilation pass that can be applied directly."""

    def __post_init__(self) -> None:
        """Ensures that the transpile_pass is a list."""
        if not isinstance(self.transpile_pass, list):
            self.transpile_pass = [self.transpile_pass]


@dataclass
class DeviceSpecificAction(Action):
    """Action that represents a device-specific compilation pass that can be applied to a specific device."""

    do_while: Callable[[dict[str, Any]], bool] | None = None


# Registry of actions
_ACTIONS: dict[str, Action] = {}


def register_action(action: Action) -> Action:
    """Registers a new action in the global actions registry."""
    if action.name in _ACTIONS:
        msg = f"Action with name {action.name} already registered."
        raise ValueError(msg)
    _ACTIONS[action.name] = action
    return action


# Static optimization passes
register_action(
    StaticPassAction(
        "Optimize1qGatesDecomposition",
        CompilationOrigin.QISKIT,
        PassType.OPT,
        Optimize1qGatesDecomposition(),
    )
)

register_action(
    StaticPassAction(
        "CommutativeCancellation",
        CompilationOrigin.QISKIT,
        PassType.OPT,
        CommutativeCancellation(),
    )
)

register_action(
    StaticPassAction(
        "CommutativeInverseCancellation",
        CompilationOrigin.QISKIT,
        PassType.OPT,
        CommutativeInverseCancellation(),
    )
)

register_action(
    StaticPassAction(
        "RemoveDiagonalGatesBeforeMeasure",
        CompilationOrigin.QISKIT,
        PassType.OPT,
        RemoveDiagonalGatesBeforeMeasure(),
    )
)

register_action(
    StaticPassAction(
        "InverseCancellation",
        CompilationOrigin.QISKIT,
        PassType.OPT,
        InverseCancellation([XGate(), ZGate()]),
    )
)

register_action(
    StaticPassAction(
        "OptimizeCliffords",
        CompilationOrigin.QISKIT,
        PassType.OPT,
        OptimizeCliffords(),
    )
)

register_action(
    StaticPassAction(
        "Opt2qBlocks",
        CompilationOrigin.QISKIT,
        PassType.OPT,
        [Collect2qBlocks(), ConsolidateBlocks()],
    )
)

register_action(
    StaticPassAction(
        "PeepholeOptimise2Q",
        CompilationOrigin.TKET,
        PassType.OPT,
        PeepholeOptimise2Q(),
    )
)

register_action(
    StaticPassAction(
        "CliffordSimp",
        CompilationOrigin.TKET,
        PassType.OPT,
        CliffordSimp(),
    )
)

register_action(
    StaticPassAction(
        "FullPeepholeOptimiseCX",
        CompilationOrigin.TKET,
        PassType.OPT,
        FullPeepholeOptimise(),
    )
)

register_action(
    StaticPassAction(
        "RemoveRedundancies",
        CompilationOrigin.TKET,
        PassType.OPT,
        RemoveRedundancies(),
    )
)

# Device-specific optimization passes
register_action(
    DeviceSpecificAction(
        "QiskitO3",
        CompilationOrigin.QISKIT,
        PassType.OPT,
        transpile_pass=lambda native_gate, coupling_map: [
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
        do_while=lambda property_set: not property_set["optimization_loop_minimum_point"],
    )
)

register_action(
    DeviceSpecificAction(
        "BQSKitO2",
        CompilationOrigin.BQSKIT,
        PassType.OPT,
        transpile_pass=lambda circuit: bqskit_compile(
            circuit,
            optimization_level=1 if os.getenv("GITHUB_ACTIONS") == "true" else 2,
            synthesis_epsilon=1e-1 if os.getenv("GITHUB_ACTIONS") == "true" else 1e-8,
            max_synthesis_size=2 if os.getenv("GITHUB_ACTIONS") == "true" else 3,
            seed=10,
            num_workers=1 if os.getenv("GITHUB_ACTIONS") == "true" else -1,
        ),
    )
)

# Final optimization passes
register_action(
    DeviceSpecificAction(
        "VF2PostLayout",
        CompilationOrigin.QISKIT,
        PassType.FINAL_OPT,
        transpile_pass=lambda device: VF2PostLayout(target=device),
    )
)

# Layout passes
register_action(
    DeviceSpecificAction(
        "TrivialLayout",
        CompilationOrigin.QISKIT,
        PassType.LAYOUT,
        transpile_pass=lambda device: [
            TrivialLayout(coupling_map=CouplingMap(device.build_coupling_map())),
            FullAncillaAllocation(coupling_map=CouplingMap(device.build_coupling_map())),
            EnlargeWithAncilla(),
            ApplyLayout(),
        ],
    )
)

register_action(
    DeviceSpecificAction(
        "DenseLayout",
        CompilationOrigin.QISKIT,
        PassType.LAYOUT,
        transpile_pass=lambda device: [
            DenseLayout(coupling_map=CouplingMap(device.build_coupling_map())),
            FullAncillaAllocation(coupling_map=CouplingMap(device.build_coupling_map())),
            EnlargeWithAncilla(),
            ApplyLayout(),
        ],
    )
)

register_action(
    DeviceSpecificAction(
        "VF2Layout",
        CompilationOrigin.QISKIT,
        PassType.LAYOUT,
        transpile_pass=lambda device: [
            VF2Layout(target=device),
            ConditionalController(
                [
                    FullAncillaAllocation(coupling_map=CouplingMap(device.build_coupling_map())),
                    EnlargeWithAncilla(),
                    ApplyLayout(),
                ],
                condition=lambda property_set: property_set["VF2Layout_stop_reason"]
                == VF2LayoutStopReason.SOLUTION_FOUND,
            ),
        ],
    )
)

# Routing passes
register_action(
    DeviceSpecificAction(
        "BasicSwap",
        CompilationOrigin.QISKIT,
        PassType.ROUTING,
        transpile_pass=lambda device: [BasicSwap(coupling_map=CouplingMap(device.build_coupling_map()))],
    )
)

register_action(
    DeviceSpecificAction(
        "RoutingPass",
        CompilationOrigin.TKET,
        PassType.ROUTING,
        transpile_pass=lambda device: [
            PreProcessTKETRoutingAfterQiskitLayout(),
            RoutingPass(Architecture(list(device.build_coupling_map()))),
        ],
    )
)

# Mapping passes
register_action(
    DeviceSpecificAction(
        "SabreMapping",
        CompilationOrigin.QISKIT,
        PassType.MAPPING,
        transpile_pass=lambda device: [
            SabreLayout(coupling_map=CouplingMap(device.build_coupling_map()), skip_routing=False)
        ],
    )
)

register_action(
    DeviceSpecificAction(
        "BQSKitMapping",
        CompilationOrigin.BQSKIT,
        PassType.MAPPING,
        transpile_pass=lambda device: lambda bqskit_circuit: bqskit_compile(
            bqskit_circuit,
            model=MachineModel(
                num_qudits=device.num_qubits,
                gate_set=get_bqskit_native_gates(device),
                coupling_graph=[(elem[0], elem[1]) for elem in device.build_coupling_map()],
            ),
            with_mapping=True,
            optimization_level=1 if os.getenv("GITHUB_ACTIONS") == "true" else 2,
            synthesis_epsilon=1e-1 if os.getenv("GITHUB_ACTIONS") == "true" else 1e-8,
            max_synthesis_size=2 if os.getenv("GITHUB_ACTIONS") == "true" else 3,
            seed=10,
            num_workers=1 if os.getenv("GITHUB_ACTIONS") == "true" else -1,
        ),
    )
)

# Synthesis passes
register_action(
    DeviceSpecificAction(
        "BasisTranslator",
        CompilationOrigin.QISKIT,
        PassType.SYNTHESIS,
        transpile_pass=lambda device: [
            BasisTranslator(StandardEquivalenceLibrary, target_basis=device.operation_names)
        ],
    )
)

register_action(
    DeviceSpecificAction(
        "BQSKitSynthesis",
        CompilationOrigin.BQSKIT,
        PassType.SYNTHESIS,
        transpile_pass=lambda device: lambda bqskit_circuit: bqskit_compile(
            bqskit_circuit,
            model=MachineModel(bqskit_circuit.num_qudits, gate_set=get_bqskit_native_gates(device)),
            optimization_level=1 if os.getenv("GITHUB_ACTIONS") == "true" else 2,
            synthesis_epsilon=1e-1 if os.getenv("GITHUB_ACTIONS") == "true" else 1e-8,
            max_synthesis_size=2 if os.getenv("GITHUB_ACTIONS") == "true" else 3,
            seed=10,
            num_workers=1 if os.getenv("GITHUB_ACTIONS") == "true" else -1,
        ),
    )
)

# Terminate action (no passes)
register_action(
    StaticPassAction(
        "terminate",
        CompilationOrigin.GENERAL,
        PassType.TERMINATE,
        transpile_pass=[],
    )
)


def get_actions_by_pass_type() -> dict[PassType, list[Action]]:
    """Returns a dictionary mapping each PassType to a list of Actions of that type."""
    result: dict[PassType, list[Action]] = defaultdict(list)
    for action in _ACTIONS.values():
        result[action.pass_type].append(action)
    return dict(result)
