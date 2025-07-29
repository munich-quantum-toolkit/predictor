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
from typing import TYPE_CHECKING

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
from qiskit.passmanager import ConditionalController
from qiskit.transpiler import (
    CouplingMap,
)
from qiskit.transpiler.passes import (
    ApplyLayout,
    BasicSwap,
    BasisTranslator,
    Collect2qBlocks,
    CollectCliffords,
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
    SabreSwap,
    Size,
    TrivialLayout,
    UnitarySynthesis,
    VF2Layout,
    VF2PostLayout,
)
from qiskit.transpiler.passes.layout.vf2_layout import VF2LayoutStopReason
from qiskit.transpiler.preset_passmanagers import common

from mqt.predictor.rl.helper import SafeAIRouting, get_openqasm_gates
from mqt.predictor.rl.parsing import (
    PreProcessTKETRoutingAfterQiskitLayout,
    get_bqskit_native_gates,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from bqskit import Circuit
    from pytket._tket.passes import BasePass as tket_BasePass
    from qiskit.transpiler.basepasses import BasePass as qiskit_BasePass


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
    transpile_pass: (
        list[qiskit_BasePass | tket_BasePass]
        | Callable[..., list[qiskit_BasePass | tket_BasePass]]
        | Callable[
            ...,
            Callable[..., tuple[Any, ...] | Circuit],
        ]
    )
    stochastic: bool = False


@dataclass
class DeviceIndependentAction(Action):
    """Action that represents a static compilation pass that can be applied directly."""


@dataclass
class DeviceDependentAction(Action):
    """Action that represents a device-specific compilation pass that can be applied to a specific device."""

    transpile_pass: (
        Callable[..., list[qiskit_BasePass | tket_BasePass]]
        | Callable[
            ...,
            Callable[..., tuple[Any, ...] | Circuit],
        ]
    )
    do_while: Callable[[dict[str, Circuit]], bool] | None = None


# Registry of actions
_ACTIONS: dict[str, Action] = {}


def register_action(action: Action) -> Action:
    """Registers a new action in the global actions registry.

    Raises:
        ValueError: If an action with the same name is already registered.
    """
    if action.name in _ACTIONS:
        msg = f"Action with name {action.name} already registered."
        raise ValueError(msg)
    _ACTIONS[action.name] = action
    return action


def remove_action(name: str) -> None:
    """Removes an action from the global actions registry by name.

    Raises:
        ValueError: If no action with the given name is registered.
    """
    if name not in _ACTIONS:
        msg = f"No action with name {name} is registered."
        raise KeyError(msg)
    del _ACTIONS[name]


register_action(
    DeviceIndependentAction(
        "Optimize1qGatesDecomposition",
        CompilationOrigin.QISKIT,
        PassType.OPT,
        [Optimize1qGatesDecomposition(basis=get_openqasm_gates())],
    )
)

register_action(
    DeviceIndependentAction(
        "CommutativeCancellation",
        CompilationOrigin.QISKIT,
        PassType.OPT,
        [CommutativeCancellation()],
    )
)

register_action(
    DeviceIndependentAction(
        "CommutativeInverseCancellation",
        CompilationOrigin.QISKIT,
        PassType.OPT,
        [CommutativeInverseCancellation()],
    )
)

register_action(
    DeviceIndependentAction(
        "RemoveDiagonalGatesBeforeMeasure",
        CompilationOrigin.QISKIT,
        PassType.OPT,
        [RemoveDiagonalGatesBeforeMeasure()],
    )
)

register_action(
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
    )
)

register_action(
    DeviceIndependentAction(
        "OptimizeCliffords",
        CompilationOrigin.QISKIT,
        PassType.OPT,
        [CollectCliffords(), OptimizeCliffords()],
    )
)

register_action(
    DeviceDependentAction(
        "Opt2qBlocks",
        CompilationOrigin.QISKIT,
        PassType.OPT,
        transpile_pass=lambda device: [
            Collect2qBlocks(),
            ConsolidateBlocks(basis_gates=device.operation_names),
            UnitarySynthesis(basis_gates=device.operation_names, coupling_map=device.build_coupling_map()),
        ],
    )
)

register_action(
    DeviceIndependentAction(
        "PeepholeOptimise2Q",
        CompilationOrigin.TKET,
        PassType.OPT,
        [PeepholeOptimise2Q()],
    )
)

register_action(
    DeviceIndependentAction(
        "CliffordSimp",
        CompilationOrigin.TKET,
        PassType.OPT,
        [CliffordSimp()],
    )
)

register_action(
    DeviceIndependentAction(
        "FullPeepholeOptimiseCX",
        CompilationOrigin.TKET,
        PassType.OPT,
        [FullPeepholeOptimise()],
    )
)

register_action(
    DeviceIndependentAction(
        "RemoveRedundancies",
        CompilationOrigin.TKET,
        PassType.OPT,
        [RemoveRedundancies()],
    )
)

register_action(
    DeviceDependentAction(
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
    DeviceDependentAction(
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

register_action(
    DeviceDependentAction(
        "VF2PostLayout",
        CompilationOrigin.QISKIT,
        PassType.FINAL_OPT,
        transpile_pass=lambda device: VF2PostLayout(target=device),
    )
)

register_action(
    DeviceDependentAction(
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
    DeviceDependentAction(
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

register_action(
    DeviceDependentAction(
        "SabreMapping",
        CompilationOrigin.QISKIT,
        PassType.MAPPING,
        stochastic=True,
        # Qiskit O3 by default uses (max_iterations, layout_trials, swap_trials) = (4, 20, 20)
        transpile_pass=lambda device, max_iteration=(20, 20): [
            SabreLayout(
                coupling_map=CouplingMap(device.build_coupling_map()),
                skip_routing=False,
                layout_trials=max_iteration[0],
                swap_trials=max_iteration[1],
                max_iterations=4,
                seed=None,
            ),
        ],
    )
)

register_action(
    DeviceDependentAction(
        "SabreLayout+BasicSwap",
        CompilationOrigin.QISKIT,
        PassType.MAPPING,
        stochastic=True,
        transpile_pass=lambda device, max_iteration=(20, 20): [
            SabreLayout(
                coupling_map=CouplingMap(device.build_coupling_map()),
                skip_routing=True,
                layout_trials=max_iteration[0],
                swap_trials=max_iteration[1],
                max_iterations=4,
                seed=None,
            ),
            FullAncillaAllocation(coupling_map=CouplingMap(device.build_coupling_map())),
            EnlargeWithAncilla(),
            ApplyLayout(),
            BasicSwap(coupling_map=CouplingMap(device.build_coupling_map())),
        ],
    )
)

register_action(
    DeviceDependentAction(
        "SabreLayout+AIRouting",
        CompilationOrigin.QISKIT,
        PassType.MAPPING,
        stochastic=True,
        transpile_pass=lambda device, max_iteration=(20, 20): [
            SabreLayout(
                coupling_map=CouplingMap(device.build_coupling_map()),
                skip_routing=True,
                layout_trials=max_iteration[0],
                swap_trials=max_iteration[1],
                max_iterations=4,
                seed=None,
            ),
            FullAncillaAllocation(coupling_map=CouplingMap(device.build_coupling_map())),
            EnlargeWithAncilla(),
            ApplyLayout(),
            SafeAIRouting(
                coupling_map=device.build_coupling_map(), optimization_level=3, layout_mode="optimize", local_mode=True
            ),
        ],
    )
)

register_action(
    DeviceDependentAction(
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

register_action(
    DeviceDependentAction(
        name="DenseLayout+BasicSwap",
        origin=CompilationOrigin.QISKIT,
        pass_type=PassType.MAPPING,
        transpile_pass=lambda device: [
            DenseLayout(coupling_map=CouplingMap(device.build_coupling_map())),
            FullAncillaAllocation(coupling_map=CouplingMap(device.build_coupling_map())),
            EnlargeWithAncilla(),
            ApplyLayout(),
            BasicSwap(coupling_map=CouplingMap(device.build_coupling_map())),
        ],
    )
)

register_action(
    DeviceDependentAction(
        name="DenseLayout+SabreSwap",
        origin=CompilationOrigin.QISKIT,
        pass_type=PassType.MAPPING,
        stochastic=True,
        transpile_pass=lambda device, max_iteration=(20, 20): [
            DenseLayout(coupling_map=CouplingMap(device.build_coupling_map())),
            FullAncillaAllocation(coupling_map=CouplingMap(device.build_coupling_map())),
            EnlargeWithAncilla(),
            ApplyLayout(),
            SabreSwap(
                coupling_map=CouplingMap(device.build_coupling_map()),
                heuristic="decay",
                trials=max_iteration[1],
                seed=None,
            ),
        ],
    )
)

register_action(
    DeviceDependentAction(
        name="DenseLayout+AIRouting",
        origin=CompilationOrigin.QISKIT,
        pass_type=PassType.MAPPING,
        stochastic=True,
        transpile_pass=lambda device: [
            DenseLayout(coupling_map=CouplingMap(device.build_coupling_map())),
            FullAncillaAllocation(coupling_map=CouplingMap(device.build_coupling_map())),
            EnlargeWithAncilla(),
            ApplyLayout(),
            SafeAIRouting(
                coupling_map=device.build_coupling_map(), optimization_level=3, layout_mode="optimize", local_mode=True
            ),
        ],
    )
)

register_action(
    DeviceDependentAction(
        name="VF2Layout+BasicSwap",
        origin=CompilationOrigin.QISKIT,
        pass_type=PassType.MAPPING,
        transpile_pass=lambda device: [
            VF2Layout(
                coupling_map=CouplingMap(device.build_coupling_map()),
                target=device,
            ),
            ConditionalController(
                [
                    FullAncillaAllocation(coupling_map=CouplingMap(device.build_coupling_map())),
                    EnlargeWithAncilla(),
                    ApplyLayout(),
                ],
                condition=lambda property_set: property_set["VF2Layout_stop_reason"]
                == VF2LayoutStopReason.SOLUTION_FOUND,
            ),
            ConditionalController(
                [
                    TrivialLayout(coupling_map=CouplingMap(device.build_coupling_map())),
                    FullAncillaAllocation(coupling_map=CouplingMap(device.build_coupling_map())),
                    EnlargeWithAncilla(),
                    ApplyLayout(),
                ],
                # Run if VF2Layout did not find a solution
                condition=lambda property_set: property_set["VF2Layout_stop_reason"]
                != VF2LayoutStopReason.SOLUTION_FOUND,
            ),
            BasicSwap(coupling_map=CouplingMap(device.build_coupling_map())),
        ],
    )
)

register_action(
    DeviceDependentAction(
        name="VF2Layout+SabreSwap",
        origin=CompilationOrigin.QISKIT,
        pass_type=PassType.MAPPING,
        stochastic=True,
        transpile_pass=lambda device, max_iteration=(20, 20): [
            VF2Layout(
                coupling_map=CouplingMap(device.build_coupling_map()),
                target=device,
            ),
            ConditionalController(
                [
                    FullAncillaAllocation(coupling_map=CouplingMap(device.build_coupling_map())),
                    EnlargeWithAncilla(),
                    ApplyLayout(),
                ],
                condition=lambda property_set: property_set["VF2Layout_stop_reason"]
                == VF2LayoutStopReason.SOLUTION_FOUND,
            ),
            ConditionalController(
                [
                    TrivialLayout(coupling_map=CouplingMap(device.build_coupling_map())),
                    FullAncillaAllocation(coupling_map=CouplingMap(device.build_coupling_map())),
                    EnlargeWithAncilla(),
                    ApplyLayout(),
                ],
                # Run if VF2Layout did not find a solution
                condition=lambda property_set: property_set["VF2Layout_stop_reason"]
                != VF2LayoutStopReason.SOLUTION_FOUND,
            ),
            SabreSwap(
                coupling_map=CouplingMap(device.build_coupling_map()),
                heuristic="decay",
                trials=max_iteration[1],
                seed=None,
            ),
        ],
    )
)

register_action(
    DeviceDependentAction(
        name="VF2Layout+AIRouting",
        origin=CompilationOrigin.QISKIT,
        pass_type=PassType.MAPPING,
        stochastic=True,
        transpile_pass=lambda device: [
            VF2Layout(
                coupling_map=CouplingMap(device.build_coupling_map()),
                target=device,
            ),
            ConditionalController(
                [
                    FullAncillaAllocation(coupling_map=CouplingMap(device.build_coupling_map())),
                    EnlargeWithAncilla(),
                    ApplyLayout(),
                ],
                condition=lambda property_set: property_set["VF2Layout_stop_reason"]
                == VF2LayoutStopReason.SOLUTION_FOUND,
            ),
            ConditionalController(
                [
                    TrivialLayout(coupling_map=CouplingMap(device.build_coupling_map())),
                    FullAncillaAllocation(coupling_map=CouplingMap(device.build_coupling_map())),
                    EnlargeWithAncilla(),
                    ApplyLayout(),
                ],
                # Run if VF2Layout did not find a solution
                condition=lambda property_set: property_set["VF2Layout_stop_reason"]
                != VF2LayoutStopReason.SOLUTION_FOUND,
            ),
            SafeAIRouting(
                coupling_map=device.build_coupling_map(), optimization_level=3, layout_mode="optimize", local_mode=True
            ),
        ],
    )
)

register_action(
    DeviceDependentAction(
        name="SabreLayout+AIRouting",
        origin=CompilationOrigin.QISKIT,
        stochastic=True,
        pass_type=PassType.MAPPING,
        transpile_pass=lambda device, max_iteration=(20, 20): [
            SabreLayout(
                coupling_map=CouplingMap(device.build_coupling_map()),
                skip_routing=True,
                layout_trials=max_iteration[0],
                swap_trials=1,
                max_iterations=4,
            ),
            FullAncillaAllocation(coupling_map=CouplingMap(device.build_coupling_map())),
            EnlargeWithAncilla(),
            ApplyLayout(),
            SafeAIRouting(
                coupling_map=device.build_coupling_map(),
                optimization_level=3,
                layout_mode="optimize",
                local_mode=True,
            ),
        ],
    )
)

register_action(
    DeviceDependentAction(
        "BasisTranslator",
        CompilationOrigin.QISKIT,
        PassType.SYNTHESIS,
        transpile_pass=lambda device: [
            BasisTranslator(StandardEquivalenceLibrary, target_basis=device.operation_names)
        ],
    )
)

register_action(
    DeviceDependentAction(
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

register_action(
    DeviceIndependentAction(
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
    return result
