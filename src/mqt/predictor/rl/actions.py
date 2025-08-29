# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""This module provides the actions that can be used in the reinforcement learning environment."""

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
from qiskit import QuantumCircuit
from qiskit.circuit import ClassicalRegister, Instruction, QuantumRegister, Qubit, StandardEquivalenceLibrary
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
from qiskit.transpiler import CouplingMap
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
from qiskit_ibm_transpiler.ai.routing import AIRouting

from mqt.predictor.rl.parsing import (
    PreProcessTKETRoutingAfterQiskitLayout,
    get_bqskit_native_gates,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from bqskit import Circuit
    from pytket._tket.passes import BasePass as tket_BasePass
    from qiskit.dagcircuit import DAGCircuit
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
        | Callable[..., list[Any]]
        | Callable[..., list[qiskit_BasePass | tket_BasePass]]
        | Callable[
            ...,
            Callable[..., tuple[Any, ...] | Circuit],
        ]
    )
    stochastic: bool | None = False
    preserve: bool | None = False


@dataclass
class DeviceIndependentAction(Action):
    """Action that represents a static compilation pass that can be applied directly."""


@dataclass
class DeviceDependentAction(Action):
    """Action that represents a device-specific compilation pass that can be applied to a specific device."""

    transpile_pass: (
        Callable[..., list[Any]]
        | Callable[..., list[qiskit_BasePass | tket_BasePass]]
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


def get_openqasm_gates() -> list[str]:
    """Returns a list of all quantum gates within the openQASM 2.0 standard header.

    According to https://github.com/Qiskit/qiskit-terra/blob/main/qiskit/qasm/libs/qelib1.inc
    Removes generic single qubit gates u1, u2, u3 since they are no meaningful features for RL

    """
    return [
        "cx",
        "id",
        "u",
        "p",
        "x",
        "y",
        "z",
        "h",
        "r",
        "s",
        "sdg",
        "t",
        "tdg",
        "rx",
        "ry",
        "rz",
        "sx",
        "sxdg",
        "cz",
        "cy",
        "swap",
        "ch",
        "ccx",
        "cswap",
        "crx",
        "cry",
        "crz",
        "cu1",
        "cp",
        "csx",
        "cu",
        "rxx",
        "rzz",
        "rccx",
    ]


register_action(
    DeviceIndependentAction(
        "Optimize1qGatesDecomposition",
        CompilationOrigin.QISKIT,
        PassType.OPT,
        preserve=True,
        transpile_pass= lambda device: [Optimize1qGatesDecomposition(basis=device.operation_names)],
    )
)

register_action(
    DeviceIndependentAction(
        "CommutativeCancellation",
        CompilationOrigin.QISKIT,
        PassType.OPT,
        [CommutativeCancellation()],
        preserve=True,
    )
)

register_action(
    DeviceIndependentAction(
        "CommutativeInverseCancellation",
        CompilationOrigin.QISKIT,
        PassType.OPT,
        [CommutativeInverseCancellation()],
        preserve=True,
    )
)

register_action(
    DeviceIndependentAction(
        "RemoveDiagonalGatesBeforeMeasure",
        CompilationOrigin.QISKIT,
        PassType.OPT,
        [RemoveDiagonalGatesBeforeMeasure()],
        preserve=True,
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
        preserve=True,
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
        transpile_pass=lambda native_gate, coupling_map: [
            Collect2qBlocks(),
            ConsolidateBlocks(basis_gates=native_gate),
            UnitarySynthesis(basis_gates=native_gate, coupling_map=coupling_map),
        ],
        preserve=True,
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

# register_action(
#     DeviceDependentAction(
#         "QiskitO3",
#         CompilationOrigin.QISKIT,
#         PassType.OPT,
#         transpile_pass=lambda native_gate, coupling_map: [
#             Collect2qBlocks(),
#             ConsolidateBlocks(basis_gates=native_gate),
#             UnitarySynthesis(basis_gates=native_gate, coupling_map=coupling_map),
#             Optimize1qGatesDecomposition(basis=native_gate),
#             CommutativeCancellation(basis_gates=native_gate),
#             GatesInBasis(native_gate),
#             ConditionalController(
#                 common.generate_translation_passmanager(
#                     target=None, basis_gates=native_gate, coupling_map=coupling_map
#                 ).to_flow_controller(),
#                 condition=lambda property_set: not property_set["all_gates_in_basis"],
#             ),
#             Depth(recurse=True),
#             FixedPoint("depth"),
#             Size(recurse=True),
#             FixedPoint("size"),
#             MinimumPoint(["depth", "size"], "optimization_loop"),
#         ],
#         do_while=lambda property_set: not property_set["optimization_loop_minimum_point"],
#     )
# )

# register_action(
#     DeviceDependentAction(
#         "BQSKitO2",
#         CompilationOrigin.BQSKIT,
#         PassType.OPT,
#         transpile_pass=lambda circuit: bqskit_compile(
#             circuit,
#             optimization_level=1 if os.getenv("GITHUB_ACTIONS") == "true" else 2,
#             synthesis_epsilon=1e-1 if os.getenv("GITHUB_ACTIONS") == "true" else 1e-8,
#             max_synthesis_size=2 if os.getenv("GITHUB_ACTIONS") == "true" else 3,
#             seed=10,
#             num_workers=1 if os.getenv("GITHUB_ACTIONS") == "true" else -1,
#         ),
#     )
# )

register_action(
    DeviceDependentAction(
        "VF2PostLayout",
        CompilationOrigin.QISKIT,
        PassType.FINAL_OPT,
        transpile_pass=lambda device: VF2PostLayout(target=device, call_limit=30000000, max_trials=250000),
    )
)

register_action(
    DeviceDependentAction(
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
            VF2Layout(target=device, call_limit=30000000, max_trials=250000),
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

# register_action(
#     DeviceDependentAction(
#         "SabreLayout",
#         CompilationOrigin.QISKIT,
#         PassType.LAYOUT,
#         transpile_pass=lambda device: [
#             SabreLayout(
#                 coupling_map=CouplingMap(device.build_coupling_map()), 
#                 skip_routing=True,
#             ),
#             FullAncillaAllocation(coupling_map=CouplingMap(device.build_coupling_map())),
#             EnlargeWithAncilla(),
#             ApplyLayout(),
#         ],
#     )
# )

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
        "BasicSwap",
        CompilationOrigin.QISKIT,
        PassType.ROUTING,
        transpile_pass=lambda device: [BasicSwap(coupling_map=CouplingMap(device.build_coupling_map()))],
    )
)
# register_action(
#     DeviceDependentAction(
#         "SabreSwap",
#         CompilationOrigin.QISKIT,
#         PassType.ROUTING,
#         stochastic=True,
#         transpile_pass=lambda device: [
#             SabreSwap(coupling_map=CouplingMap(device.build_coupling_map()), heuristic="decay")
#         ],
#     )
# )

# register_action(
#     DeviceDependentAction(
#         "AIRouting",
#         CompilationOrigin.QISKIT,
#         PassType.ROUTING,
#         stochastic=True,
#         transpile_pass=lambda device: [
#                 SafeAIRouting(
#                     coupling_map=device.build_coupling_map(),
#                     optimization_level=3,
#                     layout_mode="improve", 
#                     local_mode=True
#                 )
#             ],
#     )
# )

register_action(
    DeviceDependentAction(
        "SabreMapping",
        CompilationOrigin.QISKIT,
        PassType.MAPPING,
        stochastic=True,
        # Qiskit O3 by default uses (max_iterations, layout_trials, swap_trials) = (4, 20, 20)
        transpile_pass=lambda device: [
            SabreLayout(
                coupling_map=CouplingMap(device.build_coupling_map()),
                skip_routing=False,
                seed=None,
            ),
        ],
    )
)

# register_action(
#     DeviceDependentAction(
#         "AIRouting_opt", 
#         CompilationOrigin.QISKIT,
#         PassType.MAPPING,
#         stochastic=True,
#         transpile_pass=lambda device: [
#             ### Requires a initial layout, but "optimize" mode overwrites it
#             TrivialLayout(coupling_map=CouplingMap(device.build_coupling_map())),
#             FullAncillaAllocation(coupling_map=CouplingMap(device.build_coupling_map())),
#             EnlargeWithAncilla(),
#             ApplyLayout(),
#             SafeAIRouting(coupling_map=device.build_coupling_map(), optimization_level=3, layout_mode="optimize"),
#         ],
#     )
# )

# register_action(
#     DeviceDependentAction(
#         "BQSKitMapping",
#         CompilationOrigin.BQSKIT,
#         PassType.MAPPING,
#         transpile_pass=lambda device: lambda bqskit_circuit: bqskit_compile(
#             bqskit_circuit,
#             model=MachineModel(
#                 num_qudits=device.num_qubits,
#                 gate_set=get_bqskit_native_gates(device),
#                 coupling_graph=[(elem[0], elem[1]) for elem in device.build_coupling_map()],
#             ),
#             with_mapping=True,
#             optimization_level=1 if os.getenv("GITHUB_ACTIONS") == "true" else 2,
#             synthesis_epsilon=1e-1 if os.getenv("GITHUB_ACTIONS") == "true" else 1e-8,
#             max_synthesis_size=2 if os.getenv("GITHUB_ACTIONS") == "true" else 3,
#             seed=10,
#             num_workers=1 if os.getenv("GITHUB_ACTIONS") == "true" else -1,
#         ),
#     )
# )

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

# register_action(
#     DeviceDependentAction(
#         "BQSKitSynthesis",
#         CompilationOrigin.BQSKIT,
#         PassType.SYNTHESIS,
#         transpile_pass=lambda device: lambda bqskit_circuit: bqskit_compile(
#             bqskit_circuit,
#             model=MachineModel(bqskit_circuit.num_qudits, gate_set=get_bqskit_native_gates(device)),
#             optimization_level=1 if os.getenv("GITHUB_ACTIONS") == "true" else 2,
#             synthesis_epsilon=1e-1 if os.getenv("GITHUB_ACTIONS") == "true" else 1e-8,
#             max_synthesis_size=2 if os.getenv("GITHUB_ACTIONS") == "true" else 3,
#             seed=10,
#             num_workers=1 if os.getenv("GITHUB_ACTIONS") == "true" else -1,
#         ),
#     )
# )

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


def extract_cregs_and_measurements(
    qc: QuantumCircuit,
) -> tuple[list[ClassicalRegister], list[tuple[Instruction, list[Any], list[Any]]]]:
    """Extracts classical registers and measurement operations from a quantum circuit.

    Args:
        qc: The input QuantumCircuit.

    Returns:
        A tuple containing a list of classical registers and a list of measurement operations.
    """
    cregs = [ClassicalRegister(cr.size, name=cr.name) for cr in qc.cregs]
    measurements = [(item.operation, item.qubits, item.clbits) for item in qc.data if item.operation.name == "measure"]
    return cregs, measurements


def remove_cregs(qc: QuantumCircuit) -> QuantumCircuit:
    """Removes classical registers and measurement operations from the circuit.

    Args:
        qc: The input QuantumCircuit.

    Returns:
        A new QuantumCircuit with only quantum operations (no cregs or measurements).
    """
    qregs = [QuantumRegister(qr.size, name=qr.name) for qr in qc.qregs]
    new_qc = QuantumCircuit(*qregs)
    old_to_new = {}
    for orig_qr, new_qr in zip(qc.qregs, new_qc.qregs, strict=False):
        for idx in range(orig_qr.size):
            old_to_new[orig_qr[idx]] = new_qr[idx]
    for item in qc.data:
        instr = item.operation
        qargs = [old_to_new[q] for q in item.qubits]
        if instr.name not in ("measure", "barrier"):
            new_qc.append(instr, qargs)
    return new_qc


def add_cregs_and_measurements(
    qc: QuantumCircuit,
    cregs: list[ClassicalRegister],
    measurements: list[tuple[Instruction, list[Any], list[Any]]],
    qubit_map: dict[Qubit, Qubit] | None = None,
) -> QuantumCircuit:
    """Adds classical registers and measurement operations back to the quantum circuit.

    Args:
        qc: The quantum circuit to which cregs and measurements are added.
        cregs: List of ClassicalRegister to add.
        measurements: List of measurement instructions as tuples (Instruction, qubits, clbits).
        qubit_map: Optional dictionary mapping original qubits to new qubits.

    Returns:
        The modified QuantumCircuit with cregs and measurements added.
    """
    for cr in cregs:
        qc.add_register(cr)
    for instr, qargs, cargs in measurements:
        new_qargs = [qubit_map[q] for q in qargs] if qubit_map else qargs
        qc.append(instr, new_qargs, cargs)
    return qc


class SafeAIRouting(AIRouting):  # type: ignore[misc]
    """Custom AIRouting wrapper that removes classical registers before routing.

    This prevents failures in AIRouting when classical bits are present by
    temporarily removing classical registers and measurements and restoring
    them after routing is completed.
    """

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the routing pass on a DAGCircuit."""
        # 1. Convert input dag to circuit
        qc_orig = dag_to_circuit(dag)

        # 2. Extract classical registers and measurement instructions
        cregs, measurements = extract_cregs_and_measurements(qc_orig)

        # 3. Remove cregs and measurements
        qc_noclassical = remove_cregs(qc_orig)

        # 4. Convert back to dag and run routing (AIRouting)
        dag_noclassical = circuit_to_dag(qc_noclassical)
        dag_routed = super().run(dag_noclassical)

        # 5. Convert routed dag to circuit for restoration
        qc_routed = dag_to_circuit(dag_routed)

        # 6. Build mapping from original qubits to qubits in routed circuit
        final_layout = getattr(self, "property_set", {}).get("final_layout", None)
        if final_layout is None and hasattr(dag_routed, "property_set"):
            final_layout = dag_routed.property_set.get("final_layout", None)

        assert final_layout is not None, "final_layout is None — cannot map virtual qubits"
        qubit_map = {}
        for virt in qc_orig.qubits:
            try:
                phys = final_layout[virt]
            except KeyError as err:
                msg = f"Virtual qubit {virt} not found in final layout!"
                raise RuntimeError(msg) from err
            if isinstance(phys, int):
                try:
                    qubit_map[virt] = qc_routed.qubits[phys]
                except IndexError as err:
                    msg = f"Physical index {phys} is out of range in routed circuit!"
                    raise RuntimeError(msg) from err
            else:
                try:
                    idx = qc_routed.qubits.index(phys)
                except ValueError as err:
                    msg = f"Physical qubit {phys} not found in output circuit!"
                    raise RuntimeError(msg) from err
                qubit_map[virt] = qc_routed.qubits[idx]
                # 7. Restore classical registers and measurement instructions
        qc_final = add_cregs_and_measurements(qc_routed, cregs, measurements, qubit_map)
        # 8. Return as dag
        return circuit_to_dag(qc_final)
