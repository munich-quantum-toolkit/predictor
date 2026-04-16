# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""This module provides the actions that can be used in the reinforcement learning environment."""

from __future__ import annotations

import os
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, TypedDict, cast

import numpy as np
from bqskit import MachineModel
from bqskit.compiler import Compiler, Workflow
from bqskit.compiler.compile import (
    build_multi_qudit_retarget_workflow,
    build_partitioning_workflow,
    build_single_qudit_retarget_workflow,
    get_instantiate_options,
)
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
from pytket.architecture import Architecture
from pytket.passes import (
    CliffordSimp,
    FullPeepholeOptimise,
    KAKDecomposition,
    PeepholeOptimise2Q,
    RemoveRedundancies,
    RoutingPass,
)
from pytket.placement import GraphPlacement, NoiseAwarePlacement
from qiskit import QuantumCircuit
from qiskit.circuit import ClassicalRegister, StandardEquivalenceLibrary
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
    BasisTranslator,
    Collect2qBlocks,
    CollectCliffords,
    CommutativeCancellation,
    CommutativeInverseCancellation,
    ConsolidateBlocks,
    DenseLayout,
    ElidePermutations,
    EnlargeWithAncilla,
    FullAncillaAllocation,
    InverseCancellation,
    Optimize1qGatesDecomposition,
    OptimizeCliffords,
    RemoveDiagonalGatesBeforeMeasure,
    SabreLayout,
    SabreSwap,
    UnitarySynthesis,
    VF2Layout,
    VF2PostLayout,
)
from qiskit.transpiler.passes.layout.vf2_layout import VF2LayoutStopReason

from mqt.predictor.rl.parsing import (
    PreProcessTKETRoutingAfterQiskitLayout,
    get_bqskit_native_gates,
)

IS_WIN_PY313 = sys.platform == "win32" and sys.version_info[:2] == (3, 13)
if not IS_WIN_PY313:
    # qiskit-ibm-transpiler currently emits import-time warnings
    # these can not be ignored by the filterwarnings in pyproject.toml
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"invalid escape sequence '\\w'",
            category=DeprecationWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"invalid escape sequence '\\w'",
            category=SyntaxWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r'"is" with (?:a literal|\'str\' literal)',
            category=SyntaxWarning,
        )
        from qiskit_ibm_transpiler.ai.routing import AIRouting


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Protocol, TypeAlias

    from bqskit import Circuit
    from bqskit.compiler.basepass import BasePass as bqskit_BasePass
    from bqskit.compiler.passdata import PassData
    from bqskit.compiler.workflow import WorkflowLike
    from pytket._tket.passes import BasePass as tket_BasePass
    from qiskit.circuit import ClassicalRegister, Clbit, Instruction, Qubit
    from qiskit.dagcircuit import DAGCircuit
    from qiskit.passmanager import PropertySet
    from qiskit.transpiler import Target
    from qiskit.transpiler.basepasses import BasePass as qiskit_BasePass

    PassList: TypeAlias = list[qiskit_BasePass | tket_BasePass]
    BQSKitMapping: TypeAlias = tuple[Circuit, tuple[int, ...], tuple[int, ...]]
    BQSKitCompileFn: TypeAlias = Callable[[Circuit], Circuit | BQSKitMapping]
    TranspilePassSpec: TypeAlias = PassList | Callable[..., PassList] | Callable[..., BQSKitCompileFn]

    class BQSKitWorkflowDataLike(Protocol):
        """Minimal attribute shape used from BQSKit request-data payloads."""

        initial_mapping: Sequence[int]
        final_mapping: Sequence[int]


class _BQSKitCompileData(TypedDict):
    """Materialized mapping metadata returned from BQSKit workflows."""

    initial_mapping: tuple[int, ...]
    final_mapping: tuple[int, ...]


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
    transpile_pass: TranspilePassSpec
    stochastic: bool | None = False
    preserve_layout: bool | None = False


@dataclass
class DeviceIndependentAction(Action):
    """Action that represents a static compilation pass that can be applied directly."""


@dataclass
class DeviceDependentAction(Action):
    """Action that represents a device-specific compilation pass that can be applied to a specific device."""

    transpile_pass: Callable[..., PassList] | Callable[..., BQSKitCompileFn]
    do_while: Callable[[PropertySet], bool] | None = None


# Registry of actions
_ACTIONS: dict[str, Action] = {}

_BQSKIT_OPT_LEVEL = 1 if os.getenv("GITHUB_ACTIONS") == "true" else 2
_BQSKIT_SYNTHESIS_EPSILON = 1e-1 if os.getenv("GITHUB_ACTIONS") == "true" else 1e-8
_BQSKIT_MAX_SYNTHESIS_SIZE = 3
_BQSKIT_SEED = 10


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
) -> Circuit | tuple[Circuit, _BQSKitCompileData]:
    """Compile ``circuit`` with a custom BQSKit workflow."""
    compiler = Compiler()
    try:
        result = compiler.compile(circuit, workflow, request_data=request_data)
    finally:
        compiler.close()

    if request_data:
        compiled_circuit, data = cast("tuple[Circuit, BQSKitWorkflowDataLike]", result)
        return compiled_circuit, {
            "initial_mapping": tuple(data.initial_mapping),
            "final_mapping": tuple(data.final_mapping),
        }

    return cast("Circuit", result)


def _build_bqskit_common_prefix(model: MachineModel) -> list[bqskit_BasePass]:
    """Build the common BQSKit workflow prefix used for RL actions."""
    return [
        SetRandomSeedPass(_BQSKIT_SEED),
        UnfoldPass(),
        ExtractMeasurements(),
        SetModelPass(model),
    ]


def _build_bqskit_common_suffix() -> list[bqskit_BasePass]:
    """Build the common BQSKit workflow suffix used for RL actions."""
    return [RestoreMeasurements()]


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
    *mapping_passes: bqskit_BasePass,
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
        return (
            compiled_circuit,
            data["initial_mapping"],
            data["final_mapping"],
        )

    return _compile


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
        [Optimize1qGatesDecomposition()],
    )
)

register_action(
    DeviceDependentAction(
        "Optimize1qGatesDecomposition_preserve",
        CompilationOrigin.QISKIT,
        PassType.OPT,
        preserve_layout=True,
        transpile_pass=lambda device: [Optimize1qGatesDecomposition(basis=device.operation_names)],
    )
)

register_action(
    DeviceIndependentAction(
        "CommutativeCancellation",
        CompilationOrigin.QISKIT,
        PassType.OPT,
        [CommutativeCancellation()],
        preserve_layout=True,
    )
)

register_action(
    DeviceIndependentAction(
        "CommutativeInverseCancellation",
        CompilationOrigin.QISKIT,
        PassType.OPT,
        [CommutativeInverseCancellation()],
        preserve_layout=True,
    )
)

register_action(
    DeviceIndependentAction(
        "RemoveDiagonalGatesBeforeMeasure",
        CompilationOrigin.QISKIT,
        PassType.OPT,
        [RemoveDiagonalGatesBeforeMeasure()],
        preserve_layout=True,
    )
)

register_action(
    DeviceIndependentAction(
        "ElidePermutations",
        CompilationOrigin.QISKIT,
        PassType.OPT,
        [ElidePermutations()],
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
        preserve_layout=True,
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
    DeviceIndependentAction(
        "Opt2qBlocks",
        CompilationOrigin.QISKIT,
        PassType.OPT,
        [Collect2qBlocks(), ConsolidateBlocks(), UnitarySynthesis()],
    )
)

register_action(
    DeviceDependentAction(
        "Opt2qBlocks_preserve",
        CompilationOrigin.QISKIT,
        PassType.OPT,
        transpile_pass=lambda native_gate, coupling_map: [
            Collect2qBlocks(),
            ConsolidateBlocks(basis_gates=native_gate),
            UnitarySynthesis(basis_gates=native_gate, coupling_map=coupling_map),
        ],
        preserve_layout=True,
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
        "KAKDecomposition",
        CompilationOrigin.TKET,
        PassType.OPT,
        [KAKDecomposition(allow_swaps=False)],
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
        "VF2PostLayout",
        CompilationOrigin.QISKIT,
        PassType.FINAL_OPT,
        transpile_pass=lambda device: VF2PostLayout(target=device, call_limit=5000),
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
            VF2Layout(target=device, call_limit=5000),
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
    )
)

register_action(
    DeviceDependentAction(
        "GraphPlacement",
        CompilationOrigin.TKET,
        PassType.LAYOUT,
        transpile_pass=lambda device: [
            GraphPlacement(Architecture(list(device.build_coupling_map())), timeout=5000, maximum_matches=5000)
        ],
    )
)

register_action(
    DeviceDependentAction(
        "NoiseAwarePlacement",
        CompilationOrigin.TKET,
        PassType.LAYOUT,
        transpile_pass=lambda device, node_err, edge_err, readout_err: [
            NoiseAwarePlacement(
                Architecture(list(device.build_coupling_map())),
                node_errors=node_err,
                link_errors=edge_err,
                readout_errors=readout_err,
                timeout=5000,
                maximum_matches=5000,
            )
        ],
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
        "GeneralizedSabreRoutingPass",
        CompilationOrigin.BQSKIT,
        PassType.ROUTING,
        transpile_pass=lambda device: _bqskit_mapping_factory(
            device, GeneralizedSabreRoutingPass(), apply_placement=False
        ),
    )
)

register_action(
    DeviceDependentAction(
        "SabreSwap",
        CompilationOrigin.QISKIT,
        PassType.ROUTING,
        stochastic=True,
        transpile_pass=lambda device: [
            SabreSwap(coupling_map=CouplingMap(device.build_coupling_map()), heuristic="decay")
        ],
    )
)

if not IS_WIN_PY313:
    register_action(
        DeviceDependentAction(
            "AIRouting",
            CompilationOrigin.QISKIT,
            PassType.ROUTING,
            stochastic=True,
            transpile_pass=lambda device: [
                SafeAIRouting(
                    coupling_map=device.build_coupling_map(),
                    optimization_level=3,
                    layout_mode="improve",
                    local_mode=True,
                )
            ],
        )
    )

    register_action(
        DeviceDependentAction(
            "AIRouting_opt",
            CompilationOrigin.QISKIT,
            PassType.MAPPING,
            stochastic=True,
            transpile_pass=lambda device: [
                ### Requires an initial layout, but "optimize" mode overwrites it
                SabreLayout(coupling_map=CouplingMap(device.build_coupling_map()), skip_routing=True, max_iterations=1),
                FullAncillaAllocation(coupling_map=CouplingMap(device.build_coupling_map())),
                EnlargeWithAncilla(),
                ApplyLayout(),
                SafeAIRouting(coupling_map=device.build_coupling_map(), optimization_level=3, layout_mode="optimize"),
            ],
        )
    )

register_action(
    DeviceDependentAction(
        "SabreMapping",
        CompilationOrigin.QISKIT,
        PassType.MAPPING,
        stochastic=True,
        transpile_pass=lambda device: [
            SabreLayout(coupling_map=CouplingMap(device.build_coupling_map()), skip_routing=False, max_iterations=1),
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
) -> tuple[list[ClassicalRegister], list[tuple[Instruction, list[Qubit], list[Clbit]]]]:
    """Extract classical registers and measurement operations from a quantum circuit.

    Args:
        qc: The input QuantumCircuit.

    Returns:
        A tuple ``(cregs, measurements)`` where:
            - ``cregs`` is a list of the circuit's ClassicalRegister objects.
            - ``measurements`` is a list of tuples ``(instr, qargs, cargs)`` for each
              measurement, where:
                * ``instr`` is the measurement Instruction,
                * ``qargs`` is the list of Qubit objects measured,
                * ``cargs`` is the list of Clbit objects written to.
    """
    # IMPORTANT: reuse the original registers, do NOT clone them
    cregs = list(qc.cregs)

    measurements: list[tuple[Instruction, list[Qubit], list[Clbit]]] = []
    for item in qc.data:
        if item.operation.name == "measure":
            instr = item.operation
            qargs = list(item.qubits)
            cargs = list(item.clbits)
            measurements.append((instr, qargs, cargs))

    return cregs, measurements


def remove_cregs(qc: QuantumCircuit) -> QuantumCircuit:
    """Return a copy of ``qc`` without classical registers and measurements.

    Classical registers and measurement operations are removed, but quantum
    registers and qubit *identity* are preserved.

    Args:
        qc: The input QuantumCircuit.

    Returns:
        A new QuantumCircuit with only quantum operations (no cregs or measurements).
    """
    # Reuse the original QuantumRegister objects to preserve Qubit identity
    new_qc = QuantumCircuit(*qc.qregs)

    for item in qc.data:
        instr = item.operation
        if instr.name in ("measure", "barrier"):
            continue
        # Use the original Qubit objects directly; no remapping
        qargs = list(item.qubits)
        new_qc.append(instr, qargs)

    return new_qc


def add_cregs_and_measurements(
    qc: QuantumCircuit,
    cregs: list[ClassicalRegister],
    measurements: list[tuple[Instruction, list[Qubit], list[Clbit]]],
    qubit_map: dict[Qubit, Qubit] | None = None,
) -> QuantumCircuit:
    """Add classical registers and measurement operations back to a circuit.

    Args:
        qc:
            The quantum circuit to which cregs and measurements are added.
        cregs:
            List of ClassicalRegister objects to add (typically taken from
            :func:`extract_cregs_and_measurements`).
        measurements:
            List of measurement tuples ``(instr, qargs, cargs)`` as returned by
            :func:`extract_cregs_and_measurements`.
        qubit_map:
            Optional mapping from original Qubit objects to new Qubit objects.
            If provided, measurement qubits are remapped via this dictionary.

    Returns:
        The modified QuantumCircuit with cregs and measurements added.
    """
    # Attach the original ClassicalRegister objects so their Clbits are valid in this circuit
    for cr in cregs:
        qc.add_register(cr)

    for instr, qargs, cargs in measurements:
        new_qargs = [qubit_map[q] for q in qargs] if qubit_map is not None else qargs
        qc.append(instr, new_qargs, cargs)

    return qc


if not IS_WIN_PY313:

    class SafeAIRouting(AIRouting):
        """Custom AIRouting wrapper that removes classical registers before routing.

        This prevents failures in AIRouting when classical bits are present by
        temporarily removing classical registers and measurements and restoring
        them after routing is completed.
        """

        def run(self, dag: DAGCircuit) -> DAGCircuit:
            """Run the routing pass on a DAGCircuit."""
            qc_orig = dag_to_circuit(dag)
            # Extract classical registers and measurement instructions
            cregs, measurements = extract_cregs_and_measurements(qc_orig)
            # Remove cregs and measurements
            qc_noclassical = remove_cregs(qc_orig)
            # Convert back to dag and run routing (AIRouting)
            dag_noclassical = circuit_to_dag(qc_noclassical)
            dag_routed = super().run(dag_noclassical)
            # Convert routed dag to circuit for restoration
            qc_routed = dag_to_circuit(dag_routed)
            # Build mapping from original qubits to qubits in routed circuit
            final_layout = getattr(self, "property_set", {}).get("final_layout", None)
            if final_layout is None:
                msg = "final_layout is None — cannot map virtual qubits"
                raise RuntimeError(msg)
            qubit_map = {}
            for virt in qc_orig.qubits:
                if virt not in final_layout:
                    msg = f"Virtual qubit {virt} not found in final layout!"
                    raise RuntimeError(msg)
                phys = final_layout[virt]
                if isinstance(phys, int):
                    if not 0 <= phys < len(qc_routed.qubits):
                        msg = f"Physical index {phys} out of range in routed circuit!"
                        raise ValueError(msg)
                    qubit_map[virt] = qc_routed.qubits[phys]
                else:
                    if phys not in qc_routed.qubits:
                        msg = f"Physical qubit {phys} not found in output circuit!"
                        raise ValueError(msg)
                    qubit_map[virt] = qc_routed.qubits[qc_routed.qubits.index(phys)]
            # Restore classical registers and measurement instructions
            qc_final = add_cregs_and_measurements(qc_routed, cregs, measurements, qubit_map)
            # Return as dag
            return circuit_to_dag(qc_final)
