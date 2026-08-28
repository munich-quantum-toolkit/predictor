# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Qiskit actions and execution helpers."""

from __future__ import annotations

import atexit
import logging
import multiprocessing
from functools import cache
from importlib import import_module
from typing import TYPE_CHECKING, Any, cast

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
    BasicSwap,
    BasisTranslator,
    Collect2qBlocks,
    CollectCliffords,
    CommutativeCancellation,
    CommutativeInverseCancellation,
    ConsolidateBlocks,
    Decompose,
    DenseLayout,
    Depth,
    ElidePermutations,
    EnlargeWithAncilla,
    FixedPoint,
    FullAncillaAllocation,
    GatesInBasis,
    InverseCancellation,
    LookaheadSwap,
    MinimumPoint,
    Optimize1qGatesDecomposition,
    Optimize1qGatesSimpleCommutation,
    OptimizeCliffords,
    RemoveDiagonalGatesBeforeMeasure,
    RemoveIdentityEquivalent,
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

from mqt.predictor.rl.actions.base import (
    CompilationOrigin,
    DeferredDeviceAction,
    DeviceIndependentAction,
    PassType,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from multiprocessing.pool import Pool

    from qiskit import QuantumCircuit
    from qiskit.passmanager import PropertySet
    from qiskit.passmanager.base_tasks import Task
    from qiskit.transpiler import Layout, Target

    from mqt.predictor.rl.actions.base import Action

logger = logging.getLogger("mqt-predictor")
_QISKIT_TIMEOUT_POOL: Pool | None = None
_VF2_ACTION_NAMES = frozenset({"VF2Layout", "VF2PostLayout"})

_AI_ROUTING_ACTION_NAMES = frozenset({"AIRouting", "AIRouting_opt"})


@cache
def _load_airouting() -> type[Any]:
    """Load IBM's optional AI routing pass."""
    try:
        module = import_module("qiskit_ibm_transpiler.ai.routing")
    except ImportError as exc:
        msg = "AIRouting requires a qiskit-ibm-transpiler installation compatible with this environment."
        raise RuntimeError(msg) from exc
    return cast("type[Any]", vars(module)["AIRouting"])


def _airouting_pass(*, coupling_map: CouplingMap, layout_mode: str) -> Task:
    """Construct IBM's local AI routing pass."""
    return _load_airouting()(
        coupling_map=coupling_map,
        optimization_level=3,
        layout_mode=layout_mode,
        local_mode=True,
    )


@cache
def _is_ai_routing_available() -> bool:
    """Return whether IBM's AI routing pass can be imported."""
    try:
        _load_airouting()
    except RuntimeError:
        return False
    return True


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
            [
                CollectCliffords(),
                OptimizeCliffords(),
                Decompose(gates_to_decompose="clifford", apply_synthesis=True),
            ],
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
        DeviceIndependentAction(
            "RemoveIdentityEquivalent",
            CompilationOrigin.QISKIT,
            PassType.OPT,
            [RemoveIdentityEquivalent()],
            preserves_layout=True,
            preserves_routing=True,
            preserves_synthesis=True,
        ),
        DeferredDeviceAction(
            "Optimize1qGatesSimpleCommutation",
            CompilationOrigin.QISKIT,
            PassType.OPT,
            transpile_pass=lambda device: cast(
                "list[Task]",
                [
                    Optimize1qGatesSimpleCommutation(
                        basis=device.operation_names,
                        run_to_completion=True,
                    )
                ],
            ),
            preserves_layout=True,
            preserves_routing=True,
            preserves_synthesis=True,
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
        transpile_pass=lambda device: [VF2PostLayout(target=device)],
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
        DeferredDeviceAction(
            "TrivialLayout",
            CompilationOrigin.QISKIT,
            PassType.LAYOUT,
            transpile_pass=lambda device: cast(
                "list[Task]",
                [
                    TrivialLayout(coupling_map=CouplingMap(device.build_coupling_map())),
                    FullAncillaAllocation(coupling_map=CouplingMap(device.build_coupling_map())),
                    EnlargeWithAncilla(),
                    ApplyLayout(),
                ],
            ),
        ),
        DeferredDeviceAction(
            "ElidePermutations",
            CompilationOrigin.QISKIT,
            PassType.LAYOUT,
            transpile_pass=lambda device: cast(
                "list[Task]",
                [
                    ElidePermutations(),
                    TrivialLayout(coupling_map=CouplingMap(device.build_coupling_map())),
                    FullAncillaAllocation(coupling_map=CouplingMap(device.build_coupling_map())),
                    EnlargeWithAncilla(),
                    ApplyLayout(),
                ],
            ),
        ),
    ]


def qiskit_routing_actions() -> list[Action]:
    """Return the Qiskit routing actions."""
    return [
        DeferredDeviceAction(
            "SabreSwap",
            CompilationOrigin.QISKIT,
            PassType.ROUTING,
            transpile_pass=lambda device: cast(
                "list[Task]", [SabreSwap(coupling_map=CouplingMap(device.build_coupling_map()), heuristic="decay")]
            ),
        ),
        DeferredDeviceAction(
            "BasicSwap",
            CompilationOrigin.QISKIT,
            PassType.ROUTING,
            transpile_pass=lambda device: cast(
                "list[Task]", [BasicSwap(coupling_map=CouplingMap(device.build_coupling_map()))]
            ),
        ),
        DeferredDeviceAction(
            "LookaheadSwap",
            CompilationOrigin.QISKIT,
            PassType.ROUTING,
            transpile_pass=lambda device: cast(
                "list[Task]",
                [
                    LookaheadSwap(
                        coupling_map=CouplingMap(device.build_coupling_map()),
                        search_depth=1,
                        search_width=1,
                    )
                ],
            ),
        ),
    ]


def qiskit_ai_routing_action() -> Action:
    """Return IBM's AI routing action."""
    return DeferredDeviceAction(
        "AIRouting",
        CompilationOrigin.QISKIT,
        PassType.ROUTING,
        transpile_pass=lambda device: cast(
            "list[Task]",
            [
                _airouting_pass(
                    coupling_map=device.build_coupling_map(),
                    layout_mode="improve",
                )
            ],
        ),
    )


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


def qiskit_ai_mapping_action() -> Action:
    """Return the combined AI layout and routing action."""
    return DeferredDeviceAction(
        "AIRouting_opt",
        CompilationOrigin.QISKIT,
        PassType.MAPPING,
        transpile_pass=lambda device: cast(
            "list[Task]",
            [
                _airouting_pass(
                    coupling_map=device.build_coupling_map(),
                    layout_mode="optimize",
                ),
            ],
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


def _postprocess_layout_action(
    action: Action,
    property_set: PropertySet,
    altered_qc: QuantumCircuit,
    layout: TranspileLayout | None,
    input_qubit_count: int | None = None,
) -> tuple[QuantumCircuit, TranspileLayout | None]:
    """Update Qiskit's layout metadata after passes that can create or alter layouts."""
    if action.name == "VF2PostLayout":
        assert property_set["VF2PostLayout_stop_reason"] is not None
        post_layout = property_set["post_layout"]
        if post_layout:
            assert layout is not None
            altered_qc, apply_layout = postprocess_vf2postlayout(altered_qc, post_layout, layout)
            property_set = apply_layout.property_set
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
            _input_qubit_count=input_qubit_count,
            _output_qubit_list=altered_qc.qubits,
        )
    return altered_qc, layout


def _seed_randomized_passes(passes: list[Task], seed: int) -> None:
    """Seed randomized Qiskit passes with CPU-independent trial counts."""
    for transpiler_pass in passes:
        if isinstance(transpiler_pass, (SabreLayout, SabreSwap, VF2Layout, VF2PostLayout)):
            transpiler_pass.seed = seed
        if isinstance(transpiler_pass, SabreLayout):
            transpiler_pass.layout_trials = 1
            transpiler_pass.swap_trials = 1
        elif isinstance(transpiler_pass, SabreSwap):
            transpiler_pass.trials = 1


def _set_native_pass_time_limits(passes: Sequence[Task], pass_timeout: float | None) -> None:
    """Configure native deadlines for Qiskit passes that support them."""
    for transpiler_pass in passes:
        if isinstance(transpiler_pass, (VF2Layout, VF2PostLayout)):
            transpiler_pass.time_limit = pass_timeout


def _run_qiskit_action(
    action: Action,
    circuit: QuantumCircuit,
    device: Target,
    layout: TranspileLayout | None,
    input_qubit_count: int | None = None,
    seed: int | None = None,
    pass_timeout: float | None = None,
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

    _set_native_pass_time_limits(passes, pass_timeout)

    if seed is not None:
        _seed_randomized_passes(passes, seed)

    if action.name == "QiskitO3" and isinstance(action, DeferredDeviceAction):
        assert action.do_while is not None
        pm = PassManager([DoWhileController(passes, do_while=action.do_while)])
    else:
        pm = PassManager(passes)

    altered_qc = pm.run(circuit)

    if action.pass_type in {PassType.LAYOUT, PassType.MAPPING, PassType.FINAL_OPT}:
        altered_qc, layout = _postprocess_layout_action(action, pm.property_set, altered_qc, layout, input_qubit_count)
    elif action.pass_type == PassType.ROUTING and layout and pm.property_set["final_layout"] is not None:
        routing_layout = pm.property_set["final_layout"]
        layout.final_layout = (
            layout.final_layout.compose(routing_layout, circuit.qubits)
            if layout.final_layout is not None
            else routing_layout
        )

    if altered_qc.count_ops().get("unitary"):
        # Custom "unitary" gates can not be processed further by other passes
        altered_qc = altered_qc.decompose(gates_to_decompose="unitary")
    return altered_qc, layout


def _run_registered_vf2_action(
    action_name: str,
    circuit: QuantumCircuit,
    device: Target,
    layout: TranspileLayout | None,
    input_qubit_count: int | None,
    seed: int | None,
    pass_timeout: float,
) -> tuple[QuantumCircuit, TranspileLayout | None]:
    if action_name == "VF2Layout":
        action = next(action for action in qiskit_layout_actions() if action.name == action_name)
    else:
        action = qiskit_final_optimization_action()
    return _run_qiskit_action(action, circuit, device, layout, input_qubit_count, seed, pass_timeout)


def _close_qiskit_timeout_pool() -> None:
    global _QISKIT_TIMEOUT_POOL  # ruff: ignore[global-statement]
    if _QISKIT_TIMEOUT_POOL is not None:
        _QISKIT_TIMEOUT_POOL.terminate()
        _QISKIT_TIMEOUT_POOL.join()
        _QISKIT_TIMEOUT_POOL = None


def _get_qiskit_timeout_pool() -> Pool:
    global _QISKIT_TIMEOUT_POOL  # ruff: ignore[global-statement]
    if _QISKIT_TIMEOUT_POOL is None:
        _QISKIT_TIMEOUT_POOL = multiprocessing.get_context("forkserver").Pool(processes=1)
    return _QISKIT_TIMEOUT_POOL


atexit.register(_close_qiskit_timeout_pool)


def run_qiskit_action(
    action: Action,
    circuit: QuantumCircuit,
    device: Target,
    layout: TranspileLayout | None,
    input_qubit_count: int | None = None,
    seed: int | None = None,
    pass_timeout: float | None = None,
) -> tuple[QuantumCircuit, TranspileLayout | None]:
    """Apply a Qiskit action and return the updated circuit and layout metadata."""
    if pass_timeout is None or action.name not in _VF2_ACTION_NAMES:
        return _run_qiskit_action(action, circuit, device, layout, input_qubit_count, seed, pass_timeout)

    try:
        return (
            _get_qiskit_timeout_pool()
            .apply_async(
                _run_registered_vf2_action,
                (action.name, circuit, device, layout, input_qubit_count, seed, pass_timeout),
            )
            .get(timeout=pass_timeout)
        )
    except (multiprocessing.TimeoutError, TimeoutError) as error:
        _close_qiskit_timeout_pool()
        msg = f"Compilation pass exceeded the timeout of {pass_timeout:g} seconds."
        raise TimeoutError(msg) from error


def is_qiskit_action_available(action: Action, device: Target) -> bool:
    """Return whether a Qiskit action is available for the current device."""
    if action.name in _AI_ROUTING_ACTION_NAMES and not _is_ai_routing_available():
        return False
    # Only allow VF2PostLayout if "ibm" is in the device name # TODO: Why?
    return action.name != "VF2PostLayout" or "ibm" in device.description
