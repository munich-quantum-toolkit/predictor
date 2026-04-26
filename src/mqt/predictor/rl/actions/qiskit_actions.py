# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Qiskit RL actions and their action-local helper logic."""

from __future__ import annotations

import logging
import sys
import warnings
from typing import TYPE_CHECKING, Any, cast

from qiskit import QuantumCircuit
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
from qiskit.exceptions import QiskitError
from qiskit.passmanager import ConditionalController
from qiskit.transpiler import CouplingMap, PassManager, TranspileLayout
from qiskit.transpiler.exceptions import TranspilerError
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

from . import (
    CompilationOrigin,
    DeviceDependentAction,
    DeviceIndependentAction,
    PassType,
    register_action,
)

logger = logging.getLogger("mqt-predictor")

IS_WIN_PY313 = sys.platform == "win32" and sys.version_info[:2] == (3, 13)

AIRouting: Any = None
SafeAIRouting: Any = None
HAS_AI_ROUTING = False
if not IS_WIN_PY313:
    try:
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
            from qiskit_ibm_transpiler.local_routing.routing.inference import RoutingInference

            HAS_AI_ROUTING = True
    except ImportError:
        pass


if TYPE_CHECKING:
    from collections.abc import Callable

    from qiskit.circuit import ClassicalRegister, Clbit, Instruction, Qubit
    from qiskit.dagcircuit import DAGCircuit
    from qiskit.passmanager.base_tasks import Task
    from qiskit.transpiler import Layout, Target

    from . import (
        Action,
    )


_AI_ROUTING_RUNTIME_STATE: dict[str, bool | RuntimeError | None] = {
    "validated": False,
    "error": None,
}


def ensure_ai_routing_runtime_available() -> None:
    """Validate that AIRouting is usable at runtime and preload its model once."""
    if not HAS_AI_ROUTING or bool(_AI_ROUTING_RUNTIME_STATE["validated"]):
        return
    cached_error = _AI_ROUTING_RUNTIME_STATE["error"]
    if isinstance(cached_error, RuntimeError):
        raise cached_error

    try:
        RoutingInference()
    except Exception as exc:
        runtime_error = RuntimeError(
            "AIRouting is installed but not usable: the qiskit-ibm-transpiler routing model "
            "could not be loaded during startup. Ensure the routing model is already cached "
            "locally or that the machine has network access for the initial download."
        )
        _AI_ROUTING_RUNTIME_STATE["error"] = runtime_error
        raise runtime_error from exc

    _AI_ROUTING_RUNTIME_STATE["validated"] = True


def get_qiskit_action_passes(action: Action, device: Target, layout: TranspileLayout | None) -> list[Task]:
    """Materialize Qiskit passes, including action-specific construction details."""
    if action.name == "Opt2qBlocks_preserve" and isinstance(action, DeviceDependentAction):
        return cast(
            "list[Task]",
            action.transpile_pass(
                device.operation_names,
                CouplingMap(device.build_coupling_map()) if layout is not None else None,
            ),
        )

    if callable(action.transpile_pass):
        pass_factory = cast("Callable[[Target], list[Task]]", action.transpile_pass)
        return pass_factory(device)

    return cast("list[Task]", action.transpile_pass)


def postprocess_vf2postlayout(
    qc: QuantumCircuit, post_layout: Layout, layout_before: TranspileLayout
) -> tuple[QuantumCircuit, ApplyLayout]:
    """Apply VF2PostLayout while preserving the existing Qiskit layout bookkeeping."""
    original_qubit_indices = dict(layout_before.input_qubit_mapping)
    for virt, phys in layout_before.initial_layout.get_virtual_bits().items():
        if virt not in original_qubit_indices:
            original_qubit_indices[virt] = phys

    apply_layout = ApplyLayout()
    apply_layout.property_set["layout"] = layout_before.initial_layout
    apply_layout.property_set["original_qubit_indices"] = original_qubit_indices
    apply_layout.property_set["final_layout"] = layout_before.final_layout
    apply_layout.property_set["post_layout"] = post_layout

    altered_qc = apply_layout.run(circuit_to_dag(qc))
    return dag_to_circuit(altered_qc), apply_layout


def postprocess_qiskit_action_result(
    action: Action,
    pm_property_set: dict[str, object] | None,
    altered_qc: QuantumCircuit,
    layout_before: TranspileLayout | None,
    circuit: QuantumCircuit,
    *,
    updates_layout: bool,
    updates_routing: bool,
) -> tuple[QuantumCircuit, TranspileLayout | None]:
    """Keep Qiskit layout bookkeeping close to the Qiskit action definitions."""
    if not pm_property_set:
        return altered_qc, layout_before

    layout_after = layout_before
    if updates_layout:
        if action.name == "VF2PostLayout":
            assert pm_property_set["VF2PostLayout_stop_reason"] is not None
            post_layout = cast("Layout | None", pm_property_set.get("post_layout"))
            if post_layout is not None:
                assert layout_after is not None
                altered_qc, _ = postprocess_vf2postlayout(altered_qc, post_layout, layout_after)
        elif action.name == "VF2Layout":
            if pm_property_set["VF2Layout_stop_reason"] != VF2LayoutStopReason.SOLUTION_FOUND:
                logger.warning(
                    "VF2Layout pass did not find a solution. Reason: %s",
                    pm_property_set["VF2Layout_stop_reason"],
                )
        else:
            assert pm_property_set["layout"] is not None

        initial_layout = cast("Layout | None", pm_property_set.get("layout"))
        if initial_layout is not None:
            layout_after = TranspileLayout(
                initial_layout=initial_layout,
                input_qubit_mapping=cast("dict[object, int]", pm_property_set.get("original_qubit_indices")),
                final_layout=cast("Layout | None", pm_property_set.get("final_layout")),
                _output_qubit_list=altered_qc.qubits,
                _input_qubit_count=circuit.num_qubits,
            )

    if layout_after is not None and (updates_layout or updates_routing):
        final_layout = cast("Layout | None", pm_property_set.get("final_layout"))
        if final_layout is not None:
            layout_after.final_layout = final_layout

    return altered_qc, layout_after


def fom_aware_compile(
    action: Action,
    device: Target,
    qc: QuantumCircuit,
    score_circuit: Callable[[QuantumCircuit], tuple[float, str]],
    max_iteration: int = 20,
) -> tuple[QuantumCircuit, dict[str, Any] | None]:
    """Run a stochastic Qiskit pass multiple times and keep the best result."""
    best_result: QuantumCircuit | None = None
    best_property_set: dict[str, Any] | None = None
    best_fom = -1.0
    best_swap_count = float("inf")

    assert callable(action.transpile_pass), "Mapping action should be callable"
    pass_factory = cast("Callable[[Target], list[Task]]", action.transpile_pass)
    for i in range(max_iteration):
        passes = pass_factory(device)
        pm = PassManager(passes)
        try:
            out_circ = pm.run(qc)
            prop_set = dict(pm.property_set)

            try:
                synth_pass = PassManager([
                    BasisTranslator(StandardEquivalenceLibrary, target_basis=device.operation_names)
                ])
                synth_circ = synth_pass.run(out_circ.copy())
                fom, _ = score_circuit(synth_circ)

                if fom > best_fom:
                    best_fom = fom
                    best_result = out_circ
                    best_property_set = prop_set

            except (QiskitError, TranspilerError, RuntimeError, ValueError, TypeError) as exc:
                logger.warning("[Fallback to SWAP counts] Synthesis or fidelity computation failed: %s", exc)
                swap_count = out_circ.count_ops().get("swap", 0)
                if best_result is None or swap_count < best_swap_count:
                    best_swap_count = swap_count
                    best_result = out_circ
                    best_property_set = prop_set

        except Exception:
            logger.exception("[Error] Pass failed at iteration %d", i + 1)
            continue

    if best_result is not None:
        return best_result, best_property_set
    logger.error("All attempts failed.")
    return qc, None


def run_qiskit_action(
    *,
    action: Action,
    circuit: QuantumCircuit,
    device: Target,
    layout: TranspileLayout | None,
    max_iteration: int,
    score_circuit: Callable[[QuantumCircuit], tuple[float, str]],
) -> tuple[QuantumCircuit, TranspileLayout | None]:
    """Apply a Qiskit action and update the layout bookkeeping it owns.

    Args:
        action: The Qiskit action to apply.
        circuit: The current quantum circuit.
        device: The target device.
        layout: The current layout (if any).
        max_iteration: Maximum iterations for stochastic actions.
        score_circuit: Function to score circuits (for figure of merit).

    Returns:
        Tuple of (compiled circuit, updated layout).
    """
    pm_property_set: dict[str, Any] | None = None
    if getattr(action, "stochastic", False):
        altered_qc, pm_property_set = fom_aware_compile(
            action,
            device,
            circuit,
            score_circuit,
            max_iteration=max_iteration,
        )
    else:
        transpile_pass = get_qiskit_action_passes(action, device, layout)
        pm = PassManager(transpile_pass)
        altered_qc = pm.run(circuit)
        pm_property_set = dict(pm.property_set) if hasattr(pm, "property_set") else None

    altered_qc, layout = postprocess_qiskit_action_result(
        action,
        pm_property_set,
        altered_qc,
        layout,
        circuit,
        updates_layout=action.pass_type in (PassType.LAYOUT, PassType.MAPPING, PassType.FINAL_OPT),
        updates_routing=action.pass_type == PassType.ROUTING,
    )

    if altered_qc.count_ops().get("unitary"):
        altered_qc = altered_qc.decompose(gates_to_decompose="unitary")
    elif altered_qc.count_ops().get("clifford"):
        altered_qc = altered_qc.decompose(gates_to_decompose="clifford")

    return altered_qc, layout


register_action(
    DeviceIndependentAction(
        "Optimize1qGatesDecomposition",
        CompilationOrigin.QISKIT,
        PassType.OPT,
        [Optimize1qGatesDecomposition()],
        preserves_layout=True,
        preserves_routing=True,
        preserves_synthesis=False,
    )
)

register_action(
    DeviceDependentAction(
        "Optimize1qGatesDecomposition_preserve",
        CompilationOrigin.QISKIT,
        PassType.OPT,
        preserves_layout=True,
        preserves_routing=True,
        preserves_synthesis=True,
        transpile_pass=lambda device: [Optimize1qGatesDecomposition(basis=device.operation_names)],
    )
)

register_action(
    DeviceIndependentAction(
        "CommutativeCancellation",
        CompilationOrigin.QISKIT,
        PassType.OPT,
        [CommutativeCancellation()],
        preserves_layout=True,
        preserves_routing=True,
        preserves_synthesis=True,
    )
)

register_action(
    DeviceIndependentAction(
        "CommutativeInverseCancellation",
        CompilationOrigin.QISKIT,
        PassType.OPT,
        [CommutativeInverseCancellation()],
        preserves_layout=True,
        preserves_routing=True,
        preserves_synthesis=True,
    )
)

register_action(
    DeviceIndependentAction(
        "RemoveDiagonalGatesBeforeMeasure",
        CompilationOrigin.QISKIT,
        PassType.OPT,
        [RemoveDiagonalGatesBeforeMeasure()],
        preserves_layout=True,
        preserves_routing=True,
        preserves_synthesis=True,
    )
)

register_action(
    DeviceIndependentAction(
        "ElidePermutations",
        CompilationOrigin.QISKIT,
        PassType.OPT,
        [ElidePermutations()],
        preserves_layout=False,  # folds SWAPs into the layout's final_layout, changing the qubit mapping
        preserves_routing=False,
        preserves_synthesis=True,
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
        preserves_layout=True,
        preserves_routing=True,
        preserves_synthesis=True,
    )
)

register_action(
    DeviceIndependentAction(
        "OptimizeCliffords",
        CompilationOrigin.QISKIT,
        PassType.OPT,
        [CollectCliffords(), OptimizeCliffords()],
        preserves_layout=True,
        preserves_routing=False,
        preserves_synthesis=False,
    )
)

register_action(
    DeviceIndependentAction(
        "Opt2qBlocks",
        CompilationOrigin.QISKIT,
        PassType.OPT,
        [Collect2qBlocks(), ConsolidateBlocks(), UnitarySynthesis()],
        preserves_layout=True,
        preserves_routing=True,
        preserves_synthesis=False,
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
        preserves_layout=True,
        preserves_routing=True,
        preserves_synthesis=True,
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
        "SabreSwap",
        CompilationOrigin.QISKIT,
        PassType.ROUTING,
        stochastic=True,
        transpile_pass=lambda device: [
            SabreSwap(coupling_map=CouplingMap(device.build_coupling_map()), heuristic="decay")
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


def extract_cregs_and_measurements(
    qc: QuantumCircuit,
) -> tuple[list[ClassicalRegister], list[tuple[Instruction, list[Qubit], list[Clbit]]]]:
    """Extract classical registers and measurement operations from a quantum circuit."""
    cregs = list(qc.cregs)
    measurements: list[tuple[Instruction, list[Qubit], list[Clbit]]] = [
        (item.operation, list(item.qubits), list(item.clbits)) for item in qc.data if item.operation.name == "measure"
    ]
    return cregs, measurements


def remove_cregs(qc: QuantumCircuit) -> QuantumCircuit:
    """Return a copy of ``qc`` without classical registers and measurements."""
    new_qc = QuantumCircuit(*qc.qregs)
    for item in qc.data:
        instr = item.operation
        if instr.name in ("measure", "barrier"):
            continue
        new_qc.append(instr, list(item.qubits))
    return new_qc


def add_cregs_and_measurements(
    qc: QuantumCircuit,
    cregs: list[ClassicalRegister],
    measurements: list[tuple[Instruction, list[Qubit], list[Clbit]]],
    qubit_map: dict[Qubit, Qubit] | None = None,
) -> QuantumCircuit:
    """Add classical registers and measurement operations back to a circuit."""
    for creg in cregs:
        qc.add_register(creg)

    for instr, qargs, cargs in measurements:
        new_qargs = [qubit_map[q] for q in qargs] if qubit_map is not None else qargs
        qc.append(instr, new_qargs, cargs)

    return qc


if HAS_AI_ROUTING:
    _AIRoutingBase = cast("type[Any]", AIRouting)

    class _SafeAIRouting(_AIRoutingBase):
        """AIRouting wrapper that strips classical structure before routing and restores it afterwards."""

        def run(self, dag: DAGCircuit) -> DAGCircuit:
            """Run the routing pass on a DAGCircuit."""
            qc_orig = dag_to_circuit(dag)
            cregs, measurements = extract_cregs_and_measurements(qc_orig)
            dag_routed = super().run(circuit_to_dag(remove_cregs(qc_orig)))
            qc_routed = dag_to_circuit(dag_routed)

            final_layout = getattr(self, "property_set", {}).get("final_layout", None)
            if final_layout is None:
                msg = "final_layout is None — cannot map virtual qubits"
                raise RuntimeError(msg)

            qubit_map: dict[Qubit, Qubit] = {}
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

            return circuit_to_dag(add_cregs_and_measurements(qc_routed, cregs, measurements, qubit_map))

    SafeAIRouting = _SafeAIRouting


if HAS_AI_ROUTING:
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
                SabreLayout(coupling_map=CouplingMap(device.build_coupling_map()), skip_routing=True, max_iterations=1),
                FullAncillaAllocation(coupling_map=CouplingMap(device.build_coupling_map())),
                EnlargeWithAncilla(),
                ApplyLayout(),
                SafeAIRouting(coupling_map=device.build_coupling_map(), optimization_level=3, layout_mode="optimize"),
            ],
        )
    )


__all__ = [
    "HAS_AI_ROUTING",
    "IS_WIN_PY313",
    "ensure_ai_routing_runtime_available",
    "postprocess_vf2postlayout",
    "run_qiskit_action",
]

if HAS_AI_ROUTING:
    __all__ += ["AIRouting", "SafeAIRouting"]
