# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Evaluate locked SDK compilation pipelines on the held-out RL test split."""

from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from mqt.bench.targets import get_device
from pytket import OpType
from pytket.architecture import Architecture
from pytket.extensions.qiskit.backends.ibm_utils import _gen_lightsabre_transformation  # noqa: PLC2701
from pytket.extensions.qiskit.qiskit_convert import qiskit_to_tk, tk_to_qiskit
from pytket.passes import (
    AutoRebase,
    AutoSquash,
    CliffordSimp,
    CustomPassMap,
    DecomposeBoxes,
    FullPeepholeOptimise,
    KAKDecomposition,
    RemoveRedundancies,
    SynthesiseTket,
)
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary
from qiskit.converters import dag_to_circuit
from qiskit.transpiler import CouplingMap, PassManager, TranspileLayout
from qiskit.transpiler.basepasses import BasePass as QiskitBasePass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes import BasisTranslator, UnitarySynthesis
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from mqt.predictor.rl.approx_reward import get_basis_gates_from_target
from mqt.predictor.rl.experiments.evaluation import (
    CircuitEvaluationResult,
    collect_final_metrics,
    compute_action_effectiveness_summary,
    compute_average_metrics,
    load_test_circuits,
    resolve_test_circuit_directory,
)
from mqt.predictor.rl.helper import get_path_training_circuits_test, get_path_training_circuits_train
from mqt.predictor.rl.predictorenv import PredictorEnv

if TYPE_CHECKING:
    from pytket.passes import (
        BasePass as TketBasePass,
    )
    from qiskit import QuantumCircuit
    from qiskit.dagcircuit import DAGCircuit
    from qiskit.passmanager.compilation_status import PropertySet
    from qiskit.transpiler import Target
    from qiskit.transpiler.passmanager import StagedPassManager

    from mqt.predictor.reward import figure_of_merit
    from mqt.predictor.rl.experiments.evaluation import (
        ActionEffectivenessSummary,
        FinalCircuitMetrics,
    )


LOCKED_QISKIT_VERSION = "2.3.0"
LOCKED_PYTKET_VERSION = "2.15.0"
ALL_PRIMITIVE_1Q_GATES: set[OpType] = {OpType.Rx, OpType.Rz, OpType.SX, OpType.X}
ALL_PRIMITIVE_2Q_GATES: set[OpType] = {OpType.CX, OpType.CZ, OpType.ECR, OpType.ZZPhase}
QISKIT_GATE_NAME_TO_OPTYPE: dict[str, OpType] = {
    "cx": OpType.CX,
    "cz": OpType.CZ,
    "ecr": OpType.ECR,
    "id": OpType.noop,
    "measure": OpType.Measure,
    "reset": OpType.Reset,
    "rx": OpType.Rx,
    "rz": OpType.Rz,
    "sx": OpType.SX,
    "x": OpType.X,
    "zzphase": OpType.ZZPhase,
}
PIPELINE_EVALUATION_ERRORS = (AttributeError, KeyError, TypeError, ValueError, RuntimeError, TranspilerError)


@dataclass(slots=True)
class PipelineStepSnapshot:
    """Circuit state after one pipeline step."""

    pass_name: str
    circuit: QuantumCircuit
    layout: TranspileLayout | None = None
    laid_out_override: bool | None = None


@dataclass(slots=True)
class StepEvaluationState:
    """Reward and compilation-state flags for one pipeline snapshot."""

    reward_value: float
    reward_kind: str
    depth: int
    size: int
    synthesized: bool
    laid_out: bool
    routed: bool


@dataclass(slots=True)
class TketPipelineStep:
    """Named TKET pass used for the locked default O2 pipeline."""

    name: str
    pass_: TketBasePass
    enables_layout_tracking: bool = False


@dataclass(slots=True)
class PipelineEvaluationResult:
    """Aggregated evaluation report for one SDK pipeline."""

    pipeline_name: str
    sdk_name: str
    sdk_version: str
    device_name: str
    figure_of_merit: str
    test_directory: Path
    documented_passes: list[str]
    circuits: list[CircuitEvaluationResult]
    average_metrics: FinalCircuitMetrics
    action_effectiveness: ActionEffectivenessSummary

    def to_dict(self) -> dict[str, object]:
        """Serialize the report into JSON-compatible data."""
        data = asdict(self)
        data["test_directory"] = str(self.test_directory)
        return data


def evaluate_qiskit_o3_pipeline(
    device: Target,
    figure_of_merit_name: figure_of_merit = "expected_fidelity",
    path_training_circuits: str | Path | None = None,
    path_test_circuits: str | Path | None = None,
) -> PipelineEvaluationResult:
    """Evaluate Qiskit's locked O3 preset pipeline on the held-out test split."""
    test_dir = resolve_test_directory(path_training_circuits, path_test_circuits)
    circuits = load_test_circuits(test_dir)
    pass_manager = generate_preset_pass_manager(optimization_level=3, target=device)
    evaluator = PredictorEnv(device=device, reward_function=figure_of_merit_name)

    results = [
        evaluate_qiskit_circuit(
            qc=qc,
            device=device,
            evaluator=evaluator,
            pass_manager=pass_manager,
        )
        for qc in circuits
    ]

    return PipelineEvaluationResult(
        pipeline_name="qiskit_o3",
        sdk_name="qiskit",
        sdk_version=LOCKED_QISKIT_VERSION,
        device_name=device.description,
        figure_of_merit=str(figure_of_merit_name),
        test_directory=test_dir,
        documented_passes=extract_qiskit_documented_passes(pass_manager),
        circuits=results,
        average_metrics=compute_average_metrics(results),
        action_effectiveness=compute_action_effectiveness_summary(results),
    )


def evaluate_tket_o2_pipeline(
    device: Target,
    figure_of_merit_name: figure_of_merit = "expected_fidelity",
    path_training_circuits: str | Path | None = None,
    path_test_circuits: str | Path | None = None,
) -> PipelineEvaluationResult:
    """Evaluate pytket-qiskit's locked IBM-style O2 pipeline on the held-out test split."""
    test_dir = resolve_test_directory(path_training_circuits, path_test_circuits)
    circuits = load_test_circuits(test_dir)
    evaluator = PredictorEnv(device=device, reward_function=figure_of_merit_name)
    pipeline_steps = build_tket_o2_pipeline(device)

    results = [
        evaluate_tket_circuit(
            qc=qc,
            device=device,
            evaluator=evaluator,
            pipeline_steps=pipeline_steps,
        )
        for qc in circuits
    ]

    return PipelineEvaluationResult(
        pipeline_name="tket_o2",
        sdk_name="pytket",
        sdk_version=LOCKED_PYTKET_VERSION,
        device_name=device.description,
        figure_of_merit=str(figure_of_merit_name),
        test_directory=test_dir,
        documented_passes=[step.name for step in pipeline_steps],
        circuits=results,
        average_metrics=compute_average_metrics(results),
        action_effectiveness=compute_action_effectiveness_summary(results),
    )


def resolve_test_directory(
    path_training_circuits: str | Path | None,
    path_test_circuits: str | Path | None,
) -> Path:
    """Resolve the held-out evaluation directory for pipeline baselines."""
    training_dir = (
        Path(path_training_circuits) if path_training_circuits is not None else get_path_training_circuits_train()
    )
    return resolve_test_circuit_directory(training_dir, path_test_circuits or get_path_training_circuits_test())


def evaluate_qiskit_circuit(
    qc: QuantumCircuit,
    device: Target,
    evaluator: PredictorEnv,
    pass_manager: PassManager,
) -> CircuitEvaluationResult:
    """Run Qiskit's preset O3 pipeline on one circuit and collect evaluation metrics."""
    snapshots: list[PipelineStepSnapshot] = [PipelineStepSnapshot(pass_name="__start__", circuit=qc.copy())]

    def callback(**kwargs: object) -> None:
        pass_ = cast("QiskitBasePass", kwargs["pass_"])
        dag = cast("DAGCircuit", kwargs["dag"])
        property_set = cast("PropertySet", kwargs["property_set"])
        circuit = dag_to_circuit(dag, copy_operations=True)
        layout = TranspileLayout.from_property_set(dag, property_set)
        snapshots.append(
            PipelineStepSnapshot(
                pass_name=pass_.name(),
                circuit=circuit,
                layout=copy.deepcopy(layout) if layout is not None else None,
            )
        )

    compiled_qc = pass_manager.run(qc.copy(), callback=callback)
    return summarize_pipeline_execution(
        snapshots=snapshots,
        final_qc=compiled_qc,
        evaluator=evaluator,
        device=device,
        circuit_name=qc.name or "<unnamed>",
    )


def evaluate_tket_circuit(
    qc: QuantumCircuit,
    device: Target,
    evaluator: PredictorEnv,
    pipeline_steps: list[TketPipelineStep],
) -> CircuitEvaluationResult:
    """Run pytket-qiskit's IBM-style O2 pipeline on one circuit and collect evaluation metrics."""
    tk_circuit = qiskit_to_tk(qc, preserve_param_uuid=True)
    snapshots: list[PipelineStepSnapshot] = [PipelineStepSnapshot(pass_name="__start__", circuit=qc.copy())]
    laid_out = False

    for step in pipeline_steps:
        step.pass_.apply(tk_circuit)
        if step.enables_layout_tracking:
            laid_out = True
        snapshots.append(
            PipelineStepSnapshot(
                pass_name=step.name,
                circuit=tk_to_qiskit(tk_circuit),
                laid_out_override=laid_out,
            )
        )

    final_qc = snapshots[-1].circuit
    return summarize_pipeline_execution(
        snapshots=snapshots,
        final_qc=final_qc,
        evaluator=evaluator,
        device=device,
        circuit_name=qc.name or "<unnamed>",
    )


def summarize_pipeline_execution(
    snapshots: list[PipelineStepSnapshot],
    final_qc: QuantumCircuit,
    evaluator: PredictorEnv,
    device: Target,
    circuit_name: str,
) -> CircuitEvaluationResult:
    """Convert a recorded pass-by-pass execution trace into the shared result structure."""
    step_states = [evaluate_snapshot(snapshot, evaluator, device) for snapshot in snapshots]
    used_passes = [snapshot.pass_name for snapshot in snapshots[1:]]
    effective_passes: list[str] = []
    effective_steps = 0

    for previous_snapshot, current_snapshot, previous_state, current_state in zip(
        snapshots[:-1],
        snapshots[1:],
        step_states[:-1],
        step_states[1:],
        strict=True,
    ):
        del previous_snapshot
        if is_effective_step(previous_state, current_state):
            effective_passes.append(current_snapshot.pass_name)
            effective_steps += 1

    figure_value, figure_kind = evaluate_final_figure_of_merit(final_qc, evaluator)
    return CircuitEvaluationResult(
        circuit_name=circuit_name,
        figure_of_merit_value=figure_value,
        figure_of_merit_kind=figure_kind,
        metrics=collect_final_metrics(final_qc, device),
        used_compilation_passes=used_passes,
        effective_compilation_passes=effective_passes,
        effective_steps=effective_steps,
        terminated=True,
        hit_step_limit=False,
        decision_trace=[],
    )


def evaluate_snapshot(
    snapshot: PipelineStepSnapshot,
    evaluator: PredictorEnv,
    device: Target,
) -> StepEvaluationState:
    """Compute reward and compilation-state flags for one intermediate pipeline state."""
    evaluator.state = snapshot.circuit
    evaluator.layout = snapshot.layout
    evaluator.compilation_state_flags = None

    try:
        reward_value, reward_kind = evaluator.calculate_reward(mode="auto")
    except PIPELINE_EVALUATION_ERRORS:
        reward_value, reward_kind = retry_snapshot_reward_with_unitary_synthesis(snapshot, evaluator, device)
    synthesized = evaluator.is_circuit_synthesized(snapshot.circuit)
    laid_out = determine_laid_out_state(snapshot, evaluator)
    routed = (
        evaluator.is_circuit_routed(snapshot.circuit, CouplingMap(device.build_coupling_map())) if laid_out else False
    )

    return StepEvaluationState(
        reward_value=reward_value,
        reward_kind=reward_kind,
        depth=int(snapshot.circuit.depth()),
        size=int(snapshot.circuit.size()),
        synthesized=synthesized,
        laid_out=laid_out,
        routed=routed,
    )


def retry_snapshot_reward_with_unitary_synthesis(
    snapshot: PipelineStepSnapshot,
    evaluator: PredictorEnv,
    device: Target,
) -> tuple[float, str]:
    """Retry intermediate reward evaluation on a temporary synthesized circuit.

    This is a local workaround for pipeline evaluation only. Some intermediate
    baseline states still contain opaque `unitary` instructions that the shared
    approximate-reward path cannot rebase with `BasisTranslator` alone.
    """
    try:
        reward_eval_circuit = PassManager([
            UnitarySynthesis(target=device),
            BasisTranslator(SessionEquivalenceLibrary, get_basis_gates_from_target(device)),
        ]).run(snapshot.circuit.copy())
        evaluator.state = reward_eval_circuit
        evaluator.layout = snapshot.layout
        evaluator.compilation_state_flags = None
        reward_value, reward_kind = evaluator.calculate_reward(mode="approx")
    except PIPELINE_EVALUATION_ERRORS:
        return math.nan, "unavailable"
    else:
        return reward_value, f"{reward_kind}_unitary_synthesis"


def determine_laid_out_state(snapshot: PipelineStepSnapshot, evaluator: PredictorEnv) -> bool:
    """Determine whether a snapshot should count as laid out."""
    if snapshot.laid_out_override is not None:
        return snapshot.laid_out_override
    if snapshot.layout is None:
        return False
    try:
        return evaluator.is_circuit_laid_out(snapshot.circuit, snapshot.layout)
    except PIPELINE_EVALUATION_ERRORS:
        return False


def is_effective_step(previous_state: StepEvaluationState, current_state: StepEvaluationState) -> bool:
    """Return whether a pass is effective under the RL evaluator's bookkeeping."""
    state_progress = any(
        not before and after
        for before, after in zip(
            (
                previous_state.synthesized,
                previous_state.laid_out,
                previous_state.routed,
            ),
            (
                current_state.synthesized,
                current_state.laid_out,
                current_state.routed,
            ),
            strict=True,
        )
    )
    if state_progress:
        return True
    if not math.isnan(previous_state.reward_value) and not math.isnan(current_state.reward_value):
        return current_state.reward_value > previous_state.reward_value + 1e-12
    return current_state.size < previous_state.size or current_state.depth < previous_state.depth


def evaluate_final_figure_of_merit(final_qc: QuantumCircuit, evaluator: PredictorEnv) -> tuple[float, str]:
    """Compute the final figure of merit, preferring exact evaluation for the terminal circuit."""
    evaluator.state = final_qc
    evaluator.layout = getattr(final_qc, "_layout", None)
    evaluator.compilation_state_flags = None
    try:
        return evaluator.calculate_reward(mode="exact")
    except PIPELINE_EVALUATION_ERRORS:
        return evaluator.calculate_reward(mode="auto")


def extract_qiskit_documented_passes(pass_manager: StagedPassManager) -> list[str]:
    """Return the documented pass order of Qiskit's staged O3 pipeline."""
    documented_passes: list[str] = []
    for stage_name in pass_manager.stages:
        stage_pm = getattr(pass_manager, stage_name)
        documented_passes.extend(extract_qiskit_task_names(stage_pm._tasks))  # noqa: SLF001
    return documented_passes


def extract_qiskit_task_names(tasks: list[object] | tuple[object, ...]) -> list[str]:
    """Recursively flatten Qiskit pass-manager tasks into pass names."""
    names: list[str] = []
    for task in tasks:
        if isinstance(task, list):
            names.extend(extract_qiskit_task_names(cast("list[object]", task)))
            continue
        child_tasks = getattr(task, "tasks", None)
        if child_tasks is not None:
            names.extend(extract_qiskit_task_names(list(child_tasks)))
            continue
        if isinstance(task, QiskitBasePass):
            names.append(task.name())
            continue
        names.append(type(task).__name__)
    return names


def build_tket_o2_pipeline(device: Target) -> list[TketPipelineStep]:
    """Construct pytket-qiskit's documented IBM-style O2 pipeline for the locked SDK version."""
    architecture = Architecture(list(device.build_coupling_map()))
    gate_set = {
        QISKIT_GATE_NAME_TO_OPTYPE[name] for name in device.operation_names if name in QISKIT_GATE_NAME_TO_OPTYPE
    }.union({OpType.Measure, OpType.Reset})
    primitive_gates = gate_set & (ALL_PRIMITIVE_1Q_GATES | ALL_PRIMITIVE_2Q_GATES)
    primitive_1q_gates = gate_set & ALL_PRIMITIVE_1Q_GATES

    return [
        TketPipelineStep("DecomposeBoxes", DecomposeBoxes()),
        TketPipelineStep("FullPeepholeOptimise", FullPeepholeOptimise()),
        TketPipelineStep("AutoRebase", AutoRebase(primitive_gates)),
        TketPipelineStep(
            "lightsabrepass",
            CustomPassMap(_gen_lightsabre_transformation(architecture), "lightsabrepass"),
            enables_layout_tracking=True,
        ),
        TketPipelineStep("KAKDecomposition", KAKDecomposition(allow_swaps=False)),
        TketPipelineStep("CliffordSimp", CliffordSimp(False)),
        TketPipelineStep("SynthesiseTket", SynthesiseTket()),
        TketPipelineStep("AutoRebase", AutoRebase(primitive_gates)),
        TketPipelineStep("AutoSquash", AutoSquash(primitive_1q_gates)),
        TketPipelineStep("RemoveRedundancies", RemoveRedundancies()),
    ]


def run_selected_pipelines(
    pipeline_names: list[str],
    device_name: str,
    figure_of_merit_name: figure_of_merit = "expected_fidelity",
    path_training_circuits: str | Path | None = None,
    path_test_circuits: str | Path | None = None,
) -> list[PipelineEvaluationResult]:
    """Evaluate the selected locked SDK pipelines."""
    device = get_device(device_name)
    results: list[PipelineEvaluationResult] = []

    for pipeline_name in pipeline_names:
        if pipeline_name == "qiskit_o3":
            results.append(
                evaluate_qiskit_o3_pipeline(
                    device=device,
                    figure_of_merit_name=figure_of_merit_name,
                    path_training_circuits=path_training_circuits,
                    path_test_circuits=path_test_circuits,
                )
            )
            continue
        if pipeline_name == "tket_o2":
            results.append(
                evaluate_tket_o2_pipeline(
                    device=device,
                    figure_of_merit_name=figure_of_merit_name,
                    path_training_circuits=path_training_circuits,
                    path_test_circuits=path_test_circuits,
                )
            )
            continue
        msg = f"Unsupported pipeline '{pipeline_name}'."
        raise ValueError(msg)

    return results


def print_summary(result: PipelineEvaluationResult) -> None:
    """Print a concise terminal summary for one evaluated pipeline."""
    avg = result.average_metrics
    print(f"Pipeline: {result.pipeline_name} ({result.sdk_name} {result.sdk_version})")
    print(f"Device: {result.device_name}")
    print(f"Figure of merit: {result.figure_of_merit}")
    print(f"Test directory: {result.test_directory}")
    print(f"Evaluated circuits: {len(result.circuits)}")
    print(f"Documented passes: {', '.join(result.documented_passes)}")
    print(
        "Average metrics:",
        f"expected_fidelity={format_optional_float(avg.expected_fidelity)}",
        f"estimated_success_probability={format_optional_float(avg.estimated_success_probability)}",
        f"depth={avg.depth}",
        f"size={avg.size}",
    )
    print(
        "Overall action effectiveness:",
        f"{result.action_effectiveness.total_effective_uses}/{result.action_effectiveness.total_uses}",
        f"({result.action_effectiveness.overall_effectiveness_ratio:.1%})",
    )
    print("Per-pass effectiveness:")
    for stats in result.action_effectiveness.per_action:
        print(f"  {stats.action_name}: {stats.effective_uses}/{stats.total_uses} ({stats.effectiveness_ratio:.1%})")
    print()


def format_optional_float(value: float | None) -> str:
    """Format optional floating-point metrics for the CLI summary."""
    if value is None or math.isnan(value):
        return "nan"
    return f"{value:.6f}"


def main() -> None:
    """Run the baseline pipeline evaluator from the command line."""
    parser = argparse.ArgumentParser(
        description="Evaluate the locked Qiskit/TKET compilation pipelines on the held-out RL test split."
    )
    parser.add_argument("--device", default="ibm_falcon_127", help="Target device name.")
    parser.add_argument(
        "--figure-of-merit",
        default="expected_fidelity",
        choices=["expected_fidelity", "critical_depth", "estimated_success_probability"],
        help="Figure of merit used for evaluation.",
    )
    parser.add_argument(
        "--pipelines",
        nargs="+",
        default=["qiskit_o3", "tket_o2"],
        choices=["qiskit_o3", "tket_o2"],
        help="Pipelines to evaluate.",
    )
    parser.add_argument("--train-dir", type=Path, default=None, help="Optional training-circuits directory.")
    parser.add_argument("--test-dir", type=Path, default=None, help="Optional test-circuits directory.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    results = run_selected_pipelines(
        pipeline_names=args.pipelines,
        device_name=args.device,
        figure_of_merit_name=args.figure_of_merit,
        path_training_circuits=args.train_dir,
        path_test_circuits=args.test_dir,
    )

    for result in results:
        print_summary(result)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps([result.to_dict() for result in results], indent=2), encoding="utf-8")
        print(f"Wrote JSON report to {args.output}")


if __name__ == "__main__":
    main()
