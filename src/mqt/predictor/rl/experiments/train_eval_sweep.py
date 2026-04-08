# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Orchestrate full RL training/evaluation sweeps using the existing experiment helpers.

This script intentionally builds on top of the existing experiment modules:

- ``training.py`` via :func:`run_rl_training`
- ``evaluation.py`` via :func:`evaluate_trained_predictor`
- ``pipeline_evaluation.py`` via :func:`run_selected_pipelines`

It adds the missing cluster-oriented concerns around those helpers:

- sweep planning over all supported RL configurations
- resumable execution
- per-configuration output directories
- atomic JSON result writes
- progress and status tracking
- unique model artifact copies so different MDP runs do not overwrite each other
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import platform
import shutil
import socket
import sys
import tempfile
import traceback
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mqt.bench.targets import get_device
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

from mqt.predictor.hellinger import get_hellinger_model_path
from mqt.predictor.reward import esp_data_available
from mqt.predictor.rl.experiments.evaluation import (
    PredictorEvaluationResult,
    evaluate_trained_predictor,
)
from mqt.predictor.rl.experiments.pipeline_evaluation import (
    PipelineEvaluationResult,
    run_selected_pipelines,
)
from mqt.predictor.rl.experiments.training import run_rl_training

if TYPE_CHECKING:
    from qiskit.transpiler import Target


ALL_MDP_VARIANTS = ("paper", "flexible", "thesis", "hybrid")
SUPPORTED_RL_FIGURES_OF_MERIT = (
    "expected_fidelity",
    "critical_depth",
    "estimated_success_probability",
    "estimated_hellinger_distance",
)
PIPELINE_COMPATIBLE_FIGURES_OF_MERIT = frozenset(
    {"expected_fidelity", "critical_depth", "estimated_success_probability"}
)
SUPPORTED_PIPELINES = ("qiskit_o3", "tket_o2")
EVALUATION_MODES = {
    "stochastic": False,
    "deterministic": True,
}


@dataclass(frozen=True, slots=True)
class SweepRunSpec:
    """One RL training/evaluation configuration."""

    figure_of_merit: str
    mdp: str

    @property
    def slug(self) -> str:
        """Return a filesystem-friendly identifier."""
        return f"{self.figure_of_merit}__{self.mdp}"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run a full RL training/evaluation sweep over the existing experiment helpers and "
            "store resumable, per-configuration artifacts."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", required=True, help="Target device name, e.g. ibm_falcon_127.")
    parser.add_argument(
        "--figures-of-merit",
        nargs="+",
        default=["auto"],
        choices=["auto", *SUPPORTED_RL_FIGURES_OF_MERIT],
        help=(
            "Figures of merit to train/evaluate. Use 'auto' to include every figure supported by the selected device."
        ),
    )
    parser.add_argument(
        "--mdps",
        nargs="+",
        default=list(ALL_MDP_VARIANTS),
        choices=list(ALL_MDP_VARIANTS),
        help="MDP transition policies to sweep.",
    )
    parser.add_argument(
        "--evaluation-modes",
        nargs="+",
        default=list(EVALUATION_MODES),
        choices=list(EVALUATION_MODES),
        help="Evaluation rollout modes to execute after training.",
    )
    parser.add_argument(
        "--pipelines",
        nargs="+",
        default=[],
        choices=list(SUPPORTED_PIPELINES),
        help="Optional locked pipeline baselines to evaluate once per figure of merit.",
    )
    parser.add_argument("--timesteps", type=int, default=100000, help="Training timesteps per RL configuration.")
    parser.add_argument(
        "--checkpoint-frequency",
        type=int,
        default=2048,
        help="How often to save PPO checkpoints during training.",
    )
    parser.add_argument("--train-verbose", type=int, default=1, help="Verbosity passed to PPO training.")
    parser.add_argument("--max-steps", type=int, default=200, help="Maximum rollout steps per evaluation circuit.")
    parser.add_argument("--seed", type=int, default=0, help="Evaluation seed.")
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=None,
        help="Optional training-circuit directory passed into the existing training/evaluation helpers.",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=None,
        help="Optional held-out test-circuit directory passed into the existing evaluation helpers.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for status, logs, copied models, and JSON reports. Defaults to a timestamped folder.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Optional run-name suffix used when auto-generating the output directory.",
    )
    parser.add_argument(
        "--test-training",
        action="store_true",
        help="Use the lightweight training mode from the existing training helper for smoke runs.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore completed status files and rerun configurations.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately on the first failed training or evaluation phase.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned sweep and exit without training or evaluation.",
    )
    args = parser.parse_args()

    if "auto" in args.figures_of_merit and len(args.figures_of_merit) > 1:
        parser.error("'auto' cannot be combined with explicit figures of merit.")

    return args


def now_utc_iso() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat()


def default_output_directory(device_name: str, run_name: str | None) -> Path:
    """Return the default output directory for a new sweep."""
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    suffix = f"_{run_name}" if run_name else ""
    return Path.cwd() / "rl_experiment_runs" / f"{timestamp}_{device_name}{suffix}"


def configure_logging(output_dir: Path) -> logging.Logger:
    """Configure a sweep-local logger."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("mqt-predictor.train-eval-sweep")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(output_dir / "orchestrator.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def resolve_supported_figures_of_merit(device: Target) -> list[str]:
    """Return every RL figure of merit supported for the selected device."""
    supported = ["expected_fidelity", "critical_depth"]

    if esp_data_available(device):
        supported.append("estimated_success_probability")

    if get_hellinger_model_path(device).is_file():
        supported.append("estimated_hellinger_distance")

    return supported


def resolve_figures_of_merit(device: Target, requested: list[str]) -> list[str]:
    """Resolve and validate the figure-of-merit sweep."""
    supported = resolve_supported_figures_of_merit(device)

    if requested == ["auto"]:
        return supported

    missing_support: list[str] = []
    for figure_of_merit in requested:
        if figure_of_merit not in supported:
            missing_support.append(figure_of_merit)

    if missing_support:
        supported_values = ", ".join(supported)
        requested_values = ", ".join(missing_support)
        msg = (
            f"Requested unsupported figures of merit for {device.description}: {requested_values}. "
            f"Supported values are: {supported_values}."
        )
        raise ValueError(msg)

    return requested


def build_run_specs(figures_of_merit: list[str], mdps: list[str]) -> list[SweepRunSpec]:
    """Return the full RL sweep plan."""
    return [SweepRunSpec(figure_of_merit=figure_of_merit, mdp=mdp) for figure_of_merit in figures_of_merit for mdp in mdps]


def build_run_directory(output_dir: Path, spec: SweepRunSpec) -> Path:
    """Return the directory for one RL run."""
    return output_dir / "runs" / spec.slug


def build_pipeline_directory(output_dir: Path, figure_of_merit: str, pipeline_name: str) -> Path:
    """Return the directory for one pipeline-baseline run."""
    return output_dir / "pipeline_runs" / f"{figure_of_merit}__{pipeline_name}"


def empty_run_status(spec: SweepRunSpec, evaluation_modes: list[str]) -> dict[str, Any]:
    """Return the initial status document for one RL run."""
    return {
        "configuration": asdict(spec),
        "created_at": now_utc_iso(),
        "updated_at": now_utc_iso(),
        "status": "pending",
        "training": {"status": "pending"},
        "evaluations": {mode: {"status": "pending"} for mode in evaluation_modes},
    }


def empty_pipeline_status(figure_of_merit: str, pipeline_name: str) -> dict[str, Any]:
    """Return the initial status document for one baseline pipeline run."""
    return {
        "configuration": {"figure_of_merit": figure_of_merit, "pipeline": pipeline_name},
        "created_at": now_utc_iso(),
        "updated_at": now_utc_iso(),
        "status": "pending",
        "result": {"status": "pending"},
    }


def sanitize_json_value(value: object) -> object:
    """Convert objects into JSON-compatible values."""
    if is_dataclass(value):
        return sanitize_json_value(asdict(value))

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {str(key): sanitize_json_value(val) for key, val in value.items()}

    if isinstance(value, list | tuple):
        return [sanitize_json_value(item) for item in value]

    if isinstance(value, set | frozenset):
        return [sanitize_json_value(item) for item in sorted(value)]

    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return sanitize_json_value(value.item())
        except (TypeError, ValueError):
            pass

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    return value


def atomic_write_json(path: Path, payload: object) -> None:
    """Atomically write JSON to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f"{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(sanitize_json_value(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temp_path = Path(handle.name)

    temp_path.replace(path)


def append_jsonl(path: Path, payload: object) -> None:
    """Append one JSONL event to disk and flush it eagerly."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(sanitize_json_value(payload), sort_keys=True))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def copy_file_atomically(source: Path, destination: Path) -> Path:
    """Copy a file into place atomically."""
    if not source.is_file():
        msg = f"Expected artifact '{source}' does not exist."
        raise FileNotFoundError(msg)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb",
        dir=destination.parent,
        prefix=f"{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)

    shutil.copy2(source, temp_path)
    temp_path.replace(destination)
    return destination


def load_status(path: Path, fallback: dict[str, Any]) -> dict[str, Any]:
    """Load a status file when present, otherwise return a copy of the fallback."""
    if not path.is_file():
        return json.loads(json.dumps(fallback))

    return json.loads(path.read_text(encoding="utf-8"))


def update_run_status(path: Path, status: dict[str, Any], evaluation_modes: list[str]) -> None:
    """Update the run-level status field and persist the document."""
    training_status = status["training"]["status"]
    evaluation_statuses = [status["evaluations"][mode]["status"] for mode in evaluation_modes]

    if training_status == "failed" or any(phase_status == "failed" for phase_status in evaluation_statuses):
        overall = "failed"
    elif training_status == "completed" and all(phase_status == "completed" for phase_status in evaluation_statuses):
        overall = "completed"
    elif training_status == "running" or any(phase_status == "running" for phase_status in evaluation_statuses):
        overall = "running"
    elif training_status == "completed" and any(
        phase_status in {"pending", "running"} for phase_status in evaluation_statuses
    ):
        overall = "partial"
    else:
        overall = "pending"

    status["status"] = overall
    status["updated_at"] = now_utc_iso()
    atomic_write_json(path, status)


def update_pipeline_status(path: Path, status: dict[str, Any]) -> None:
    """Update and persist one pipeline-baseline status document."""
    status["status"] = status["result"]["status"]
    status["updated_at"] = now_utc_iso()
    atomic_write_json(path, status)


def checkpoint_sort_key(path: Path) -> int:
    """Return the timestep encoded in a checkpoint filename."""
    for part in reversed(path.stem.split("_")):
        if part.isdigit():
            return int(part)
    return -1


def find_latest_checkpoint(checkpoint_directory: Path) -> Path | None:
    """Return the most recent PPO checkpoint in a directory."""
    if not checkpoint_directory.is_dir():
        return None

    checkpoint_files = sorted(
        checkpoint_directory.glob("model_checkpoint_*_steps.zip"),
        key=checkpoint_sort_key,
    )
    if not checkpoint_files:
        return None
    return checkpoint_files[-1]


def get_model_num_timesteps(model_path: Path) -> int:
    """Load a PPO checkpoint and return the completed timesteps."""
    model = MaskablePPO.load(model_path)
    return int(model.num_timesteps)


class TrainingHeartbeatCallback(BaseCallback):
    """Persist training heartbeat information during long PPO runs."""

    def __init__(
        self,
        *,
        spec: SweepRunSpec,
        target_timesteps: int,
        status: dict[str, Any],
        status_path: Path,
        evaluation_modes: list[str],
        event_log_path: Path,
        logger: logging.Logger,
        log_frequency: int,
    ) -> None:
        """Initialize the callback."""
        super().__init__()
        self.spec = spec
        self.target_timesteps = target_timesteps
        self.status = status
        self.status_path = status_path
        self.evaluation_modes = evaluation_modes
        self.event_log_path = event_log_path
        self.logger = logger
        self.log_frequency = max(1, log_frequency)
        self.last_logged_timesteps = 0

    def _on_step(self) -> bool:
        current_timesteps = int(self.model.num_timesteps)
        if current_timesteps - self.last_logged_timesteps < self.log_frequency:
            return True

        self.last_logged_timesteps = current_timesteps
        self.status["training"]["completed_timesteps"] = current_timesteps
        self.status["training"]["last_progress_at"] = now_utc_iso()
        update_run_status(self.status_path, self.status, self.evaluation_modes)

        progress_ratio = current_timesteps / self.target_timesteps if self.target_timesteps > 0 else 1.0
        self.logger.info(
            "Training progress for %s: %d/%d timesteps (%.1f%%).",
            self.spec.slug,
            current_timesteps,
            self.target_timesteps,
            progress_ratio * 100.0,
        )
        append_jsonl(
            self.event_log_path,
            {
                "timestamp": now_utc_iso(),
                "event": "training_progress",
                "configuration": asdict(self.spec),
                "completed_timesteps": current_timesteps,
                "target_timesteps": self.target_timesteps,
            },
        )
        return True


def summarize_predictor_evaluation(result: PredictorEvaluationResult) -> dict[str, Any]:
    """Return a compact summary for quick inspection and CSV reporting."""
    figure_values = [circuit.figure_of_merit_value for circuit in result.circuits]
    figure_kinds = sorted({circuit.figure_of_merit_kind for circuit in result.circuits})

    average_figure_value = None
    if figure_values:
        average_figure_value = sum(figure_values) / len(figure_values)

    return {
        "evaluated_circuits": len(result.circuits),
        "test_directory": str(result.test_directory),
        "average_figure_of_merit_value": average_figure_value,
        "figure_of_merit_kinds": figure_kinds,
        "step_limit_hits": sum(1 for circuit in result.circuits if circuit.hit_step_limit),
        "average_metrics": sanitize_json_value(result.average_metrics),
        "feature_importance": {
            "baseline_mean_reward": result.feature_importance.baseline_mean_reward,
            "average_original_feature_importance": result.feature_importance.average_original_feature_importance,
            "average_gate_count_feature_importance": result.feature_importance.average_gate_count_feature_importance,
        },
        "action_effectiveness": {
            "total_uses": result.action_effectiveness.total_uses,
            "total_effective_uses": result.action_effectiveness.total_effective_uses,
            "overall_effectiveness_ratio": result.action_effectiveness.overall_effectiveness_ratio,
        },
    }


def summarize_pipeline_evaluation(result: PipelineEvaluationResult) -> dict[str, Any]:
    """Return a compact summary for one pipeline baseline run."""
    figure_values = [circuit.figure_of_merit_value for circuit in result.circuits]
    average_figure_value = None
    if figure_values:
        average_figure_value = sum(figure_values) / len(figure_values)

    return {
        "pipeline_name": result.pipeline_name,
        "sdk_name": result.sdk_name,
        "sdk_version": result.sdk_version,
        "device_name": result.device_name,
        "figure_of_merit": result.figure_of_merit,
        "test_directory": str(result.test_directory),
        "evaluated_circuits": len(result.circuits),
        "average_figure_of_merit_value": average_figure_value,
        "average_metrics": sanitize_json_value(result.average_metrics),
        "action_effectiveness": {
            "total_uses": result.action_effectiveness.total_uses,
            "total_effective_uses": result.action_effectiveness.total_effective_uses,
            "overall_effectiveness_ratio": result.action_effectiveness.overall_effectiveness_ratio,
        },
    }


@contextmanager
def working_directory(path: Path) -> Any:
    """Temporarily change the working directory."""
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def run_training_phase(
    *,
    spec: SweepRunSpec,
    status_path: Path,
    status: dict[str, Any],
    run_dir: Path,
    args: argparse.Namespace,
    device: Target,
    logger: logging.Logger,
    event_log_path: Path,
) -> Path:
    """Execute or resume one training phase and return the copied model path."""
    model_copy_path = run_dir / "artifacts" / f"model_{spec.slug}_{device.description}.zip"
    training_result_path = run_dir / "training_result.json"
    checkpoint_directory = run_dir / "checkpoints"

    if (
        not args.force
        and status["training"]["status"] == "completed"
        and model_copy_path.is_file()
        and training_result_path.is_file()
    ):
        logger.info("Skipping completed training for %s.", spec.slug)
        return model_copy_path

    checkpoint_path: Path | None = None
    completed_timesteps = 0
    if not args.force:
        checkpoint_path = find_latest_checkpoint(checkpoint_directory)
        if checkpoint_path is not None:
            completed_timesteps = get_model_num_timesteps(checkpoint_path)

    if completed_timesteps >= args.timesteps:
        logger.info(
            "Training checkpoints for %s already reached %d/%d timesteps. Finalizing from checkpoint.",
            spec.slug,
            completed_timesteps,
            args.timesteps,
        )
        copied_model_path = copy_file_atomically(checkpoint_path, model_copy_path)
        training_payload = {
            "completed_at": now_utc_iso(),
            "configuration": asdict(spec),
            "device": device.description,
            "training_directory": status["training"].get("training_directory", str(args.train_dir or "")),
            "source_model_path": str(checkpoint_path),
            "saved_model_path": str(copied_model_path),
            "timesteps": args.timesteps,
            "completed_timesteps": completed_timesteps,
            "verbose": args.train_verbose,
            "test_training": args.test_training,
            "checkpoint_directory": str(checkpoint_directory),
            "resumed_from_checkpoint": str(checkpoint_path),
        }
        atomic_write_json(training_result_path, training_payload)
        status["training"] = {
            "status": "completed",
            "started_at": status["training"].get("started_at", now_utc_iso()),
            "finished_at": now_utc_iso(),
            "timesteps": args.timesteps,
            "completed_timesteps": completed_timesteps,
            "verbose": args.train_verbose,
            "test_training": args.test_training,
            "training_directory": status["training"].get("training_directory", str(args.train_dir or "")),
            "source_model_path": str(checkpoint_path),
            "saved_model_path": str(copied_model_path),
            "checkpoint_directory": str(checkpoint_directory),
            "resumed_from_checkpoint": str(checkpoint_path),
            "result_path": str(training_result_path),
        }
        update_run_status(status_path, status, args.evaluation_modes)
        return copied_model_path

    remaining_timesteps = args.timesteps - completed_timesteps
    if checkpoint_path is None:
        logger.info("Training %s for %s timesteps.", spec.slug, args.timesteps)
    else:
        logger.info(
            "Resuming training for %s from checkpoint %s at %d/%d timesteps.",
            spec.slug,
            checkpoint_path.name,
            completed_timesteps,
            args.timesteps,
        )
    append_jsonl(
        event_log_path,
        {
            "timestamp": now_utc_iso(),
            "event": "training_started",
            "configuration": asdict(spec),
            "timesteps": args.timesteps,
            "remaining_timesteps": remaining_timesteps,
            "resumed_from_checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        },
    )

    status["training"] = {
        "status": "running",
        "started_at": status["training"].get("started_at", now_utc_iso()),
        "timesteps": args.timesteps,
        "completed_timesteps": completed_timesteps,
        "verbose": args.train_verbose,
        "test_training": args.test_training,
        "checkpoint_directory": str(checkpoint_directory),
        "checkpoint_frequency": args.checkpoint_frequency,
        "resumed_from_checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
    }
    update_run_status(status_path, status, args.evaluation_modes)

    try:
        heartbeat_callback = TrainingHeartbeatCallback(
            spec=spec,
            target_timesteps=args.timesteps,
            status=status,
            status_path=status_path,
            evaluation_modes=args.evaluation_modes,
            event_log_path=event_log_path,
            logger=logger,
            log_frequency=args.checkpoint_frequency,
        )
        with working_directory(run_dir):
            training_result = run_rl_training(
                device=device,
                figure_of_merit=spec.figure_of_merit,
                mdp=spec.mdp,
                timesteps=remaining_timesteps,
                verbose=args.train_verbose,
                test=args.test_training,
                path_training_circuits=args.train_dir,
                checkpoint_directory=checkpoint_directory,
                checkpoint_frequency=args.checkpoint_frequency,
                resume_from_checkpoint=checkpoint_path,
                callback=heartbeat_callback,
            )

        copied_model_path = copy_file_atomically(training_result.model_path, model_copy_path)
        training_payload = {
            "completed_at": now_utc_iso(),
            "configuration": asdict(spec),
            "device": device.description,
            "training_directory": str(training_result.training_directory),
            "source_model_path": str(training_result.model_path),
            "saved_model_path": str(copied_model_path),
            "timesteps": args.timesteps,
            "completed_timesteps": training_result.total_timesteps,
            "verbose": args.train_verbose,
            "test_training": args.test_training,
            "checkpoint_directory": str(checkpoint_directory),
            "resumed_from_checkpoint": (
                str(training_result.resumed_from_checkpoint) if training_result.resumed_from_checkpoint is not None else None
            ),
        }
        atomic_write_json(training_result_path, training_payload)

        status["training"] = {
            "status": "completed",
            "started_at": status["training"]["started_at"],
            "finished_at": now_utc_iso(),
            "timesteps": args.timesteps,
            "completed_timesteps": training_result.total_timesteps,
            "verbose": args.train_verbose,
            "test_training": args.test_training,
            "training_directory": str(training_result.training_directory),
            "source_model_path": str(training_result.model_path),
            "saved_model_path": str(copied_model_path),
            "checkpoint_directory": str(checkpoint_directory),
            "checkpoint_frequency": args.checkpoint_frequency,
            "resumed_from_checkpoint": (
                str(training_result.resumed_from_checkpoint) if training_result.resumed_from_checkpoint is not None else None
            ),
            "result_path": str(training_result_path),
        }
        update_run_status(status_path, status, args.evaluation_modes)
        append_jsonl(
            event_log_path,
            {
                "timestamp": now_utc_iso(),
                "event": "training_completed",
                "configuration": asdict(spec),
                "saved_model_path": str(copied_model_path),
                "completed_timesteps": training_result.total_timesteps,
            },
        )
        return copied_model_path
    except Exception as exc:  # noqa: BLE001
        status["training"] = {
            "status": "failed",
            "started_at": status["training"]["started_at"],
            "finished_at": now_utc_iso(),
            "timesteps": args.timesteps,
            "completed_timesteps": status["training"].get("completed_timesteps", completed_timesteps),
            "checkpoint_directory": str(checkpoint_directory),
            "checkpoint_frequency": args.checkpoint_frequency,
            "resumed_from_checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        update_run_status(status_path, status, args.evaluation_modes)
        append_jsonl(
            event_log_path,
            {
                "timestamp": now_utc_iso(),
                "event": "training_failed",
                "configuration": asdict(spec),
                "error": str(exc),
            },
        )
        raise


def run_evaluation_phase(
    *,
    spec: SweepRunSpec,
    mode_name: str,
    deterministic: bool,
    status_path: Path,
    status: dict[str, Any],
    run_dir: Path,
    args: argparse.Namespace,
    device: Target,
    model_path: Path,
    logger: logging.Logger,
    event_log_path: Path,
) -> None:
    """Execute or resume one evaluation phase."""
    evaluation_result_path = run_dir / "evaluations" / f"{mode_name}.json"

    if (
        not args.force
        and status["evaluations"][mode_name]["status"] == "completed"
        and evaluation_result_path.is_file()
    ):
        logger.info("Skipping completed %s evaluation for %s.", mode_name, spec.slug)
        return

    logger.info("Evaluating %s with %s rollout.", spec.slug, mode_name)
    append_jsonl(
        event_log_path,
        {
            "timestamp": now_utc_iso(),
            "event": "evaluation_started",
            "configuration": asdict(spec),
            "evaluation_mode": mode_name,
        },
    )

    status["evaluations"][mode_name] = {
        "status": "running",
        "started_at": now_utc_iso(),
        "deterministic": deterministic,
        "seed": args.seed,
        "max_steps": args.max_steps,
    }
    update_run_status(status_path, status, args.evaluation_modes)

    try:
        evaluation_result = evaluate_trained_predictor(
            model_path=model_path,
            device=device,
            figure_of_merit=spec.figure_of_merit,
            mdp=spec.mdp,
            path_training_circuits=args.train_dir,
            path_test_circuits=args.test_dir,
            max_steps=args.max_steps,
            deterministic=deterministic,
            seed=args.seed,
        )
        evaluation_payload = {
            "completed_at": now_utc_iso(),
            "configuration": asdict(spec),
            "device": device.description,
            "evaluation_mode": mode_name,
            "deterministic": deterministic,
            "seed": args.seed,
            "max_steps": args.max_steps,
            "model_path": str(model_path),
            "result": evaluation_result,
        }
        atomic_write_json(evaluation_result_path, evaluation_payload)

        status["evaluations"][mode_name] = {
            "status": "completed",
            "started_at": status["evaluations"][mode_name]["started_at"],
            "finished_at": now_utc_iso(),
            "deterministic": deterministic,
            "seed": args.seed,
            "max_steps": args.max_steps,
            "result_path": str(evaluation_result_path),
            "summary": summarize_predictor_evaluation(evaluation_result),
        }
        update_run_status(status_path, status, args.evaluation_modes)
        append_jsonl(
            event_log_path,
            {
                "timestamp": now_utc_iso(),
                "event": "evaluation_completed",
                "configuration": asdict(spec),
                "evaluation_mode": mode_name,
                "result_path": str(evaluation_result_path),
            },
        )
    except Exception as exc:  # noqa: BLE001
        status["evaluations"][mode_name] = {
            "status": "failed",
            "started_at": status["evaluations"][mode_name]["started_at"],
            "finished_at": now_utc_iso(),
            "deterministic": deterministic,
            "seed": args.seed,
            "max_steps": args.max_steps,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        update_run_status(status_path, status, args.evaluation_modes)
        append_jsonl(
            event_log_path,
            {
                "timestamp": now_utc_iso(),
                "event": "evaluation_failed",
                "configuration": asdict(spec),
                "evaluation_mode": mode_name,
                "error": str(exc),
            },
        )
        raise


def run_pipeline_phase(
    *,
    figure_of_merit: str,
    pipeline_name: str,
    output_dir: Path,
    args: argparse.Namespace,
    logger: logging.Logger,
    event_log_path: Path,
) -> None:
    """Execute or resume one baseline pipeline evaluation."""
    pipeline_dir = build_pipeline_directory(output_dir, figure_of_merit, pipeline_name)
    status_path = pipeline_dir / "status.json"
    result_path = pipeline_dir / "result.json"
    pipeline_dir.mkdir(parents=True, exist_ok=True)

    status = load_status(status_path, empty_pipeline_status(figure_of_merit, pipeline_name))

    if figure_of_merit not in PIPELINE_COMPATIBLE_FIGURES_OF_MERIT:
        logger.info(
            "Skipping pipeline baseline %s for %s because the figure of merit is not supported there.",
            pipeline_name,
            figure_of_merit,
        )
        status["result"] = {
            "status": "skipped",
            "finished_at": now_utc_iso(),
            "reason": (
                f"Pipeline baselines are only implemented for: "
                f"{', '.join(sorted(PIPELINE_COMPATIBLE_FIGURES_OF_MERIT))}."
            ),
        }
        update_pipeline_status(status_path, status)
        return

    if not args.force and status["result"]["status"] == "completed" and result_path.is_file():
        logger.info("Skipping completed pipeline baseline %s for %s.", pipeline_name, figure_of_merit)
        return

    logger.info("Evaluating pipeline baseline %s for %s.", pipeline_name, figure_of_merit)
    append_jsonl(
        event_log_path,
        {
            "timestamp": now_utc_iso(),
            "event": "pipeline_started",
            "figure_of_merit": figure_of_merit,
            "pipeline": pipeline_name,
        },
    )

    status["result"] = {
        "status": "running",
        "started_at": now_utc_iso(),
    }
    update_pipeline_status(status_path, status)

    try:
        results = run_selected_pipelines(
            pipeline_names=[pipeline_name],
            device_name=args.device,
            figure_of_merit_name=figure_of_merit,
            path_training_circuits=args.train_dir,
            path_test_circuits=args.test_dir,
        )
        if len(results) != 1:
            msg = f"Expected exactly one pipeline result for '{pipeline_name}', received {len(results)}."
            raise RuntimeError(msg)

        pipeline_result = results[0]
        result_payload = {
            "completed_at": now_utc_iso(),
            "configuration": {"figure_of_merit": figure_of_merit, "pipeline": pipeline_name},
            "result": pipeline_result,
        }
        atomic_write_json(result_path, result_payload)

        status["result"] = {
            "status": "completed",
            "started_at": status["result"]["started_at"],
            "finished_at": now_utc_iso(),
            "result_path": str(result_path),
            "summary": summarize_pipeline_evaluation(pipeline_result),
        }
        update_pipeline_status(status_path, status)
        append_jsonl(
            event_log_path,
            {
                "timestamp": now_utc_iso(),
                "event": "pipeline_completed",
                "figure_of_merit": figure_of_merit,
                "pipeline": pipeline_name,
                "result_path": str(result_path),
            },
        )
    except Exception as exc:  # noqa: BLE001
        status["result"] = {
            "status": "failed",
            "started_at": status["result"]["started_at"],
            "finished_at": now_utc_iso(),
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        update_pipeline_status(status_path, status)
        append_jsonl(
            event_log_path,
            {
                "timestamp": now_utc_iso(),
                "event": "pipeline_failed",
                "figure_of_merit": figure_of_merit,
                "pipeline": pipeline_name,
                "error": str(exc),
            },
        )
        raise


def write_global_progress(
    *,
    output_dir: Path,
    specs: list[SweepRunSpec],
    evaluation_modes: list[str],
    pipeline_figures: list[str],
    pipeline_names: list[str],
) -> dict[str, Any]:
    """Write a global progress snapshot."""
    training_completed = 0
    training_failed = 0
    evaluation_completed = 0
    evaluation_failed = 0
    run_completed = 0
    run_failed = 0

    for spec in specs:
        status_path = build_run_directory(output_dir, spec) / "status.json"
        status = load_status(status_path, empty_run_status(spec, evaluation_modes))

        if status["training"]["status"] == "completed":
            training_completed += 1
        elif status["training"]["status"] == "failed":
            training_failed += 1

        for mode_name in evaluation_modes:
            phase_status = status["evaluations"][mode_name]["status"]
            if phase_status == "completed":
                evaluation_completed += 1
            elif phase_status == "failed":
                evaluation_failed += 1

        if status["status"] == "completed":
            run_completed += 1
        elif status["status"] == "failed":
            run_failed += 1

    pipeline_completed = 0
    pipeline_failed = 0
    pipeline_skipped = 0
    for figure_of_merit in pipeline_figures:
        for pipeline_name in pipeline_names:
            status_path = build_pipeline_directory(output_dir, figure_of_merit, pipeline_name) / "status.json"
            status = load_status(status_path, empty_pipeline_status(figure_of_merit, pipeline_name))
            phase_status = status["result"]["status"]
            if phase_status == "completed":
                pipeline_completed += 1
            elif phase_status == "failed":
                pipeline_failed += 1
            elif phase_status == "skipped":
                pipeline_skipped += 1

    progress_payload = {
        "updated_at": now_utc_iso(),
        "training_runs_total": len(specs),
        "training_runs_completed": training_completed,
        "training_runs_failed": training_failed,
        "evaluation_runs_total": len(specs) * len(evaluation_modes),
        "evaluation_runs_completed": evaluation_completed,
        "evaluation_runs_failed": evaluation_failed,
        "rl_runs_completed": run_completed,
        "rl_runs_failed": run_failed,
        "pipeline_runs_total": len(pipeline_figures) * len(pipeline_names),
        "pipeline_runs_completed": pipeline_completed,
        "pipeline_runs_failed": pipeline_failed,
        "pipeline_runs_skipped": pipeline_skipped,
    }
    atomic_write_json(output_dir / "progress.json", progress_payload)
    return progress_payload


def log_progress_snapshot(logger: logging.Logger, progress_payload: dict[str, Any]) -> None:
    """Emit a concise progress line to stdout/stderr."""
    logger.info(
        (
            "Progress: training %s/%s completed, evaluations %s/%s completed, "
            "pipeline baselines %s/%s completed, failed rl runs=%s, failed evaluations=%s, failed pipelines=%s."
        ),
        progress_payload["training_runs_completed"],
        progress_payload["training_runs_total"],
        progress_payload["evaluation_runs_completed"],
        progress_payload["evaluation_runs_total"],
        progress_payload["pipeline_runs_completed"],
        progress_payload["pipeline_runs_total"],
        progress_payload["rl_runs_failed"],
        progress_payload["evaluation_runs_failed"],
        progress_payload["pipeline_runs_failed"],
    )


def write_summary_csv(
    *,
    output_dir: Path,
    specs: list[SweepRunSpec],
    evaluation_modes: list[str],
    pipeline_figures: list[str],
    pipeline_names: list[str],
) -> None:
    """Write a flat CSV summary over completed and partial runs."""
    rows: list[dict[str, Any]] = []

    for spec in specs:
        status_path = build_run_directory(output_dir, spec) / "status.json"
        status = load_status(status_path, empty_run_status(spec, evaluation_modes))
        training = status["training"]

        for mode_name in evaluation_modes:
            evaluation = status["evaluations"][mode_name]
            summary = evaluation.get("summary", {})
            average_metrics = summary.get("average_metrics", {})
            action_effectiveness = summary.get("action_effectiveness", {})
            feature_importance = summary.get("feature_importance", {})
            rows.append(
                {
                    "run_type": "rl",
                    "figure_of_merit": spec.figure_of_merit,
                    "mdp": spec.mdp,
                    "evaluation_mode": mode_name,
                    "pipeline": "",
                    "run_status": status["status"],
                    "training_status": training["status"],
                    "evaluation_status": evaluation["status"],
                    "saved_model_path": training.get("saved_model_path", ""),
                    "result_path": evaluation.get("result_path", ""),
                    "evaluated_circuits": summary.get("evaluated_circuits", ""),
                    "average_figure_of_merit_value": summary.get("average_figure_of_merit_value", ""),
                    "average_expected_fidelity": average_metrics.get("expected_fidelity", ""),
                    "average_estimated_success_probability": average_metrics.get(
                        "estimated_success_probability", ""
                    ),
                    "average_depth": average_metrics.get("depth", ""),
                    "average_size": average_metrics.get("size", ""),
                    "overall_effectiveness_ratio": action_effectiveness.get("overall_effectiveness_ratio", ""),
                    "average_original_feature_importance": feature_importance.get(
                        "average_original_feature_importance", ""
                    ),
                    "average_gate_count_feature_importance": feature_importance.get(
                        "average_gate_count_feature_importance", ""
                    ),
                }
            )

    for figure_of_merit in pipeline_figures:
        for pipeline_name in pipeline_names:
            status_path = build_pipeline_directory(output_dir, figure_of_merit, pipeline_name) / "status.json"
            status = load_status(status_path, empty_pipeline_status(figure_of_merit, pipeline_name))
            summary = status["result"].get("summary", {})
            average_metrics = summary.get("average_metrics", {})
            action_effectiveness = summary.get("action_effectiveness", {})
            rows.append(
                {
                    "run_type": "pipeline",
                    "figure_of_merit": figure_of_merit,
                    "mdp": "",
                    "evaluation_mode": "",
                    "pipeline": pipeline_name,
                    "run_status": status["status"],
                    "training_status": "",
                    "evaluation_status": status["result"]["status"],
                    "saved_model_path": "",
                    "result_path": status["result"].get("result_path", ""),
                    "evaluated_circuits": summary.get("evaluated_circuits", ""),
                    "average_figure_of_merit_value": summary.get("average_figure_of_merit_value", ""),
                    "average_expected_fidelity": average_metrics.get("expected_fidelity", ""),
                    "average_estimated_success_probability": average_metrics.get(
                        "estimated_success_probability", ""
                    ),
                    "average_depth": average_metrics.get("depth", ""),
                    "average_size": average_metrics.get("size", ""),
                    "overall_effectiveness_ratio": action_effectiveness.get("overall_effectiveness_ratio", ""),
                    "average_original_feature_importance": "",
                    "average_gate_count_feature_importance": "",
                }
            )

    summary_path = output_dir / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_type",
        "figure_of_merit",
        "mdp",
        "evaluation_mode",
        "pipeline",
        "run_status",
        "training_status",
        "evaluation_status",
        "saved_model_path",
        "result_path",
        "evaluated_circuits",
        "average_figure_of_merit_value",
        "average_expected_fidelity",
        "average_estimated_success_probability",
        "average_depth",
        "average_size",
        "overall_effectiveness_ratio",
        "average_original_feature_importance",
        "average_gate_count_feature_importance",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    figures_of_merit: list[str],
    specs: list[SweepRunSpec],
) -> None:
    """Write a static manifest for the planned sweep."""
    manifest_payload = {
        "created_at": now_utc_iso(),
        "device": args.device,
        "figures_of_merit": figures_of_merit,
        "mdps": args.mdps,
        "evaluation_modes": args.evaluation_modes,
        "pipelines": args.pipelines,
        "timesteps": args.timesteps,
        "checkpoint_frequency": args.checkpoint_frequency,
        "train_verbose": args.train_verbose,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "train_dir": args.train_dir,
        "test_dir": args.test_dir,
        "output_dir": output_dir,
        "test_training": args.test_training,
        "force": args.force,
        "python_version": platform.python_version(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "argv": sys.argv,
        "planned_rl_runs": [asdict(spec) for spec in specs],
    }
    atomic_write_json(output_dir / "manifest.json", manifest_payload)


def print_plan(
    *,
    output_dir: Path,
    figures_of_merit: list[str],
    specs: list[SweepRunSpec],
    args: argparse.Namespace,
) -> None:
    """Print the planned sweep to stdout."""
    print(f"Output directory: {output_dir}")
    print(f"Device: {args.device}")
    print(f"Figures of merit: {', '.join(figures_of_merit)}")
    print(f"MDPs: {', '.join(args.mdps)}")
    print(f"Evaluation modes: {', '.join(args.evaluation_modes)}")
    print(f"Pipelines: {', '.join(args.pipelines) if args.pipelines else 'none'}")
    print(f"Training timesteps per RL configuration: {args.timesteps}")
    print(f"Checkpoint frequency: {args.checkpoint_frequency}")
    print(f"Planned RL runs: {len(specs)}")
    for index, spec in enumerate(specs, start=1):
        print(f"  [{index:02d}] {spec.slug}")


def main() -> None:
    """Run the full sweep."""
    args = parse_args()
    device = get_device(args.device)
    figures_of_merit = resolve_figures_of_merit(device, args.figures_of_merit)
    specs = build_run_specs(figures_of_merit, args.mdps)
    output_dir = args.output_dir or default_output_directory(args.device, args.name)

    if args.dry_run:
        print_plan(output_dir=output_dir, figures_of_merit=figures_of_merit, specs=specs, args=args)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(output_dir)
    event_log_path = output_dir / "events.jsonl"

    write_manifest(output_dir=output_dir, args=args, figures_of_merit=figures_of_merit, specs=specs)
    append_jsonl(
        event_log_path,
        {
            "timestamp": now_utc_iso(),
            "event": "sweep_started",
            "device": args.device,
            "figures_of_merit": figures_of_merit,
            "mdps": args.mdps,
            "evaluation_modes": args.evaluation_modes,
            "pipelines": args.pipelines,
            "timesteps": args.timesteps,
            "output_dir": str(output_dir),
        },
    )
    logger.info(
        "Starting RL sweep for device=%s with %d training configurations and %d evaluation phases.",
        args.device,
        len(specs),
        len(specs) * len(args.evaluation_modes),
    )
    try:
        for index, spec in enumerate(specs, start=1):
            run_dir = build_run_directory(output_dir, spec)
            run_dir.mkdir(parents=True, exist_ok=True)

            config_payload = {
                "configuration": asdict(spec),
                "device": args.device,
                "timesteps": args.timesteps,
                "checkpoint_frequency": args.checkpoint_frequency,
                "train_verbose": args.train_verbose,
                "max_steps": args.max_steps,
                "seed": args.seed,
                "train_dir": args.train_dir,
                "test_dir": args.test_dir,
                "evaluation_modes": args.evaluation_modes,
                "output_directory": run_dir,
            }
            atomic_write_json(run_dir / "config.json", config_payload)

            status_path = run_dir / "status.json"
            status = load_status(status_path, empty_run_status(spec, args.evaluation_modes))
            update_run_status(status_path, status, args.evaluation_modes)

            logger.info("[%d/%d] Processing %s.", index, len(specs), spec.slug)
            try:
                model_path = run_training_phase(
                    spec=spec,
                    status_path=status_path,
                    status=status,
                    run_dir=run_dir,
                    args=args,
                    device=device,
                    logger=logger,
                    event_log_path=event_log_path,
                )

                for mode_name, deterministic in EVALUATION_MODES.items():
                    if mode_name not in args.evaluation_modes:
                        continue
                    try:
                        run_evaluation_phase(
                            spec=spec,
                            mode_name=mode_name,
                            deterministic=deterministic,
                            status_path=status_path,
                            status=status,
                            run_dir=run_dir,
                            args=args,
                            device=device,
                            model_path=model_path,
                            logger=logger,
                            event_log_path=event_log_path,
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.exception(
                            "Evaluation mode %s for %s failed: %s",
                            mode_name,
                            spec.slug,
                            exc,
                        )
                        if args.fail_fast:
                            raise
            except Exception as exc:  # noqa: BLE001
                logger.exception("Run %s failed: %s", spec.slug, exc)
                if args.fail_fast:
                    raise
            finally:
                progress_payload = write_global_progress(
                    output_dir=output_dir,
                    specs=specs,
                    evaluation_modes=args.evaluation_modes,
                    pipeline_figures=figures_of_merit,
                    pipeline_names=args.pipelines,
                )
                write_summary_csv(
                    output_dir=output_dir,
                    specs=specs,
                    evaluation_modes=args.evaluation_modes,
                    pipeline_figures=figures_of_merit,
                    pipeline_names=args.pipelines,
                )
                log_progress_snapshot(logger, progress_payload)

        for figure_of_merit in figures_of_merit:
            for pipeline_name in args.pipelines:
                try:
                    run_pipeline_phase(
                        figure_of_merit=figure_of_merit,
                        pipeline_name=pipeline_name,
                        output_dir=output_dir,
                        args=args,
                        logger=logger,
                        event_log_path=event_log_path,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.exception(
                        "Pipeline baseline %s for %s failed: %s",
                        pipeline_name,
                        figure_of_merit,
                        exc,
                    )
                    if args.fail_fast:
                        raise
                finally:
                    progress_payload = write_global_progress(
                        output_dir=output_dir,
                        specs=specs,
                        evaluation_modes=args.evaluation_modes,
                        pipeline_figures=figures_of_merit,
                        pipeline_names=args.pipelines,
                    )
                    write_summary_csv(
                        output_dir=output_dir,
                        specs=specs,
                        evaluation_modes=args.evaluation_modes,
                        pipeline_figures=figures_of_merit,
                        pipeline_names=args.pipelines,
                    )
                    log_progress_snapshot(logger, progress_payload)
    except KeyboardInterrupt:
        append_jsonl(
            event_log_path,
            {
                "timestamp": now_utc_iso(),
                "event": "sweep_interrupted",
                "output_dir": str(output_dir),
            },
        )
        progress_payload = write_global_progress(
            output_dir=output_dir,
            specs=specs,
            evaluation_modes=args.evaluation_modes,
            pipeline_figures=figures_of_merit,
            pipeline_names=args.pipelines,
        )
        write_summary_csv(
            output_dir=output_dir,
            specs=specs,
            evaluation_modes=args.evaluation_modes,
            pipeline_figures=figures_of_merit,
            pipeline_names=args.pipelines,
        )
        log_progress_snapshot(logger, progress_payload)
        logger.error("Sweep interrupted. Partial results remain in %s", output_dir)
        raise
    except Exception as exc:  # noqa: BLE001
        append_jsonl(
            event_log_path,
            {
                "timestamp": now_utc_iso(),
                "event": "sweep_failed",
                "output_dir": str(output_dir),
                "error": str(exc),
            },
        )
        progress_payload = write_global_progress(
            output_dir=output_dir,
            specs=specs,
            evaluation_modes=args.evaluation_modes,
            pipeline_figures=figures_of_merit,
            pipeline_names=args.pipelines,
        )
        write_summary_csv(
            output_dir=output_dir,
            specs=specs,
            evaluation_modes=args.evaluation_modes,
            pipeline_figures=figures_of_merit,
            pipeline_names=args.pipelines,
        )
        log_progress_snapshot(logger, progress_payload)
        logger.exception("Sweep failed. Partial results remain in %s", output_dir)
        raise
    else:
        append_jsonl(
            event_log_path,
            {
                "timestamp": now_utc_iso(),
                "event": "sweep_completed",
                "output_dir": str(output_dir),
            },
        )
        logger.info("Sweep finished. Artifacts written to %s", output_dir)


if __name__ == "__main__":
    main()
