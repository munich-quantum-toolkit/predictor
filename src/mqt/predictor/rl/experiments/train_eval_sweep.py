# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Run a resumable ESP-focused RL sweep over the selected devices and MDP variants.

The script is intentionally narrow:

- figure of merit is fixed to ``estimated_success_probability``
- devices default to ``ibm_boston_156`` and ``iqm_garnet_20``
- all RL MDP variants are trained
- both stochastic and deterministic evaluations are run
- optional locked baseline pipelines are evaluated

The goal is a clean SLURM-friendly entry point that:

- starts a fresh sweep when the output directory is empty
- resumes from saved checkpoints when a previous batch did not finish
- logs progress to one human-readable log file
- saves per-run JSON results that can be inspected later
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import signal
import tempfile
import traceback
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mqt.bench.targets import get_device
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

from mqt.predictor.reward import esp_data_available
from mqt.predictor.rl.experiments.evaluation import evaluate_trained_predictor
from mqt.predictor.rl.experiments.gnn_tuning import run_gnn_hyperparameter_tuning
from mqt.predictor.rl.experiments.pipeline_evaluation import run_selected_pipelines
from mqt.predictor.rl.experiments.training import run_rl_training

if TYPE_CHECKING:
    from collections.abc import Generator

    from qiskit.transpiler import Target

    from mqt.predictor.reward import figure_of_merit
    from mqt.predictor.rl.experiments.evaluation import PredictorEvaluationResult
    from mqt.predictor.rl.experiments.pipeline_evaluation import PipelineEvaluationResult


FIGURE_OF_MERIT: figure_of_merit = "estimated_success_probability"
DEFAULT_DEVICES = ("ibm_boston_156", "iqm_garnet_20")
DEFAULT_MDPS = ("paper", "flexible", "thesis", "hybrid")
DEFAULT_PIPELINES = ("qiskit_o3", "tket_o2")
EVALUATION_MODES = (("stochastic", False), ("deterministic", True))


@dataclass(frozen=True, slots=True)
class RLRunSpec:
    """One RL training and evaluation configuration."""

    device_name: str
    mdp: str

    @property
    def label(self) -> str:
        """Return a short human-readable run label."""
        return f"{self.device_name}/{self.mdp}"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a resumable RL sweep for estimated_success_probability.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        default=list(DEFAULT_DEVICES),
        help="Target devices to process.",
    )
    parser.add_argument(
        "--mdps",
        nargs="+",
        default=list(DEFAULT_MDPS),
        choices=list(DEFAULT_MDPS),
        help="MDP variants to train and evaluate.",
    )
    parser.add_argument("--timesteps", type=int, default=100000, help="Training timesteps per RL model.")
    parser.add_argument(
        "--checkpoint-frequency",
        type=int,
        default=2048,
        help="PPO checkpoint cadence in timesteps.",
    )
    parser.add_argument("--train-verbose", type=int, default=1, help="Verbosity passed to PPO.")
    parser.add_argument("--max-steps", type=int, default=200, help="Maximum evaluation rollout steps.")
    parser.add_argument("--seed", type=int, default=0, help="Evaluation seed.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "rl_experiment_runs" / "estimated_success_probability_sweep",
        help="Stable output directory used for logs, checkpoints, and results.",
    )
    parser.add_argument("--train-dir", type=Path, default=None, help="Optional training-circuit directory.")
    parser.add_argument("--test-dir", type=Path, default=None, help="Optional test-circuit directory.")
    parser.add_argument(
        "--skip-pipelines",
        action="store_true",
        help="Skip the locked baseline pipeline evaluations.",
    )
    parser.add_argument(
        "--test-training",
        action="store_true",
        help="Use the lightweight training mode from the existing training helper.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore completed markers and rerun every phase.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first failed training, evaluation, or pipeline run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned sweep and exit.",
    )
    parser.add_argument(
        "--gnn-tuning",
        action="store_true",
        help="Enable GNN hyperparameter tuning mode (replaces MDP sweep).",
    )
    parser.add_argument(
        "--gnn-trials",
        type=int,
        default=10,
        help="Number of Optuna trials for GNN tuning.",
    )
    parser.add_argument(
        "--gnn-trial-steps",
        type=int,
        default=2000,
        help="Training steps per GNN tuning trial.",
    )
    return parser.parse_args()


def now_utc_iso() -> str:
    """Return the current UTC timestamp in ISO format."""
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat()


def configure_logging(output_dir: Path) -> logging.Logger:
    """Configure file and console logging for the sweep."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("mqt-predictor.esp-sweep")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_handler = logging.FileHandler(output_dir / "sweep.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def sanitize_json_value(value: object) -> object:
    """Convert objects into JSON-compatible values."""
    if is_dataclass(value) and not isinstance(value, type):
        return sanitize_json_value(asdict(value))

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {str(key): sanitize_json_value(val) for key, val in value.items()}

    if isinstance(value, list | tuple):
        return [sanitize_json_value(item) for item in value]

    if isinstance(value, set | frozenset):
        return [sanitize_json_value(item) for item in sorted(value)]

    if hasattr(value, "item"):
        try:
            item_method = value.item
            if callable(item_method):
                return sanitize_json_value(item_method())  # type: ignore[misc]
        except (TypeError, ValueError):
            pass

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    return value


def atomic_write_json(path: Path, payload: object) -> None:
    """Atomically write one JSON file."""
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


def load_json(path: Path, fallback: dict[str, Any]) -> dict[str, Any]:
    """Load a JSON file or return the provided fallback value."""
    if not path.is_file():
        return json.loads(json.dumps(fallback))
    return json.loads(path.read_text(encoding="utf-8"))


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


def install_signal_handlers() -> None:
    """Convert TERM/INT signals into KeyboardInterrupt for graceful shutdown."""

    def handle_interrupt(signum: int, _frame: object) -> None:
        msg = f"Received signal {signum}."
        raise KeyboardInterrupt(msg)

    signal.signal(signal.SIGTERM, handle_interrupt)
    signal.signal(signal.SIGINT, handle_interrupt)


def rl_run_directory(output_dir: Path, spec: RLRunSpec) -> Path:
    """Return the output directory for one RL run."""
    return output_dir / spec.device_name / "rl" / spec.mdp


def pipeline_directory(output_dir: Path, device_name: str, pipeline_name: str) -> Path:
    """Return the output directory for one pipeline run."""
    return output_dir / device_name / "pipelines" / pipeline_name


def empty_rl_status(spec: RLRunSpec) -> dict[str, Any]:
    """Return the initial status document for one RL run."""
    return {
        "device": spec.device_name,
        "figure_of_merit": FIGURE_OF_MERIT,
        "mdp": spec.mdp,
        "created_at": now_utc_iso(),
        "updated_at": now_utc_iso(),
        "training": {"status": "pending", "completed_timesteps": 0},
        "evaluations": {mode_name: {"status": "pending"} for mode_name, _ in EVALUATION_MODES},
    }


def empty_pipeline_status(device_name: str, pipeline_name: str) -> dict[str, Any]:
    """Return the initial status document for one pipeline run."""
    return {
        "device": device_name,
        "figure_of_merit": FIGURE_OF_MERIT,
        "pipeline": pipeline_name,
        "created_at": now_utc_iso(),
        "updated_at": now_utc_iso(),
        "result": {"status": "pending"},
    }


def save_rl_status(path: Path, status: dict[str, Any]) -> None:
    """Persist one RL run status document."""
    status["updated_at"] = now_utc_iso()
    atomic_write_json(path, status)


def save_pipeline_status(path: Path, status: dict[str, Any]) -> None:
    """Persist one pipeline status document."""
    status["updated_at"] = now_utc_iso()
    atomic_write_json(path, status)


def checkpoint_sort_key(path: Path) -> int:
    """Return the timestep suffix encoded in a checkpoint filename."""
    for part in reversed(path.stem.split("_")):
        if part.isdigit():
            return int(part)
    return -1


def find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Return the newest PPO checkpoint in a directory."""
    if not checkpoint_dir.is_dir():
        return None

    checkpoint_files = sorted(checkpoint_dir.glob("model_checkpoint_*_steps.zip"), key=checkpoint_sort_key)
    if not checkpoint_files:
        return None
    return checkpoint_files[-1]


def get_model_num_timesteps(model_path: Path) -> int:
    """Load a PPO checkpoint and return its completed timestep count."""
    return int(MaskablePPO.load(model_path).num_timesteps)


class TrainingHeartbeatCallback(BaseCallback):
    """Update the on-disk training status during long PPO runs."""

    def __init__(
        self,
        *,
        spec: RLRunSpec,
        target_timesteps: int,
        initial_timesteps: int,
        status: dict[str, Any],
        status_path: Path,
        logger: logging.Logger,
        log_frequency: int,
    ) -> None:
        """Initialize the heartbeat callback."""
        super().__init__()
        self.spec = spec
        self.target_timesteps = target_timesteps
        self.status = status
        self.status_path = status_path
        self.progress_logger = logger
        self.log_frequency = max(1, log_frequency)
        self.last_logged_timesteps = initial_timesteps

    def _on_step(self) -> bool:
        current_timesteps = int(self.model.num_timesteps)
        if current_timesteps - self.last_logged_timesteps < self.log_frequency:
            return True

        self.last_logged_timesteps = current_timesteps
        self.status["training"]["completed_timesteps"] = current_timesteps
        self.status["training"]["last_progress_at"] = now_utc_iso()
        save_rl_status(self.status_path, self.status)
        self.progress_logger.info(
            "Training %s: %d/%d timesteps (%.1f%%).",
            self.spec.label,
            current_timesteps,
            self.target_timesteps,
            (current_timesteps / self.target_timesteps) * 100.0 if self.target_timesteps > 0 else 100.0,
        )
        return True


def summarize_predictor_evaluation(result: PredictorEvaluationResult) -> dict[str, Any]:
    """Return a compact summary for one RL evaluation result."""
    figure_values = [circuit.figure_of_merit_value for circuit in result.circuits]
    average_figure_value = sum(figure_values) / len(figure_values) if figure_values else None
    return {
        "evaluated_circuits": len(result.circuits),
        "average_figure_of_merit_value": average_figure_value,
        "average_metrics": sanitize_json_value(result.average_metrics),
        "step_limit_hits": sum(1 for circuit in result.circuits if circuit.hit_step_limit),
        "action_effectiveness": {
            "total_uses": result.action_effectiveness.total_uses,
            "total_effective_uses": result.action_effectiveness.total_effective_uses,
            "overall_effectiveness_ratio": result.action_effectiveness.overall_effectiveness_ratio,
        },
    }


def summarize_pipeline_evaluation(result: PipelineEvaluationResult) -> dict[str, Any]:
    """Return a compact summary for one pipeline result."""
    figure_values = [circuit.figure_of_merit_value for circuit in result.circuits]
    average_figure_value = sum(figure_values) / len(figure_values) if figure_values else None
    return {
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
def working_directory(path: Path) -> Generator[None, None, None]:
    """Temporarily change the working directory."""
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def resolve_device(device_name: str) -> Target:
    """Load one target device and ensure ESP is supported."""
    device = get_device(device_name)
    if not esp_data_available(device):
        msg = f"Device '{device_name}' does not provide the calibration data required for ESP."
        raise ValueError(msg)
    return device


def get_single_pipeline_result(
    results: list[PipelineEvaluationResult],
    pipeline_name: str,
) -> PipelineEvaluationResult:
    """Return the single expected pipeline result."""
    if len(results) != 1:
        msg = f"Expected one pipeline result for '{pipeline_name}', received {len(results)}."
        raise RuntimeError(msg)
    return results[0]


def finalize_training_from_checkpoint(
    *,
    spec: RLRunSpec,
    status_path: Path,
    status: dict[str, Any],
    checkpoint_path: Path,
    checkpoint_dir: Path,
    model_copy_path: Path,
    training_result_path: Path,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> Path:
    """Convert an already-complete checkpoint into the final saved model artifact."""
    completed_timesteps = get_model_num_timesteps(checkpoint_path)
    copied_model_path = copy_file_atomically(checkpoint_path, model_copy_path)
    training_payload = {
        "completed_at": now_utc_iso(),
        "device": spec.device_name,
        "figure_of_merit": FIGURE_OF_MERIT,
        "mdp": spec.mdp,
        "timesteps": args.timesteps,
        "completed_timesteps": completed_timesteps,
        "saved_model_path": str(copied_model_path),
        "source_model_path": str(checkpoint_path),
        "checkpoint_directory": str(checkpoint_dir),
    }
    atomic_write_json(training_result_path, training_payload)

    status["training"] = {
        "status": "completed",
        "finished_at": now_utc_iso(),
        "timesteps": args.timesteps,
        "completed_timesteps": completed_timesteps,
        "verbose": args.train_verbose,
        "checkpoint_directory": str(checkpoint_dir),
        "checkpoint_frequency": args.checkpoint_frequency,
        "resumed_from_checkpoint": str(checkpoint_path),
        "saved_model_path": str(copied_model_path),
        "result_path": str(training_result_path),
    }
    save_rl_status(status_path, status)
    logger.info("Finalized completed checkpoint for %s.", spec.label)
    return copied_model_path


def run_training_phase(
    *,
    spec: RLRunSpec,
    device: Target,
    run_dir: Path,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> Path:
    """Train one RL configuration or resume it from the latest checkpoint."""
    status_path = run_dir / "status.json"
    status = load_json(status_path, empty_rl_status(spec))

    model_copy_path = run_dir / "artifacts" / f"model_{spec.device_name}_{spec.mdp}.zip"
    training_result_path = run_dir / "training.json"
    checkpoint_dir = run_dir / "checkpoints"

    if (
        not args.force
        and status["training"]["status"] == "completed"
        and model_copy_path.is_file()
        and training_result_path.is_file()
    ):
        logger.info("Skipping completed training for %s.", spec.label)
        return model_copy_path

    checkpoint_path = None if args.force else find_latest_checkpoint(checkpoint_dir)
    completed_timesteps = 0 if checkpoint_path is None else get_model_num_timesteps(checkpoint_path)

    if checkpoint_path is not None and completed_timesteps >= args.timesteps:
        return finalize_training_from_checkpoint(
            spec=spec,
            status_path=status_path,
            status=status,
            checkpoint_path=checkpoint_path,
            checkpoint_dir=checkpoint_dir,
            model_copy_path=model_copy_path,
            training_result_path=training_result_path,
            args=args,
            logger=logger,
        )

    remaining_timesteps = args.timesteps - completed_timesteps
    if checkpoint_path is None:
        logger.info("Training %s for %d timesteps.", spec.label, args.timesteps)
    else:
        logger.info(
            "Resuming %s from %s at %d/%d timesteps.",
            spec.label,
            checkpoint_path.name,
            completed_timesteps,
            args.timesteps,
        )

    status["training"] = {
        "status": "running",
        "started_at": status["training"].get("started_at", now_utc_iso()),
        "timesteps": args.timesteps,
        "completed_timesteps": completed_timesteps,
        "verbose": args.train_verbose,
        "checkpoint_directory": str(checkpoint_dir),
        "checkpoint_frequency": args.checkpoint_frequency,
        "resumed_from_checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
    }
    save_rl_status(status_path, status)

    try:
        heartbeat = TrainingHeartbeatCallback(
            spec=spec,
            target_timesteps=args.timesteps,
            initial_timesteps=completed_timesteps,
            status=status,
            status_path=status_path,
            logger=logger,
            log_frequency=args.checkpoint_frequency,
        )
        with working_directory(run_dir):
            training_result = run_rl_training(
                device=device,
                figure_of_merit=FIGURE_OF_MERIT,
                mdp=spec.mdp,
                timesteps=remaining_timesteps,
                verbose=args.train_verbose,
                test=args.test_training,
                path_training_circuits=args.train_dir,
                checkpoint_directory=checkpoint_dir,
                checkpoint_frequency=args.checkpoint_frequency,
                resume_from_checkpoint=checkpoint_path,
                callback=heartbeat,
            )

        copied_model_path = copy_file_atomically(training_result.model_path, model_copy_path)
        training_payload = {
            "completed_at": now_utc_iso(),
            "device": spec.device_name,
            "figure_of_merit": FIGURE_OF_MERIT,
            "mdp": spec.mdp,
            "timesteps": args.timesteps,
            "completed_timesteps": training_result.total_timesteps,
            "saved_model_path": str(copied_model_path),
            "source_model_path": str(training_result.model_path),
            "checkpoint_directory": str(checkpoint_dir),
            "resumed_from_checkpoint": (
                str(training_result.resumed_from_checkpoint)
                if training_result.resumed_from_checkpoint is not None
                else None
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
            "checkpoint_directory": str(checkpoint_dir),
            "checkpoint_frequency": args.checkpoint_frequency,
            "resumed_from_checkpoint": (
                str(training_result.resumed_from_checkpoint)
                if training_result.resumed_from_checkpoint is not None
                else None
            ),
            "saved_model_path": str(copied_model_path),
            "result_path": str(training_result_path),
        }
        save_rl_status(status_path, status)
    except KeyboardInterrupt:
        latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
        latest_timesteps = (
            completed_timesteps if latest_checkpoint is None else get_model_num_timesteps(latest_checkpoint)
        )
        status["training"] = {
            "status": "interrupted",
            "started_at": status["training"]["started_at"],
            "finished_at": now_utc_iso(),
            "timesteps": args.timesteps,
            "completed_timesteps": latest_timesteps,
            "checkpoint_directory": str(checkpoint_dir),
            "checkpoint_frequency": args.checkpoint_frequency,
            "resumed_from_checkpoint": str(latest_checkpoint) if latest_checkpoint is not None else None,
        }
        save_rl_status(status_path, status)
        raise
    except Exception as exc:
        latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
        latest_timesteps = (
            completed_timesteps if latest_checkpoint is None else get_model_num_timesteps(latest_checkpoint)
        )
        status["training"] = {
            "status": "failed",
            "started_at": status["training"]["started_at"],
            "finished_at": now_utc_iso(),
            "timesteps": args.timesteps,
            "completed_timesteps": latest_timesteps,
            "checkpoint_directory": str(checkpoint_dir),
            "checkpoint_frequency": args.checkpoint_frequency,
            "resumed_from_checkpoint": str(latest_checkpoint) if latest_checkpoint is not None else None,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        save_rl_status(status_path, status)
        raise
    else:
        logger.info("Completed training for %s.", spec.label)
        return copied_model_path


def run_evaluation_phase(
    *,
    spec: RLRunSpec,
    device: Target,
    run_dir: Path,
    model_path: Path,
    mode_name: str,
    deterministic: bool,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> None:
    """Evaluate one trained RL model in one rollout mode."""
    status_path = run_dir / "status.json"
    status = load_json(status_path, empty_rl_status(spec))
    result_path = run_dir / f"evaluation_{mode_name}.json"

    if not args.force and status["evaluations"][mode_name]["status"] == "completed" and result_path.is_file():
        logger.info("Skipping completed %s evaluation for %s.", mode_name, spec.label)
        return

    logger.info("Evaluating %s with %s rollout.", spec.label, mode_name)
    status["evaluations"][mode_name] = {
        "status": "running",
        "started_at": now_utc_iso(),
        "seed": args.seed,
        "max_steps": args.max_steps,
        "deterministic": deterministic,
    }
    save_rl_status(status_path, status)

    try:
        result = evaluate_trained_predictor(
            model_path=model_path,
            device=device,
            figure_of_merit=FIGURE_OF_MERIT,
            mdp=spec.mdp,
            path_training_circuits=args.train_dir,
            path_test_circuits=args.test_dir,
            max_steps=args.max_steps,
            deterministic=deterministic,
            seed=args.seed,
        )
        atomic_write_json(
            result_path,
            {
                "completed_at": now_utc_iso(),
                "device": spec.device_name,
                "figure_of_merit": FIGURE_OF_MERIT,
                "mdp": spec.mdp,
                "evaluation_mode": mode_name,
                "deterministic": deterministic,
                "result": result,
            },
        )
        status["evaluations"][mode_name] = {
            "status": "completed",
            "started_at": status["evaluations"][mode_name]["started_at"],
            "finished_at": now_utc_iso(),
            "seed": args.seed,
            "max_steps": args.max_steps,
            "deterministic": deterministic,
            "result_path": str(result_path),
            "summary": summarize_predictor_evaluation(result),
        }
        save_rl_status(status_path, status)
        logger.info("Completed %s evaluation for %s.", mode_name, spec.label)
    except KeyboardInterrupt:
        status["evaluations"][mode_name] = {
            "status": "interrupted",
            "started_at": status["evaluations"][mode_name]["started_at"],
            "finished_at": now_utc_iso(),
            "seed": args.seed,
            "max_steps": args.max_steps,
            "deterministic": deterministic,
        }
        save_rl_status(status_path, status)
        raise
    except Exception as exc:
        status["evaluations"][mode_name] = {
            "status": "failed",
            "started_at": status["evaluations"][mode_name]["started_at"],
            "finished_at": now_utc_iso(),
            "seed": args.seed,
            "max_steps": args.max_steps,
            "deterministic": deterministic,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        save_rl_status(status_path, status)
        raise


def run_pipeline_phase(
    *,
    device_name: str,
    pipeline_name: str,
    output_dir: Path,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> None:
    """Evaluate one locked baseline pipeline for one device."""
    status_path = pipeline_directory(output_dir, device_name, pipeline_name) / "status.json"
    result_path = pipeline_directory(output_dir, device_name, pipeline_name) / "result.json"
    status = load_json(status_path, empty_pipeline_status(device_name, pipeline_name))

    if not args.force and status["result"]["status"] == "completed" and result_path.is_file():
        logger.info("Skipping completed pipeline %s for %s.", pipeline_name, device_name)
        return

    logger.info("Evaluating pipeline %s for %s.", pipeline_name, device_name)
    status["result"] = {"status": "running", "started_at": now_utc_iso()}
    save_pipeline_status(status_path, status)

    try:
        results = run_selected_pipelines(
            pipeline_names=[pipeline_name],
            device_name=device_name,
            figure_of_merit_name=FIGURE_OF_MERIT,
            path_training_circuits=args.train_dir,
            path_test_circuits=args.test_dir,
        )
        result = get_single_pipeline_result(results, pipeline_name)
        atomic_write_json(
            result_path,
            {
                "completed_at": now_utc_iso(),
                "device": device_name,
                "figure_of_merit": FIGURE_OF_MERIT,
                "pipeline": pipeline_name,
                "result": result,
            },
        )
        status["result"] = {
            "status": "completed",
            "started_at": status["result"]["started_at"],
            "finished_at": now_utc_iso(),
            "result_path": str(result_path),
            "summary": summarize_pipeline_evaluation(result),
        }
        save_pipeline_status(status_path, status)
        logger.info("Completed pipeline %s for %s.", pipeline_name, device_name)
    except KeyboardInterrupt:
        status["result"] = {
            "status": "interrupted",
            "started_at": status["result"]["started_at"],
            "finished_at": now_utc_iso(),
        }
        save_pipeline_status(status_path, status)
        raise
    except Exception as exc:
        status["result"] = {
            "status": "failed",
            "started_at": status["result"]["started_at"],
            "finished_at": now_utc_iso(),
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        save_pipeline_status(status_path, status)
        raise


def write_manifest(output_dir: Path, args: argparse.Namespace) -> None:
    """Write the static sweep configuration."""
    manifest_data = {
        "created_at": now_utc_iso(),
        "figure_of_merit": FIGURE_OF_MERIT,
        "devices": args.devices,
        "mdps": args.mdps,
        "timesteps": args.timesteps,
        "checkpoint_frequency": args.checkpoint_frequency,
        "train_verbose": args.train_verbose,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "output_dir": str(args.output_dir),
        "train_dir": args.train_dir,
        "test_dir": args.test_dir,
        "skip_pipelines": args.skip_pipelines,
        "test_training": args.test_training,
        "force": args.force,
        "gnn_tuning": args.gnn_tuning,
        "gnn_trials": args.gnn_trials,
        "gnn_trial_steps": args.gnn_trial_steps,
    }
    atomic_write_json(output_dir / "manifest.json", manifest_data)


def write_progress(output_dir: Path, devices: list[str], mdps: list[str], include_pipelines: bool) -> dict[str, Any]:
    """Compute and persist a simple progress summary."""
    training_total = len(devices) * len(mdps)
    evaluation_total = training_total * len(EVALUATION_MODES)
    pipeline_total = len(devices) * len(DEFAULT_PIPELINES) if include_pipelines else 0

    training_completed = 0
    training_failed = 0
    evaluation_completed = 0
    evaluation_failed = 0
    pipeline_completed = 0
    pipeline_failed = 0

    for device_name in devices:
        for mdp in mdps:
            status = load_json(
                rl_run_directory(output_dir, RLRunSpec(device_name=device_name, mdp=mdp)) / "status.json",
                empty_rl_status(RLRunSpec(device_name=device_name, mdp=mdp)),
            )
            if status["training"]["status"] == "completed":
                training_completed += 1
            elif status["training"]["status"] == "failed":
                training_failed += 1

            for mode_name, _ in EVALUATION_MODES:
                phase_status = status["evaluations"][mode_name]["status"]
                if phase_status == "completed":
                    evaluation_completed += 1
                elif phase_status == "failed":
                    evaluation_failed += 1

        if include_pipelines:
            for pipeline_name in DEFAULT_PIPELINES:
                status = load_json(
                    pipeline_directory(output_dir, device_name, pipeline_name) / "status.json",
                    empty_pipeline_status(device_name, pipeline_name),
                )
                phase_status = status["result"]["status"]
                if phase_status == "completed":
                    pipeline_completed += 1
                elif phase_status == "failed":
                    pipeline_failed += 1

    progress = {
        "updated_at": now_utc_iso(),
        "figure_of_merit": FIGURE_OF_MERIT,
        "training": {
            "completed": training_completed,
            "failed": training_failed,
            "total": training_total,
        },
        "evaluations": {
            "completed": evaluation_completed,
            "failed": evaluation_failed,
            "total": evaluation_total,
        },
        "pipelines": {
            "completed": pipeline_completed,
            "failed": pipeline_failed,
            "total": pipeline_total,
        },
    }
    atomic_write_json(output_dir / "progress.json", progress)
    return progress


def log_progress(logger: logging.Logger, progress: dict[str, Any]) -> None:
    """Log a compact progress summary."""
    logger.info(
        "Progress: training %d/%d, evaluations %d/%d, pipelines %d/%d.",
        progress["training"]["completed"],
        progress["training"]["total"],
        progress["evaluations"]["completed"],
        progress["evaluations"]["total"],
        progress["pipelines"]["completed"],
        progress["pipelines"]["total"],
    )


def run_gnn_tuning_phase(
    *,
    device_name: str,
    output_dir: Path,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> None:
    """Run GNN hyperparameter tuning for one device."""
    device = resolve_device(device_name)
    tuning_dir = output_dir / device_name / "gnn_tuning"
    tuning_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting GNN hyperparameter tuning for %s.", device_name)
    logger.info("Number of trials: %d", args.gnn_trials)
    logger.info("Steps per trial: %d", args.gnn_trial_steps)

    try:
        results = run_gnn_hyperparameter_tuning(
            device=device,
            figure_of_merit=FIGURE_OF_MERIT,
            output_dir=tuning_dir,
            num_trials=args.gnn_trials,
            trial_steps=args.gnn_trial_steps,
            mdp="paper",  # Fixed MDP for tuning
            logger=logger,
        )
        logger.info(
            "GNN tuning completed for %s. Best score: %.4f (trial %d)",
            device_name,
            results["best_score"],
            results["best_trial"],
        )
    except Exception:
        logger.exception("GNN tuning failed for %s.", device_name)
        raise


def print_plan(args: argparse.Namespace) -> None:
    """Print the planned sweep and exit."""
    if args.gnn_tuning:
        print(f"Figure of merit: {FIGURE_OF_MERIT}")
        print("Mode: GNN hyperparameter tuning")
        print(f"Devices: {', '.join(args.devices)}")
        print(f"Trials per device: {args.gnn_trials}")
        print(f"Steps per trial: {args.gnn_trial_steps}")
        print("MDP: paper (fixed)")
        print(f"Output directory: {args.output_dir}")
        print(f"Planned GNN tuning runs: {len(args.devices)}")
        for index, device_name in enumerate(args.devices, start=1):
            print(f"  [{index:02d}] {device_name}")
    else:
        specs = [RLRunSpec(device_name=device_name, mdp=mdp) for device_name in args.devices for mdp in args.mdps]
        print(f"Figure of merit: {FIGURE_OF_MERIT}")
        print(f"Devices: {', '.join(args.devices)}")
        print(f"MDPs: {', '.join(args.mdps)}")
        print(f"Timesteps: {args.timesteps}")
        print(f"Checkpoint frequency: {args.checkpoint_frequency}")
        print(f"Output directory: {args.output_dir}")
        print(f"Pipelines: {'disabled' if args.skip_pipelines else ', '.join(DEFAULT_PIPELINES)}")
        print(f"Planned RL runs: {len(specs)}")
        for index, spec in enumerate(specs, start=1):
            print(f"  [{index:02d}] {spec.label}")


def main() -> None:
    """Run the full ESP sweep."""
    args = parse_args()
    args.output_dir = args.output_dir.resolve()

    if args.dry_run:
        print_plan(args)
        return

    install_signal_handlers()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(args.output_dir)
    write_manifest(args.output_dir, args)

    logger.info("Starting ESP RL sweep.")
    logger.info("Devices: %s", ", ".join(args.devices))
    logger.info("MDPs: %s", ", ".join(args.mdps))
    logger.info("Output directory: %s", args.output_dir)

    include_pipelines = not args.skip_pipelines

    # Handle GNN tuning mode separately
    if args.gnn_tuning:
        logger.info("Running in GNN hyperparameter tuning mode.")
        try:
            for device_name in args.devices:
                run_gnn_tuning_phase(
                    device_name=device_name,
                    output_dir=args.output_dir,
                    args=args,
                    logger=logger,
                )
        except KeyboardInterrupt:
            logger.exception("GNN tuning interrupted.")
            raise
        except Exception:
            logger.exception("GNN tuning phase failed.")
            if args.fail_fast:
                raise

        logger.info("GNN hyperparameter tuning finished. Results are in %s", args.output_dir)
        return

    # Standard RL MDP sweep mode (existing behavior)
    try:
        for device_name in args.devices:
            device = resolve_device(device_name)
            logger.info("Processing device %s.", device_name)

            for mdp in args.mdps:
                spec = RLRunSpec(device_name=device_name, mdp=mdp)
                run_dir = rl_run_directory(args.output_dir, spec)
                run_dir.mkdir(parents=True, exist_ok=True)

                try:
                    model_path = run_training_phase(
                        spec=spec,
                        device=device,
                        run_dir=run_dir,
                        args=args,
                        logger=logger,
                    )
                    for mode_name, deterministic in EVALUATION_MODES:
                        run_evaluation_phase(
                            spec=spec,
                            device=device,
                            run_dir=run_dir,
                            model_path=model_path,
                            mode_name=mode_name,
                            deterministic=deterministic,
                            args=args,
                            logger=logger,
                        )
                except Exception:
                    logger.exception("RL run failed for %s.", spec.label)
                    if args.fail_fast:
                        raise
                finally:
                    progress = write_progress(args.output_dir, args.devices, args.mdps, include_pipelines)
                    log_progress(logger, progress)

            if include_pipelines:
                for pipeline_name in DEFAULT_PIPELINES:
                    pipeline_run_dir = pipeline_directory(args.output_dir, device_name, pipeline_name)
                    pipeline_run_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        run_pipeline_phase(
                            device_name=device_name,
                            pipeline_name=pipeline_name,
                            output_dir=args.output_dir,
                            args=args,
                            logger=logger,
                        )
                    except Exception:
                        logger.exception("Pipeline %s failed for %s.", pipeline_name, device_name)
                        if args.fail_fast:
                            raise
                    finally:
                        progress = write_progress(args.output_dir, args.devices, args.mdps, include_pipelines)
                        log_progress(logger, progress)
    except KeyboardInterrupt:
        progress = write_progress(args.output_dir, args.devices, args.mdps, include_pipelines)
        log_progress(logger, progress)
        logger.exception("Sweep interrupted. Rerun the same command to continue from saved checkpoints and results.")
        raise

    progress = write_progress(args.output_dir, args.devices, args.mdps, include_pipelines)
    log_progress(logger, progress)
    logger.info("Sweep finished. Results are in %s", args.output_dir)


if __name__ == "__main__":
    main()
