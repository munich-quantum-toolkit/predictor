# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Train a small RL model that exercises the combined Predictor v3 features."""

from __future__ import annotations

import argparse
import getpass
import json
import re
import subprocess
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gymnasium.spaces import Dict as DictSpace
from mqt.bench.targets import get_device
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from mqt.predictor.rl.helper import get_path_training_circuits
from mqt.predictor.rl.predictorenv import PredictorEnv

if TYPE_CHECKING:
    from collections.abc import Sequence


DEFAULT_OUTPUT_DIR = Path("/tmp") / getpass.getuser() / "mqt-predictor-v3"
CIRCUIT_NAME_PATTERN = re.compile(r"(?P<family>.+)_indep_qiskit_(?P<qubits>\d+)\.qasm$")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=2048)
    parser.add_argument("--device", default="ibm_falcon_27")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--stochastic-action-trials", type=int, default=3)
    parser.add_argument("--max-circuits", type=int, default=12)
    parser.add_argument("--max-circuit-qubits", type=int, default=5)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=512)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    """Reject invalid training configurations early."""
    positive_values = {
        "timesteps": args.timesteps,
        "max_steps": args.max_steps,
        "stochastic_action_trials": args.stochastic_action_trials,
        "max_circuits": args.max_circuits,
        "max_circuit_qubits": args.max_circuit_qubits,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "checkpoint_every": args.checkpoint_every,
    }
    invalid = [name for name, value in positive_values.items() if value < 1]
    if invalid:
        msg = f"Expected positive values for: {', '.join(invalid)}."
        raise ValueError(msg)
    if args.n_steps < 2:
        msg = "n_steps must be at least two."
        raise ValueError(msg)
    if args.batch_size > args.n_steps:
        msg = "batch_size must not exceed n_steps."
        raise ValueError(msg)


def prepare_training_circuits(destination: Path, max_qubits: int, max_circuits: int) -> list[str]:
    """Extract a deterministic, small circuit sample outside the source tree."""
    source_archive = get_path_training_circuits() / "training_data_compilation.zip"
    destination.mkdir(parents=True)

    with zipfile.ZipFile(source_archive) as archive:
        candidates: list[tuple[int, str, str]] = []
        for member in archive.namelist():
            filename = Path(member).name
            if filename.startswith("._"):
                continue
            match = CIRCUIT_NAME_PATTERN.fullmatch(filename)
            if match is None:
                continue
            num_qubits = int(match.group("qubits"))
            if 3 <= num_qubits <= max_qubits:
                candidates.append((num_qubits, match.group("family"), member))

        selected = sorted(candidates)[:max_circuits]
        if not selected:
            msg = f"No training circuits with 3-{max_qubits} qubits found in {source_archive}."
            raise RuntimeError(msg)

        filenames = []
        for _num_qubits, _family, member in selected:
            filename = Path(member).name
            (destination / filename).write_bytes(archive.read(member))
            filenames.append(filename)

    return filenames


def git_state() -> tuple[str, bool]:
    """Return the current Git revision and whether tracked files are dirty."""
    revision_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    status_result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        check=False,
        capture_output=True,
        text=True,
    )
    revision = revision_result.stdout.strip() if revision_result.returncode == 0 else "unknown"
    dirty = status_result.returncode != 0 or bool(status_result.stdout.strip())
    return revision, dirty


def write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    """Write the run manifest atomically enough for progress inspection."""
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    """Run the small v3 training experiment."""
    args = parse_args(argv)
    validate_args(args)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_name = args.run_name or f"{timestamp}-{args.device}-seed{args.seed}"
    run_dir = args.output_dir.expanduser().resolve() / run_name
    run_dir.mkdir(parents=True)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir()
    circuit_names = prepare_training_circuits(
        run_dir / "circuits",
        max_qubits=args.max_circuit_qubits,
        max_circuits=args.max_circuits,
    )

    device = get_device(args.device)
    predictor_env = PredictorEnv(
        device=device,
        reward_function="expected_fidelity",
        path_training_circuits=run_dir / "circuits",
        max_steps=args.max_steps,
        mdp="v3",
        stochastic_action_trials=args.stochastic_action_trials,
        intermediate_reward=True,
    )
    if not isinstance(predictor_env.observation_space, DictSpace):
        msg = "Expected the Predictor v3 observation space to be a dictionary."
        raise TypeError(msg)
    env = Monitor(predictor_env, filename=str(run_dir / "monitor.csv"))
    revision, dirty = git_state()
    metadata: dict[str, Any] = {
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "git_revision": revision,
        "git_dirty": dirty,
        "device": args.device,
        "figure_of_merit": "expected_fidelity",
        "mdp": "v3",
        "intermediate_reward": True,
        "stochastic_action_trials": args.stochastic_action_trials,
        "stochastic_actions": sorted(action.name for action in predictor_env.action_set.values() if action.stochastic),
        "observation_keys": sorted(predictor_env.observation_space.spaces),
        "training_circuits": circuit_names,
        "timesteps_requested": args.timesteps,
        "max_steps_per_episode": args.max_steps,
        "ppo": {
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "n_epochs": args.n_epochs,
            "policy_net_arch": {"pi": [8], "vf": [8]},
            "gamma": 0.98,
            "seed": args.seed,
            "device": "cpu",
        },
    }
    metadata_path = run_dir / "run.json"
    write_metadata(metadata_path, metadata)

    model = MaskablePPO(
        MaskableMultiInputActorCriticPolicy,
        env,
        verbose=1,
        tensorboard_log=str(run_dir / "tensorboard"),
        gamma=0.98,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        seed=args.seed,
        device="cpu",
        policy_kwargs={"net_arch": {"pi": [8], "vf": [8]}},
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_every,
        save_path=str(checkpoint_dir),
        name_prefix="v3_predictor",
        verbose=2,
    )

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=checkpoint_callback,
            progress_bar=True,
            tb_log_name="training",
        )
        model.save(run_dir / "final_model")
        metadata.update(
            status="completed",
            completed_at=datetime.now(UTC).isoformat(),
            timesteps_completed=model.num_timesteps,
        )
    except Exception as exc:
        metadata.update(
            status="failed",
            completed_at=datetime.now(UTC).isoformat(),
            error=f"{type(exc).__name__}: {exc}",
            timesteps_completed=model.num_timesteps,
        )
        raise
    finally:
        write_metadata(metadata_path, metadata)
        env.close()

    print(f"Run artifacts: {run_dir}")


if __name__ == "__main__":
    main()
