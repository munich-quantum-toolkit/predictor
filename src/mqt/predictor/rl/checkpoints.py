# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Small rolling checkpoint support for SB3 training runs."""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING

from stable_baselines3.common.callbacks import BaseCallback

if TYPE_CHECKING:
    from pathlib import Path

_PREFIX = "model_checkpoint"
_MAX_CHECKPOINTS = 3


def _checkpoint_step(path: Path, prefix: str) -> int | None:
    start = f"{prefix}_"
    end = "_steps.zip"
    if not path.name.startswith(start) or not path.name.endswith(end):
        return None
    try:
        return int(path.name[len(start) : -len(end)])
    except ValueError:
        return None


def latest_checkpoint(directory: Path, prefix: str = _PREFIX) -> Path | None:
    """Return the checkpoint with the largest encoded step count."""
    checkpoints = [
        (step, path)
        for path in directory.glob(f"{prefix}_*_steps.zip")
        if (step := _checkpoint_step(path, prefix)) is not None
    ]
    return max(checkpoints, default=(0, None), key=operator.itemgetter(0))[1]


def _prune_checkpoints(directory: Path, prefix: str, keep: int) -> None:
    checkpoints = sorted(
        (
            (step, path)
            for path in directory.glob(f"{prefix}_*_steps.zip")
            if (step := _checkpoint_step(path, prefix)) is not None
        ),
        key=operator.itemgetter(0),
    )
    for _, path in checkpoints[:-keep]:
        path.unlink()


def prune_checkpoints(directory: Path, prefix: str = _PREFIX) -> None:
    """Remove incomplete saves and all but the newest three checkpoints."""
    for path in directory.glob(f".{prefix}_*_steps.tmp.zip"):
        path.unlink()
    _prune_checkpoints(directory, prefix, _MAX_CHECKPOINTS)


class RollingCheckpointCallback(BaseCallback):
    """Save native SB3 checkpoints and retain only the newest three."""

    def __init__(self, save_interval: int, save_path: Path, prefix: str = _PREFIX) -> None:
        """Initialize a rollout-boundary checkpoint callback."""
        if save_interval <= 0:
            msg = "save_interval must be positive"
            raise ValueError(msg)
        super().__init__()
        self._save_interval = save_interval
        self._directory = save_path
        self._prefix = prefix
        self._next_save = save_interval

    def _on_training_start(self) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
        prune_checkpoints(self._directory, self._prefix)
        completed_steps = int(self.model.num_timesteps)
        if completed_steps == 0 and latest_checkpoint(self._directory, self._prefix) is None:
            self._save()
        self._next_save = (completed_steps // self._save_interval + 1) * self._save_interval

    def _save(self) -> Path:
        path = self._directory / f"{self._prefix}_{self.model.num_timesteps}_steps.zip"
        temporary_path = path.with_name(f".{path.stem}.tmp.zip")
        _prune_checkpoints(self._directory, self._prefix, _MAX_CHECKPOINTS - 1)
        temporary_path.unlink(missing_ok=True)
        self.model.save(temporary_path)
        temporary_path.replace(path)
        return path

    def _on_rollout_start(self) -> None:
        completed_steps = int(self.model.num_timesteps)
        if completed_steps >= self._next_save:
            self._save()
            self._next_save = (completed_steps // self._save_interval + 1) * self._save_interval

    def _on_step(self) -> bool:
        return True

    def save_final(self) -> Path:
        """Save the current model state and apply the rolling retention limit."""
        path = self._directory / f"{self._prefix}_{self.model.num_timesteps}_steps.zip"
        if not path.exists():
            return self._save()
        prune_checkpoints(self._directory, self._prefix)
        return path
