# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for lightweight training checkpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from mqt.predictor.rl.checkpoints import RollingCheckpointCallback, latest_checkpoint

if TYPE_CHECKING:
    from pathlib import Path

    from stable_baselines3.common.base_class import BaseAlgorithm


class _FakeModel:
    def __init__(self, num_timesteps: int = 0) -> None:
        self.num_timesteps = num_timesteps
        self.saved_steps: list[int] = []

    def save(self, path: Path) -> None:
        self.saved_steps.append(self.num_timesteps)
        path.write_text(str(self.num_timesteps), encoding="utf-8")


def _start_callback(callback: RollingCheckpointCallback, model: _FakeModel) -> None:
    callback.init_callback(cast("BaseAlgorithm", model))
    callback.on_training_start({}, {})


def test_rolling_checkpoints_save_completed_rollouts_and_retain_three(tmp_path: Path) -> None:
    """Save at step zero and completed rollout intervals while retaining three files."""
    model = _FakeModel()
    callback = RollingCheckpointCallback(2, tmp_path)
    _start_callback(callback, model)

    for step in range(1, 9):
        model.num_timesteps = step
        callback.on_step()
        if step % 2 == 0:
            callback.on_rollout_start()

    assert model.saved_steps == [0, 2, 4, 6, 8]
    assert [path.name for path in sorted(tmp_path.glob("*.zip"))] == [
        "model_checkpoint_4_steps.zip",
        "model_checkpoint_6_steps.zip",
        "model_checkpoint_8_steps.zip",
    ]
    assert not list(tmp_path.glob(".*.tmp.zip"))
    assert latest_checkpoint(tmp_path) == tmp_path / "model_checkpoint_8_steps.zip"


def test_rolling_checkpoints_resume_at_the_next_interval(tmp_path: Path) -> None:
    """Continue checkpoint scheduling from a restored model's timestep."""
    (tmp_path / "model_checkpoint_8_steps.zip").write_bytes(b"checkpoint")
    model = _FakeModel(8)
    callback = RollingCheckpointCallback(2, tmp_path)
    _start_callback(callback, model)

    model.num_timesteps = 9
    callback.on_step()
    callback.on_rollout_start()
    model.num_timesteps = 10
    callback.on_step()
    callback.on_rollout_start()

    assert model.saved_steps == [10]
    assert latest_checkpoint(tmp_path) == tmp_path / "model_checkpoint_10_steps.zip"
