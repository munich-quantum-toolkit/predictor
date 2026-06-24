# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Clean-up fixtures for the tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mqt.predictor.ml.helper import get_path_training_data as ml_get_path_training_data
from mqt.predictor.rl.helper import get_path_trained_model as rl_get_path_trained_model

if TYPE_CHECKING:
    import pytest


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:  # noqa: ARG001
    """Clean up trained RL and ML models after test session."""
    for model in rl_get_path_trained_model().glob("*.zip"):
        model.unlink()

    for model in (ml_get_path_training_data() / "trained_model").glob("*.joblib"):
        model.unlink()

    for data in (ml_get_path_training_data() / "training_data_aggregated").glob("*.npy"):
        data.unlink()
