# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Clean-up fixtures for the tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from mqt.predictor.ml.helper import get_path_training_data as ml_get_path_training_data
from mqt.predictor.rl.helper import get_path_trained_model as rl_get_path_trained_model

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(scope="session", autouse=True)
def clean_rl_models() -> Generator[None]:
    """Clean up trained RL models after test session."""
    yield

    for model in rl_get_path_trained_model().glob("*.zip"):
        model.unlink()


@pytest.fixture(scope="session", autouse=True)
def clean_ml_models() -> Generator[None]:
    """Clean up trained ML models after test session."""
    yield

    for model in (ml_get_path_training_data() / "trained_model").glob("*.joblib"):
        model.unlink()

    for data in (ml_get_path_training_data() / "training_data_aggregated").glob("*.npy"):
        data.unlink()
