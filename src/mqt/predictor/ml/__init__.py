# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""MQT Predictor - Automatic Device Selection with Device-Specific Circuit Compilation for Quantum Computing."""

from __future__ import annotations

from mqt.predictor.ml import helper
from mqt.predictor.ml.predictor import Predictor, predict_device_for_figure_of_merit, setup_device_predictor

__all__ = [
    "Predictor",
    "helper",
    "predict_device_for_figure_of_merit",
    "setup_device_predictor",
]


def __getattr__(name: str) -> Any:
    """Lightweight lazy loader for deprecated direct imports from mqt.predictor.ml."""
    if name in {"Predictor", "predict_device_for_figure_of_merit", "setup_device_predictor"}:
        from . import predictor as _predictor

        return getattr(_predictor, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
