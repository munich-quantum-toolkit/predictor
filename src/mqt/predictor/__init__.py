# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""MQT Predictor - Automatic Device Selection with Device-Specific Circuit Compilation for Quantum Computing."""

from __future__ import annotations

import logging
from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mqt.predictor.qcompile import qcompile

__all__ = [
    "qcompile",
]

logger = logging.getLogger("mqt-predictor")

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logger_formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(logger_formatter)
logger.addHandler(console_handler)
logger.setLevel(logging.DEBUG)
logger.propagate = False


def __getattr__(name: str) -> object:
    """Lazily import public package exports."""
    if name != "qcompile":
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)

    module = import_module("mqt.predictor.qcompile")
    return getattr(module, name)
