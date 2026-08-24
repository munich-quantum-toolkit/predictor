# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Actions (i.e. compiler passes) available in the reinforcement learning environment."""

from __future__ import annotations

from mqt.predictor.rl.actions import bqskit_actions, qiskit_actions, tket_actions
from mqt.predictor.rl.actions.base import (
    Action,
    CompilationOrigin,
    DeferredDeviceAction,
    DeviceIndependentAction,
    PassType,
)
from mqt.predictor.rl.actions.registry import get_actions_by_pass_type, register_action

__all__ = [
    "Action",
    "CompilationOrigin",
    "DeferredDeviceAction",
    "DeviceIndependentAction",
    "PassType",
    "bqskit_actions",
    "get_actions_by_pass_type",
    "qiskit_actions",
    "register_action",
    "tket_actions",
]
