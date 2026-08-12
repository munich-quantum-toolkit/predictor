# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Action types describing the compiler passes available in the reinforcement learning environment."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from qiskit.passmanager import PropertySet


class CompilationOrigin(StrEnum):
    """Enumeration of the origin of the compilation action."""

    QISKIT = "qiskit"
    TKET = "tket"
    BQSKIT = "bqskit"


class PassType(StrEnum):
    """Enumeration of the type of compilation pass."""

    OPT = "optimization"
    SYNTHESIS = "synthesis"
    MAPPING = "mapping"
    LAYOUT = "layout"
    ROUTING = "routing"
    FINAL_OPT = "final_optimization"
    TERMINATE = "terminate"


@dataclass
class Action:
    """Base class for all actions in the reinforcement learning environment.

    Attributes:
        name: Unique action name.
        origin: SDK origin of the action; ``None`` for terminate.
        pass_type: Category of pass represented by this action.
        transpile_pass: Pass object(s) executed for this action.
        preserves_layout: Whether action preserves existing layout.
        preserves_routing: Whether action preserves existing routing.
        preserves_synthesis: Whether action preserves synthesis state.
        stochastic: Whether repeated execution can yield different results.
    """

    name: str
    origin: CompilationOrigin | None
    pass_type: PassType
    transpile_pass: Any
    preserves_layout: bool = False
    preserves_routing: bool = False
    preserves_synthesis: bool = False
    stochastic: bool = False


@dataclass
class DeviceIndependentAction(Action):
    """Action that represents a static compilation pass that can be applied directly."""


@dataclass
class DeferredDeviceAction(Action):
    """Action that defers construction of a device-specific pass.

    Attributes:
        do_while: Optional do-while predicate for pass-manager execution.
    """

    transpile_pass: Any
    do_while: Callable[[PropertySet], bool] | None = None
