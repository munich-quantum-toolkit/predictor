# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Actions (i.e. compiler passes) available in the reinforcement learning environment."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from qiskit.passmanager import PropertySet


class CompilationOrigin(str, Enum):
    """Enumeration of the origin of the compilation action."""

    QISKIT = "qiskit"
    TKET = "tket"
    BQSKIT = "bqskit"


class PassType(str, Enum):
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
    """

    name: str
    origin: CompilationOrigin | None
    pass_type: PassType
    transpile_pass: Any
    preserves_layout: bool = False
    preserves_routing: bool = False
    preserves_synthesis: bool = False


@dataclass
class DeviceIndependentAction(Action):
    """Action that represents a static compilation pass that can be applied directly."""


@dataclass
class DeviceDependentAction(Action):
    """Device-specific action that depends on a target device.

    Attributes:
        do_while: Optional do-while predicate for pass-manager execution.
    """

    transpile_pass: Any
    do_while: Callable[[PropertySet], bool] | None = None


_ACTIONS: dict[str, Action] = {}


def register_action(action: Action) -> Action:
    """Registers a new Action in the global _ACTIONS registry.

    Args:
        action: Action to register.

    Returns:
        The registered Action.

    Raises:
        ValueError: If an action with the same name is already registered.
    """
    if action.name in _ACTIONS:
        msg = f"Action with name {action.name} already registered."
        raise ValueError(msg)
    _ACTIONS[action.name] = action
    return action


def get_actions_by_pass_type() -> dict[PassType, list[Action]]:
    """Groups registered Actions from the global _ACTIONS registry by PassType.

    Returns:
        A dictionary mapping each PassType to the list of registered Actions of that type.
    """
    result: dict[PassType, list[Action]] = defaultdict(list)
    for action in _ACTIONS.values():
        result[action.pass_type].append(action)
    return result


from mqt.predictor.rl.actions import bqskit_actions, qiskit_actions, tket_actions

for _action in (
    *qiskit_actions.qiskit_optimization_actions(),
    *tket_actions.tket_optimization_actions(),
    qiskit_actions.qiskit_o3_action(),
    qiskit_actions.qiskit_final_optimization_action(),
    *qiskit_actions.qiskit_layout_actions(),
    *bqskit_actions.bqskit_layout_actions(),
    tket_actions.tket_routing_action(),
    bqskit_actions.bqskit_routing_action(),
    qiskit_actions.qiskit_mapping_action(),
    bqskit_actions.bqskit_pam_mapping_action(),
    qiskit_actions.qiskit_synthesis_action(),
    *bqskit_actions.bqskit_synthesis_actions(),
    DeviceIndependentAction(
        "terminate",
        None,
        PassType.TERMINATE,
        transpile_pass=[],
    ),
):
    register_action(_action)

__all__ = [
    "Action",
    "CompilationOrigin",
    "DeviceDependentAction",
    "DeviceIndependentAction",
    "PassType",
    "bqskit_actions",
    "get_actions_by_pass_type",
    "qiskit_actions",
    "register_action",
    "tket_actions",
]
