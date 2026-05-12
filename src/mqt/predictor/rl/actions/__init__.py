# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Actions available in the reinforcement learning environment."""

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
    GENERAL = "general"


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
    """Base class for all actions in the reinforcement learning environment."""

    name: str
    origin: CompilationOrigin
    pass_type: PassType
    transpile_pass: Any


@dataclass
class DeviceIndependentAction(Action):
    """Action that represents a static compilation pass that can be applied directly."""


@dataclass
class DeviceDependentAction(Action):
    """Action that represents a device-specific compilation pass that can be applied to a specific device."""

    transpile_pass: Any
    do_while: Callable[[PropertySet], bool] | None = None


_ACTIONS: dict[str, Action] = {}


def register_action(action: Action) -> Action:
    """Registers a new action in the global actions registry.

    Raises:
        ValueError: If an action with the same name is already registered.
    """
    if action.name in _ACTIONS:
        msg = f"Action with name {action.name} already registered."
        raise ValueError(msg)
    _ACTIONS[action.name] = action
    return action


def remove_action(name: str) -> None:
    """Removes an action from the global actions registry by name.

    Raises:
        ValueError: If no action with the given name is registered.
    """
    if name not in _ACTIONS:
        msg = f"No action with name {name} is registered."
        raise KeyError(msg)
    del _ACTIONS[name]


def get_actions_by_pass_type() -> dict[PassType, list[Action]]:
    """Returns a dictionary mapping each PassType to a list of Actions of that type."""
    result: dict[PassType, list[Action]] = defaultdict(list)
    for action in _ACTIONS.values():
        result[action.pass_type].append(action)
    return result


from mqt.predictor.rl.actions import bqskit_actions as _bqskit_actions
from mqt.predictor.rl.actions import qiskit_actions as _qiskit_actions
from mqt.predictor.rl.actions import tket_actions as _tket_actions

for _action in (
    *_qiskit_actions.qiskit_optimization_actions(),
    *_tket_actions.tket_optimization_actions(),
    _qiskit_actions.qiskit_o3_action(),
    _bqskit_actions.bqskit_optimization_action(),
    _qiskit_actions.qiskit_final_optimization_action(),
    *_qiskit_actions.qiskit_layout_actions(),
    _tket_actions.tket_routing_action(),
    _qiskit_actions.qiskit_mapping_action(),
    _bqskit_actions.bqskit_mapping_action(),
    _qiskit_actions.qiskit_synthesis_action(),
    _bqskit_actions.bqskit_synthesis_action(),
    DeviceIndependentAction(
        "terminate",
        CompilationOrigin.GENERAL,
        PassType.TERMINATE,
        transpile_pass=[],
    ),
):
    register_action(_action)

PreProcessTKETRoutingAfterQiskitLayout = _tket_actions.PreProcessTKETRoutingAfterQiskitLayout
final_layout_pytket_to_qiskit = _tket_actions.final_layout_pytket_to_qiskit
run_tket_action = _tket_actions.run_tket_action

final_layout_bqskit_to_qiskit = _bqskit_actions.final_layout_bqskit_to_qiskit
bqskit_to_qiskit = _bqskit_actions.bqskit_to_qiskit
get_bqskit_native_gates = _bqskit_actions.get_bqskit_native_gates
run_bqskit_action = _bqskit_actions.run_bqskit_action

postprocess_vf2postlayout = _qiskit_actions.postprocess_vf2postlayout
run_qiskit_action = _qiskit_actions.run_qiskit_action

__all__ = [
    "Action",
    "CompilationOrigin",
    "DeviceDependentAction",
    "DeviceIndependentAction",
    "PassType",
    "PreProcessTKETRoutingAfterQiskitLayout",
    "bqskit_to_qiskit",
    "final_layout_bqskit_to_qiskit",
    "final_layout_pytket_to_qiskit",
    "get_actions_by_pass_type",
    "get_bqskit_native_gates",
    "postprocess_vf2postlayout",
    "register_action",
    "remove_action",
    "run_bqskit_action",
    "run_qiskit_action",
    "run_tket_action",
]
