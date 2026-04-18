# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Actions package for the reinforcement learning environment."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import TypeAlias

    from bqskit import Circuit
    from qiskit.passmanager import PropertySet

    from .qiskit_actions import AIRouting, SafeAIRouting

    PassList: TypeAlias = list[object]
    BQSKitMapping: TypeAlias = tuple[Circuit, tuple[int, ...], tuple[int, ...]]
    BQSKitCompileFn: TypeAlias = Callable[[Circuit], Circuit | BQSKitMapping]
    TranspilePassSpec: TypeAlias = PassList | Callable[..., PassList] | Callable[..., BQSKitCompileFn]


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
    transpile_pass: TranspilePassSpec
    stochastic: bool | None = False
    preserves_layout: bool | None = False
    preserves_routing: bool | None = False
    preserves_synthesis: bool | None = False


@dataclass
class DeviceIndependentAction(Action):
    """Action that represents a static compilation pass that can be applied directly."""


@dataclass
class DeviceDependentAction(Action):
    """Action that represents a device-specific compilation pass that can be applied to a specific device."""

    transpile_pass: Callable[..., PassList] | Callable[..., BQSKitCompileFn]
    do_while: Callable[[PropertySet], bool] | None = None


_ACTIONS: dict[str, Action] = {}


def register_action(action: Action) -> Action:
    """Register a new action in the global actions registry."""
    if action.name in _ACTIONS:
        msg = f"Action with name {action.name} already registered."
        raise ValueError(msg)
    _ACTIONS[action.name] = action
    return action


def remove_action(name: str) -> None:
    """Remove an action from the global actions registry by name."""
    if name not in _ACTIONS:
        msg = f"No action with name {name} is registered."
        raise KeyError(msg)
    del _ACTIONS[name]


def get_actions_by_pass_type() -> dict[PassType, list[Action]]:
    """Return a dictionary mapping each pass type to the registered actions of that type."""
    result: dict[PassType, list[Action]] = defaultdict(list)
    for action in _ACTIONS.values():
        result[action.pass_type].append(action)
    return result


from . import bqskit_actions as _bqskit_actions
from . import qiskit_actions as _qiskit_actions
from . import tket_actions as _tket_actions

HAS_AI_ROUTING = _qiskit_actions.HAS_AI_ROUTING
IS_WIN_PY313 = _qiskit_actions.IS_WIN_PY313
ensure_ai_routing_runtime_available = _qiskit_actions.ensure_ai_routing_runtime_available
fom_aware_compile = _qiskit_actions.fom_aware_compile
postprocess_vf2postlayout = _qiskit_actions.postprocess_vf2postlayout
run_qiskit_action = _qiskit_actions.run_qiskit_action

PreProcessTKETRoutingAfterQiskitLayout = _tket_actions.PreProcessTKETRoutingAfterQiskitLayout
final_layout_pytket_to_qiskit = _tket_actions.final_layout_pytket_to_qiskit
prepare_noise_data = _tket_actions.prepare_noise_data
run_tket_action = _tket_actions.run_tket_action
translate_tket_placement_to_qiskit_layout = _tket_actions.translate_tket_placement_to_qiskit_layout

final_layout_bqskit_routing_to_qiskit = _bqskit_actions.final_layout_bqskit_routing_to_qiskit
final_layout_bqskit_to_qiskit = _bqskit_actions.final_layout_bqskit_to_qiskit
run_bqskit_action = _bqskit_actions.run_bqskit_action

if HAS_AI_ROUTING:
    AIRouting = _qiskit_actions.AIRouting
    SafeAIRouting = _qiskit_actions.SafeAIRouting


register_action(
    DeviceIndependentAction(
        "terminate",
        CompilationOrigin.GENERAL,
        PassType.TERMINATE,
        transpile_pass=[],
    )
)


__all__ = [
    "HAS_AI_ROUTING",
    "IS_WIN_PY313",
    "Action",
    "CompilationOrigin",
    "DeviceDependentAction",
    "DeviceIndependentAction",
    "PassType",
    "PreProcessTKETRoutingAfterQiskitLayout",
    "ensure_ai_routing_runtime_available",
    "final_layout_bqskit_routing_to_qiskit",
    "final_layout_bqskit_to_qiskit",
    "final_layout_pytket_to_qiskit",
    "fom_aware_compile",
    "get_actions_by_pass_type",
    "postprocess_vf2postlayout",
    "prepare_noise_data",
    "register_action",
    "remove_action",
    "run_bqskit_action",
    "run_qiskit_action",
    "run_tket_action",
    "translate_tket_placement_to_qiskit_layout",
]

if HAS_AI_ROUTING:
    __all__ += ["AIRouting", "SafeAIRouting"]
