# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Registry of the actions (i.e. compiler passes) available in the reinforcement learning environment."""

from __future__ import annotations

from collections import defaultdict

from mqt.predictor.rl.actions import bqskit_actions, qiskit_actions, tket_actions
from mqt.predictor.rl.actions.base import Action, DeviceIndependentAction, PassType

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


for _action in (
    *qiskit_actions.qiskit_layout_actions(),
    qiskit_actions.qiskit_mapping_action(),
    qiskit_actions.qiskit_synthesis_action(),
    qiskit_actions.qiskit_o3_action(),
    *qiskit_actions.qiskit_optimization_actions(),
    qiskit_actions.qiskit_final_optimization_action(),
    tket_actions.tket_routing_action(),
    *tket_actions.tket_optimization_actions(),
    *bqskit_actions.bqskit_layout_actions(),
    *bqskit_actions.bqskit_routing_actions(),
    *bqskit_actions.bqskit_mapping_actions(),
    *bqskit_actions.bqskit_synthesis_actions(),
    DeviceIndependentAction(
        "terminate",
        None,
        PassType.TERMINATE,
        transpile_pass=[],
    ),
):
    register_action(_action)
