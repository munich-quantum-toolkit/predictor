# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Qiskit optimization actions matching KIT's ICSE pass set."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qiskit.transpiler.optimization_metric import OptimizationMetric
from qiskit.transpiler.passes import (
    Collect1qRuns,
    Collect2qBlocks,
    CollectCliffords,
    CollectLinearFunctions,
    CollectMultiQBlocks,
    CommutationAnalysis,
    CommutativeCancellation,
    CommutativeInverseCancellation,
    CommutativeOptimization,
    ConsolidateBlocks,
    ContractIdleWiresInControlFlow,
    ElidePermutations,
    HighLevelSynthesis,
    HoareOptimizer,
    InverseCancellation,
    LightCone,
    Optimize1qGates,
    Optimize1qGatesDecomposition,
    Optimize1qGatesSimpleCommutation,
    OptimizeAnnotated,
    OptimizeCliffords,
    OptimizeCliffordT,
    OptimizeSwapBeforeMeasure,
    RemoveDiagonalGatesBeforeMeasure,
    RemoveFinalReset,
    RemoveIdentityEquivalent,
    RemoveResetInZeroState,
    ResetAfterMeasureSimplification,
    Split2QUnitaries,
    SubstitutePi4Rotations,
    TemplateOptimization,
    UnitarySynthesis,
)

from mqt.predictor.rl.actions import Action, CompilationOrigin, DeviceIndependentAction, PassType

if TYPE_CHECKING:
    from qiskit.passmanager.base_tasks import Task
    from qiskit.transpiler import Target


KIT_PASS_NAMES: tuple[str, ...] = (
    "Collect1qRuns",
    "CommutationAnalysis",
    "Collect2qBlocks",
    "CollectMultiQBlocks",
    "CollectCliffords",
    "CollectLinearFunctions",
    "ConsolidateBlocks",
    "Optimize1qGates",
    "Optimize1qGatesDecomposition",
    "Optimize1qGatesSimpleCommutation",
    "Split2QUnitaries",
    "SubstitutePi4Rotations",
    "OptimizeCliffordT",
    "InverseCancellation",
    "CommutativeCancellation",
    "CommutativeInverseCancellation",
    "CommutativeOptimization",
    "HoareOptimizer",
    "OptimizeAnnotated",
    "TemplateOptimization",
    "OptimizeCliffords",
    "RemoveIdentityEquivalent",
    "RemoveFinalReset",
    "RemoveResetInZeroState",
    "RemoveDiagonalGatesBeforeMeasure",
    "ResetAfterMeasureSimplification",
    "OptimizeSwapBeforeMeasure",
    "ElidePermutations",
    "ContractIdleWiresInControlFlow",
    "LightCone",
)

KIT_ACTION_NAMES: tuple[str, ...] = (
    "Collect1qRuns",
    "Collect2qBlocks",
    "CollectMultiQBlocks",
    "CommutationAnalysis",
    "CollectCliffords",
    "CollectLinearFunctions",
    "ConsolidateBlocks",
    "HoareOptimizer",
    "OptimizeAnnotated",
    "OptimizeCliffords",
    "TemplateOptimization",
    "CommutativeCancellation",
    "CommutativeInverseCancellation",
    "CommutativeOptimization",
    "InverseCancellation",
    "Optimize1qGates",
    "Optimize1qGatesDecomposition",
    "Optimize1qGatesSimpleCommutation",
    "OptimizeCliffordT",
    "Split2QUnitaries",
    "SubstitutePi4Rotations",
    "ContractIdleWiresInControlFlow",
    "ElidePermutations",
    "LightCone",
    "OptimizeSwapBeforeMeasure",
    "RemoveDiagonalGatesBeforeMeasure",
    "RemoveFinalReset",
    "RemoveIdentityEquivalent",
    "RemoveResetInZeroState",
    "ResetAfterMeasureSimplification",
)


def kit_optimization_actions() -> list[Action]:
    """Return optimization actions for KIT's all-to-all, basis-S protocol.

    The action names match the concrete Qiskit passes in the ICSE Qiskit-ML
    ``CustomPassManagerBuilder.PASS_MAP``. Aggregation anchors include the
    required synthesis step so every action leaves the circuit executable by the
    next action and by the reward translation boundary.

    Returns:
        A list of device-independent Qiskit optimization actions.
    """
    return [_kit_action(name) for name in KIT_ACTION_NAMES]


def _kit_action(name: str) -> Action:
    """Create one opt-only action by pass name."""
    pass_builders = {
        "Collect1qRuns": lambda: [Collect1qRuns()],
        "Collect2qBlocks": lambda: [Collect2qBlocks()],
        "CollectMultiQBlocks": lambda: [CollectMultiQBlocks()],
        "CommutationAnalysis": lambda: [CommutationAnalysis()],
        "CollectCliffords": lambda: [
            CollectCliffords(),
            OptimizeCliffords(),
            HighLevelSynthesis(optimization_metric=OptimizationMetric.COUNT_2Q),
        ],
        "CollectLinearFunctions": lambda: [
            CollectLinearFunctions(),
            HighLevelSynthesis(optimization_metric=OptimizationMetric.COUNT_2Q),
        ],
        "ConsolidateBlocks": lambda: [
            Collect1qRuns(),
            Collect2qBlocks(),
            ConsolidateBlocks(),
            UnitarySynthesis(approximation_degree=1.0),
        ],
        "OptimizeAnnotated": lambda: [OptimizeAnnotated()],
        "OptimizeCliffords": lambda: [OptimizeCliffords()],
        "TemplateOptimization": lambda: [TemplateOptimization()],
        "CommutativeCancellation": lambda: [CommutativeCancellation()],
        "CommutativeInverseCancellation": lambda: [CommutativeInverseCancellation()],
        "CommutativeOptimization": lambda: [CommutativeOptimization()],
        "InverseCancellation": lambda: [InverseCancellation()],
        "Optimize1qGates": lambda: [Optimize1qGates()],
        "Optimize1qGatesDecomposition": lambda: [Optimize1qGatesDecomposition()],
        "Optimize1qGatesSimpleCommutation": lambda: [Optimize1qGatesSimpleCommutation()],
        "OptimizeCliffordT": lambda: [OptimizeCliffordT()],
        "Split2QUnitaries": lambda: [Split2QUnitaries()],
        "SubstitutePi4Rotations": lambda: [SubstitutePi4Rotations()],
        "ContractIdleWiresInControlFlow": lambda: [ContractIdleWiresInControlFlow()],
        "ElidePermutations": lambda: [ElidePermutations()],
        "LightCone": lambda: [LightCone()],
        "OptimizeSwapBeforeMeasure": lambda: [OptimizeSwapBeforeMeasure()],
        "RemoveDiagonalGatesBeforeMeasure": lambda: [RemoveDiagonalGatesBeforeMeasure()],
        "RemoveFinalReset": lambda: [RemoveFinalReset()],
        "RemoveIdentityEquivalent": lambda: [RemoveIdentityEquivalent()],
        "RemoveResetInZeroState": lambda: [RemoveResetInZeroState()],
        "ResetAfterMeasureSimplification": lambda: [ResetAfterMeasureSimplification()],
    }

    transpile_pass = _hoare_optimizer_passes if name == "HoareOptimizer" else pass_builders[name]()

    return DeviceIndependentAction(
        name,
        CompilationOrigin.QISKIT,
        PassType.OPT,
        transpile_pass,
    )


def _hoare_optimizer_passes(_device: Target) -> list[Task]:
    """Build HoareOptimizer lazily because it requires the optional z3 package."""
    return [HoareOptimizer()]
