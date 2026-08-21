# Upgrade Guide

This document describes breaking changes and how to upgrade. For a complete list
of changes including minor and patch releases, please refer to the
[changelog](CHANGELOG.md).

## [Unreleased]

### Expanded Qiskit action set

The RL action space now includes the following Qiskit passes:

- the `TrivialLayout` and `ElidePermutations` layout actions;
- the `SabreSwap`, `BasicSwap`, and `LookaheadSwap` routing actions; and
- the `RemoveIdentityEquivalent` and `Optimize1qGatesSimpleCommutation`
  optimization actions.

`ElidePermutations` establishes a trivial layout in the same action so its
output permutation remains part of the canonical layout. `OptimizeCliffords` now
collects standard Clifford gates before optimizing and decomposes the result for
subsequent passes.

Existing RL models must be retrained because the action-space size and the
indices of later actions have changed. Code that persists or selects actions by
numeric index must be updated.

### Compact scalar RL observations

The `num_qubits` and `depth` entries in RL observations are now one-element
`float32` arrays in `[0, 1]` instead of discrete integers. The qubit count is
linearly scaled by the target device's qubit count; the depth is `log1p`-scaled
and capped at 999,999. Existing RL models must be retrained, and code that
consumes `PredictorEnv` observations directly must handle the new array values.

### RL operation features

The RL observation now includes normalized frequencies for supported OpenQASM
gates and measurements. Existing RL models must be retrained, and code that
consumes `PredictorEnv` observations directly must handle the additional keys.

### End of support for Python 3.10

Starting with this release, MQT Predictor no longer supports Python 3.10. As a
result, MQT Predictor is no longer tested under Python 3.10 and requires Python
3.11 or later.

### Repeated stochastic Qiskit actions

The stochastic `QiskitSabreMapping` action now evaluates 20 candidates by
default and retains the candidate with the best configured figure of merit. This
can increase compilation time and change the selected circuit. Set
`stochastic_action_trials=1` when constructing `PredictorEnv` to retain the
previous single-attempt behavior.

### Atomic BQSKit compilation actions

The composite actions `BQSKitO2`, `BQSKitSynthesis`, and `BQSKitMapping` are no
longer available. Each of them ran several compilation steps at once. They have
been replaced by smaller actions for individual BQSKit passes, allowing the RL
predictor to choose layout, routing, mapping, and synthesis steps independently.

Code that selects actions by name must use the new pass-specific names. Existing
RL models must also be retrained because the available action space has changed.
The new actions include:

- the `GreedyPlacementPass`, `TrivialPlacementPass`, `StaticPlacementPass`, and
  `GeneralizedSabreLayoutPass` layout actions;
- the `GeneralizedSabreRoutingPass` routing action;
- the `BQSKitSABREMapping` mapping action; and
- the `QSearchSynthesisPass`, `LEAPSynthesisPass`, `WalshDiagonalSynthesisPass`,
  `FullQSDPass`, `BlockZXZPass`, and `FullBlockZXZPass` synthesis actions.

See the
[framework setup](docs/setup.md#step-2-train-reinforcement-learning-models) for
how RL actions are used and the
[BQSKit pass documentation](https://bqskit.readthedocs.io/en/latest/source/passes.html)
for details about the individual passes.

### Default RL MDP strategy

`PredictorEnv`, `Predictor`, and `rl_compile` now use the `v3` MDP strategy by
default. To retain the original MQT Predictor behavior, pass `mdp="v2"`
explicitly.

Existing models trained with the previous MQT Predictor strategy can be reused
with the `v2` policy. Copy or rename `model_<figure_of_merit>_<device>.zip` to
`model_<figure_of_merit>_<device>_v2.zip`, then pass `mdp="v2"` to `Predictor`
or `rl_compile`. Models for the new default `v3` policy must be retrained.

## [2.4.0]

### Trained RL model names

`Predictor.train_model` no longer accepts `model_name`; models now always use
`model_<figure_of_merit>_<device>`. Custom names were removed because
`compile_as_predicted` expects this fixed name when loading a model.

### Reproducible RL training

`test=True` no longer sets a random seed implicitly. Pass `seed=0` explicitly if
you require the previous deterministic behavior:

```python
predictor.train_model(timesteps=10, test=True, seed=0)
```

### Low-level RL modules

The SDK-specific action and conversion helpers have moved out of
`mqt.predictor.rl.actions` and `mqt.predictor.rl.parsing`:

- Qiskit helpers now live in `mqt.predictor.rl.actions.qiskit_actions`.
- TKET helpers now live in `mqt.predictor.rl.actions.tket_actions`.
- BQSKit helpers now live in `mqt.predictor.rl.actions.bqskit_actions`.

The shared action types and registry functions remain available from
`mqt.predictor.rl.actions`. The `remove_action` function and the
`CompilationOrigin.GENERAL` enum member have been removed. The termination
action now uses `None` as its origin.

### End of support for x86 macOS systems

Starting with this release, MQT Predictor no longer supports x86 macOS systems.
This step is necessary to ensure compatibility with PyTorch. x86 macOS systems
are no longer tested in our CI.

## [2.3.0]

In this release, we have migrated to using Qiskit's `Target` class to represent
quantum devices. This change allows for better compatibility with the latest MQT
Bench version and improves the overall usability of the library. Beyond that, we
also support Qiskit v2 now.

Furthermore, both the ML and RL parts of MQT Predictor have been refactored to
enhance their functionality and usability: The ML setup has been simplified and
streamlined, making it easier to use and integrate into your workflows. The RL
action handling has been updated to utilize dataclasses, which improves the
structure and clarity of the code, making it easier to understand and maintain.

### General

MQT Predictor has moved to the
[munich-quantum-toolkit](https://github.com/munich-quantum-toolkit) GitHub
organization under <https://github.com/munich-quantum-toolkit/predictor>. While
most links should be automatically redirected, please update any links in your
code to point to the new location. All links in the documentation have been
updated accordingly.

<!-- Version links -->

[unreleased]: https://github.com/munich-quantum-toolkit/predictor/compare/v2.4.0...HEAD
[2.4.0]: https://github.com/munich-quantum-toolkit/predictor/compare/v2.3.0...v2.4.0
[2.3.0]: https://github.com/munich-quantum-toolkit/predictor/compare/v2.2.0...v2.3.0
