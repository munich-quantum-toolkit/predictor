# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the compilation with reinforcement learning."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from gymnasium.spaces import Dict, Discrete
from mqt.bench import BenchmarkLevel, get_benchmark
from mqt.bench.targets import get_device
from qiskit import QuantumCircuit
from qiskit.circuit.library import CXGate
from qiskit.qasm2 import dump
from qiskit.quantum_info import Clifford, Operator
from qiskit.transpiler import CouplingMap, InstructionProperties, Layout, PassManager, Target, TranspileLayout
from qiskit.transpiler.passes import BasicSwap, GatesInBasis, SabreLayout, SabreSwap

from mqt.predictor.rl import Predictor, rl_compile
from mqt.predictor.rl import predictor as predictor_module
from mqt.predictor.rl import predictorenv as predictorenv_module
from mqt.predictor.rl.actions import (
    CompilationOrigin,
    DeviceIndependentAction,
    PassType,
    get_actions_by_pass_type,
    qiskit_actions,
    register_action,
)
from mqt.predictor.rl.actions import registry as actions_registry_module
from mqt.predictor.rl.helper import create_feature_dict, get_path_trained_model

if TYPE_CHECKING:
    from collections.abc import Sequence

    from qiskit.passmanager.base_tasks import Task


def test_predictor_env_reset_from_string() -> None:
    """Test the reset function of the predictor environment with a quantum circuit given as a string as input."""
    device = get_device("ibm_heron_156")
    predictor = Predictor(figure_of_merit="expected_fidelity", device=device)
    qasm_path = Path("test.qasm")
    qc = get_benchmark("dj", BenchmarkLevel.ALG, 3)
    with qasm_path.open("w", encoding="utf-8") as f:
        dump(qc, f)
    assert predictor.env.reset(qc=qasm_path)[0] == create_feature_dict(qc)


@pytest.mark.parametrize(("device_name", "num_qubit_values"), [("quantinuum_h2_56", 128), ("ibm_heron_156", 157)])
def test_predictor_env_observation_space_covers_device(device_name: str, num_qubit_values: int) -> None:
    """Test device-width observations while preserving the encoding for smaller devices."""
    device = get_device(device_name)
    env = predictorenv_module.PredictorEnv(device=device)
    qc = QuantumCircuit(device.num_qubits)
    qc.h(0)
    qc.cx(0, 1)

    observation, _ = env.reset(qc=qc)

    assert env.observation_space.contains(observation)
    assert isinstance(env.observation_space, Dict)
    num_qubits_space = env.observation_space["num_qubits"]
    assert isinstance(num_qubits_space, Discrete)
    assert num_qubits_space.n == num_qubit_values


def test_predictor_env_esp_error() -> None:
    """Test the predictor environment with ESP as figure of merit and missing calibration data."""
    device = get_device("quantinuum_h2_56")
    with pytest.raises(
        ValueError, match=re.escape("Missing calibration data for ESP calculation on quantinuum_h2_56.")
    ):
        Predictor(figure_of_merit="estimated_success_probability", device=device)


def test_predictor_env_hellinger_error() -> None:
    """Test the predictor environment with the Estimated Hellinger Distance as figure of merit and a missing model."""
    device = get_device("ibm_heron_156")
    with pytest.raises(
        ValueError, match=re.escape("Missing trained model for Hellinger distance estimates on ibm_heron_156.")
    ):
        Predictor(figure_of_merit="estimated_hellinger_distance", device=device)


@pytest.mark.model_training
def test_qcompile_with_newly_trained_models() -> None:
    """Test the qcompile function with a newly trained model.

    Important: Those trained models are used in later tests and must not be deleted.
    To test ESP as well, training must be done with a device that provides all relevant information (i.e. T1, T2 and gate times).
    """
    figure_of_merit = "expected_fidelity"
    device = get_device("ibm_heron_156")
    qc = get_benchmark("ghz", BenchmarkLevel.ALG, 3)
    predictor = Predictor(figure_of_merit=figure_of_merit, device=device)

    model_name = "model_" + figure_of_merit + "_" + device.description
    model_path = Path(get_path_trained_model() / (model_name + ".zip"))
    if not model_path.exists():
        with pytest.raises(
            FileNotFoundError,
            match=re.escape(
                "The RL model 'model_expected_fidelity_ibm_heron_156' is not trained yet. Please train the model before using it."
            ),
        ):
            rl_compile(qc, device=device, figure_of_merit=figure_of_merit)

    predictor.train_model(timesteps=512, test=True, seed=0)

    qc_compiled, compilation_information = rl_compile(qc, device=device, figure_of_merit=figure_of_merit)

    check_nat_gates = GatesInBasis(basis_gates=device.operation_names)
    check_nat_gates(qc_compiled)
    only_nat_gates = check_nat_gates.property_set["all_gates_in_basis"]

    assert qc_compiled.layout is not None
    assert compilation_information is not None
    assert only_nat_gates, "Circuit should only contain native gates but was not detected as such"


def test_qcompile_with_false_input() -> None:
    """Test the qcompile function with false input."""
    qc = get_benchmark("dj", BenchmarkLevel.ALG, 5)
    with pytest.raises(ValueError, match=re.escape("figure_of_merit must not be None if predictor_singleton is None.")):
        rl_compile(qc, device=get_device("quantinuum_h2_56"), figure_of_merit=None)
    with pytest.raises(ValueError, match=re.escape("device must not be None if predictor_singleton is None.")):
        rl_compile(qc, device=None, figure_of_merit="expected_fidelity")


def test_warning_for_unidirectional_device() -> None:
    """Test the warning for a unidirectional device."""
    target = Target()
    target.add_instruction(CXGate(), {(0, 1): InstructionProperties()})
    target.description = "uni-directional device"

    msg = "The connectivity of the device 'uni-directional device' is uni-directional and MQT Predictor might return a compiled circuit that assumes bi-directionality."
    with pytest.warns(UserWarning, match=re.escape(msg)):
        Predictor(figure_of_merit="expected_fidelity", device=target)


def test_predictor_env_action_indices_follow_pass_type_order() -> None:
    """Policy indices group actions by pass type and preserve order within each group."""
    env = predictorenv_module.PredictorEnv(device=get_device("ibm_heron_156"))
    action_groups = (
        (PassType.SYNTHESIS, env.actions_synthesis_indices),
        (PassType.OPT, env.actions_opt_indices),
        (PassType.LAYOUT, env.actions_layout_indices),
        (PassType.ROUTING, env.actions_routing_indices),
        (PassType.MAPPING, env.actions_mapping_indices),
        (PassType.FINAL_OPT, env.actions_final_optimization_indices),
    )
    registered_actions = get_actions_by_pass_type()

    for pass_type, indices in action_groups:
        assert [env.action_set[index] for index in indices] == registered_actions[pass_type]
    assert [index for _, indices in action_groups for index in indices] == list(range(env.action_terminate_index))
    assert env.action_set[env.action_terminate_index].name == "terminate"
    assert env.action_space.contains(env.action_terminate_index)
    assert not env.action_space.contains(env.action_terminate_index + 1)


def test_predictor_env_truncates_at_max_steps() -> None:
    """Test that the environment truncates episodes that hit the step limit."""
    device = get_device("ibm_heron_156")
    env = predictorenv_module.PredictorEnv(device=device, max_steps=1)
    qc = QuantumCircuit(1)
    qc.h(0)
    env.reset(qc)

    _, reward_val, terminated, truncated, info = env.step(env.actions_opt_indices[0])

    assert reward_val == 0
    assert not terminated
    assert truncated
    assert info["truncation_reason"] == "max_steps_exceeded"


def test_predictor_env_actions_after_layout_with_non_native_unrouted_circuit() -> None:
    """Test valid actions for a laid-out circuit that still needs synthesis and routing."""
    device = get_device("ibm_heron_156")
    env = predictorenv_module.PredictorEnv(device=device)
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 2)
    env.reset(qc)

    env.layout = TranspileLayout(
        initial_layout=Layout({qubit: index for index, qubit in enumerate(qc.qubits)}),
        input_qubit_mapping={qubit: index for index, qubit in enumerate(qc.qubits)},
        final_layout=None,
        _output_qubit_list=qc.qubits,
        _input_qubit_count=qc.num_qubits,
    )

    valid_actions = env.determine_valid_actions_for_state()

    assert set(env.actions_synthesis_indices).issubset(valid_actions)
    assert set(env.actions_routing_indices).issubset(valid_actions)
    assert set(env.actions_opt_indices).issubset(valid_actions)
    assert env.action_terminate_index not in valid_actions


@pytest.mark.parametrize("already_routed", [False, True])
def test_predictor_env_qiskit_routing_updates_final_layout(already_routed: bool) -> None:
    """Test that Qiskit routing actions update the tracked final layout."""
    device = get_device("ibm_heron_156")
    env = predictorenv_module.PredictorEnv(device=device)
    qc = QuantumCircuit(3)
    qc.cx(0, 2)
    env.reset(qc)

    initial_layout = Layout({qubit: index for index, qubit in enumerate(qc.qubits)})
    final_layout = Layout({qc.qubits[0]: 1, qc.qubits[1]: 2, qc.qubits[2]: 0})
    env.layout = TranspileLayout(
        initial_layout=initial_layout,
        input_qubit_mapping={qubit: index for index, qubit in enumerate(qc.qubits)},
        final_layout=final_layout if already_routed else None,
        _output_qubit_list=qc.qubits,
        _input_qubit_count=qc.num_qubits,
    )

    action = DeviceIndependentAction(
        name="BasicSwap",
        pass_type=PassType.ROUTING,
        transpile_pass=[BasicSwap(CouplingMap.from_line(3))],
        origin=CompilationOrigin.QISKIT,
    )
    routing_action_index = next(iter(env.actions_routing_indices))
    env.action_set[routing_action_index] = action
    altered_qc = env.apply_action(action_index=routing_action_index)

    assert env.layout is altered_qc.layout
    assert env.layout.initial_layout == initial_layout
    assert env.layout.routing_permutation() == ([0, 2, 1] if already_routed else [1, 0, 2])
    assert final_layout == Layout({qc.qubits[0]: 1, qc.qubits[1]: 2, qc.qubits[2]: 0})


@pytest.mark.parametrize("wrapped", [False, True])
def test_clifford_optimization_collects_and_decomposes(wrapped: bool) -> None:
    """Test Clifford optimization on standard gates and existing Clifford blocks."""
    env = predictorenv_module.PredictorEnv(device=get_device("ibm_heron_156"))
    definition = QuantumCircuit(1)
    definition.h(0)
    for _ in range(4):
        definition.s(0)
    circuit = QuantumCircuit(1)
    if wrapped:
        circuit.append(Clifford(definition), [0])
    else:
        circuit = definition
    env.reset(circuit)

    if wrapped:
        cancellation_index = next(
            index for index, action in env.action_set.items() if action.name == "CommutativeCancellation"
        )
        assert env.apply_action(cancellation_index).count_ops() == {"clifford": 1}

    optimize_cliffords_index = next(
        index for index, action in env.action_set.items() if action.name == "OptimizeCliffords"
    )
    optimized = env.apply_action(optimize_cliffords_index)
    assert "clifford" not in optimized.count_ops()
    assert optimized.size() == 1
    assert Operator(optimized).equiv(Operator(circuit))


def test_seeded_resets_sample_a_repeatable_sequence(tmp_path: Path) -> None:
    """The initial seed controls circuit selection across subsequent episode resets."""
    filenames = ["c_2.qasm", "a_2.qasm", "b_2.qasm"]
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    for filename in filenames:
        with (tmp_path / filename).open("w", encoding="utf-8") as stream:
            dump(qc, stream)

    env = predictorenv_module.PredictorEnv(device=get_device("ibm_heron_156"), path_training_circuits=tmp_path)
    samples = []
    for seed in (123, None, None, None):
        env.reset(seed=seed)
        samples.append(Path(env.filename).name)

    rng = np.random.default_rng(123)
    assert samples == [sorted(filenames)[rng.integers(len(filenames))] for _ in samples]


def test_rl_compile_seeds_policy_and_layout_per_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """A seeded compilation repeats its SABRE layout without changing later calls."""
    device = get_device("ibm_heron_156")
    predictor = Predictor("expected_fidelity", device, max_steps=2)
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    modes: list[bool] = []
    settings: list[tuple[int | None, int, int]] = []

    class MappingPolicy:
        """Choose synthesis, then SABRE mapping."""

        @staticmethod
        def predict(_obs: object, *, action_masks: Sequence[bool], deterministic: bool) -> tuple[int, None]:
            modes.append(deterministic)
            name = "QiskitSabreMapping" if predictor.env.used_actions else "BasisTranslator"
            index = next(i for i, action in predictor.env.action_set.items() if action.name == name)
            assert action_masks[index]
            return index, None

    def capture_passes(passes: list[Task]) -> PassManager:
        settings.extend(
            (task.seed, task.layout_trials, task.swap_trials) for task in passes if isinstance(task, SabreLayout)
        )
        return PassManager(passes)

    monkeypatch.setattr(predictor_module, "load_model", lambda _name: MappingPolicy())
    monkeypatch.setattr(qiskit_actions, "PassManager", capture_passes)
    layouts = []
    for seed in (123, None, 123):
        compiled, actions = rl_compile(qc, device, predictor_singleton=predictor, seed=seed)
        assert actions == ["BasisTranslator", "QiskitSabreMapping"]
        assert compiled.layout is not None
        layouts.append(compiled.layout.final_index_layout())

    assert modes == [True, True, False, False, True, True]
    assert settings[0] == settings[2]
    assert settings[0][0] is not None
    assert settings[0][1:] == (1, 1)
    assert settings[1][0] is None
    assert layouts[0] == layouts[2]


def test_seeded_sabre_swap_preserves_registered_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """Seeding a registered routing action affects only that invocation."""
    swap = SabreSwap(CouplingMap.from_line(3), seed=17, trials=4)
    action = DeviceIndependentAction("SabreSwap", CompilationOrigin.QISKIT, PassType.ROUTING, [swap])
    qc = QuantumCircuit(3)
    qc.cx(0, 2)
    device = get_device("ibm_heron_156")
    settings = []

    def capture_passes(passes: list[Task]) -> PassManager:
        task = passes[0]
        assert isinstance(task, SabreSwap)
        settings.append((task.seed, task.trials))
        return PassManager(passes)

    monkeypatch.setattr(qiskit_actions, "PassManager", capture_passes)
    for seed in (123, None):
        qiskit_actions.run_qiskit_action(action, qc, device, layout=None, seed=seed)

    assert settings == [(123, 1), (17, 4)]


@pytest.fixture
def line_env() -> predictorenv_module.PredictorEnv:
    """Provide a three-qubit target with nearest-neighbor interactions."""
    target = Target.from_configuration(
        basis_gates=["rz", "sx", "x", "cx"], coupling_map=CouplingMap.from_line(3), num_qubits=3
    )
    target.description = "bidirectional line"
    return predictorenv_module.PredictorEnv(device=target)


def _apply_qiskit_action(env: predictorenv_module.PredictorEnv, name: str) -> QuantumCircuit:
    index = next(i for i, action in env.action_set.items() if action.name == name)
    env.state = env.apply_action(index)
    return env.state


def test_qiskit_trivial_layout_keeps_logical_order(line_env: predictorenv_module.PredictorEnv) -> None:
    """Layout pads a narrow circuit and keeps its logical qubits in order."""
    circuit = QuantumCircuit(2)
    circuit.cx(0, 1)
    line_env.reset(circuit)

    laid_out = _apply_qiskit_action(line_env, "TrivialLayout")

    assert laid_out.num_qubits == 3
    assert line_env.layout is not None
    assert line_env.layout.final_index_layout() == [0, 1]


def test_qiskit_sabre_routes_and_tracks_measurements(line_env: predictorenv_module.PredictorEnv) -> None:
    """SABRE routes a nonadjacent interaction and preserves logical measurement results."""
    circuit = QuantumCircuit(3)
    circuit.x(0)
    circuit.cx(0, 2)
    circuit.measure_all()
    line_env.reset(circuit)
    _apply_qiskit_action(line_env, "TrivialLayout")

    routed = _apply_qiskit_action(line_env, "SabreSwap")

    assert line_env.is_circuit_routed(routed, CouplingMap.from_line(3))
    assert line_env.layout is not None
    final = line_env.layout.final_index_layout()
    assert sorted(final) == [0, 1, 2]
    assert routed.count_ops().get("swap", 0) >= 1
    measured = {
        routed.find_bit(item.clbits[0]).index: routed.find_bit(item.qubits[0]).index
        for item in routed.data
        if item.operation.name == "measure"
    }
    assert measured == dict(enumerate(final))


def test_run_qiskit_action_skips_sabre_on_unrestricted_target() -> None:
    """The standalone Qiskit action preserves layout when the target has no coupling map."""
    target = Target.from_configuration(basis_gates=["rz", "sx", "x", "cx"], num_qubits=3)
    circuit = QuantumCircuit(3)
    circuit.cx(0, 2)
    trivial_layout = next(action for action in qiskit_actions.qiskit_layout_actions() if action.name == "TrivialLayout")
    placed, layout = qiskit_actions.run_qiskit_action(trivial_layout, circuit, target, None)

    routed, layout = qiskit_actions.run_qiskit_action(qiskit_actions.qiskit_routing_action(), placed, target, layout)

    assert routed.count_ops() == {"cx": 1}
    assert layout is not None
    assert layout.final_index_layout() == [0, 1, 2]


@pytest.mark.parametrize("name", ["RemoveIdentityEquivalent", "Optimize1qGatesSimpleCommutation"])
def test_qiskit_optimizations_preserve_unitary(line_env: predictorenv_module.PredictorEnv, name: str) -> None:
    """The identity and commutation passes remove redundant rotations across a CX."""
    circuit = QuantumCircuit(2)
    angle = np.pi / 3 if name == "Optimize1qGatesSimpleCommutation" else 0
    circuit.rz(angle, 0)
    circuit.cx(0, 1)
    if angle:
        circuit.rz(-angle, 0)
    line_env.reset(circuit)

    optimized = _apply_qiskit_action(line_env, name)

    assert optimized.count_ops() == {"cx": 1}
    assert Operator(optimized).equiv(Operator(circuit))


def test_register_action(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test the register_action function."""
    actions_registry = vars(actions_registry_module)
    monkeypatch.setitem(actions_registry, "_ACTIONS", actions_registry["_ACTIONS"].copy())
    action = DeviceIndependentAction(
        name="test_action", pass_type=PassType.OPT, transpile_pass=[], origin=CompilationOrigin.QISKIT
    )
    assert action not in get_actions_by_pass_type()[PassType.OPT]
    register_action(action)
    assert action in get_actions_by_pass_type()[PassType.OPT]

    with pytest.raises(ValueError, match=re.escape("Action with name test_action already registered.")):
        register_action(action)


@pytest.mark.model_training
def test_qcompile_generates_trace_file(tmp_path: Path) -> None:
    """Test that rl_compile correctly generates a trace JSON file when tracing is enabled."""
    figure_of_merit = "expected_fidelity"
    device = get_device("ibm_heron_156")

    # Trace a circuit that the model-training test already compiled successfully.
    qc = get_benchmark("ghz", BenchmarkLevel.ALG, 3)

    qc_compiled, compilation_information = rl_compile(
        qc, device=device, figure_of_merit=figure_of_merit, tracer_output_path=tmp_path
    )

    assert qc_compiled is not None
    assert compilation_information is not None

    generated_json_files = list(tmp_path.glob("*.json"))
    assert len(generated_json_files) == 1, f"Expected exactly 1 trace JSON file, but found {len(generated_json_files)}."

    trace_file = generated_json_files[0]
    assert trace_file.exists()
    assert trace_file.is_file()
