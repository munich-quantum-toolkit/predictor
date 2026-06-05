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

import pytest
from mqt.bench import BenchmarkLevel, get_benchmark
from mqt.bench.targets import get_device
from qiskit import QuantumCircuit
from qiskit.circuit.library import CXGate
from qiskit.qasm2 import dump
from qiskit.transpiler import InstructionProperties, Layout, Target, TranspileLayout
from qiskit.transpiler.passes import GatesInBasis

from mqt.predictor.rl import Predictor, rl_compile
from mqt.predictor.rl import predictorenv as predictorenv_module
from mqt.predictor.rl.actions import (
    CompilationOrigin,
    DeviceIndependentAction,
    PassType,
    get_actions_by_pass_type,
    qiskit_actions,
    register_action,
    remove_action,
)
from mqt.predictor.rl.helper import create_feature_dict, get_path_trained_model


def test_predictor_env_reset_from_string() -> None:
    """Test the reset function of the predictor environment with a quantum circuit given as a string as input."""
    device = get_device("ibm_eagle_127")
    predictor = Predictor(figure_of_merit="expected_fidelity", device=device)
    qasm_path = Path("test.qasm")
    qc = get_benchmark("dj", BenchmarkLevel.ALG, 3)
    with qasm_path.open("w", encoding="utf-8") as f:
        dump(qc, f)
    assert predictor.env.reset(qc=qasm_path)[0] == create_feature_dict(qc)


def test_predictor_env_esp_error() -> None:
    """Test the predictor environment with ESP as figure of merit and missing calibration data."""
    device = get_device("quantinuum_h2_56")
    with pytest.raises(
        ValueError, match=re.escape("Missing calibration data for ESP calculation on quantinuum_h2_56.")
    ):
        Predictor(figure_of_merit="estimated_success_probability", device=device)


def test_predictor_env_hellinger_error() -> None:
    """Test the predictor environment with the Estimated Hellinger Distance as figure of merit and a missing model."""
    device = get_device("ibm_falcon_27")
    with pytest.raises(
        ValueError, match=re.escape("Missing trained model for Hellinger distance estimates on ibm_falcon_27.")
    ):
        Predictor(figure_of_merit="estimated_hellinger_distance", device=device)


def test_qcompile_with_newly_trained_models() -> None:
    """Test the qcompile function with a newly trained model.

    Important: Those trained models are used in later tests and must not be deleted.
    To test ESP as well, training must be done with a device that provides all relevant information (i.e. T1, T2 and gate times).
    """
    figure_of_merit = "expected_fidelity"
    device = get_device("ibm_falcon_127")
    qc = get_benchmark("ghz", BenchmarkLevel.ALG, 3)
    predictor = Predictor(figure_of_merit=figure_of_merit, device=device)

    model_name = "model_" + figure_of_merit + "_" + device.description
    model_path = Path(get_path_trained_model() / (model_name + ".zip"))
    if not model_path.exists():
        with pytest.raises(
            FileNotFoundError,
            match=re.escape(
                "The RL model 'model_expected_fidelity_ibm_falcon_127' is not trained yet. Please train the model before using it."
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


def test_predictor_env_actions_after_layout_with_non_native_unrouted_circuit() -> None:
    """Test valid actions for a laid-out circuit that still needs synthesis and routing."""
    device = get_device("ibm_falcon_27")
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


def test_predictor_env_qiskit_routing_updates_final_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that Qiskit routing actions update the tracked final layout."""
    device = get_device("ibm_falcon_27")
    env = predictorenv_module.PredictorEnv(device=device)
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    env.reset(qc)

    initial_layout = Layout({qubit: index for index, qubit in enumerate(qc.qubits)})
    final_layout = Layout({qc.qubits[0]: 1, qc.qubits[1]: 0})
    env.layout = TranspileLayout(
        initial_layout=initial_layout,
        input_qubit_mapping={qubit: index for index, qubit in enumerate(qc.qubits)},
        final_layout=None,
        _output_qubit_list=qc.qubits,
        _input_qubit_count=qc.num_qubits,
    )

    class FakePassManager:
        """Minimal PassManager replacement that exposes a final layout."""

        def __init__(self, _passes: object) -> None:
            self.property_set = {"final_layout": final_layout}

        def run(self, circuit: QuantumCircuit) -> QuantumCircuit:
            return circuit

    monkeypatch.setattr(qiskit_actions, "PassManager", FakePassManager)
    action = DeviceIndependentAction(
        name="SyntheticQiskitRouting",
        pass_type=PassType.ROUTING,
        transpile_pass=[],
        origin=CompilationOrigin.QISKIT,
    )
    env.action_set[0] = action
    altered_qc = env.apply_action(action_index=0)

    assert altered_qc is env.state
    assert env.layout.final_layout is final_layout


def test_register_action() -> None:
    """Test the register_action function."""
    action = DeviceIndependentAction(
        name="test_action", pass_type=PassType.OPT, transpile_pass=[], origin=CompilationOrigin.QISKIT
    )
    assert action not in get_actions_by_pass_type()[PassType.OPT]
    register_action(action)
    assert action in get_actions_by_pass_type()[PassType.OPT]

    with pytest.raises(ValueError, match=re.escape("Action with name test_action already registered.")):
        register_action(action)

    remove_action(action.name)

    with pytest.raises(KeyError, match=re.escape("No action with name wrong_action_name is registered")):
        remove_action("wrong_action_name")
