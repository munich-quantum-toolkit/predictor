# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the compilation with reinforcement learning."""

from __future__ import annotations

import re
from pathlib import Path
import pandas as pd

import pytest
from qiskit_ibm_runtime import QiskitRuntimeService
from mqt.bench import BenchmarkLevel, get_benchmark
from mqt.bench.targets import get_device
from qiskit.circuit.library import CXGate
from qiskit.qasm2 import dump
from qiskit.transpiler import InstructionProperties, Target
from qiskit.transpiler.passes import GatesInBasis

from mqt.predictor.rl import Predictor, rl_compile
from mqt.predictor.rl.actions import (
    CompilationOrigin,
    DeviceIndependentAction,
    PassType,
    get_actions_by_pass_type,
    register_action,
    remove_action,
)
from mqt.predictor.rl.helper import create_feature_dict, get_path_training_circuits
from qiskit import QuantumCircuit


def test_predictor_env_reset_from_string() -> None:
    """Test the reset function of the predictor environment with a quantum circuit given as a string as input."""
    device = get_device("ibm_eagle_127")
    predictor = Predictor(figure_of_merit="expected_fidelity", device=device)
    qasm_path = Path("test.qasm")
    qc = get_benchmark("dj", BenchmarkLevel.INDEP, 3)
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
    # figure_of_merit = "expected_fidelity"
    # device = get_device("ibm_eagle_127")
    # qc = get_benchmark("ghz", BenchmarkLevel.INDEP, 20)
    # predictor = Predictor(figure_of_merit=figure_of_merit, device=device)

    # model_name = "model_" + figure_of_merit + "_" + device.description
    # model_path = Path(get_path_trained_model() / (model_name + ".zip"))
    # if not model_path.exists():
    #     with pytest.raises(
    #         FileNotFoundError,
    #         match=re.escape(
    #             "The RL model 'model_expected_fidelity_ibm_falcon_127' is not trained yet. Please train the model before using it."
    #         ),
    #     ):
    #         rl_compile(qc, device=device, figure_of_merit=figure_of_merit)
    figure_of_merit = "expected_fidelity"

    api_token = ""
    available_devices = ["ibm_brisbane", "ibm_torino"]
    device = available_devices[1]

    service = QiskitRuntimeService(channel="ibm_cloud", token=api_token)
    backend = service.backend(device)
    backend.target.description = "ibm_torino"  # HACK
    predictor = Predictor(figure_of_merit=figure_of_merit, device=backend.target)
    qc = get_benchmark("ghz", BenchmarkLevel.INDEP, 17, target=backend.target)

    predictor.train_model(
        timesteps=10000,
        test=False,
        model_name="model_new_actions"
    )

    qc_compiled, compilation_information = rl_compile(
        qc, device=backend.target, figure_of_merit=figure_of_merit
    )

    check_nat_gates = GatesInBasis(basis_gates=backend.target.operation_names)
    check_nat_gates(qc_compiled)
    only_nat_gates = check_nat_gates.property_set["all_gates_in_basis"]

    assert qc_compiled.layout is not None
    assert compilation_information is not None
    assert only_nat_gates, "Circuit should only contain native gates but was not detected as such"


def test_qcompile_with_false_input() -> None:
    """Test the qcompile function with false input."""
    qc = get_benchmark("dj", BenchmarkLevel.INDEP, 5)
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


def test_evaluations() -> None:
    test_dir = get_path_training_circuits() / "new_indep_circuits" / "test"
    results_dir = Path(__file__).resolve().parent / "results" / "new_actions"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "info.csv"
    figure_of_merit = "expected_fidelity"

    api_token = ""
    available_devices = ["ibm_brisbane", "ibm_torino"]
    device = available_devices[1]

    service = QiskitRuntimeService(channel="ibm_cloud", token=api_token)
    backend = service.backend(device)
    backend.target.description = "ibm_torino"  # HACK
    model_results = []
    model_label= "new_actions"
    for file_path in test_dir.glob("*.qasm"):
        file_name = file_path.name
        print(f"File: {file_name}")
        qc = QuantumCircuit.from_qasm_file(str(file_path))
        qc_compiled, reward, info, depth, critical_depth, esp = rl_compile(
            qc, device=backend.target, figure_of_merit=figure_of_merit
        )
        model_results.append({
            "model": model_label,
            "file": file_path.name,
            "depth": depth,
            "crit_depth": critical_depth,
            "esp": esp,
            "reward": reward,
            "ep_length_mean": len(info)
        })
        print(f"✅ Size {qc.num_qubits} | File: {file_path.name} | "
            f"Reward: {reward:.4f} | "
            f"Depth: {depth} | "
            f"Critical Depth: {critical_depth} |"
            f"ESP: {esp} |"
            f"Mean Steps: {len(info):.1f}")
        
    df = pd.DataFrame(model_results)
    df.sort_values(by=["depth", "model"], inplace=True)
    df.to_csv(output_path, index=False)
    print(f"📁 Results saved to: {output_path}")

