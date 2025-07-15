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

import pytest
from mqt.bench import BenchmarkLevel, get_benchmark
from mqt.bench.targets import get_device
from qiskit.circuit.library import CXGate
from qiskit.qasm2 import dump
from qiskit.transpiler import InstructionProperties, Target

from mqt.predictor import rl
from mqt.predictor.rl.actions import (
    CompilationOrigin,
    DeviceIndependentAction,
    PassType,
    get_actions_by_pass_type,
    register_action,
)


def test_predictor_env_reset_from_string() -> None:
    """Test the reset function of the predictor environment with a quantum circuit given as a string as input."""
    device = get_device("ibm_eagle_127")
    predictor = rl.Predictor(figure_of_merit="expected_fidelity", device=device)
    qasm_path = Path("test.qasm")
    qc = get_benchmark("dj", BenchmarkLevel.ALG, 3)
    with qasm_path.open("w", encoding="utf-8") as f:
        dump(qc, f)
    assert predictor.env.reset(qc=qasm_path)[0] == rl.helper.create_feature_dict(qc)


def test_predictor_env_esp_error() -> None:
    """Test the predictor environment with ESP as figure of merit and missing calibration data."""
    device = get_device("quantinuum_h2_56")
    with pytest.raises(
        ValueError, match=re.escape("Missing calibration data for ESP calculation on quantinuum_h2_56.")
    ):
        rl.Predictor(figure_of_merit="estimated_success_probability", device=device)


def test_predictor_env_hellinger_error() -> None:
    """Test the predictor environment with the Estimated Hellinger Distance as figure of merit and a missing model."""
    device = get_device("ibm_falcon_27")
    with pytest.raises(
        ValueError, match=re.escape("Missing trained model for Hellinger distance estimates on ibm_falcon_27.")
    ):
        rl.Predictor(figure_of_merit="estimated_hellinger_distance", device=device)


def test_qcompile_with_newly_trained_models() -> None:
    """Test the qcompile function with a newly trained model.

    Important: Those trained models are used in later tests and must not be deleted.
    To test ESP as well, training must be done with a device that provides all relevant information (i.e. T1, T2 and gate times).
    """
    figure_of_merit = "expected_fidelity"
    device = get_device("ibm_falcon_127")
    qc = get_benchmark("ghz", BenchmarkLevel.ALG, 3)
    predictor = rl.Predictor(figure_of_merit=figure_of_merit, device=device)

    model_name = "model_" + figure_of_merit + "_" + device.description
    model_path = Path(rl.helper.get_path_trained_model() / (model_name + ".zip"))
    if not model_path.exists():
        with pytest.raises(
            FileNotFoundError,
            match=re.escape(
                "The RL model 'model_expected_fidelity_ibm_falcon_127' is not trained yet. Please train the model before using it."
            ),
        ):
            rl.qcompile(qc, device=device, figure_of_merit=figure_of_merit)

    predictor.train_model(
        timesteps=100,
        test=True,
    )

    qc_compiled, compilation_information = rl.qcompile(qc, device=device, figure_of_merit=figure_of_merit)
    assert qc_compiled.layout is not None
    assert compilation_information is not None


def test_qcompile_with_false_input() -> None:
    """Test the qcompile function with false input."""
    qc = get_benchmark("dj", BenchmarkLevel.ALG, 5)
    with pytest.raises(ValueError, match=re.escape("figure_of_merit must not be None if predictor_singleton is None.")):
        rl.helper.qcompile(qc, device=get_device("quantinuum_h2_56"), figure_of_merit=None)
    with pytest.raises(ValueError, match=re.escape("device must not be None if predictor_singleton is None.")):
        rl.helper.qcompile(qc, device=None, figure_of_merit="expected_fidelity")


def test_warning_for_unidirectional_device() -> None:
    """Test the warning for a unidirectional device."""
    target = Target()
    target.add_instruction(CXGate(), {(0, 1): InstructionProperties()})
    target.description = "uni-directional device"

    msg = "The connectivity of the device 'uni-directional device' is uni-directional and MQT Predictor might return a compiled circuit that assumes bi-directionality."
    with pytest.warns(UserWarning, match=re.escape(msg)):
        rl.Predictor(figure_of_merit="expected_fidelity", device=target)


def test_register_action() -> None:
    """Test the register_action function."""
    action = DeviceIndependentAction(
        name="test_action", pass_type=PassType.OPT, transpile_pass=[], origin=CompilationOrigin.QISKIT
    )
    assert action not in get_actions_by_pass_type()[PassType.OPT]
    register_action(action)
    assert action in get_actions_by_pass_type()[PassType.OPT]
