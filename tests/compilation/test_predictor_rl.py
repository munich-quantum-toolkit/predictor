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
from mqt.bench import get_benchmark
from qiskit.qasm2 import dump

from mqt.predictor import rl


def test_predictor_env_reset_from_string() -> None:
    """Test the reset function of the predictor environment with a quantum circuit given as a string as input."""
    predictor = rl.Predictor(figure_of_merit="expected_fidelity", device_name="ionq_harmony")
    qasm_path = Path("test.qasm")
    qc = get_benchmark("dj", 1, 3)
    with qasm_path.open("w", encoding="utf-8") as f:
        dump(qc, f)
    assert predictor.env.reset(qc=qasm_path)[0] == rl.helper.create_feature_dict(qc)


def test_warning_for_unidirectional_device() -> None:
    """Test the warning for a unidirectional device."""
    msg = "The connectivity of the device 'oqc_lucy' is uni-directional and MQT Predictor might return a compiled circuit that assumes bi-directionality."
    with pytest.warns(UserWarning, match=re.escape(msg)):
        rl.Predictor(figure_of_merit="expected_fidelity", device_name="oqc_lucy")


def test_predictor_env_esp_error() -> None:
    """Test the predictor environment with ESP as figure of merit and missing calibration data."""
    with pytest.raises(ValueError, match=re.escape("Missing calibration data for ESP calculation on ibm_montreal.")):
        rl.Predictor(figure_of_merit="estimated_success_probability", device_name="ibm_montreal")


def test_predictor_env_hellinger_error() -> None:
    """Test the predictor environment with the Estimated Hellinger Distance as figure of merit and a missing model."""
    with pytest.raises(
        ValueError, match=re.escape("Missing trained model for Hellinger distance estimates on ibm_montreal.")
    ):
        rl.Predictor(figure_of_merit="estimated_hellinger_distance", device_name="ibm_montreal")


def test_qcompile_with_newly_trained_models() -> None:
    """Test the qcompile function with a newly trained model.

    Important: Those trained models are used in later tests and must not be deleted.
    To test ESP as well, training must be done with a device that provides all relevant information (i.e. T1, T2 and gate times).
    """
    figure_of_merit = "expected_fidelity"
    device = "ionq_harmony"  # fully specified calibration data
    qc = get_benchmark("ghz", 1, 3)
    predictor = rl.Predictor(figure_of_merit=figure_of_merit, device_name=device)

    model_name = "model_" + figure_of_merit + "_" + device
    model_path = Path(rl.helper.get_path_trained_model() / (model_name + ".zip"))
    if not model_path.exists():
        with pytest.raises(
            FileNotFoundError,
            match=re.escape(
                "The RL model 'model_expected_fidelity_ionq_harmony' is not trained yet. Please train the model before using it."
            ),
        ):
            rl.qcompile(qc, figure_of_merit=figure_of_merit, device_name=device)

    predictor.train_model(
        timesteps=100,
        test=True,
    )

    qc_compiled, compilation_information = rl.qcompile(qc, figure_of_merit=figure_of_merit, device_name=device)
    assert qc_compiled.layout is not None
    assert compilation_information is not None


def test_qcompile_with_false_input() -> None:
    """Test the qcompile function with false input."""
    qc = get_benchmark("dj", 1, 5)
    with pytest.raises(ValueError, match=re.escape("figure_of_merit must not be None if predictor_singleton is None.")):
        rl.helper.qcompile(qc, None, "quantinuum_h2")
    with pytest.raises(ValueError, match=re.escape("device_name must not be None if predictor_singleton is None.")):
        rl.helper.qcompile(qc, "expected_fidelity", None)
