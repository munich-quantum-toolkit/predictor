# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the CompilationTracer."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from mqt.bench import BenchmarkLevel, get_benchmark
from mqt.bench.targets.devices import get_device

from mqt.predictor.rl.helper import get_path_trained_model
from mqt.predictor.rl.predictor import Predictor, rl_compile
from mqt.predictor.rl.tracer import CompilationStep, DeviceMetadata

if TYPE_CHECKING:
    from pathlib import Path


def test_compilation_tracer_generates_valid_json(tmp_path: Path) -> None:
    """Test that the compilation tracer correctly generates a JSON file when a path is provided.

    Args:
        tmp_path: Pytest-provided temporary directory used for the trace output file.
    """
    trace_file = tmp_path / "test_trace.json"
    qc = get_benchmark("ghz", level=BenchmarkLevel.INDEP, circuit_size=3)

    figure_of_merit = "expected_fidelity"
    device = get_device("ibm_falcon_127")
    model_name = "model_" + figure_of_merit + "_" + device.description
    model_path = get_path_trained_model() / (model_name + ".zip")

    if not model_path.exists():
        predictor = Predictor(figure_of_merit=figure_of_merit, device=device)
        predictor.train_model(timesteps=1000, test=True)

    _compiled_qc, _compilation_info = rl_compile(
        qc, device=device, figure_of_merit=figure_of_merit, tracer_output_path=str(trace_file)
    )

    assert trace_file.exists(), "Tracer JSON file was not generated."
    assert trace_file.is_file(), "Tracer output path is not a valid file."

    with trace_file.open(encoding="utf-8") as f:
        trace_data = json.load(f)

    assert "circuit_name" in trace_data, "Tracer JSON is missing the circuit name."
    assert "mdp_policy" in trace_data, "Tracer JSON is missing the mdp policy."
    assert "device" in trace_data, "Tracer JSON is missing the device information."
    assert "schema_version" in trace_data, "Tracer JSON is missing the schema version."
    assert "timestamp" in trace_data, "Tracer JSON is missing the timestamp."
    assert "steps" in trace_data, "Tracer JSON is missing the steps array."

    assert len(trace_data["steps"]) > 1, "Tracer should record subsequent compilation steps beyond the Baseline."
    assert trace_data["steps"][0]["action"] == "Baseline", "First step must be Baseline."
    assert trace_data["schema_version"] == "1.0.0"

    last_step_data = trace_data["steps"][-1]
    assert last_step_data.get("is_terminal") is True, "The final compilation step must be marked as terminal."

    # Verify Figures of Merit
    fom_data = last_step_data.get("figures_of_merit")
    assert fom_data is not None, "Figures of merit dictionary is missing from the trace step."

    # always calculated ones
    assert fom_data.get("expected_fidelity") is not None, "Expected fidelity failed to populate."
    assert fom_data.get("critical_depth") is not None, "Critical depth fallback failed."

    # for this device ESP should be populated
    assert fom_data.get("success_probability") is not None, "ESP fallback calculation failed."
    assert "value" in fom_data["success_probability"], "ESP is missing its float value."
    assert "kind" in fom_data["success_probability"], "ESP is missing its kind string."

    # It is valid for HD to be None (model missing) or a populated dictionary (model exists)
    hd_metric = fom_data.get("hellinger_distance")
    if hd_metric is not None:
        assert "value" in hd_metric, "Hellinger distance is missing its float value."
        assert "kind" in hd_metric, "Hellinger distance is missing its kind string."

    try:
        # Initialize from JSON (throws if the structures don't match)
        DeviceMetadata(**trace_data["device"])

        # Semantically validate both the first and the last steps
        CompilationStep(**trace_data["steps"][0])
        CompilationStep(**last_step_data)

    except TypeError as e:
        pytest.fail(
            f"Semantic Validation Failed! The generated JSON does not match your Python dataclasses. Error: {e}"
        )
