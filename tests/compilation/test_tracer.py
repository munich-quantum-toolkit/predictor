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

from mqt.predictor.qcompile import qcompile
from mqt.predictor.rl.tracer import CompilationStep, DeviceMetadata

if TYPE_CHECKING:
    from pathlib import Path


def test_compilation_tracer_generates_valid_json(tmp_path: Path) -> None:
    """Test that the compilation tracer correctly generates a JSON file when a path is provided."""
    trace_file = tmp_path / "test_trace.json"
    qc = get_benchmark("ghz", level=BenchmarkLevel.INDEP, circuit_size=3)
    _compiled_qc, _compilation_info, _selected_device = qcompile(
        qc, figure_of_merit="expected_fidelity", tracer_output_path=str(trace_file)
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

    assert len(trace_data["steps"]) > 0, "Tracer did not record any compilation steps."
    assert trace_data["steps"][0]["action"] == "Baseline"
    assert trace_data["schema_version"] == "1.0.0"

    try:
        # initialize from JSON (throws if the structures don't match)
        DeviceMetadata(**trace_data["device"])
        CompilationStep(**trace_data["steps"][0])

    except TypeError as e:
        # pytest.fail instantly stops the test and prints your custom error message
        pytest.fail(
            f"Semantic Validation Failed! The generated JSON does not match your Python dataclasses. Error: {e}"
        )
