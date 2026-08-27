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
from mqt.bench.targets.devices import get_device
from qiskit import QuantumCircuit

from mqt.predictor.rl.helper import create_feature_dict
from mqt.predictor.rl.tracer import (
    CompilationStep,
    CompilationTracer,
    DeviceMetadata,
    FigureOfMeritMetric,
    FigureOfMeritMetrics,
    GateCalibration,
    TopologyEdge,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_compilation_tracer_generates_valid_json(tmp_path: Path) -> None:
    """Test that the compilation tracer correctly generates a JSON file using stubbed trace data.

    Args:
        tmp_path: Pytest-provided temporary directory used for the trace output file.
    """
    trace_file = tmp_path / "test_trace.json"

    # Obtain real device data and setup other mock data
    device = get_device("ibm_falcon_127")

    tracer = CompilationTracer.from_initial_state(
        device=device,
        circuit_name="ghz_3",
        figure_of_merit="expected_fidelity",
        mdp_policy="mock_policy",
    )

    fom_metrics_baseline = FigureOfMeritMetrics(
        expected_fidelity=FigureOfMeritMetric(value=0.95, kind="exact"),
        critical_depth=FigureOfMeritMetric(value=12.0, kind="exact"),
        estimated_hellinger_distance=FigureOfMeritMetric(value=0.05, kind="approx"),
        estimated_success_probability=FigureOfMeritMetric(value=0.88, kind="exact"),
    )

    # Setup terminal metrics as "unavailable" with 0.0 values to test carry-over logic
    fom_metrics_terminal = FigureOfMeritMetrics(
        expected_fidelity=FigureOfMeritMetric(value=0.0, kind="unavailable"),
        critical_depth=FigureOfMeritMetric(value=0.0, kind="unavailable"),
        estimated_hellinger_distance=FigureOfMeritMetric(value=0.0, kind="unavailable"),
        estimated_success_probability=FigureOfMeritMetric(value=0.0, kind="unavailable"),
    )

    # 1. Create dummy Qiskit circuits for the tracer to parse
    qc_baseline = QuantumCircuit(3)
    qc_baseline.h(0)
    qc_baseline.cx(0, 1)

    qc_terminal = QuantumCircuit(3)
    qc_terminal.h(0)
    qc_terminal.cx(0, 1)
    qc_terminal.cx(1, 2)
    qc_terminal.rz(0.5, 2)

    # 2. Mock the feature dictionaries that the RL environment normally passes
    features_baseline = create_feature_dict(qc_baseline)
    features_terminal = create_feature_dict(qc_terminal)

    # 3. Use the actual `record_step` API instead of manually building CompilationSteps
    tracer.record_step(
        step_index=0,
        action_name="Baseline",
        action_type="INITIAL",
        action_duration=0.0,
        reward=0.0,
        current_qc=qc_baseline,
        figures_of_merit=fom_metrics_baseline,
        features=features_baseline,
        synthesized=True,
        laid_out=False,
        routed=False,
        done=False,
    )

    tracer.record_step(
        step_index=1,
        action_name="qiskit_routing_pass",
        action_type="ROUTING",
        action_duration=0.15,
        reward=1.234567,
        current_qc=qc_terminal,
        figures_of_merit=fom_metrics_terminal,
        features=features_terminal,
        synthesized=True,
        laid_out=True,
        routed=True,
        done=True,
    )

    tracer.total_duration = 0.15
    tracer.save_to_json(trace_file)

    assert trace_file.exists(), "Tracer JSON file was not generated."
    assert trace_file.is_file(), "Tracer output path is not a valid file."

    with trace_file.open(encoding="utf-8") as f:
        trace_data = json.load(f)

    # Validate Top-Level Metadata and Values
    assert trace_data["circuit_name"] == "ghz_3"
    assert trace_data["mdp_policy"] == "mock_policy"
    assert trace_data["schema_version"] == "1.0.0"
    assert trace_data["total_duration"] == pytest.approx(0.15)
    assert trace_data["device"]["name"] == "ibm_falcon_127"
    assert "timestamp" in trace_data, "Tracer JSON is missing the timestamp."

    # Validate Step Array Length
    assert len(trace_data["steps"]) == 2

    # Validate Baseline (First Step)
    first_step = trace_data["steps"][0]
    assert first_step["action_name"] == "Baseline"
    assert first_step["action_type"] == "INITIAL"
    assert first_step["action_duration"] == pytest.approx(0.0)

    # Check that record_step correctly parsed the Qiskit circuit
    assert first_step["num_qubits"] == 3
    assert first_step["total_gates"] == 2

    # Validate Terminal Step (Last Step)
    last_step_data = trace_data["steps"][-1]
    assert last_step_data.get("is_terminal") is True
    assert last_step_data["action_duration"] == pytest.approx(0.15)
    assert last_step_data["total_gates"] == 4

    # Verify that unavailable metrics successfully carried over the values from the previous step
    terminal_fom = last_step_data["figures_of_merit"]
    assert terminal_fom["expected_fidelity"]["value"] == pytest.approx(0.95)
    assert terminal_fom["expected_fidelity"]["kind"] == "unavailable"
    assert terminal_fom["critical_depth"]["value"] == pytest.approx(12.0)
    assert terminal_fom["critical_depth"]["kind"] == "unavailable"
    assert terminal_fom["estimated_hellinger_distance"]["value"] == pytest.approx(0.05)
    assert terminal_fom["estimated_hellinger_distance"]["kind"] == "unavailable"
    assert terminal_fom["estimated_success_probability"]["value"] == pytest.approx(0.88)
    assert terminal_fom["estimated_success_probability"]["kind"] == "unavailable"

    # Re-Validation via Dataclasses
    # 1. DeviceMetadata
    device_args = trace_data["device"].copy()

    # Replace the raw nested dictionaries with strictly validated Dataclass objects
    device_args["topology"] = [TopologyEdge(**edge) for edge in device_args.get("topology", [])]
    device_args["calibration_data"] = {
        gate: [GateCalibration(**cal) for cal in cals] for gate, cals in device_args.get("calibration_data", {}).items()
    }

    DeviceMetadata(**device_args)

    # 2. CompilationStep
    for step_data in (first_step, last_step_data):
        fom_raw = step_data["figures_of_merit"]
        hd_raw = fom_raw.get("estimated_hellinger_distance")
        esp_raw = fom_raw.get("estimated_success_probability")

        fom_metrics_obj = FigureOfMeritMetrics(
            expected_fidelity=FigureOfMeritMetric(**fom_raw["expected_fidelity"]),
            critical_depth=FigureOfMeritMetric(**fom_raw["critical_depth"]),
            estimated_hellinger_distance=FigureOfMeritMetric(**hd_raw) if hd_raw is not None else None,
            estimated_success_probability=FigureOfMeritMetric(**esp_raw) if esp_raw is not None else None,
        )

        step_args = step_data.copy()
        step_args["figures_of_merit"] = fom_metrics_obj
        CompilationStep(**step_args)
