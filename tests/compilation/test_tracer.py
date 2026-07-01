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

    fom_metrics = FigureOfMeritMetrics(
        expected_fidelity=FigureOfMeritMetric(value=0.95, kind="exact"),
        critical_depth=FigureOfMeritMetric(value=12.0, kind="exact"),
        hellinger_distance=FigureOfMeritMetric(value=0.05, kind="approx"),
        success_probability=FigureOfMeritMetric(value=0.88, kind="exact"),
    )

    baseline_step = CompilationStep(
        step_index=0,
        action_name="Baseline",
        action_type="INITIAL",
        action_duration=0.0,
        reward=0.0,
        current_depth=5,
        num_qubits=3,
        gates_per_operation={"cx": 2, "h": 1},
        total_gates=3,
        figures_of_merit=fom_metrics,
        synthesized=True,
        laid_out=False,
        routed=False,
        is_terminal=False,
        circuit_qasm3='OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[3] q;\n',
        program_communication=0.5,
        raw_critical_depth=5.0,
        entanglement_ratio=0.33,
        parallelism=0.66,
        liveness=0.8,
    )

    terminal_step = CompilationStep(
        step_index=1,
        action_name="qiskit_routing_pass",
        action_type="ROUTING",
        action_duration=0.15,
        reward=1.234567,
        current_depth=8,
        num_qubits=3,
        gates_per_operation={"cx": 4, "h": 1, "rz": 2},
        total_gates=7,
        figures_of_merit=fom_metrics,
        synthesized=True,
        laid_out=True,
        routed=True,
        is_terminal=True,
        circuit_qasm3="// fully routed",
        program_communication=0.2,
        raw_critical_depth=8.0,
        entanglement_ratio=0.5,
        parallelism=0.4,
        liveness=0.9,
    )

    tracer.steps.extend([baseline_step, terminal_step])
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

    # Validate Terminal Step (Last Step)
    last_step_data = trace_data["steps"][-1]
    assert last_step_data.get("is_terminal") is True
    assert last_step_data["action_duration"] == pytest.approx(0.15)

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
        hd_raw = fom_raw.get("hellinger_distance")
        esp_raw = fom_raw.get("success_probability")

        fom_metrics_obj = FigureOfMeritMetrics(
            expected_fidelity=FigureOfMeritMetric(**fom_raw["expected_fidelity"]),
            critical_depth=FigureOfMeritMetric(**fom_raw["critical_depth"]),
            hellinger_distance=FigureOfMeritMetric(**hd_raw) if hd_raw is not None else None,
            success_probability=FigureOfMeritMetric(**esp_raw) if esp_raw is not None else None,
        )

        step_args = step_data.copy()
        step_args["figures_of_merit"] = fom_metrics_obj
        CompilationStep(**step_args)
