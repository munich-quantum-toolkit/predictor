# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the helper functions of the reinforcement learning predictor."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
from bqskit.ir import gates
from bqskit.ir.circuit import Circuit
from mqt.bench import BenchmarkLevel, get_benchmark
from mqt.bench.targets import get_device
from qiskit import transpile
from qiskit.transpiler import PassManager, PropertySet
from qiskit.transpiler.passes.layout.vf2_post_layout import VF2PostLayoutStopReason

from mqt.predictor.rl.actions import (
    PassType,
    get_actions_by_pass_type,
)
from mqt.predictor.rl.actions.bqskit_actions import bqskit_to_qiskit, get_bqskit_native_gates
from mqt.predictor.rl.actions.qiskit_actions import run_qiskit_action
from mqt.predictor.rl.helper import create_feature_dict, get_path_trained_model, get_path_training_circuits

if TYPE_CHECKING:
    from collections.abc import Callable

    import pytest
    from qiskit.passmanager.base_tasks import Task
    from qiskit.transpiler import Target


def test_create_feature_dict() -> None:
    """Test the creation of a feature dictionary."""
    qc = get_benchmark("dj", BenchmarkLevel.ALG, 5)
    features = create_feature_dict(qc)
    for feature in features.values():
        assert isinstance(feature, np.ndarray | int)


def test_get_path_trained_model() -> None:
    """Test the retrieval of the path to the trained model."""
    path = get_path_trained_model()
    assert path.exists()
    assert isinstance(path, Path)


def test_get_path_training_circuits() -> None:
    """Test the retrieval of the path to the training circuits."""
    path = get_path_training_circuits()
    assert path.exists()
    assert isinstance(path, Path)


def test_get_bqskit_native_gates_supports_iqm_r_gate() -> None:
    """Test that IQM's native RGate is represented in BQSKit."""
    native_gates = get_bqskit_native_gates(get_device("iqm_crystal_20"))

    assert any(isinstance(gate, gates.U1qGate) for gate in native_gates)


def test_bqskit_to_qiskit_converts_u1q_to_r_gate() -> None:
    """Test that BQSKit's U1qGate is converted to Qiskit's RGate."""
    circuit = Circuit(1)
    circuit.append_gate(gates.U1qGate(), [0], [0.1, 0.2])

    qc = bqskit_to_qiskit(circuit)

    assert qc.data[0].operation.name == "r"
    assert qc.data[0].operation.params == [0.1, 0.2]


def test_vf2_layout_and_postlayout(caplog: pytest.LogCaptureFixture) -> None:
    """Test the VF2Layout and VF2PostLayout passes."""
    qc = get_benchmark("ghz", BenchmarkLevel.ALG, 3)
    vf2_layout_action = next(
        action for action in get_actions_by_pass_type()[PassType.LAYOUT] if action.name == "VF2Layout"
    )

    for dev in [get_device("ibm_falcon_27"), get_device("quantinuum_h2_56")]:
        layouted_qc, _ = run_qiskit_action(vf2_layout_action, qc, dev, None)
        assert layouted_qc.layout is not None
        assert len(layouted_qc.layout.initial_layout) == dev.num_qubits

    dev_success = get_device("ibm_falcon_27")
    qft_qc = get_benchmark("qft", BenchmarkLevel.ALG, 3).decompose()
    _, layout = run_qiskit_action(vf2_layout_action, qft_qc, dev_success, None)
    assert layout is None
    assert "VF2Layout pass did not find a solution. Reason: VF2LayoutStopReason.NO_SOLUTION_FOUND" in caplog.text

    qc_transpiled = transpile(qc, target=dev_success, optimization_level=0)
    assert qc_transpiled.layout is not None

    initial_layout_before = qc_transpiled.layout.initial_layout

    post_layout_passes: list[Task] | None = None
    for layout_action in get_actions_by_pass_type()[PassType.FINAL_OPT]:
        if layout_action.name == "VF2PostLayout":
            factory = cast("Callable[[Target], list[Task]]", layout_action.transpile_pass)
            post_layout_passes = factory(dev_success)
            break
    assert post_layout_passes is not None

    pm = PassManager(post_layout_passes)
    property_set = PropertySet()
    qc_transpiled.layout.write_into_property_set(property_set)
    altered_qc = pm.run(qc_transpiled, property_set=property_set)

    assert pm.property_set["VF2PostLayout_stop_reason"] == VF2PostLayoutStopReason.SOLUTION_FOUND

    assert altered_qc.layout is not None
    assert initial_layout_before != altered_qc.layout.initial_layout
    assert altered_qc.layout.input_qubit_mapping == qc_transpiled.layout.input_qubit_mapping
    assert len(altered_qc.layout.final_index_layout()) == qc.num_qubits
