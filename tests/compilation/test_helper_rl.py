# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the helper functions of the reinforcement learning predictor."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from mqt.bench import get_benchmark
from mqt.bench.devices import get_device_by_name
from qiskit import transpile
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes.layout.vf2_post_layout import VF2PostLayoutStopReason

from mqt.predictor import rl


def test_create_feature_dict() -> None:
    """Test the creation of a feature dictionary."""
    qc = get_benchmark("dj", 1, 5)
    features = rl.helper.create_feature_dict(qc)
    for feature in features.values():
        assert isinstance(feature, np.ndarray | int)


def test_get_path_trained_model() -> None:
    """Test the retrieval of the path to the trained model."""
    path = rl.helper.get_path_trained_model()
    assert path.exists()
    assert isinstance(path, Path)


def test_get_path_training_circuits() -> None:
    """Test the retrieval of the path to the training circuits."""
    path = rl.helper.get_path_training_circuits()
    assert path.exists()
    assert isinstance(path, Path)


def test_vf2_layout_and_postlayout() -> None:
    """Test the VF2Layout and VF2PostLayout passes."""
    qc = get_benchmark("ghz", 1, 3)

    for dev in [get_device_by_name("ibm_montreal"), get_device_by_name("ionq_harmony")]:
        layout_pass = None
        for layout_action in rl.helper.get_actions_layout():
            if layout_action["name"] == "VF2Layout":
                layout_pass = layout_action["transpile_pass"](dev)
                break
        pm = PassManager(layout_pass)
        layouted_qc = pm.run(qc)
        assert len(layouted_qc.layout.initial_layout) == dev.num_qubits

    dev_success = get_device_by_name("ibm_montreal")
    qc_transpiled = transpile(
        qc, basis_gates=dev_success.basis_gates, coupling_map=dev_success.coupling_map, optimization_level=0
    )
    assert qc_transpiled.layout is not None

    initial_layout_before = qc_transpiled.layout.initial_layout

    post_layout_pass = None
    for layout_action in rl.helper.get_actions_final_optimization():
        if layout_action["name"] == "VF2PostLayout":
            post_layout_pass = layout_action["transpile_pass"](dev_success)
            break

    pm = PassManager(post_layout_pass)
    altered_qc = pm.run(qc_transpiled)

    assert pm.property_set["VF2PostLayout_stop_reason"] == VF2PostLayoutStopReason.SOLUTION_FOUND

    postprocessed_vf2postlayout_qc, _ = rl.helper.postprocess_vf2postlayout(
        altered_qc, pm.property_set["post_layout"], qc_transpiled.layout
    )

    assert initial_layout_before != postprocessed_vf2postlayout_qc.layout.initial_layout
