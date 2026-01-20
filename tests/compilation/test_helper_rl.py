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

import numpy as np
from mqt.bench import BenchmarkLevel, get_benchmark
from mqt.bench.targets import get_device
from qiskit import transpile
from qiskit.transpiler import PassManager
from qiskit.transpiler.basepasses import BasePass as QiskitBasePass
from qiskit.transpiler.passes.layout.vf2_post_layout import VF2PostLayoutStopReason

from mqt.predictor.rl.actions import PassType, get_actions_by_pass_type
from mqt.predictor.rl.helper import create_feature_dict, get_path_trained_model, get_path_training_circuits
from mqt.predictor.rl.parsing import postprocess_vf2postlayout


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


def test_vf2_layout_and_postlayout() -> None:
    """Test the VF2Layout and VF2PostLayout passes."""
    qc = get_benchmark("ghz", BenchmarkLevel.ALG, 3)

    for dev in [get_device("ibm_falcon_27"), get_device("quantinuum_h2_56")]:
        layout_pass: QiskitBasePass | None = None
        for layout_action in get_actions_by_pass_type()[PassType.LAYOUT]:
            if layout_action.name == "VF2Layout":
                assert callable(layout_action.transpile_pass)
                layout_pass_ = layout_action.transpile_pass(dev)
                assert isinstance(layout_pass_, QiskitBasePass)
                layout_pass = layout_pass_
                break
        assert isinstance(layout_pass, QiskitBasePass)
        pm = PassManager(layout_pass)
        layouted_qc = pm.run(qc)
        assert layouted_qc.layout is not None
        assert len(layouted_qc.layout.initial_layout) == dev.num_qubits

    dev_success = get_device("ibm_falcon_27")
    qc_transpiled = transpile(qc, target=dev_success, optimization_level=0)
    assert qc_transpiled.layout is not None

    initial_layout_before = qc_transpiled.layout.initial_layout

    post_layout_pass: QiskitBasePass | None = None
    for layout_action in get_actions_by_pass_type()[PassType.FINAL_OPT]:
        if layout_action.name == "VF2PostLayout":
            assert callable(layout_action.transpile_pass)
            post_layout_pass_ = layout_action.transpile_pass(dev_success)
            assert isinstance(post_layout_pass_, QiskitBasePass)
            post_layout_pass = post_layout_pass_
            break
    assert isinstance(post_layout_pass, QiskitBasePass)

    pm = PassManager(post_layout_pass)
    altered_qc = pm.run(qc_transpiled)

    assert pm.property_set["VF2PostLayout_stop_reason"] == VF2PostLayoutStopReason.SOLUTION_FOUND

    _, pass_manager = postprocess_vf2postlayout(altered_qc, pm.property_set["post_layout"], qc_transpiled.layout)

    assert initial_layout_before != pass_manager.property_set["initial_layout"]
