# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the compilation with reinforcement learning."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import torch
from mqt.bench import BenchmarkLevel, get_benchmark
from mqt.bench.targets import get_device
from qiskit import QuantumCircuit
from qiskit.circuit.library import CXGate
from qiskit.qasm2 import dump
from qiskit.transpiler import InstructionProperties, Target
from qiskit.transpiler.passes import GatesInBasis
from torch_geometric.data import Data

from mqt.predictor.rl import Predictor, rl_compile
from mqt.predictor.rl.actions import (
    Action,
    CompilationOrigin,
    DeviceIndependentAction,
    PassType,
    fom_aware_compile,
    get_actions_by_pass_type,
    register_action,
    remove_action,
    run_tket_action,
)
from mqt.predictor.rl.helper import create_feature_dict, get_path_trained_model

if TYPE_CHECKING:
    from collections.abc import Generator

    from _pytest.monkeypatch import MonkeyPatch

    from mqt.predictor.reward import figure_of_merit


def test_predictor_env_reset_from_string() -> None:
    """Test the reset function of the predictor environment with a quantum circuit given as a string as input."""
    device = get_device("ibm_eagle_127")
    predictor = Predictor(figure_of_merit="expected_fidelity", device=device)
    qasm_path = Path("test.qasm")
    qc = get_benchmark("dj", BenchmarkLevel.ALG, 3)
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


def test_qcompile_with_false_input() -> None:
    """Test the qcompile function with false input."""
    qc = get_benchmark("dj", BenchmarkLevel.ALG, 5)
    with pytest.raises(ValueError, match=re.escape("figure_of_merit must not be None if predictor_singleton is None.")):
        rl_compile(qc, device=get_device("quantinuum_h2_56"), figure_of_merit=None)
    with pytest.raises(ValueError, match=re.escape("device must not be None if predictor_singleton is None.")):
        rl_compile(qc, device=None, figure_of_merit="expected_fidelity")


@pytest.mark.parametrize(
    ("graph", "device_name", "train_kwargs"),
    [
        pytest.param(False, "ibm_falcon_127", {"timesteps": 10}, id="ppo-ibm_falcon_127"),
        pytest.param(False, "quantinuum_h2_56", {"timesteps": 10}, id="ppo-quantinuum_h2_56"),
        pytest.param(
            True,
            "ibm_falcon_127",
            {"iterations": 2, "steps": 5, "num_epochs": 1, "minibatch_size": 4},
            id="gnn-ibm_falcon_127",
        ),
    ],
)
@pytest.mark.usefixtures("_cleanup_gnn_model_expected_fidelity_ibm_falcon_127")
def test_train_and_compile(graph: bool, device_name: str, train_kwargs: dict[str, Any]) -> None:
    """Test the training and compilation pipeline for both MaskablePPO and GNN approaches.

    Important: The non-GNN trained models are used in later tests and must not be deleted.
    To test ESP as well, training must be done with a device that provides all relevant information (i.e. T1, T2 and gate times).
    """
    figure_of_merit = "expected_fidelity"
    device = get_device(device_name)
    qc = get_benchmark("ghz", BenchmarkLevel.ALG, 3)
    predictor = Predictor(figure_of_merit=figure_of_merit, device=device, graph=graph)
    predictor.train_model(test=True, **train_kwargs)

    qc_compiled, compilation_info = predictor.compile_as_predicted(qc)

    assert isinstance(qc_compiled, QuantumCircuit)
    assert compilation_info is not None
    assert len(compilation_info) > 0
    if not graph:
        check_nat_gates = GatesInBasis(basis_gates=device.operation_names)
        check_nat_gates(qc_compiled)
        assert check_nat_gates.property_set["all_gates_in_basis"], (
            "Circuit should only contain native gates but was not detected as such"
        )


def test_warning_for_unidirectional_device() -> None:
    """Test the warning for a unidirectional device."""
    target = Target()
    target.add_instruction(CXGate(), {(0, 1): InstructionProperties()})
    target.description = "uni-directional device"

    msg = "The connectivity of the device 'uni-directional device' is uni-directional and MQT Predictor might return a compiled circuit that assumes bi-directionality."
    with pytest.warns(UserWarning, match=re.escape(msg)):
        Predictor(figure_of_merit="expected_fidelity", device=target)


def test_fom_aware_compile_fallback(monkeypatch: MonkeyPatch) -> None:
    """Test fallback of the fom_aware_compile function in case of a compilation failure."""
    qc = QuantumCircuit(2)
    qc.swap(0, 1)

    dummy_action = Action(
        name="DummyAction",
        origin=CompilationOrigin.QISKIT,
        pass_type=PassType.MAPPING,
        transpile_pass=lambda _device: [],  # no passes applied
    )

    predictor = Predictor(figure_of_merit="critical_depth", device=get_device("ibm_eagle_127"))
    monkeypatch.setattr(
        predictor.env, "calculate_reward", lambda _circ: (_ for _ in ()).throw(RuntimeError("fake error"))
    )
    compiled_qc, prop_set = fom_aware_compile(
        dummy_action,
        predictor.env.device,
        qc,
        predictor.env.calculate_reward,
        max_iteration=1,
    )

    assert isinstance(compiled_qc, QuantumCircuit)
    assert isinstance(prop_set, dict)
    assert "swap" in compiled_qc.count_ops()


def test_tket_action_layout_failure() -> None:
    """Test fallback in case of TKET layout placement failure."""
    qc = QuantumCircuit(1)

    class FakePass:
        def get_placement_map(self, _: object) -> None:
            msg = "fake placement failure"
            raise RuntimeError(msg)

        def apply(self, _: object) -> None:
            pass

    dummy_action = Action(
        name="DummyLayout",
        origin=CompilationOrigin.TKET,
        pass_type=PassType.LAYOUT,
        transpile_pass=lambda _device: [FakePass()],
    )

    predictor = Predictor(figure_of_merit="estimated_success_probability", device=get_device("ibm_eagle_127"))
    predictor.env.actions_layout_indices.append(0)
    predictor.env.state = qc
    result_qc, _ = run_tket_action(
        action=dummy_action,
        action_index=0,
        state=predictor.env.state,
        device=predictor.env.device,
        layout=predictor.env.layout,
        num_qubits_uncompiled_circuit=predictor.env.num_qubits_uncompiled_circuit,
        actions_layout_indices=predictor.env.actions_layout_indices,
        actions_routing_indices=predictor.env.actions_routing_indices,
    )

    assert isinstance(result_qc, QuantumCircuit)
    assert result_qc.num_qubits == 1


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


@pytest.mark.parametrize(
    "fom",
    ["expected_fidelity", "estimated_success_probability"],
)
def test_approx_reward_paths_use_cached_per_gate_maps(fom: figure_of_merit) -> None:
    """Ensure approx reward path runs and uses cached per-basis-gate calibration maps.

    We don't test exact numeric values (backend-dependent), only that:
      - approx path runs,
      - cached maps are populated,
      - output is a valid probability in [0, 1].
    """
    qc = get_benchmark("ghz", BenchmarkLevel.ALG, 3)
    device = get_device("ibm_heron_133")
    predictor = Predictor(figure_of_merit=fom, device=device)

    val, kind = predictor.env.calculate_reward(qc=qc, mode="approx")
    assert kind == "approx"
    assert 0.0 <= val <= 1.0

    # Ensure caching produced per-gate mappings
    assert predictor.env.dev_avgs_cached
    assert isinstance(predictor.env.err_by_gate, dict)
    assert isinstance(predictor.env.dur_by_gate, dict)
    assert len(predictor.env.err_by_gate) > 0

    if fom == "estimated_success_probability":
        assert len(predictor.env.dur_by_gate) > 0
        # tbar is optional depending on backend calibration; just sanity-check type
        assert predictor.env.tbar is None or predictor.env.tbar > 0.0


@pytest.mark.parametrize("graph", [False, True])
def test_predictor_env_observation_type(graph: bool) -> None:
    """Test that PredictorEnv returns the correct observation type based on graph mode."""
    device = get_device("ibm_falcon_127")
    predictor = Predictor(figure_of_merit="expected_fidelity", device=device, graph=graph)
    qc = get_benchmark("ghz", BenchmarkLevel.ALG, 3)
    obs, _ = predictor.env.reset(qc=qc)

    if graph:
        assert isinstance(obs, Data)
        assert isinstance(obs.x, torch.Tensor)
        assert isinstance(obs.edge_index, torch.Tensor)
        assert obs.x.shape[0] > 0
        assert obs.edge_index.shape[0] == 2
    else:
        assert isinstance(obs, dict)


@pytest.mark.parametrize(
    ("graph", "model_file", "error_match"),
    [
        pytest.param(
            False,
            "model_critical_depth_ibm_falcon_127.zip",
            "The RL model 'model_critical_depth_ibm_falcon_127' is not trained yet.",
            id="ppo",
        ),
        pytest.param(
            True,
            "gnn_critical_depth_ibm_falcon_127.pt",
            "The GNN RL model 'gnn_critical_depth_ibm_falcon_127' is not trained yet.",
            id="gnn",
        ),
    ],
)
def test_model_not_trained_raises(graph: bool, model_file: str, error_match: str) -> None:
    """Test that compile_as_predicted raises FileNotFoundError when the model is not trained."""
    device = get_device("ibm_falcon_127")
    predictor = Predictor(figure_of_merit="critical_depth", device=device, graph=graph)
    model_path = get_path_trained_model() / model_file
    if model_path.exists():
        model_path.unlink()

    qc = get_benchmark("ghz", BenchmarkLevel.ALG, 3)
    with pytest.raises(FileNotFoundError, match=re.escape(error_match)):
        predictor.compile_as_predicted(qc)


@pytest.fixture
def _cleanup_gnn_model_expected_fidelity_ibm_falcon_127() -> Generator[None, None, None]:
    """Remove the GNN checkpoint created by GNN training tests."""
    yield
    model_path = get_path_trained_model() / "gnn_expected_fidelity_ibm_falcon_127.pt"
    if model_path.exists():
        model_path.unlink()


@pytest.mark.usefixtures("_cleanup_gnn_model_expected_fidelity_ibm_falcon_127")
def test_gnn_checkpoint_config_saved_correctly() -> None:
    """Test that the GNN checkpoint stores config metadata required for model reload."""
    figure_of_merit = "expected_fidelity"
    device = get_device("ibm_falcon_127")

    predictor = Predictor(figure_of_merit=figure_of_merit, device=device, graph=True)
    predictor.train_model(test=True, iterations=2, steps=5, num_epochs=1, minibatch_size=4)

    model_path = get_path_trained_model() / f"gnn_{figure_of_merit}_{device.description}.pt"
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    assert "state_dict" in checkpoint
    assert "config" in checkpoint
    cfg = checkpoint["config"]
    for key in (
        "hidden_dim",
        "num_conv_wo_resnet",
        "num_resnet_layers",
        "dropout_p",
        "bidirectional",
        "node_feature_dim",
        "num_actions",
    ):
        assert key in cfg, f"Missing config key: {key}"
