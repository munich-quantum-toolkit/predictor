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
from typing import TYPE_CHECKING

import pytest
from mqt.bench import BenchmarkLevel, get_benchmark
from mqt.bench.targets import get_device
from qiskit import QuantumCircuit
from qiskit.circuit.library import CXGate
from qiskit.qasm2 import dump
from qiskit.transpiler import CouplingMap, InstructionProperties, Target
from qiskit.transpiler.passes import CheckMap, GatesInBasis

from mqt.predictor.rl import Predictor, rl_compile
from mqt.predictor.rl.actions import (
    Action,
    CompilationOrigin,
    DeviceIndependentAction,
    PassType,
    get_actions_by_pass_type,
    register_action,
    remove_action,
)
from mqt.predictor.rl.helper import create_feature_dict, get_path_trained_model

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch


def test_predictor_env_reset_from_string() -> None:
    """Test the reset function of the predictor environment with a quantum circuit given as a string as input."""
    device = get_device("ibm_eagle_127")
    predictor = Predictor(figure_of_merit="expected_fidelity", device=device)
    qasm_path = Path("test.qasm")
    qc = get_benchmark("dj", BenchmarkLevel.INDEP, 3)
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


def test_qcompile_with_newly_trained_models() -> None:
    """Test the qcompile function with a newly trained model.

    Important: Those trained models are used in later tests and must not be deleted.
    To test ESP as well, training must be done with a device that provides all relevant information (i.e. T1, T2 and gate times).
    """
    figure_of_merit = "expected_fidelity"
    device = get_device("ibm_falcon_127")
    qc = get_benchmark("ghz", BenchmarkLevel.ALG, 3)

    predictor = Predictor(figure_of_merit=figure_of_merit, device=device)

    model_name = "model_" + figure_of_merit + "_" + device.description
    model_path = Path(get_path_trained_model() / (model_name + ".zip"))
    if not model_path.exists():
        with pytest.raises(
            FileNotFoundError,
            match=re.escape(
                "The RL model 'model_expected_fidelity_ibm_falcon_127' is not trained yet. Please train the model before using it."
            ),
        ):
            rl_compile(qc, device=device, figure_of_merit=figure_of_merit)

    predictor.train_model(
        timesteps=100,
        test=True,
    )

    qc_compiled, compilation_information = rl_compile(qc, device=device, figure_of_merit=figure_of_merit)

    check_nat_gates = GatesInBasis(basis_gates=device.operation_names)
    check_nat_gates(qc_compiled)
    only_nat_gates = check_nat_gates.property_set["all_gates_in_basis"]
    check_mapping = CheckMap(coupling_map=CouplingMap(device.build_coupling_map()))
    check_mapping(qc_compiled)
    mapped = check_mapping.property_set["is_swap_mapped"]

    assert qc_compiled.layout is not None
    assert compilation_information is not None
    assert only_nat_gates, "Circuit should only contain native gates but was not detected as such."
    assert mapped, "Circuit should be mapped to the device's coupling map."


def test_qcompile_with_false_input() -> None:
    """Test the qcompile function with false input."""
    qc = get_benchmark("dj", BenchmarkLevel.INDEP, 5)
    with pytest.raises(ValueError, match=re.escape("figure_of_merit must not be None if predictor_singleton is None.")):
        rl_compile(qc, device=get_device("quantinuum_h2_56"), figure_of_merit=None)
    with pytest.raises(ValueError, match=re.escape("device must not be None if predictor_singleton is None.")):
        rl_compile(qc, device=None, figure_of_merit="expected_fidelity")


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
    compiled_qc, prop_set = predictor.env.fom_aware_compile(dummy_action, None, qc, max_iteration=1)

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
    apply_tket = predictor.env._apply_tket_action  # noqa: SLF001
    result_qc = apply_tket(dummy_action, 0)

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
