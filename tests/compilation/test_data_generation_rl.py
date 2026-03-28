# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for RL train/test data generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from mqt.bench import BenchmarkLevel
from qiskit import QuantumCircuit
from qiskit.circuit.exceptions import CircuitError

from mqt.predictor.rl.data_generation import (
    GeneratedBenchmarkCircuit,
    build_benchmark_circuit,
    collect_circuits_for_benchmark,
    collect_working_benchmark_circuits,
    generate_rl_train_test_data,
    get_complexity_limit_reason,
    prepare_generated_circuit,
    slugify,
    split_generated_circuits,
)
from mqt.predictor.utils import get_openqasm_gates_for_rl

if TYPE_CHECKING:
    from pathlib import Path


def test_slugify() -> None:
    """Test filename normalization."""
    assert slugify("grover-noancilla") == "grover_noancilla"


def test_split_generated_circuits_90_10() -> None:
    """Test deterministic 90/10 split behavior within one family."""
    circuits = [
        GeneratedBenchmarkCircuit(
            benchmark_name="bench",
            circuit=QuantumCircuit(2),
            circuit_size=index + 2,
        )
        for index in range(10)
    ]

    split = split_generated_circuits(circuits, test_fraction=0.1, seed=3)

    assert len(split.train_circuits) == 9
    assert len(split.test_circuits) == 1


def test_split_generated_circuits_is_stratified_by_benchmark_name() -> None:
    """Ensure each sufficiently large benchmark family contributes to the test set."""
    circuits = [
        GeneratedBenchmarkCircuit("alpha", QuantumCircuit(2), circuit_size=index + 2) for index in range(10)
    ] + [GeneratedBenchmarkCircuit("beta", QuantumCircuit(2), circuit_size=index + 2) for index in range(10)]

    split = split_generated_circuits(circuits, test_fraction=0.1, seed=3)

    assert len(split.train_circuits) == 18
    assert len(split.test_circuits) == 2
    assert sorted(circuit.benchmark_name for circuit in split.test_circuits) == ["alpha", "beta"]


def test_split_generated_circuits_keeps_singletons_in_train() -> None:
    """Do not place singleton benchmark families into the test split."""
    circuits = [
        GeneratedBenchmarkCircuit("alpha", QuantumCircuit(2), circuit_size=2),
        GeneratedBenchmarkCircuit("beta", QuantumCircuit(2), circuit_size=2),
        GeneratedBenchmarkCircuit("beta", QuantumCircuit(2), circuit_size=3),
    ]

    split = split_generated_circuits(circuits, test_fraction=0.5, seed=1)

    assert len(split.train_circuits) == 2
    assert len(split.test_circuits) == 1
    assert sorted(circuit.benchmark_name for circuit in split.train_circuits) == ["alpha", "beta"]
    assert [circuit.benchmark_name for circuit in split.test_circuits] == ["beta"]


def test_split_generated_circuits_rejects_invalid_fraction() -> None:
    """Test invalid test split fractions."""
    with pytest.raises(ValueError, match="test_fraction must be strictly between 0 and 1."):
        split_generated_circuits([], test_fraction=1.0)


def test_collect_working_benchmark_circuits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test dynamic benchmark collection with the MQT Bench 2.1.0 API."""

    def fake_supported_benchmarks() -> list[str]:
        return ["alpha", "broken"]

    def fake_build(
        benchmark_name: str,
        level: BenchmarkLevel | str = BenchmarkLevel.INDEP,
        circuit_size: int | None = None,
    ) -> QuantumCircuit:
        assert level == BenchmarkLevel.INDEP
        if benchmark_name == "alpha" and circuit_size in {2, 3}:
            return QuantumCircuit(circuit_size)
        msg = "unsupported"
        raise ValueError(msg)

    monkeypatch.setattr(
        "mqt.predictor.rl.data_generation.list_supported_benchmark_names",
        fake_supported_benchmarks,
    )
    monkeypatch.setattr(
        "mqt.predictor.rl.data_generation.build_benchmark_circuit",
        fake_build,
    )

    circuits = collect_working_benchmark_circuits(max_qubits=5)

    assert [circuit.benchmark_name for circuit in circuits] == ["alpha", "alpha"]


def test_collect_circuits_for_benchmark_stops_after_complexity_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop trying larger sizes once the generated circuit exceeds the configured limits."""
    seen_sizes: list[int] = []

    def fake_build(
        benchmark_name: str,
        level: BenchmarkLevel | str = BenchmarkLevel.INDEP,
        circuit_size: int | None = None,
    ) -> QuantumCircuit:
        assert benchmark_name == "alpha"
        assert level == BenchmarkLevel.INDEP
        assert circuit_size is not None
        seen_sizes.append(circuit_size)
        qc = QuantumCircuit(circuit_size)
        for qubit in range(circuit_size):
            qc.h(qubit)
        return qc

    monkeypatch.setattr(
        "mqt.predictor.rl.data_generation.build_benchmark_circuit",
        fake_build,
    )

    circuits = collect_circuits_for_benchmark(
        benchmark_name="alpha",
        min_qubits=2,
        max_qubits=5,
        max_circuit_size=2,
        max_circuit_depth=None,
    )

    assert seen_sizes == [2, 3]
    assert [circuit.circuit_size for circuit in circuits] == [2]


def test_collect_circuits_for_benchmark_skips_invalid_sizes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip benchmark sizes that are rejected by the underlying generator."""
    seen_sizes: list[int] = []

    def fake_build(
        benchmark_name: str,
        level: BenchmarkLevel | str = BenchmarkLevel.INDEP,
        circuit_size: int | None = None,
    ) -> QuantumCircuit:
        assert benchmark_name == "alpha"
        assert level == BenchmarkLevel.INDEP
        assert circuit_size is not None
        seen_sizes.append(circuit_size)
        if circuit_size % 2 == 1:
            msg = "Number of qubits must be divisible by 2."
            raise AssertionError(msg)
        return QuantumCircuit(circuit_size)

    monkeypatch.setattr(
        "mqt.predictor.rl.data_generation.build_benchmark_circuit",
        fake_build,
    )

    circuits = collect_circuits_for_benchmark(
        benchmark_name="alpha",
        min_qubits=2,
        max_qubits=5,
        max_circuit_size=None,
        max_circuit_depth=None,
    )

    assert seen_sizes == [2, 3, 4, 5]
    assert [circuit.circuit_size for circuit in circuits] == [2, 4]


def test_collect_circuits_for_benchmark_skips_qiskit_circuit_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip benchmark sizes that fail with a Qiskit CircuitError."""
    seen_sizes: list[int] = []

    def fake_build(
        benchmark_name: str,
        level: BenchmarkLevel | str = BenchmarkLevel.INDEP,
        circuit_size: int | None = None,
    ) -> QuantumCircuit:
        assert benchmark_name == "alpha"
        assert level == BenchmarkLevel.INDEP
        assert circuit_size is not None
        seen_sizes.append(circuit_size)
        if circuit_size == 3:
            msg = "One or more of the arguments are empty"
            raise CircuitError(msg)
        return QuantumCircuit(circuit_size)

    monkeypatch.setattr(
        "mqt.predictor.rl.data_generation.build_benchmark_circuit",
        fake_build,
    )

    circuits = collect_circuits_for_benchmark(
        benchmark_name="alpha",
        min_qubits=2,
        max_qubits=4,
        max_circuit_size=None,
        max_circuit_depth=None,
    )

    assert seen_sizes == [2, 3, 4]
    assert [circuit.circuit_size for circuit in circuits] == [2, 4]


def test_generate_rl_train_test_data_writes_qasm_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test writing train and test split files."""
    generated = [
        GeneratedBenchmarkCircuit("alpha", QuantumCircuit(2), circuit_size=index + 2) for index in range(10)
    ] + [GeneratedBenchmarkCircuit("beta", QuantumCircuit(2), circuit_size=index + 2) for index in range(10)]

    monkeypatch.setattr(
        "mqt.predictor.rl.data_generation.collect_working_benchmark_circuits",
        lambda **_kwargs: generated,
    )

    result = generate_rl_train_test_data(
        path_training_circuits=tmp_path / "training",
        path_test_circuits=tmp_path / "test",
        max_qubits=20,
        seed=7,
    )

    assert len(result.train_circuits) == 18
    assert len(result.test_circuits) == 2
    assert all(path.suffix == ".qasm" for path in result.train_circuits + result.test_circuits)
    assert result.test_directory.exists()


def test_build_benchmark_circuit_requires_qiskit_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test rejection of non-Qiskit benchmark objects."""
    monkeypatch.setattr(
        "mqt.predictor.rl.data_generation.get_benchmark",
        lambda **_kwargs: object(),
    )

    with pytest.raises(TypeError, match="did not return a Qiskit circuit"):
        build_benchmark_circuit("alpha")


def test_prepare_generated_circuit_translates_to_rl_basis_gates() -> None:
    """Translate custom composite instructions into the RL feature-space gate basis."""
    subcircuit = QuantumCircuit(2, name="custom")
    subcircuit.h(0)
    subcircuit.cx(0, 1)

    qc = QuantumCircuit(2)
    qc.append(subcircuit.to_instruction(), [0, 1])

    prepared_qc = prepare_generated_circuit(qc)

    assert "custom" not in prepared_qc.count_ops()
    assert set(prepared_qc.count_ops()).issubset(set(get_openqasm_gates_for_rl()) | {"measure", "barrier"})
    assert prepared_qc.size() >= 2


def test_get_complexity_limit_reason() -> None:
    """Return a readable reason when size or depth exceeds the configured limits."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.h(1)

    assert get_complexity_limit_reason(qc, max_circuit_size=2, max_circuit_depth=None) == (
        "circuit size 3 exceeds the limit of 2"
    )
    assert get_complexity_limit_reason(qc, max_circuit_size=None, max_circuit_depth=2) == (
        f"circuit depth {qc.depth()} exceeds the limit of 2"
    )
