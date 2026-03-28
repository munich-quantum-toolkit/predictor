# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Utilities to generate RL training and test circuits from MQT Bench."""

from __future__ import annotations

import argparse
import logging
import random
import re
import time
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING

from mqt.bench import BenchmarkLevel, get_benchmark
from mqt.bench.benchmarks import get_available_benchmark_names
from qiskit import transpile
from qiskit.circuit.exceptions import CircuitError
from qiskit.qasm2 import dump

from mqt.predictor.rl.helper import (
    ensure_training_circuit_directories,
)
from mqt.predictor.utils import get_openqasm_gates_for_rl

if TYPE_CHECKING:
    from qiskit import QuantumCircuit

logger = logging.getLogger("mqt-predictor")

try:
    from networkx.exception import NetworkXError
except ImportError:  # pragma: no cover - networkx is an indirect dependency in this workflow
    NetworkXError = RuntimeError

GENERATABLE_BENCHMARK_ERRORS = (
    AssertionError,
    AttributeError,
    CircuitError,
    NetworkXError,
    NotImplementedError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


@dataclass(slots=True)
class GeneratedBenchmarkCircuit:
    """A generated benchmark circuit together with its metadata."""

    benchmark_name: str
    circuit: QuantumCircuit
    circuit_size: int | None = None
    benchmark_instance_name: str | None = None

    @property
    def file_stem(self) -> str:
        """Return a deterministic file stem for the generated circuit."""
        parts = [slugify(self.benchmark_name)]
        if self.benchmark_instance_name is not None:
            parts.append(slugify(self.benchmark_instance_name))
        if self.circuit_size is not None:
            parts.append(str(self.circuit_size))
        else:
            parts.append(str(self.circuit.num_qubits))
        parts.append("indep")
        return "_".join(parts)


@dataclass(slots=True)
class TrainTestSplit:
    """Split result for generated benchmark circuits."""

    train_circuits: list[GeneratedBenchmarkCircuit]
    test_circuits: list[GeneratedBenchmarkCircuit]


@dataclass(slots=True)
class TrainTestGenerationResult:
    """Result of generating and saving RL train/test circuits."""

    training_directory: Path
    test_directory: Path
    train_circuits: list[Path]
    test_circuits: list[Path]
    staging_directory: Path | None
    benchmark_names: list[str]


@dataclass(slots=True)
class BenchmarkGenerationAttempt:
    """Outcome of attempting to generate one benchmark circuit."""

    generated_circuit: GeneratedBenchmarkCircuit | None
    stop_benchmark: bool = False
    reason: str | None = None


def generate_rl_train_test_data(
    path_training_circuits: str | Path | None = None,
    path_test_circuits: str | Path | None = None,
    benchmark_names: list[str] | None = None,
    min_qubits: int = 2,
    max_qubits: int = 20,
    max_circuit_size: int | None = 200,
    max_circuit_depth: int | None = 200,
    test_fraction: float = 0.1,
    seed: int = 0,
) -> TrainTestGenerationResult:
    """Generate RL train/test circuit data from all available working MQT Bench benchmarks."""
    default_train_directory, default_test_directory = ensure_training_circuit_directories()
    training_directory = Path(path_training_circuits) if path_training_circuits is not None else default_train_directory
    test_directory = Path(path_test_circuits) if path_test_circuits is not None else default_test_directory
    staging_directory = training_directory.parent / "_generated_all"

    resolved_benchmark_names = benchmark_names or list_supported_benchmark_names()
    logger.info(
        "Generating RL train/test data for %d benchmarks with qubits in [%d, %d].",
        len(resolved_benchmark_names),
        min_qubits,
        max_qubits,
    )

    circuits = collect_working_benchmark_circuits(
        benchmark_names=resolved_benchmark_names,
        min_qubits=min_qubits,
        max_qubits=max_qubits,
        max_circuit_size=max_circuit_size,
        max_circuit_depth=max_circuit_depth,
        staging_directory=staging_directory,
    )
    split = split_generated_circuits(circuits, test_fraction=test_fraction, seed=seed)

    training_directory.mkdir(parents=True, exist_ok=True)
    test_directory.mkdir(parents=True, exist_ok=True)

    train_paths = save_generated_circuits(split.train_circuits, training_directory)
    test_paths = save_generated_circuits(split.test_circuits, test_directory)

    logger.info(
        "Saved %d training and %d test circuits to %s and %s.",
        len(train_paths),
        len(test_paths),
        training_directory,
        test_directory,
    )

    return TrainTestGenerationResult(
        training_directory=training_directory,
        test_directory=test_directory,
        train_circuits=train_paths,
        test_circuits=test_paths,
        staging_directory=staging_directory,
        benchmark_names=sorted({circuit.benchmark_name for circuit in circuits}),
    )


def collect_working_benchmark_circuits(
    benchmark_names: list[str] | None = None,
    min_qubits: int = 2,
    max_qubits: int = 20,
    max_circuit_size: int | None = 200,
    max_circuit_depth: int | None = 200,
    level: BenchmarkLevel | str = BenchmarkLevel.INDEP,
    staging_directory: Path | None = None,
) -> list[GeneratedBenchmarkCircuit]:
    """Collect all supported and working MQT Bench circuits up to a qubit limit."""
    candidate_names = benchmark_names or list_supported_benchmark_names()
    generated_circuits: list[GeneratedBenchmarkCircuit] = []

    for index, benchmark_name in enumerate(candidate_names, start=1):
        start_time = time.perf_counter()
        logger.info(
            "[%d/%d] Generating benchmark '%s' for qubits in [%d, %d].",
            index,
            len(candidate_names),
            benchmark_name,
            min_qubits,
            max_qubits,
        )
        benchmark_circuits = collect_circuits_for_benchmark(
            benchmark_name=benchmark_name,
            min_qubits=min_qubits,
            max_qubits=max_qubits,
            max_circuit_size=max_circuit_size,
            max_circuit_depth=max_circuit_depth,
            level=level,
            staging_directory=staging_directory,
        )
        generated_circuits.extend(benchmark_circuits)
        logger.info(
            "[%d/%d] Finished '%s': %d circuits kept in %.2fs.",
            index,
            len(candidate_names),
            benchmark_name,
            len(benchmark_circuits),
            time.perf_counter() - start_time,
        )

    unique_circuits: dict[str, GeneratedBenchmarkCircuit] = {}
    for circuit in generated_circuits:
        unique_circuits[circuit.file_stem] = circuit

    return [unique_circuits[key] for key in sorted(unique_circuits)]


def collect_circuits_for_benchmark(
    benchmark_name: str,
    min_qubits: int = 2,
    max_qubits: int = 20,
    max_circuit_size: int | None = 200,
    max_circuit_depth: int | None = 200,
    level: BenchmarkLevel | str = BenchmarkLevel.INDEP,
    staging_directory: Path | None = None,
) -> list[GeneratedBenchmarkCircuit]:
    """Collect all working circuits for a single benchmark."""
    generated_circuits: list[GeneratedBenchmarkCircuit] = []
    for circuit_size in range(min_qubits, max_qubits + 1):
        circuit_start_time = time.perf_counter()
        logger.info("  - Trying '%s' with circuit size %d.", benchmark_name, circuit_size)
        attempt = try_generate_benchmark_circuit(
            benchmark_name=benchmark_name,
            level=level,
            circuit_size=circuit_size,
            max_qubits=max_qubits,
            max_circuit_size=max_circuit_size,
            max_circuit_depth=max_circuit_depth,
        )
        elapsed_time = time.perf_counter() - circuit_start_time
        if attempt.generated_circuit is None:
            if attempt.stop_benchmark:
                logger.info(
                    "  - Stopping '%s' at size %d: %s.",
                    benchmark_name,
                    circuit_size,
                    attempt.reason or "circuit exceeded generation limits",
                )
                break
            logger.info(
                "  - Skipped '%s' size %d after %.2fs.",
                benchmark_name,
                circuit_size,
                elapsed_time,
            )
            continue
        generated_circuit = attempt.generated_circuit
        generated_circuits.append(generated_circuit)
        circuit_depth = generated_circuit.circuit.depth() or 0
        circuit_size_value = generated_circuit.circuit.size()
        if staging_directory is not None:
            staged_path = save_generated_circuit(generated_circuit, staging_directory)
            logger.info(
                "  - Kept '%s' size %d as %s in %.2fs (circuit size=%d, depth=%d).",
                benchmark_name,
                circuit_size,
                staged_path.name,
                elapsed_time,
                circuit_size_value,
                circuit_depth,
            )
        else:
            logger.info(
                "  - Kept '%s' size %d in %.2fs (circuit size=%d, depth=%d).",
                benchmark_name,
                circuit_size,
                elapsed_time,
                circuit_size_value,
                circuit_depth,
            )
    return generated_circuits


def try_generate_benchmark_circuit(
    benchmark_name: str,
    level: BenchmarkLevel | str = BenchmarkLevel.INDEP,
    circuit_size: int | None = None,
    max_qubits: int = 20,
    max_circuit_size: int | None = 200,
    max_circuit_depth: int | None = 200,
) -> BenchmarkGenerationAttempt:
    """Try to generate a benchmark circuit and keep only QASM2-serializable circuits."""
    try:
        qc = build_benchmark_circuit(
            benchmark_name=benchmark_name,
            level=level,
            circuit_size=circuit_size,
        )
        qc = prepare_generated_circuit(qc)
        if qc.num_qubits > max_qubits:
            return BenchmarkGenerationAttempt(
                generated_circuit=None,
                stop_benchmark=True,
                reason=f"circuit uses {qc.num_qubits} qubits which exceeds the limit of {max_qubits}",
            )
        complexity_reason = get_complexity_limit_reason(
            qc=qc,
            max_circuit_size=max_circuit_size,
            max_circuit_depth=max_circuit_depth,
        )
        if complexity_reason is not None:
            return BenchmarkGenerationAttempt(
                generated_circuit=None,
                stop_benchmark=True,
                reason=complexity_reason,
            )
        assert_qasm2_serializable(qc)
    except GENERATABLE_BENCHMARK_ERRORS:
        return BenchmarkGenerationAttempt(generated_circuit=None)

    return BenchmarkGenerationAttempt(
        generated_circuit=GeneratedBenchmarkCircuit(
            benchmark_name=benchmark_name,
            circuit=qc,
            circuit_size=circuit_size,
        )
    )


def build_benchmark_circuit(
    benchmark_name: str,
    level: BenchmarkLevel | str = BenchmarkLevel.INDEP,
    circuit_size: int | None = None,
) -> QuantumCircuit:
    """Build one benchmark circuit from MQT Bench."""
    normalized_level = normalize_benchmark_level(level)
    if circuit_size is None:
        msg = "MQT Bench circuit generation requires a concrete circuit size."
        raise ValueError(msg)
    qc = get_benchmark(benchmark_name, normalized_level, circuit_size, opt_level=0)
    if not hasattr(qc, "num_qubits"):
        msg = f"Benchmark '{benchmark_name}' did not return a Qiskit circuit."
        raise TypeError(msg)
    return qc


def prepare_generated_circuit(qc: QuantumCircuit) -> QuantumCircuit:
    """Translate circuits into the RL feature-space gate basis before saving.

    Some MQT Bench generators return compact circuits with custom gate definitions.
    For RL training data we want circuits expressed in the same gate vocabulary as
    the RL observation features, so size/depth and gate counts are measured on the
    translated circuit rather than the compact macro representation.
    """
    prepared_qc = transpile(
        qc,
        basis_gates=get_openqasm_gates_for_rl(),
        optimization_level=1,
        seed_transpiler=0,
    )
    validate_feature_space_gates(prepared_qc)
    return prepared_qc


def validate_feature_space_gates(qc: QuantumCircuit) -> None:
    """Ensure the translated circuit only contains supported RL feature-space gates."""
    allowed_gates = set(get_openqasm_gates_for_rl()) | {"measure", "barrier"}
    unsupported_gates = sorted({instruction.operation.name for instruction in qc.data} - allowed_gates)
    if unsupported_gates:
        msg = f"Prepared circuit still contains unsupported gates: {unsupported_gates}"
        raise ValueError(msg)


def get_complexity_limit_reason(
    qc: QuantumCircuit,
    max_circuit_size: int | None = 200,
    max_circuit_depth: int | None = 200,
) -> str | None:
    """Return a human-readable reason when a circuit exceeds the configured limits."""
    circuit_size = qc.size()
    circuit_depth = qc.depth() or 0
    if max_circuit_size is not None and circuit_size > max_circuit_size:
        return f"circuit size {circuit_size} exceeds the limit of {max_circuit_size}"
    if max_circuit_depth is not None and circuit_depth > max_circuit_depth:
        return f"circuit depth {circuit_depth} exceeds the limit of {max_circuit_depth}"
    return None


def split_generated_circuits(
    circuits: list[GeneratedBenchmarkCircuit],
    test_fraction: float = 0.1,
    seed: int = 0,
) -> TrainTestSplit:
    """Split generated circuits into train and test subsets per benchmark family."""
    if not 0.0 < test_fraction < 1.0:
        msg = "test_fraction must be strictly between 0 and 1."
        raise ValueError(msg)

    rng = random.Random(seed)
    benchmark_families: dict[str, list[GeneratedBenchmarkCircuit]] = {}
    for circuit in circuits:
        benchmark_families.setdefault(circuit.benchmark_name, []).append(circuit)

    train_circuits: list[GeneratedBenchmarkCircuit] = []
    test_circuits: list[GeneratedBenchmarkCircuit] = []

    for benchmark_name in sorted(benchmark_families):
        family_circuits = list(benchmark_families[benchmark_name])
        rng.shuffle(family_circuits)

        if len(family_circuits) <= 1:
            train_circuits.extend(family_circuits)
            continue

        test_count = max(1, round(len(family_circuits) * test_fraction))
        test_count = min(test_count, len(family_circuits) - 1)
        split_index = len(family_circuits) - test_count
        train_circuits.extend(family_circuits[:split_index])
        test_circuits.extend(family_circuits[split_index:])

    rng.shuffle(train_circuits)
    rng.shuffle(test_circuits)

    return TrainTestSplit(
        train_circuits=train_circuits,
        test_circuits=test_circuits,
    )


def save_generated_circuits(circuits: list[GeneratedBenchmarkCircuit], output_directory: Path) -> list[Path]:
    """Save generated circuits as QASM2 files."""
    output_directory.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = [
        save_generated_circuit(generated_circuit, output_directory) for generated_circuit in circuits
    ]

    return saved_paths


def save_generated_circuit(generated_circuit: GeneratedBenchmarkCircuit, output_directory: Path) -> Path:
    """Save one generated circuit as a QASM2 file."""
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"{generated_circuit.file_stem}.qasm"
    with output_path.open("w", encoding="utf-8") as file:
        dump(generated_circuit.circuit, file)
    return output_path


def list_supported_benchmark_names() -> list[str]:
    """Return benchmark names supported by the installed MQT Bench version."""
    return sorted(get_available_benchmark_names())


def normalize_benchmark_level(level: BenchmarkLevel | str) -> BenchmarkLevel:
    """Normalize a user-provided level to the installed MQT Bench API."""
    if isinstance(level, BenchmarkLevel):
        return level

    normalized = level.strip().lower()
    if normalized == "alg":
        return BenchmarkLevel.ALG
    if normalized == "indep":
        return BenchmarkLevel.INDEP
    if normalized == "nativegates":
        return BenchmarkLevel.NATIVEGATES
    if normalized == "mapped":
        return BenchmarkLevel.MAPPED
    msg = f"Unsupported benchmark level '{level}'."
    raise ValueError(msg)


def assert_qasm2_serializable(qc: QuantumCircuit) -> None:
    """Raise if a circuit cannot be serialized to OpenQASM 2."""
    dump(qc, StringIO())


def slugify(value: str) -> str:
    """Create a stable filename-safe slug."""
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()


def main() -> None:
    """Run RL train/test data generation from the command line."""
    parser = argparse.ArgumentParser(description="Generate RL train/test benchmark circuits.")
    parser.add_argument(
        "--benchmarks",
        default=None,
        help="Comma-separated benchmark names. If omitted, all available working benchmarks are used.",
    )
    parser.add_argument("--min-qubits", type=int, default=2, help="Minimum benchmark size to generate.")
    parser.add_argument("--max-qubits", type=int, default=20, help="Maximum benchmark size to generate.")
    parser.add_argument(
        "--max-circuit-size",
        type=int,
        default=250,
        help="Maximum circuit size to keep. Use 0 to disable the limit.",
    )
    parser.add_argument(
        "--max-circuit-depth",
        type=int,
        default=250,
        help="Maximum circuit depth to keep. Use 0 to disable the limit.",
    )
    parser.add_argument("--test-fraction", type=float, default=0.1, help="Fraction of circuits used for testing.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the train/test split.")
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=None,
        help="Optional output directory for training circuits.",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=None,
        help="Optional output directory for test circuits.",
    )
    args = parser.parse_args()

    benchmark_names = None
    if args.benchmarks:
        benchmark_names = [name.strip() for name in args.benchmarks.split(",") if name.strip()]

    max_circuit_size = None if args.max_circuit_size <= 0 else args.max_circuit_size
    max_circuit_depth = None if args.max_circuit_depth <= 0 else args.max_circuit_depth

    result = generate_rl_train_test_data(
        path_training_circuits=args.train_dir,
        path_test_circuits=args.test_dir,
        benchmark_names=benchmark_names,
        min_qubits=args.min_qubits,
        max_qubits=args.max_qubits,
        max_circuit_size=max_circuit_size,
        max_circuit_depth=max_circuit_depth,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )

    print("Benchmarks:", result.benchmark_names)
    print("Num benchmarks:", len(result.benchmark_names))
    print("Train circuits:", len(result.train_circuits))
    print("Test circuits:", len(result.test_circuits))
    print("Train dir:", result.training_directory)
    print("Test dir:", result.test_directory)


if __name__ == "__main__":
    main()
