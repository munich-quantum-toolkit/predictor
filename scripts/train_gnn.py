# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Generate the experiment split and train or resume the GNN predictor."""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import replace
from datetime import UTC
from importlib.metadata import version
from io import StringIO
from pathlib import Path

from mqt.bench import BenchmarkLevel, get_benchmark
from mqt.bench.benchmarks import get_available_benchmark_names
from networkx import NetworkXError
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ControlFlowOp
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
from qiskit.exceptions import QiskitError
from qiskit.qasm2 import dump
from qiskit.transpiler import Target
from qiskit_ibm_runtime.fake_provider import FakeBoston
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.utils import set_random_seed

from mqt.predictor.rl.checkpoints import RollingCheckpointCallback, latest_checkpoint, prune_checkpoints
from mqt.predictor.rl.gnn import GNNConfig, GNNMaskablePPO, GNNObservationWrapper, create_gnn_model
from mqt.predictor.rl.predictor import Predictor
from mqt.predictor.utils import get_openqasm_gates

REFERENCE_QPU = "ibm_boston"
FIGURE_OF_MERIT = "estimated_success_probability"
TEST_FRACTION = 0.1
MAX_CIRCUIT_DEPTH = 256
PASS_TIMEOUT_SECONDS = 30
RL_BASIS_GATES = [gate for gate in get_openqasm_gates() if gate in get_standard_gate_name_mapping()]
ESP_OPERATIONS = ("cz", "delay", "id", "measure", "reset", "rz", "sx", "x")
EXPECTED_GENERATION_ERRORS = (
    AssertionError,
    AttributeError,
    NetworkXError,
    NotImplementedError,
    OSError,
    QiskitError,
    RuntimeError,
    TypeError,
    ValueError,
)


class _CircuitTooDeepError(Exception):
    pass


class _TensorBoardProgressCallback(BaseCallback):
    def __init__(self, target_steps: int) -> None:
        super().__init__()
        self._target_steps = target_steps

    def _on_step(self) -> bool:
        self.logger.record("progress/total_timesteps", self.num_timesteps)
        self.logger.record("progress/fraction_complete", min(self.num_timesteps / self._target_steps, 1.0))
        self.model.dump_logs()
        return True


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()


def _write_config(path: Path, config: dict[str, object]) -> None:
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_boston_snapshot() -> tuple[Target, dict[str, object]]:
    backend = FakeBoston()
    source = backend.target
    target = Target(
        description=REFERENCE_QPU,
        num_qubits=source.num_qubits,
        dt=source.dt,
        granularity=source.granularity,
        min_length=source.min_length,
        pulse_alignment=source.pulse_alignment,
        acquire_alignment=source.acquire_alignment,
        qubit_properties=source.qubit_properties,
        concurrent_measurements=source.concurrent_measurements,
    )
    for name in ESP_OPERATIONS:
        target.add_instruction(source.operation_from_name(name), source[name], name=name)

    calibration_timestamp = backend.properties().last_update_date.astimezone(UTC).isoformat().replace("+00:00", "Z")
    metadata: dict[str, object] = {
        "backend": REFERENCE_QPU,
        "backend_version": backend.backend_version,
        "calibration_timestamp_utc": calibration_timestamp,
        "qiskit_ibm_runtime": version("qiskit-ibm-runtime"),
        "source": "qiskit_ibm_runtime.fake_provider.FakeBoston",
    }
    return target, metadata


def _record_run_metadata(output_dir: Path, device: Target, calibration_snapshot: dict[str, object]) -> None:
    metadata: dict[str, object] = {
        "benchmark_max_depth": MAX_CIRCUIT_DEPTH,
        "calibration_snapshot": calibration_snapshot,
        "exclude_control_flow": True,
        "figure_of_merit": FIGURE_OF_MERIT,
        "gnn_config": "paper",
        "max_episode_steps": 100,
        "mdp": "v3",
        "pass_timeout_seconds": PASS_TIMEOUT_SECONDS,
        "training_target": {
            "name": REFERENCE_QPU,
            "num_qubits": device.num_qubits,
            "operations": sorted(device.operation_names),
        },
    }
    path = output_dir / "run_metadata.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != metadata:
        msg = f"Run configuration differs from {path}; use a new output directory."
        raise ValueError(msg)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_config(path, metadata)


def _generate_circuit(
    benchmark_name: str, requested_size: int, max_qubits: int, seed: int
) -> tuple[QuantumCircuit, str] | None:
    generation_options = {"seed": seed} if benchmark_name == "graphstate" else {}
    circuit = get_benchmark(
        benchmark_name,
        BenchmarkLevel.INDEP,
        requested_size,
        opt_level=0,
        **generation_options,
    )
    circuit = transpile(
        circuit,
        basis_gates=RL_BASIS_GATES,
        optimization_level=1,
        seed_transpiler=0,
    )
    if any(isinstance(instruction.operation, ControlFlowOp) for instruction in circuit.data):
        return None
    if circuit.num_qubits > max_qubits:
        return None
    if (circuit.depth() or 0) > MAX_CIRCUIT_DEPTH:
        raise _CircuitTooDeepError
    stream = StringIO()
    dump(circuit, stream)
    qasm = stream.getvalue()
    if QuantumCircuit.from_qasm_str(qasm).num_qubits != circuit.num_qubits:
        return None
    return circuit, qasm


def _split_circuits(circuits: dict[str, list[Path]], train_dir: Path, test_dir: Path, seed: int) -> None:
    rng = random.Random(seed)
    train: list[Path] = []
    test: list[Path] = []
    for family in sorted(circuits):
        family_circuits = sorted(circuits[family])
        rng.shuffle(family_circuits)
        if len(family_circuits) <= 1:
            train.extend(family_circuits)
            continue
        test_count = min(max(1, round(len(family_circuits) * TEST_FRACTION)), len(family_circuits) - 1)
        train.extend(family_circuits[:-test_count])
        test.extend(family_circuits[-test_count:])

    rng.shuffle(train)
    rng.shuffle(test)
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    for paths, destination in ((train, train_dir), (test, test_dir)):
        for path in paths:
            target = destination / path.name
            if path != target:
                path.replace(target)


def _prepare_data(output_dir: Path, max_qubits: int, seed: int, *, smoke: bool) -> tuple[Path, Path]:
    data_dir = output_dir / "data"
    all_dir = data_dir / "all"
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    config_path = data_dir / "config.json"
    benchmark_names = ["ghz", "qft"] if smoke else get_available_benchmark_names()
    max_requested_size = min(max_qubits, 4) if smoke else max_qubits
    config: dict[str, object] = {
        "benchmark_names": benchmark_names,
        "complete": False,
        "exclude_control_flow": True,
        "max_circuit_depth": MAX_CIRCUIT_DEPTH,
        "max_qubits": max_requested_size,
        "mqt_bench": version("mqt-bench"),
        "seed": seed,
        "test_fraction": TEST_FRACTION,
    }

    if config_path.exists():
        saved_config = json.loads(config_path.read_text(encoding="utf-8"))
        expected_config = {key: value for key, value in config.items() if key != "complete"}
        saved_expected = {
            key: value for key, value in saved_config.items() if key not in {"complete", "test_count", "train_count"}
        }
        if saved_expected != expected_config:
            msg = f"Dataset configuration differs from {config_path}; use a new output directory."
            raise ValueError(msg)
        train_count = len(list(train_dir.glob("*.qasm")))
        test_count = len(list(test_dir.glob("*.qasm")))
        if (
            saved_config.get("complete")
            and train_count == saved_config.get("train_count")
            and test_count == saved_config.get("test_count")
        ):
            return train_dir, test_dir
    else:
        data_dir.mkdir(parents=True, exist_ok=True)
        _write_config(config_path, config)

    all_dir.mkdir(parents=True, exist_ok=True)
    circuits: dict[str, list[Path]] = {}
    search_dirs = (all_dir, train_dir, test_dir)
    for benchmark_name in benchmark_names:
        family_paths: dict[str, Path] = {}
        slug = _slug(benchmark_name)
        for requested_size in range(2, max_requested_size + 1):
            pattern = f"{slug}_indep_req{requested_size}_*.qasm"
            existing = next((path for directory in search_dirs for path in directory.glob(pattern)), None)
            if existing is not None:
                family_paths[existing.name] = existing
                continue
            try:
                generated = _generate_circuit(benchmark_name, requested_size, max_qubits, seed)
            except _CircuitTooDeepError:
                break
            except EXPECTED_GENERATION_ERRORS:
                continue
            if generated is None:
                continue
            circuit, qasm = generated

            path = all_dir / f"{slug}_indep_req{requested_size}_{circuit.num_qubits}.qasm"
            path.write_text(qasm, encoding="utf-8")
            family_paths[path.name] = path

        circuits[benchmark_name] = list(family_paths.values())
        print(f"{benchmark_name}: {len(family_paths)} compatible circuits")

    if not any(circuits.values()):
        msg = "MQT Bench did not produce any RL-compatible circuits."
        raise RuntimeError(msg)

    _split_circuits(circuits, train_dir, test_dir, seed)
    config.update(
        complete=True,
        train_count=len(list(train_dir.glob("*.qasm"))),
        test_count=len(list(test_dir.glob("*.qasm"))),
    )
    _write_config(config_path, config)
    return train_dir, test_dir


def _train(output_dir: Path, total_steps: int, save_interval: int, seed: int, *, smoke: bool) -> Path:
    device, calibration_snapshot = _load_boston_snapshot()
    _record_run_metadata(output_dir, device, calibration_snapshot)
    train_dir, test_dir = _prepare_data(output_dir, device.num_qubits, seed, smoke=smoke)
    config = GNNConfig.paper()
    if smoke:
        config = replace(
            config, hidden_dim=16, num_conv_wo_resnet=1, num_resnet_layers=0, n_steps=4, batch_size=4, n_epochs=1
        )

    predictor = Predictor(
        figure_of_merit=FIGURE_OF_MERIT,
        device=device,
        path_training_circuits=train_dir,
        max_steps=100,
        mdp="v3",
        graph=True,
        gnn_config=config,
    )
    predictor.env.pass_timeout = PASS_TIMEOUT_SECONDS
    predictor.env.configure_qiskit_action_seeding(enabled=True)
    set_random_seed(seed)
    graph_env = GNNObservationWrapper(predictor.env)
    checkpoint_dir = output_dir / "checkpoints"
    prune_checkpoints(checkpoint_dir)
    checkpoint = latest_checkpoint(checkpoint_dir)
    if checkpoint is None:
        model = create_gnn_model(
            graph_env,
            config,
            verbose=1,
            tensorboard_log=str(output_dir / "tensorboard"),
            seed=seed,
        )
    else:
        model = GNNMaskablePPO.load(checkpoint, env=graph_env)

    remaining_steps = max(0, total_steps - int(model.num_timesteps))
    if remaining_steps:
        checkpoint_callback = RollingCheckpointCallback(save_interval, checkpoint_dir)
        model.learn(
            total_timesteps=remaining_steps,
            reset_num_timesteps=checkpoint is None,
            callback=CallbackList([checkpoint_callback, _TensorBoardProgressCallback(total_steps)]),
            progress_bar=True,
        )
        checkpoint = checkpoint_callback.save_final()

    assert checkpoint is not None
    print(f"Target: {REFERENCE_QPU} ({calibration_snapshot['calibration_timestamp_utc']})")
    print(f"Figure of merit: {FIGURE_OF_MERIT}")
    print(f"Pass timeout: {PASS_TIMEOUT_SECONDS} seconds")
    print(f"Train/test circuits: {len(list(train_dir.glob('*.qasm')))}/{len(list(test_dir.glob('*.qasm')))}")
    print(f"Completed timesteps: {model.num_timesteps}")
    print(f"Latest checkpoint: {checkpoint}")
    return checkpoint


def main() -> None:
    """Run the experiment training entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total-steps", type=int, default=100_000, help="Total environment-step target.")
    parser.add_argument(
        "--save-interval",
        type=int,
        default=10_000,
        help="Minimum steps between checkpoints; saves occur after completed PPO rollouts.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="Use two small benchmarks and a tiny rollout.")
    args = parser.parse_args()
    if args.total_steps <= 0:
        parser.error("--total-steps must be positive")
    if args.save_interval <= 0:
        parser.error("--save-interval must be positive")
    _train(args.output_dir, args.total_steps, args.save_interval, args.seed, smoke=args.smoke)


if __name__ == "__main__":
    main()
