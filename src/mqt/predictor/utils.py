# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Utility functions for the mqt.predictor module."""

from __future__ import annotations

import logging
import signal
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from warnings import warn

import networkx as nx
import numpy as np
from qiskit.converters import circuit_to_dag

if TYPE_CHECKING:
    from collections.abc import Callable

    from qiskit import QuantumCircuit

    from mqt.predictor.reward import figure_of_merit
    from mqt.predictor.rl.predictor import Predictor as RL_Predictor

logger = logging.getLogger("mqt-predictor")


def timeout_watcher(
    func: Callable[..., bool | QuantumCircuit],
    args: list[QuantumCircuit | figure_of_merit | str | RL_Predictor],
    timeout: int,
) -> tuple[QuantumCircuit, list[str]] | bool:
    """Method that stops a function call after a given timeout limit."""
    if sys.platform == "win32":
        warn("Timeout is not supported on Windows.", category=RuntimeWarning, stacklevel=2)
        return func(*args) if isinstance(args, tuple | list) else func(args)

    class TimeoutExceptionError(Exception):  # Custom exception class
        pass

    def timeout_handler(_signum: int, _frame: Any) -> None:  # noqa: ANN401
        raise TimeoutExceptionError

    # Change the behavior of SIGALRM
    signal.signal(signal.SIGALRM, timeout_handler)

    signal.alarm(timeout)
    try:
        res = func(*args)
    except TimeoutExceptionError:
        logger.debug("Calculation/Generation exceeded timeout limit for " + func.__module__ + ", " + str(args[1:]))
        return False
    except Exception:
        logger.exception("Something else went wrong")
        return False
    else:
        # Reset the alarm
        signal.alarm(0)

    return res


@dataclass
class SupermarqFeatures:
    """Data class for the Supermarq features of a quantum circuit."""

    program_communication: float
    critical_depth: float
    entanglement_ratio: float
    parallelism: float
    liveness: float


def calc_supermarq_features(
    qc: QuantumCircuit,
) -> SupermarqFeatures:
    """Calculates the Supermarq features for a given quantum circuit. Code adapted from https://github.com/Infleqtion/client-superstaq/blob/91d947f8cc1d99f90dca58df5248d9016e4a5345/supermarq-benchmarks/supermarq/converters.py."""
    num_qubits = qc.num_qubits
    dag = circuit_to_dag(qc)
    dag.remove_all_ops_named("barrier")

    # Program communication = circuit's average qubit degree / degree of a complete graph.
    graph = nx.Graph()
    for op in dag.two_qubit_ops():
        q1, q2 = op.qargs
        graph.add_edge(qc.find_bit(q1).index, qc.find_bit(q2).index)
    degree_sum = sum(graph.degree(n) for n in graph.nodes)
    program_communication = degree_sum / (num_qubits * (num_qubits - 1)) if num_qubits > 1 else 0

    # Liveness feature = sum of all entries in the liveness matrix / (num_qubits * depth).
    activity_matrix = np.zeros((num_qubits, dag.depth()))
    for i, layer in enumerate(dag.layers()):
        for op in layer["partition"]:
            for qubit in op:
                activity_matrix[qc.find_bit(qubit).index, i] = 1
    liveness = np.sum(activity_matrix) / (num_qubits * dag.depth()) if dag.depth() > 0 else 0

    #  Parallelism feature = max((((# of gates / depth) -1) /(# of qubits -1)), 0).
    parallelism = (
        max(((len(dag.gate_nodes()) / dag.depth()) - 1) / (num_qubits - 1), 0)
        if num_qubits > 1 and dag.depth() > 0
        else 0
    )

    # Entanglement-ratio = ratio between # of 2-qubit gates and total number of gates in the circuit.
    entanglement_ratio = len(dag.two_qubit_ops()) / len(dag.gate_nodes()) if len(dag.gate_nodes()) > 0 else 0

    # Critical depth = # of 2-qubit gates along the critical path / total # of 2-qubit gates.
    longest_paths = dag.count_ops_longest_path()
    n_ed = sum(longest_paths[name] for name in {op.name for op in dag.two_qubit_ops()} if name in longest_paths)
    n_e = len(dag.two_qubit_ops())
    critical_depth = n_ed / n_e if n_e != 0 else 0

    assert 0 <= program_communication <= 1
    assert 0 <= critical_depth <= 1
    assert 0 <= entanglement_ratio <= 1
    assert 0 <= parallelism <= 1
    assert 0 <= liveness <= 1

    return SupermarqFeatures(
        program_communication,
        critical_depth,
        entanglement_ratio,
        parallelism,
        liveness,
    )
