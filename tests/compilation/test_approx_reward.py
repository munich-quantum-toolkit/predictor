# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for approximate reward helper functions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import pytest

from mqt.predictor.rl.approx_reward import compute_device_averages_from_target

if TYPE_CHECKING:
    from qiskit.transpiler import Target


@dataclass
class FakeInstructionProperties:
    """Minimal stand-in for Qiskit's InstructionProperties."""

    error: float | None = None
    duration: float | None = None


@dataclass
class FakeQubitProperties:
    """Minimal stand-in for Qiskit's QubitProperties."""

    t1: float | None = None
    t2: float | None = None


@dataclass
class FakeOperation:
    """Minimal operation object exposing `num_qubits`."""

    num_qubits: int


class FakeCouplingMap:
    """Minimal coupling map exposing directed edges."""

    def __init__(self, edges: list[tuple[int, int]]) -> None:
        """Store directed coupling edges."""
        self._edges = edges

    def get_edges(self) -> list[tuple[int, int]]:
        """Return directed coupling edges."""
        return self._edges


class FakeTarget:
    """Minimal Target-like object for testing branch behavior."""

    def __init__(
        self,
        *,
        num_qubits: int,
        operation_names: list[str],
        op_props: dict[str, dict[tuple[int, ...], FakeInstructionProperties]],
        arities: dict[str, int],
        edges: list[tuple[int, int]],
        qubit_properties: list[FakeQubitProperties | None] | None,
    ) -> None:
        """Initialize operation data, arities, connectivity, and qubit properties."""
        self.num_qubits = num_qubits
        self.operation_names = operation_names
        self._op_props = op_props
        self._arities = arities
        self._edges = edges
        self.qubit_properties = qubit_properties

    def build_coupling_map(self) -> FakeCouplingMap:
        """Return a minimal coupling map."""
        return FakeCouplingMap(self._edges)

    def operation_from_name(self, name: str) -> FakeOperation:
        """Return operation metadata."""
        return FakeOperation(num_qubits=self._arities[name])

    def __getitem__(self, name: str) -> dict[tuple[int, ...], FakeInstructionProperties]:
        """Return calibration map for an operation."""
        return self._op_props[name]


def test_compute_device_averages_nominal_path() -> None:
    """Compute per-gate means and qubit coherence median on a nominal target."""
    target = FakeTarget(
        num_qubits=2,
        operation_names=["measure", "x", "cx"],
        op_props={
            "measure": {
                (0,): FakeInstructionProperties(error=0.01, duration=100.0),
                (1,): FakeInstructionProperties(error=0.03, duration=120.0),
            },
            "x": {
                (0,): FakeInstructionProperties(error=0.1, duration=10.0),
                (1,): FakeInstructionProperties(error=0.3, duration=30.0),
            },
            "cx": {
                (0, 1): FakeInstructionProperties(error=0.4, duration=40.0),
            },
        },
        arities={"measure": 1, "x": 1, "cx": 2},
        edges=[(0, 1)],
        qubit_properties=[
            FakeQubitProperties(t1=100.0, t2=50.0),  # min = 50
            FakeQubitProperties(t1=200.0, t2=None),  # min = 200
        ],
    )

    err_by_gate, dur_by_gate, tbar = compute_device_averages_from_target(cast("Target", target))

    assert err_by_gate == {
        "measure": pytest.approx(0.02),
        "x": pytest.approx(0.2),
        "cx": pytest.approx(0.4),
    }
    assert dur_by_gate == {
        "measure": pytest.approx(110.0),
        "x": pytest.approx(20.0),
        "cx": pytest.approx(40.0),
    }
    assert tbar == pytest.approx(125.0)


def test_compute_device_averages_missing_target_api_raises() -> None:
    """Raise a clear RuntimeError if the object is not Target-compatible."""
    with pytest.raises(RuntimeError, match="required Target API"):
        compute_device_averages_from_target(cast("Target", object()))


def test_compute_device_averages_without_calibration_data_raises() -> None:
    """Raise if calibration data contains neither error nor duration samples."""
    target = FakeTarget(
        num_qubits=2,
        operation_names=["x"],
        op_props={
            "x": {
                (0,): FakeInstructionProperties(error=None, duration=None),
                (1,): FakeInstructionProperties(error=None, duration=None),
            }
        },
        arities={"x": 1},
        edges=[],
        qubit_properties=[None, None],
    )

    with pytest.raises(RuntimeError, match="No valid calibration data found in Target"):
        compute_device_averages_from_target(cast("Target", target))
