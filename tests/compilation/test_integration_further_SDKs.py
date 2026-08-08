# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Integration tests for the compilation actions using further SDKs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from mqt.bench.targets import get_device
from qiskit import QuantumCircuit
from qiskit.circuit import StandardEquivalenceLibrary
from qiskit.transpiler import PassManager, TranspileLayout
from qiskit.transpiler.passes import (
    ApplyLayout,
    BasisTranslator,
    EnlargeWithAncilla,
    FullAncillaAllocation,
    SabreSwap,
    TrivialLayout,
)

from mqt.predictor.rl.actions import PassType
from mqt.predictor.rl.predictorenv import PredictorEnv

if TYPE_CHECKING:
    from qiskit.transpiler import Target


def _setup_env(env: PredictorEnv, circuit: QuantumCircuit, layout: TranspileLayout | None, n_qubits: int) -> None:
    """Reset env to the given circuit/layout state without starting a full RL episode."""
    env.reset(qc=circuit.copy())
    env.layout = layout
    env.num_qubits_uncompiled_circuit = n_qubits


def _is_available(env: PredictorEnv, idx: int) -> bool:
    """Return whether action idx is structurally and SDK-valid for the current env state."""
    env.valid_actions = env.determine_valid_actions_for_state()
    return env.action_masks()[idx]


def _lay_out(circuit: QuantumCircuit, target: Target) -> tuple[QuantumCircuit, TranspileLayout]:
    """Apply a trivial Qiskit layout to the circuit."""
    coupling_map = target.build_coupling_map()
    layout_pm = PassManager([
        TrivialLayout(coupling_map),
        FullAncillaAllocation(coupling_map),
        EnlargeWithAncilla(),
        ApplyLayout(),
    ])
    laid_out = layout_pm.run(circuit.copy())
    layout = TranspileLayout(
        initial_layout=layout_pm.property_set["layout"],
        input_qubit_mapping=dict(layout_pm.property_set["original_qubit_indices"]),
        final_layout=layout_pm.property_set.get("final_layout"),
        _output_qubit_list=laid_out.qubits,
        _input_qubit_count=circuit.num_qubits,
    )
    return laid_out, layout


def _route(circuit: QuantumCircuit, layout: TranspileLayout, target: Target) -> tuple[QuantumCircuit, TranspileLayout]:
    """Route the laid-out circuit with SabreSwap."""
    coupling_map = target.build_coupling_map()
    routing_pm = PassManager([SabreSwap(coupling_map=coupling_map)])
    routed = routing_pm.run(circuit.copy())
    routed_layout = TranspileLayout(
        initial_layout=layout.initial_layout,
        input_qubit_mapping=dict(layout.input_qubit_mapping),
        final_layout=routing_pm.property_set.get("final_layout"),
        _output_qubit_list=routed.qubits,
        _input_qubit_count=len(layout.input_qubit_mapping),
    )
    return routed, routed_layout


def _synthesize(
    circuit: QuantumCircuit, layout: TranspileLayout, target: Target
) -> tuple[QuantumCircuit, TranspileLayout]:
    """Translate the circuit to the target basis without changing its layout."""
    synthesis_pm = PassManager([BasisTranslator(StandardEquivalenceLibrary, target_basis=target.operation_names)])
    synthesized = synthesis_pm.run(circuit.copy())
    synthesized_layout = TranspileLayout(
        initial_layout=layout.initial_layout,
        input_qubit_mapping=dict(layout.input_qubit_mapping),
        final_layout=layout.final_layout,
        _output_qubit_list=synthesized.qubits,
        _input_qubit_count=len(layout.input_qubit_mapping),
    )
    return synthesized, synthesized_layout


@pytest.fixture
def target() -> Target:
    """Fixture to provide the target device for testing."""
    return get_device("ibm_falcon_27")


@pytest.fixture
def simple_circuit() -> QuantumCircuit:
    """Return a small circuit used to probe action invariants.

    CX(0, 2) is intentional: qubits 0 and 2 are not adjacent on ibm_falcon_27
    (qubit 0 only connects to 1), so SabreSwap inserts at least one SWAP.
    This ensures the routed fixture carries a real SWAP so routing-preservation
    checks are non-trivial.
    """
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 2)
    qc.cx(1, 2)
    return qc


@pytest.fixture
def env(target: Target) -> PredictorEnv:
    """Create a PredictorEnv for state-based invariant checking."""
    return PredictorEnv(device=target, reward_function="expected_fidelity")


def test_synthesis_actions_produce_native_gates(
    simple_circuit: QuantumCircuit,
    env: PredictorEnv,
) -> None:
    """Invariant: every synthesis action produces only native gates for all circuit states."""
    n_qubits = simple_circuit.num_qubits
    qc_laid_out, laid_layout = _lay_out(simple_circuit, env.device)
    qc_routed, routed_layout = _route(qc_laid_out, laid_layout, env.device)

    test_cases = [
        ("uncompiled", simple_circuit, None),
        ("laid-out", qc_laid_out, laid_layout),
        ("routed", qc_routed, routed_layout),
    ]

    for idx, action in env.action_set.items():
        if action.pass_type != PassType.SYNTHESIS:
            continue
        for kind, circuit, layout in test_cases:
            _setup_env(env, circuit, layout, n_qubits)
            if not _is_available(env, idx):
                continue
            compiled = env.apply_action(idx)
            assert env.is_circuit_synthesized(compiled), (
                f"{action.name} on {env.device.description} VIOLATED INVARIANT: "
                f"synthesis produced non-native gates for {kind} circuit. "
                f"Device native gates: {env.device.operation_names}. "
                f"Circuit gates: {set(compiled.count_ops().keys())}"
            )


def test_layout_actions_establish_layout(
    simple_circuit: QuantumCircuit,
    env: PredictorEnv,
) -> None:
    """Invariant: every layout action establishes a valid qubit assignment."""
    n_qubits = simple_circuit.num_qubits

    for idx, action in env.action_set.items():
        if action.pass_type != PassType.LAYOUT:
            continue
        _setup_env(env, simple_circuit, None, n_qubits)
        if not _is_available(env, idx):
            continue
        compiled = env.apply_action(idx)
        assert env.layout is not None, (
            f"{action.name} on {env.device.description} VIOLATED INVARIANT: failed to establish layout"
        )
        assert env.is_circuit_laid_out(compiled, env.layout), (
            f"{action.name} on {env.device.description} VIOLATED INVARIANT: "
            f"did not establish valid layout. Layout: {env.layout}"
        )


def test_mapping_actions_establish_layout(
    simple_circuit: QuantumCircuit,
    env: PredictorEnv,
) -> None:
    """Invariant: every mapping action establishes a valid qubit assignment."""
    synthesis_pm = PassManager([BasisTranslator(StandardEquivalenceLibrary, target_basis=env.device.operation_names)])
    synthesized = synthesis_pm.run(simple_circuit.copy())
    n_qubits = synthesized.num_qubits

    for idx, action in env.action_set.items():
        if action.pass_type != PassType.MAPPING:
            continue
        _setup_env(env, synthesized, None, n_qubits)
        if not _is_available(env, idx):
            continue
        compiled = env.apply_action(idx)
        assert env.layout is not None, (
            f"{action.name} on {env.device.description} VIOLATED INVARIANT: failed to establish layout"
        )
        assert env.is_circuit_laid_out(compiled, env.layout), (
            f"{action.name} on {env.device.description} VIOLATED INVARIANT: "
            f"did not establish valid layout. Layout: {env.layout}"
        )


def test_routing_actions_route_circuit(
    simple_circuit: QuantumCircuit,
    env: PredictorEnv,
) -> None:
    """Invariant: every routing action produces a circuit where all 2-qubit gates respect the coupling map."""
    qc_laid_out, layout = _lay_out(simple_circuit, env.device)
    n_qubits = qc_laid_out.num_qubits
    coupling_map = env.device.build_coupling_map()

    for idx, action in env.action_set.items():
        if action.pass_type != PassType.ROUTING:
            continue
        _setup_env(env, qc_laid_out, layout, n_qubits)
        if not _is_available(env, idx):
            continue
        routed = env.apply_action(idx)
        assert env.is_circuit_routed(routed, coupling_map), (
            f"{action.name} on {env.device.description} VIOLATED INVARIANT: circuit not properly routed after action"
        )


def test_optimization_actions_preserve_invariants(
    simple_circuit: QuantumCircuit,
    env: PredictorEnv,
) -> None:
    """Invariant: OPT actions honour their declared preserves_layout/routing/synthesis contracts."""
    qc_laid_out, laid_layout = _lay_out(simple_circuit, env.device)
    qc_routed, layout = _route(qc_laid_out, laid_layout, env.device)
    qc_synthesized, layout_synth = _synthesize(qc_routed, layout, env.device)
    n_qubits = qc_routed.num_qubits
    coupling_map = env.device.build_coupling_map()

    for idx, action in env.action_set.items():
        if action.pass_type != PassType.OPT:
            continue

        if action.preserves_layout:
            pre_v2p = dict(layout.initial_layout.get_virtual_bits())
            _setup_env(env, qc_routed, layout, n_qubits)
            if _is_available(env, idx):
                compiled = env.apply_action(idx)
                assert env.is_circuit_laid_out(compiled, layout), (
                    f"{action.name} on {env.device.description} VIOLATED INVARIANT preserves_layout: "
                    f"circuit no longer has a valid layout after action"
                )
                assert env.layout is not None, (
                    f"{action.name} on {env.device.description} VIOLATED INVARIANT preserves_layout: "
                    f"layout metadata was removed"
                )
                post_v2p = dict(env.layout.initial_layout.get_virtual_bits())
                assert post_v2p == pre_v2p, (
                    f"{action.name} on {env.device.description} VIOLATED INVARIANT preserves_layout: "
                    f"initial qubit assignment changed. Before: {pre_v2p}, After: {post_v2p}"
                )

        if action.preserves_routing:
            _setup_env(env, qc_routed, layout, n_qubits)
            if _is_available(env, idx):
                compiled = env.apply_action(idx)
                assert env.is_circuit_routed(compiled, coupling_map), (
                    f"{action.name} on {env.device.description} VIOLATED INVARIANT preserves_routing: "
                    f"produced gates on non-adjacent qubits"
                )

        if action.preserves_synthesis:
            _setup_env(env, qc_synthesized, layout_synth, n_qubits)
            if _is_available(env, idx):
                compiled = env.apply_action(idx)
                assert env.is_circuit_synthesized(compiled), (
                    f"{action.name} on {env.device.description} VIOLATED INVARIANT preserves_synthesis: "
                    f"introduced non-native gates. "
                    f"Device native gates: {env.device.operation_names}. "
                    f"Circuit gates: {set(compiled.count_ops().keys())}"
                )
