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

from mqt.predictor.rl import Predictor
from mqt.predictor.rl.actions import CompilationOrigin, PassType, get_actions_by_pass_type, run_qiskit_action
from mqt.predictor.rl.actions.bqskit_actions import (
    run_bqskit_action,
)
from mqt.predictor.rl.actions.tket_actions import (
    run_tket_action,
)

if TYPE_CHECKING:
    from qiskit.transpiler import Target

    from mqt.predictor.rl.actions import Action


@pytest.fixture
def available_actions_dict() -> dict[PassType, list[Action]]:
    """Return a dictionary of available actions."""
    return get_actions_by_pass_type()


@pytest.fixture
def target() -> Target:
    """Fixture to provide the target device for testing."""
    return get_device("ibm_falcon_27")


@pytest.fixture
def simple_circuit() -> QuantumCircuit:
    """Return a small circuit used to probe action invariants."""
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    return qc


@pytest.fixture
def laid_out_circuit(
    simple_circuit: QuantumCircuit,
    target: Target,
) -> tuple[QuantumCircuit, TranspileLayout]:
    """Return the simple circuit together with its layout."""
    coupling_map = target.build_coupling_map()

    pm = PassManager([
        TrivialLayout(coupling_map),
        FullAncillaAllocation(coupling_map),
        EnlargeWithAncilla(),
        ApplyLayout(),
    ])
    laid_out = pm.run(simple_circuit.copy())
    layout = TranspileLayout(
        initial_layout=pm.property_set["layout"],
        input_qubit_mapping=dict(pm.property_set["original_qubit_indices"]),
        final_layout=pm.property_set.get("final_layout"),
        _output_qubit_list=laid_out.qubits,
        _input_qubit_count=simple_circuit.num_qubits,
    )
    return laid_out, layout


@pytest.fixture
def laid_out_and_routed_circuit(
    laid_out_circuit: tuple[QuantumCircuit, TranspileLayout],
    target: Target,
) -> tuple[QuantumCircuit, TranspileLayout]:
    """Return the laid-out circuit after routing, together with its layout."""
    coupling_map = target.build_coupling_map()
    laid_out, layout_before = laid_out_circuit

    pm = PassManager([
        SabreSwap(coupling_map=coupling_map),
    ])
    routed = pm.run(laid_out.copy())
    layout_after = TranspileLayout(
        initial_layout=layout_before.initial_layout,
        input_qubit_mapping=dict(layout_before.input_qubit_mapping),
        final_layout=pm.property_set.get("final_layout"),
        _output_qubit_list=routed.qubits,
        _input_qubit_count=layout_before._input_qubit_count,  # noqa: SLF001
    )
    return routed, layout_after


@pytest.fixture
def laid_out_and_routed_and_synthesized_circuit(
    laid_out_and_routed_circuit: tuple[QuantumCircuit, TranspileLayout],
    target: Target,
) -> tuple[QuantumCircuit, TranspileLayout]:
    """Return the routed circuit translated to the device basis with the same layout."""
    routed, layout = laid_out_and_routed_circuit
    pm = PassManager([BasisTranslator(StandardEquivalenceLibrary, target_basis=target.operation_names)])
    synthesized = pm.run(routed.copy())
    synthesized_layout = TranspileLayout(
        initial_layout=layout.initial_layout,
        input_qubit_mapping=dict(layout.input_qubit_mapping),
        final_layout=layout.final_layout,
        _output_qubit_list=synthesized.qubits,
        _input_qubit_count=layout._input_qubit_count,  # noqa: SLF001
    )
    return synthesized, synthesized_layout


@pytest.fixture
def predictor(target: Target) -> Predictor:
    """Create a predictor instance for state checking."""
    return Predictor(figure_of_merit="expected_fidelity", device=target)


@pytest.mark.parametrize("sdk", ["qiskit", "bqskit", "tket"])
def test_synthesis_actions_produce_native_gates(
    simple_circuit: QuantumCircuit,
    laid_out_circuit: tuple[QuantumCircuit, TranspileLayout],
    laid_out_and_routed_circuit: tuple[QuantumCircuit, TranspileLayout],
    available_actions_dict: dict[PassType, list[Action]],
    predictor: Predictor,
    sdk: str,
) -> None:
    """Test all synthesis actions produce native gates."""
    qc = simple_circuit.copy()
    qc_laid_out, _ = laid_out_circuit
    qc_routed, _ = laid_out_and_routed_circuit
    circuits = {
        "uncompiled": qc,
        "laid-out": qc_laid_out.copy(),
        "routed": qc_routed.copy(),
    }

    for action in available_actions_dict[PassType.SYNTHESIS]:
        for kind, circ in circuits.items():
            if sdk == "qiskit" and action.origin == CompilationOrigin.QISKIT:
                compiled, _ = run_qiskit_action(
                    action=action,
                    circuit=circ.copy(),
                    device=predictor.env.device,
                    layout=None,
                    max_iteration=10,
                    score_circuit=predictor.env.calculate_reward,
                )
            elif sdk == "bqskit" and action.origin == CompilationOrigin.BQSKIT:
                compiled, _ = run_bqskit_action(
                    action=action, circuit=circ.copy(), device=predictor.env.device, layout=None
                )
            elif sdk == "tket" and action.origin == CompilationOrigin.TKET:
                compiled, _ = run_tket_action(
                    action=action, circuit=circ.copy(), device=predictor.env.device, layout=None
                )
            else:
                continue

            assert isinstance(compiled, QuantumCircuit)
            assert predictor.env.is_circuit_synthesized(compiled), (
                f"{action.name} on {predictor.env.device.description} VIOLATED INVARIANT: "
                f"synthesis produced non-native gates for {kind} circuit. "
                f"Device native gates: {predictor.env.device.operation_names}. "
                f"Circuit gates: {set(compiled.count_ops().keys())}"
            )


@pytest.mark.parametrize("sdk", ["qiskit", "bqskit", "tket"])
def test_layout_actions_establish_layout(
    simple_circuit: QuantumCircuit,
    available_actions_dict: dict[PassType, list[Action]],
    predictor: Predictor,
    sdk: str,
) -> None:
    """Test all layout actions establish valid layouts."""
    qc = simple_circuit.copy()

    for action in available_actions_dict[PassType.LAYOUT]:
        if sdk == "qiskit" and action.origin == CompilationOrigin.QISKIT:
            compiled, layout = run_qiskit_action(
                action=action,
                circuit=qc.copy(),
                device=predictor.env.device,
                layout=None,
                max_iteration=10,
                score_circuit=predictor.env.calculate_reward,
            )
        elif sdk == "bqskit" and action.origin == CompilationOrigin.BQSKIT:
            compiled, layout = run_bqskit_action(
                action=action, circuit=qc.copy(), device=predictor.env.device, layout=None
            )
        elif sdk == "tket" and action.origin == CompilationOrigin.TKET:
            compiled, layout = run_tket_action(
                action=action, circuit=qc.copy(), device=predictor.env.device, layout=None
            )
        else:
            continue

        assert layout is not None, (
            f"{action.name} on {predictor.env.device.description} VIOLATED INVARIANT: failed to establish layout"
        )
        assert predictor.env.is_circuit_laid_out(compiled, layout), (
            f"{action.name} on {predictor.env.device.description} VIOLATED INVARIANT: "
            f"did not establish valid layout. Layout: {layout}"
        )


@pytest.mark.parametrize("sdk", ["qiskit", "bqskit", "tket"])
def test_routing_actions_route_circuit(
    laid_out_circuit: tuple[QuantumCircuit, TranspileLayout],
    available_actions_dict: dict[PassType, list[Action]],
    predictor: Predictor,
    sdk: str,
) -> None:
    """Test all routing actions result in is_circuit_routed returning True."""
    qc_laid_out, layout = laid_out_circuit

    for action in available_actions_dict[PassType.ROUTING]:
        if sdk == "qiskit" and action.origin == CompilationOrigin.QISKIT:
            routed, _ = run_qiskit_action(
                action=action,
                circuit=qc_laid_out.copy(),
                device=predictor.env.device,
                layout=layout,
                max_iteration=10,
                score_circuit=predictor.env.calculate_reward,
            )
        elif sdk == "bqskit" and action.origin == CompilationOrigin.BQSKIT:
            routed, _ = run_bqskit_action(
                action=action, circuit=qc_laid_out.copy(), device=predictor.env.device, layout=layout
            )
        elif sdk == "tket" and action.origin == CompilationOrigin.TKET:
            routed, _ = run_tket_action(
                action=action, circuit=qc_laid_out.copy(), device=predictor.env.device, layout=layout
            )
        else:
            continue

        assert predictor.env.is_circuit_routed(routed, predictor.env.device.build_coupling_map()), (
            f"{action.name} on {predictor.env.device.description} VIOLATED INVARIANT: "
            f"circuit not properly routed. Found {sum(len(instr.qubits) == 2 for instr in routed.data)} 2-qubit gates "
            f"not on valid edges"
        )


@pytest.mark.parametrize("sdk", ["qiskit", "bqskit", "tket"])
def test_optimization_actions_preserve_invariants(
    laid_out_circuit: tuple[QuantumCircuit, TranspileLayout],
    available_actions_dict: dict[PassType, list[Action]],
    predictor: Predictor,
    sdk: str,
) -> None:
    """Test OPT actions preserve claimed invariants."""
    laid_out_qc, layout = laid_out_circuit

    for action in available_actions_dict[PassType.OPT]:
        if sdk == "qiskit" and action.origin == CompilationOrigin.QISKIT:
            compiled, out_layout = run_qiskit_action(
                action=action,
                circuit=laid_out_qc.copy(),
                device=predictor.env.device,
                layout=layout,
                max_iteration=10,
                score_circuit=predictor.env.calculate_reward,
            )
        elif sdk == "bqskit" and action.origin == CompilationOrigin.BQSKIT:
            compiled, out_layout = run_bqskit_action(
                action=action, circuit=laid_out_qc.copy(), device=predictor.env.device, layout=layout
            )
        elif sdk == "tket" and action.origin == CompilationOrigin.TKET:
            compiled, out_layout = run_tket_action(
                action=action, circuit=laid_out_qc.copy(), device=predictor.env.device, layout=layout
            )
        else:
            continue

        if action.preserve_layout and layout is not None:
            assert out_layout is not None
            assert predictor.env.is_circuit_laid_out(compiled, out_layout), (
                f"{action.name} claims preserve_layout=True but layout was not preserved"
            )


@pytest.mark.parametrize("sdk", ["qiskit", "bqskit", "tket"])
def test_final_opt_actions_maintain_state(
    laid_out_and_routed_and_synthesized_circuit: tuple[QuantumCircuit, TranspileLayout],
    available_actions_dict: dict[PassType, list[Action]],
    predictor: Predictor,
    sdk: str,
) -> None:
    """Test FINAL_OPT actions maintain is_circuit_synthesized, is_circuit_laid_out, and is_circuit_routed."""
    final_qc, layout = laid_out_and_routed_and_synthesized_circuit

    for action in available_actions_dict[PassType.FINAL_OPT]:
        if sdk == "qiskit" and action.origin == CompilationOrigin.QISKIT:
            compiled, out_layout = run_qiskit_action(
                action=action,
                circuit=final_qc.copy(),
                device=predictor.env.device,
                layout=layout,
                max_iteration=10,
                score_circuit=predictor.env.calculate_reward,
            )
        elif sdk == "bqskit" and action.origin == CompilationOrigin.BQSKIT:
            compiled, out_layout = run_bqskit_action(
                action=action, circuit=final_qc.copy(), device=predictor.env.device, layout=layout
            )
        elif sdk == "tket" and action.origin == CompilationOrigin.TKET:
            compiled, out_layout = run_tket_action(
                action=action, circuit=final_qc.copy(), device=predictor.env.device, layout=layout
            )
        else:
            continue

        assert predictor.env.is_circuit_synthesized(compiled), (
            f"{action.name} on {predictor.env.device.description} VIOLATED INVARIANT: "
            f"final optimization introduced non-native gates. "
            f"Gates in output: {set(compiled.count_ops().keys())}"
        )
        assert out_layout is not None
        assert predictor.env.is_circuit_laid_out(compiled, layout=out_layout), (
            f"{action.name} on {predictor.env.device.description} VIOLATED INVARIANT: "
            f"final optimization did not maintain valid layout"
        )
        assert predictor.env.is_circuit_routed(compiled, predictor.env.device.build_coupling_map()), (
            f"{action.name} on {predictor.env.device.description} VIOLATED INVARIANT: "
            f"final optimization did not maintain valid routing"
        )
