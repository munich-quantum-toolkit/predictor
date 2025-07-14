from __future__ import annotations

import os
from typing import Any

from bqskit import compile as bqskit_compile, MachineModel
from pytket._tket.architecture import Architecture
from pytket._tket.passes import PeepholeOptimise2Q, CliffordSimp, FullPeepholeOptimise, RemoveRedundancies, RoutingPass
from qiskit import QuantumCircuit
from qiskit.circuit import StandardEquivalenceLibrary
from qiskit.circuit.library import XGate, ZGate
from qiskit.passmanager import ConditionalController
from qiskit.transpiler import CouplingMap, Layout, TranspileLayout, PassManager
from qiskit.transpiler.passes import Optimize1qGatesDecomposition, CommutativeCancellation, \
    CommutativeInverseCancellation, RemoveDiagonalGatesBeforeMeasure, InverseCancellation, OptimizeCliffords, \
    Collect2qBlocks, ConsolidateBlocks, UnitarySynthesis, GatesInBasis, Depth, FixedPoint, Size, MinimumPoint, \
    VF2PostLayout, TrivialLayout, FullAncillaAllocation, EnlargeWithAncilla, ApplyLayout, DenseLayout, VF2Layout, \
    BasicSwap, SabreLayout, BasisTranslator
from qiskit.transpiler.passes.layout.vf2_layout import VF2LayoutStopReason
from qiskit.transpiler.preset_passmanagers import common

from mqt.predictor.rl.parsing import PreProcessTKETRoutingAfterQiskitLayout, get_bqskit_native_gates



def get_actions_opt() -> list[dict[str, Any]]:
    """Returns a list of dictionaries containing information about the optimization passes that are available."""
    return [
        {
            "name": "Optimize1qGatesDecomposition",
            "transpile_pass": [Optimize1qGatesDecomposition()],
            "origin": "qiskit",
        },
        {
            "name": "CommutativeCancellation",
            "transpile_pass": [CommutativeCancellation()],
            "origin": "qiskit",
        },
        {
            "name": "CommutativeInverseCancellation",
            "transpile_pass": [CommutativeInverseCancellation()],
            "origin": "qiskit",
        },
        {
            "name": "RemoveDiagonalGatesBeforeMeasure",
            "transpile_pass": [RemoveDiagonalGatesBeforeMeasure()],
            "origin": "qiskit",
        },
        {
            "name": "InverseCancellation",
            "transpile_pass": [InverseCancellation([XGate(), ZGate()])],
            "origin": "qiskit",
        },
        {
            "name": "OptimizeCliffords",
            "transpile_pass": [OptimizeCliffords()],
            "origin": "qiskit",
        },
        {
            "name": "Opt2qBlocks",
            "transpile_pass": [Collect2qBlocks(), ConsolidateBlocks()],
            "origin": "qiskit",
        },
        {
            "name": "PeepholeOptimise2Q",
            "transpile_pass": [PeepholeOptimise2Q()],
            "origin": "tket",
        },
        {
            "name": "CliffordSimp",
            "transpile_pass": [CliffordSimp()],
            "origin": "tket",
        },
        {
            "name": "FullPeepholeOptimiseCX",
            "transpile_pass": [FullPeepholeOptimise()],
            "origin": "tket",
        },
        {
            "name": "RemoveRedundancies",
            "transpile_pass": [RemoveRedundancies()],
            "origin": "tket",
        },
        {
            "name": "QiskitO3",
            "transpile_pass": lambda native_gate, coupling_map: [
                Collect2qBlocks(),
                ConsolidateBlocks(basis_gates=native_gate),
                UnitarySynthesis(basis_gates=native_gate, coupling_map=coupling_map),
                Optimize1qGatesDecomposition(basis=native_gate),
                CommutativeCancellation(basis_gates=native_gate),
                GatesInBasis(native_gate),
                ConditionalController(
                    common.generate_translation_passmanager(
                        target=None, basis_gates=native_gate, coupling_map=coupling_map
                    ).to_flow_controller(),
                    condition=lambda property_set: not property_set["all_gates_in_basis"],
                ),
                Depth(recurse=True),
                FixedPoint("depth"),
                Size(recurse=True),
                FixedPoint("size"),
                MinimumPoint(["depth", "size"], "optimization_loop"),
            ],
            "origin": "qiskit",
            "do_while": lambda property_set: (not property_set["optimization_loop_minimum_point"]),
        },
        {
            "name": "BQSKitO2",
            "transpile_pass": lambda circuit: bqskit_compile(
                circuit,
                optimization_level=1 if os.getenv("GITHUB_ACTIONS") == "true" else 2,
                synthesis_epsilon=1e-1 if os.getenv("GITHUB_ACTIONS") == "true" else 1e-8,
                max_synthesis_size=2 if os.getenv("GITHUB_ACTIONS") == "true" else 3,
                seed=10,
                num_workers=1 if os.getenv("GITHUB_ACTIONS") == "true" else -1,
            ),
            "origin": "bqskit",
        },
    ]


def get_actions_final_optimization() -> list[dict[str, Any]]:
    """Returns a list of dictionaries containing information about the optimization passes that are available."""
    return [
        {
            "name": "VF2PostLayout",
            "transpile_pass": lambda device: VF2PostLayout(
                target=device,
            ),
            "origin": "qiskit",
        }
    ]


def get_actions_layout() -> list[dict[str, Any]]:
    """Returns a list of dictionaries containing information about the layout passes that are available."""
    return [
        {
            "name": "TrivialLayout",
            "transpile_pass": lambda device: [
                TrivialLayout(coupling_map=CouplingMap(device.build_coupling_map())),
                FullAncillaAllocation(coupling_map=CouplingMap(device.build_coupling_map())),
                EnlargeWithAncilla(),
                ApplyLayout(),
            ],
            "origin": "qiskit",
        },
        {
            "name": "DenseLayout",
            "transpile_pass": lambda device: [
                DenseLayout(coupling_map=CouplingMap(device.build_coupling_map())),
                FullAncillaAllocation(coupling_map=CouplingMap(device.build_coupling_map())),
                EnlargeWithAncilla(),
                ApplyLayout(),
            ],
            "origin": "qiskit",
        },
        {
            "name": "VF2Layout",
            "transpile_pass": lambda device: [
                VF2Layout(
                    target=device,
                ),
                ConditionalController(
                    [
                        FullAncillaAllocation(coupling_map=CouplingMap(device.build_coupling_map())),
                        EnlargeWithAncilla(),
                        ApplyLayout(),
                    ],
                    condition=lambda property_set: property_set["VF2Layout_stop_reason"]
                    == VF2LayoutStopReason.SOLUTION_FOUND,
                ),
            ],
            "origin": "qiskit",
        },
    ]


def get_actions_routing() -> list[dict[str, Any]]:
    """Returns a list of dictionaries containing information about the routing passes that are available."""
    return [
        {
            "name": "BasicSwap",
            "transpile_pass": lambda device: [BasicSwap(coupling_map=CouplingMap(device.build_coupling_map()))],
            "origin": "qiskit",
        },
        {
            "name": "RoutingPass",
            "transpile_pass": lambda device: [
                PreProcessTKETRoutingAfterQiskitLayout(),
                RoutingPass(Architecture(list(device.build_coupling_map()))),
            ],
            "origin": "tket",
        },
    ]


def get_actions_mapping() -> list[dict[str, Any]]:
    """Returns a list of dictionaries containing information about the mapping passes that are available."""
    return [
        {
            "name": "SabreMapping",
            "transpile_pass": lambda device: [
                SabreLayout(coupling_map=CouplingMap(device.build_coupling_map()), skip_routing=False),
            ],
            "origin": "qiskit",
        },
        {
            "name": "BQSKitMapping",
            "transpile_pass": lambda device: lambda bqskit_circuit: bqskit_compile(
                bqskit_circuit,
                model=MachineModel(
                    num_qudits=device.num_qubits,
                    gate_set=get_bqskit_native_gates(device),
                    coupling_graph=[(elem[0], elem[1]) for elem in device.build_coupling_map()],
                ),
                with_mapping=True,
                optimization_level=1 if os.getenv("GITHUB_ACTIONS") == "true" else 2,
                synthesis_epsilon=1e-1 if os.getenv("GITHUB_ACTIONS") == "true" else 1e-8,
                max_synthesis_size=2 if os.getenv("GITHUB_ACTIONS") == "true" else 3,
                seed=10,
                num_workers=1 if os.getenv("GITHUB_ACTIONS") == "true" else -1,
            ),
            "origin": "bqskit",
        },
    ]


def get_actions_synthesis() -> list[dict[str, Any]]:
    """Returns a list of dictionaries containing information about the synthesis passes that are available."""
    return [
        {
            "name": "BasisTranslator",
            "transpile_pass": lambda device: [
                BasisTranslator(StandardEquivalenceLibrary, target_basis=device.operation_names)
            ],
            "origin": "qiskit",
        },
        {
            "name": "BQSKitSynthesis",
            "transpile_pass": lambda device: lambda bqskit_circuit: bqskit_compile(
                bqskit_circuit,
                model=MachineModel(bqskit_circuit.num_qudits, gate_set=get_bqskit_native_gates(device)),
                optimization_level=1 if os.getenv("GITHUB_ACTIONS") == "true" else 2,
                synthesis_epsilon=1e-1 if os.getenv("GITHUB_ACTIONS") == "true" else 1e-8,
                max_synthesis_size=2 if os.getenv("GITHUB_ACTIONS") == "true" else 3,
                seed=10,
                num_workers=1 if os.getenv("GITHUB_ACTIONS") == "true" else -1,
            ),
            "origin": "bqskit",
        },
    ]


def get_action_terminate() -> dict[str, Any]:
    """Returns a dictionary containing information about the terminate pass that is available."""
    return {"name": "terminate"}


def postprocess_vf2postlayout(
    qc: QuantumCircuit, post_layout: Layout, layout_before: TranspileLayout
) -> tuple[QuantumCircuit, PassManager]:
    """Postprocesses the given quantum circuit with the post_layout and returns the altered quantum circuit and the respective PassManager."""
    apply_layout = ApplyLayout()
    assert layout_before is not None
    apply_layout.property_set["layout"] = layout_before.initial_layout
    apply_layout.property_set["original_qubit_indices"] = layout_before.input_qubit_mapping
    apply_layout.property_set["final_layout"] = layout_before.final_layout
    apply_layout.property_set["post_layout"] = post_layout

    altered_qc = apply_layout(qc)
    return altered_qc, apply_layout
