# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Helper functions of the reinforcement learning compilation predictor."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from bqskit import MachineModel
from pytket.architecture import Architecture
from pytket.circuit import Circuit, Node, Qubit
from pytket.passes import (
    CliffordSimp,
    FullPeepholeOptimise,
    PeepholeOptimise2Q,
    RemoveRedundancies,
    RoutingPass,
)
from pytket.placement import place_with_map
from qiskit import QuantumCircuit
from qiskit.circuit.equivalence_library import StandardEquivalenceLibrary
from qiskit.circuit.library import XGate, ZGate
from qiskit.transpiler import CouplingMap, Layout, PassManager, TranspileLayout
from qiskit.transpiler.passes import (
    ApplyLayout,
    BasicSwap,
    BasisTranslator,
    Collect2qBlocks,
    CommutativeCancellation,
    CommutativeInverseCancellation,
    ConsolidateBlocks,
    DenseLayout,
    Depth,
    EnlargeWithAncilla,
    FixedPoint,
    FullAncillaAllocation,
    GatesInBasis,
    InverseCancellation,
    MinimumPoint,
    Optimize1qGatesDecomposition,
    OptimizeCliffords,
    RemoveDiagonalGatesBeforeMeasure,
    SabreLayout,
    Size,
    TrivialLayout,
    UnitarySynthesis,
    VF2Layout,
    VF2PostLayout,
)
from qiskit.transpiler.passes.layout.vf2_layout import VF2LayoutStopReason
from sb3_contrib import MaskablePPO

from mqt.predictor import reward, rl
from mqt.predictor.utils import calc_supermarq_features

if TYPE_CHECKING:
    from mqt.bench.devices import Device
    from numpy.random import Generator
    from numpy.typing import NDArray


import operator
import os
import zipfile
from importlib import resources

from bqskit import compile as bqskit_compile
from bqskit.ir import gates
from qiskit import QuantumRegister
from qiskit.passmanager import ConditionalController
from qiskit.transpiler.preset_passmanagers import common

logger = logging.getLogger("mqt-predictor")


def qcompile(
    qc: QuantumCircuit | str,
    figure_of_merit: reward.figure_of_merit | None = "expected_fidelity",
    device_name: str | None = "ibm_washington",
    predictor_singleton: rl.Predictor | None = None,
) -> tuple[QuantumCircuit, list[str]]:
    """Compiles a given quantum circuit to a device optimizing for the given figure of merit.

    Arguments:
        qc: The quantum circuit to be compiled. If a string is given, it is assumed to be a path to a qasm file.
        figure_of_merit: The figure of merit to be used for compilation. Defaults to "expected_fidelity".
        device_name: The name of the device to compile to. Defaults to "ibm_washington".
        predictor_singleton: A predictor object that is used for compilation to reduce compilation time when compiling multiple quantum circuits. If None, a new predictor object is created. Defaults to None.

    Returns:
        A tuple containing the compiled quantum circuit and the compilation information. If compilation fails, False is returned.
    """
    if predictor_singleton is None:
        if figure_of_merit is None:
            msg = "figure_of_merit must not be None if predictor_singleton is None."
            raise ValueError(msg)
        if device_name is None:
            msg = "device_name must not be None if predictor_singleton is None."
            raise ValueError(msg)
        predictor = rl.Predictor(figure_of_merit=figure_of_merit, device_name=device_name)
    else:
        predictor = predictor_singleton

    return predictor.compile_as_predicted(qc)


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
            ),
            "origin": "bqskit",
        },
    ]


def get_action_terminate() -> dict[str, Any]:
    """Returns a dictionary containing information about the terminate pass that is available."""
    return {"name": "terminate"}


def get_state_sample(max_qubits: int, rng: Generator) -> tuple[QuantumCircuit, str]:
    """Returns a random quantum circuit from the training circuits folder.

    Arguments:
        max_qubits: The maximum number of qubits the returned quantum circuit may have. If no limit is set, it defaults to None.
        rng: A random number generator to select a random quantum circuit.

    Returns:
        A tuple containing the random quantum circuit and the path to the file from which it was read.
    """
    file_list = list(get_path_training_circuits().glob("*.qasm"))

    path_zip = get_path_training_circuits() / "training_data_compilation.zip"
    if len(file_list) == 0 and path_zip.exists():
        with zipfile.ZipFile(str(path_zip), "r") as zip_ref:
            zip_ref.extractall(get_path_training_circuits())

        file_list = list(get_path_training_circuits().glob("*.qasm"))
        assert len(file_list) > 0

    found_suitable_qc = False
    while not found_suitable_qc:
        random_index = rng.integers(len(file_list))
        num_qubits = int(str(file_list[random_index]).split("_")[-1].split(".")[0])
        if max_qubits and num_qubits > max_qubits:
            continue
        found_suitable_qc = True

    try:
        qc = QuantumCircuit.from_qasm_file(str(file_list[random_index]))
    except Exception:
        raise RuntimeError("Could not read QuantumCircuit from: " + str(file_list[random_index])) from None

    print("QC: ", str(file_list[random_index]))
    return qc, str(file_list[random_index])


def create_feature_dict(qc: QuantumCircuit) -> dict[str, int | NDArray[np.float64]]:
    """Creates a feature dictionary for a given quantum circuit.

    Arguments:
        qc: The quantum circuit for which the feature dictionary is created.

    Returns:
        The feature dictionary for the given quantum circuit.
    """
    feature_dict = {
        "num_qubits": qc.num_qubits,
        "depth": qc.depth(),
    }

    supermarq_features = calc_supermarq_features(qc)
    # for all dict values, put them in a list each
    feature_dict["program_communication"] = np.array([supermarq_features.program_communication], dtype=np.float32)
    feature_dict["critical_depth"] = np.array([supermarq_features.critical_depth], dtype=np.float32)
    feature_dict["entanglement_ratio"] = np.array([supermarq_features.entanglement_ratio], dtype=np.float32)
    feature_dict["parallelism"] = np.array([supermarq_features.parallelism], dtype=np.float32)
    feature_dict["liveness"] = np.array([supermarq_features.liveness], dtype=np.float32)

    return feature_dict


def get_path_training_data() -> Path:
    """Returns the path to the training data folder used for RL training."""
    return Path(str(resources.files("mqt.predictor"))) / "rl" / "training_data"


def get_path_trained_model() -> Path:
    """Returns the path to the trained model folder used for RL training."""
    return get_path_training_data() / "trained_model"


def get_path_training_circuits() -> Path:
    """Returns the path to the training circuits folder used for RL training."""
    return get_path_training_data() / "training_circuits"


def load_model(model_name: str) -> MaskablePPO:
    """Loads a trained model from the trained model folder.

    Arguments:
        model_name: The name of the model to be loaded.

    Returns:
        The loaded model.
    """
    path = get_path_trained_model()
    if Path(path / (model_name + ".zip")).is_file():
        return MaskablePPO.load(path / (model_name + ".zip"))

    error_msg = f"The RL model '{model_name}' is not trained yet. Please train the model before using it."
    logger.error(error_msg)
    raise FileNotFoundError(error_msg)


class PreProcessTKETRoutingAfterQiskitLayout:
    """Pre-processing step to route a circuit with TKET after a Qiskit Layout pass has been applied.

        The reason why we can apply the trivial layout here is that the circuit already got assigned a layout by qiskit.
        Implicitly, Qiskit is reordering its qubits in a sequential manner, i.e., the qubit with the lowest *physical* qubit
        first.

        Assuming, the layouted circuit is given by

                       ┌───┐           ░       ┌─┐
              q_2 -> 0 ┤ H ├──■────────░───────┤M├
                       └───┘┌─┴─┐      ░    ┌─┐└╥┘
              q_1 -> 1 ─────┤ X ├──■───░────┤M├─╫─
                            └───┘┌─┴─┐ ░ ┌─┐└╥┘ ║
              q_0 -> 2 ──────────┤ X ├─░─┤M├─╫──╫─
                                 └───┘ ░ └╥┘ ║  ║
        ancilla_0 -> 3 ───────────────────╫──╫──╫─
                                          ║  ║  ║
        ancilla_1 -> 4 ───────────────────╫──╫──╫─
                                          ║  ║  ║
               meas: 3/═══════════════════╩══╩══╩═
                                          0  1  2

        Applying the trivial layout, we get the same qubit order as in the original circuit and can be respectively
        routed. This results int:
                ┌───┐           ░       ┌─┐
           q_0: ┤ H ├──■────────░───────┤M├
                └───┘┌─┴─┐      ░    ┌─┐└╥┘
           q_1: ─────┤ X ├──■───░────┤M├─╫─
                     └───┘┌─┴─┐ ░ ┌─┐└╥┘ ║
           q_2: ──────────┤ X ├─░─┤M├─╫──╫─
                          └───┘ ░ └╥┘ ║  ║
           q_3: ───────────────────╫──╫──╫─
                                   ║  ║  ║
           q_4: ───────────────────╫──╫──╫─
                                   ║  ║  ║
        meas: 3/═══════════════════╩══╩══╩═
                                   0  1  2


        If we would not apply the trivial layout, no layout would be considered resulting, e.g., in the followiong circuit:
                 ┌───┐         ░    ┌─┐
       q_0: ─────┤ X ├─────■───░────┤M├───
            ┌───┐└─┬─┘   ┌─┴─┐ ░ ┌─┐└╥┘
       q_1: ┤ H ├──■───X─┤ X ├─░─┤M├─╫────
            └───┘      │ └───┘ ░ └╥┘ ║ ┌─┐
       q_2: ───────────X───────░──╫──╫─┤M├
                               ░  ║  ║ └╥┘
       q_3: ──────────────────────╫──╫──╫─
                                  ║  ║  ║
       q_4: ──────────────────────╫──╫──╫─
                                  ║  ║  ║
    meas: 3/══════════════════════╩══╩══╩═
                                  0  1  2

    """

    def apply(self, circuit: Circuit) -> None:
        """Applies the pre-processing step to route a circuit with tket after a Qiskit Layout pass has been applied."""
        mapping = {Qubit(i): Node(i) for i in range(circuit.n_qubits)}
        place_with_map(circuit=circuit, qmap=mapping)


def get_bqskit_native_gates(device: Device) -> list[gates.Gate] | None:
    """Returns the native gates of the given device.

    Arguments:
        device: The device for which the native gates are returned.

    Returns:
        The native gates of the given provider.
    """
    native_gatesets = {
        "ibm_heron": [gates.RZGate(), gates.SXGate(), gates.XGate(), gates.CZGate()],
        "ibm_eagle": [gates.RZGate(), gates.SXGate(), gates.XGate(), gates.ECRGate()],
        "ibm_falcon": [gates.RZGate(), gates.SXGate(), gates.XGate(), gates.CXGate()],
        "quantinuum": [gates.RZZGate(), gates.RZGate(), gates.RYGate(), gates.RXGate()],
        "iqm": [gates.U3Gate(), gates.CZGate()],
    }

    for key, value in native_gatesets.items():
        if device.description.startswith(key):
            return value
    msg = f"No native gates found for device {device.description} in BQSKIT"
    raise ValueError(msg)


def final_layout_pytket_to_qiskit(pytket_circuit: Circuit, qiskit_circuit: QuantumCircuit) -> Layout:
    """Converts a final layout from pytket to qiskit."""
    pytket_layout = pytket_circuit.qubit_readout
    size_circuit = pytket_circuit.n_qubits
    qiskit_layout = {}
    qiskit_qreg = qiskit_circuit.qregs[0]

    pytket_layout = dict(sorted(pytket_layout.items(), key=operator.itemgetter(1)))

    for node, qubit_index in pytket_layout.items():
        qiskit_layout[node.index[0]] = qiskit_qreg[qubit_index]

    for i in range(size_circuit):
        if i not in set(pytket_layout.values()):
            qiskit_layout[i] = qiskit_qreg[i]

    return Layout(input_dict=qiskit_layout)


def final_layout_bqskit_to_qiskit(
    bqskit_initial_layout: list[int],
    bqskit_final_layout: list[int],
    compiled_qc: QuantumCircuit,
    initial_qc: QuantumCircuit,
) -> TranspileLayout:
    """Converts a final layout from bqskit to qiskit.

    BQSKit provides an initial layout as a list[int] where each virtual qubit is mapped to a physical qubit
    similarly, it provides a final layout as a list[int] representing where each virtual qubit is mapped to at the end
    of the circuit.
    """
    ancilla = QuantumRegister(compiled_qc.num_qubits - initial_qc.num_qubits, "ancilla")
    qiskit_initial_layout = {}
    counter_ancilla_qubit = 0
    for i in range(compiled_qc.num_qubits):
        if i in bqskit_initial_layout:
            qiskit_initial_layout[i] = initial_qc.qubits[bqskit_initial_layout.index(i)]
        else:
            qiskit_initial_layout[i] = ancilla[counter_ancilla_qubit]
            counter_ancilla_qubit += 1

    initial_qubit_mapping = {bit: index for index, bit in enumerate(compiled_qc.qubits)}

    if bqskit_initial_layout == bqskit_final_layout:
        qiskit_final_layout = None
    else:
        qiskit_final_layout = {}
        for i in range(compiled_qc.num_qubits):
            if i in bqskit_final_layout:
                qiskit_final_layout[i] = compiled_qc.qubits[bqskit_initial_layout[bqskit_final_layout.index(i)]]
            else:
                qiskit_final_layout[i] = compiled_qc.qubits[i]

    return TranspileLayout(
        initial_layout=Layout(input_dict=qiskit_initial_layout),
        input_qubit_mapping=initial_qubit_mapping,
        final_layout=Layout(input_dict=qiskit_final_layout) if qiskit_final_layout else None,
        _output_qubit_list=compiled_qc.qubits,
        _input_qubit_count=initial_qc.num_qubits,
    )


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
