# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Helper functions for the machine learning device selection predictor."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mqt.predictor import ml, reward, rl

if TYPE_CHECKING:
    from qiskit import QuantumCircuit


def qcompile(
    qc: QuantumCircuit,
    figure_of_merit: reward.figure_of_merit = "expected_fidelity",
) -> tuple[QuantumCircuit, list[str], str]:
    """Compiles a given quantum circuit to a device with the highest predicted figure of merit.

    Arguments:
        qc: The quantum circuit to be compiled.
        figure_of_merit: The figure of merit to be used for compilation. Defaults to "expected_fidelity".

    Returns:
        A tuple containing the compiled quantum circuit, the compilation information, and the name of the device used for compilation.
    """
    predicted_device = ml.predict_device_for_figure_of_merit(qc, figure_of_merit=figure_of_merit)
    res = rl.qcompile(qc, device=predicted_device, figure_of_merit=figure_of_merit)
    return *res, predicted_device
