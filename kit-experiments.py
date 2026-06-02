# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Train the optimization-only RL predictor on the KIT/ICSE corpus."""

from pathlib import Path

from mqt.predictor.rl import OptOnlyPredictor

pred = OptOnlyPredictor(
    path_training_circuits=Path("/Users/patrickhopf/Code/icse-paper-2026-qiskit-ml/data/raw"),
    test_circuits_csv=Path("/Users/patrickhopf/Code/icse-paper-2026-qiskit-ml/data/archive/tables/test_circuits.csv"),
)

pred.train_model(timesteps=1000, test=True)
