# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

from __future__ import annotations

from predictor import Predictor

rl_pred = Predictor(figure_of_merit="expected_fidelity", device_name="ibm_washington")
rl_pred.train_model(timesteps=100000, model_name="sample_model_rl")
