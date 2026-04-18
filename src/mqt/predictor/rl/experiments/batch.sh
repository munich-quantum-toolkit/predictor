#!/bin/bash
# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

#SBATCH --job-name=mqt_predictor
#SBATCH --output=predictor_%j.log
#SBATCH --error=predictor_%j.err

#SBATCH --partition=cm4_inter
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=08:00:00

set -euo pipefail

REPO_DIR="/dss/dsshome1/0C/ge87mij2/mqt/predictor"
SWEEP_MODULE="mqt.predictor.rl.experiments.train_eval_sweep"


# Activate Conda environment
eval "$($HOME/miniconda/bin/conda shell.bash hook)"
conda activate mqt

export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

#########################################
####### never change header above #######
#########################################

DEFAULT_ARGS=(
  --device ibm_falcon_127
  --figure-of-merit expected_fidelity
  --mdp hybrid
  --timesteps 1000
  --verbose 1
  --train-dir src/mqt/predictor/rl/training_data/training_circuits/train
  --reward-scale 1.0
  --no-effect-penalty -0.001
  --checkpoint-dir src/mqt/predictor/rl/training_data/checkpoints/gnn_expected_fidelity_ibm_falcon_127_hybrid
  --checkpoint-frequency 2048
  --max-episode-steps 100
  --graph
  --iterations 1000
  --steps 2048
  --num-epochs 10
  --minibatch-size 64
  --hidden-dim 119
  --num-conv-wo-resnet 1
  --num-resnet-layers 5
  --dropout-p 0.1
  --lr 1e-3
  --gnn-lr 1e-3
)

cd "$REPO_DIR"

if [ "$#" -gt 0 ]; then
  ARGS=("$@")
else
  ARGS=("${DEFAULT_ARGS[@]}")
fi

printf 'Resolved command: python3 -u src/mqt/predictor/rl/experiments/training.py'
printf ' %q' "${ARGS[@]}"
printf '\n'

python3 -u src/mqt/predictor/rl/experiments/training.py "${ARGS[@]}"
