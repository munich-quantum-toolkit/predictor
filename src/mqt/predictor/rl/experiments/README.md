# RL Experiments

This directory contains the RL experiment entry points.

The important distinction is:

- `training.py`, `evaluate.py`, and `pipeline_evaluation.py` are single-purpose tools
- `train_eval_sweep.py` is the SLURM-friendly orchestration script for the full ESP sweep

## Files

- `data_generation.py`
  Generates and splits RL train/test data from MQT Bench.
- `training.py`
  Trains one RL model for one device, one figure of merit, and one MDP.
  Supports PPO checkpoints and resume-from-checkpoint.
- `evaluate.py`
  Evaluates one trained RL model on a held-out test split.
- `evaluation.py`
  Shared RL evaluation logic.
- `pipeline_evaluation.py`
  Evaluates the locked baseline pipelines `qiskit_o3` and `tket_o2`.
- `train_eval_sweep.py`
  Runs the practical full sweep used for SLURM jobs.

## What The Sweep Does

`train_eval_sweep.py` is intentionally narrow.

It always targets:

- figure of merit: `estimated_success_probability`
- devices: by default `ibm_boston_156` and `iqm_garnet_20`
- RL variants: all MDPs `paper`, `flexible`, `thesis`, `hybrid`
- evaluation modes: both `stochastic` and `deterministic`
- baselines: by default `qiskit_o3` and `tket_o2`

It is built to:

- start a fresh sweep in an empty output directory
- continue an interrupted sweep in an existing output directory
- resume unfinished training runs from saved PPO checkpoints
- skip phases that already completed

## Single-Run Tools

### Train One RL Model

```bash
python3 -u src/mqt/predictor/rl/experiments/training.py \
  --device ibm_boston_156 \
  --figure-of-merit estimated_success_probability \
  --mdp paper \
  --timesteps 100000 \
  --checkpoint-dir /tmp/mqt_rl_checkpoints \
  --checkpoint-frequency 2048
```

Resume from a saved PPO checkpoint:

```bash
python3 -u src/mqt/predictor/rl/experiments/training.py \
  --device ibm_boston_156 \
  --figure-of-merit estimated_success_probability \
  --mdp paper \
  --timesteps 50000 \
  --checkpoint-dir /tmp/mqt_rl_checkpoints \
  --checkpoint-frequency 2048 \
  --resume-from-checkpoint /tmp/mqt_rl_checkpoints/model_checkpoint_49152_steps.zip
```

### Evaluate One RL Model

```bash
python3 -u src/mqt/predictor/rl/experiments/evaluate.py \
  --device ibm_boston_156 \
  --figure-of-merit estimated_success_probability \
  --mdp paper \
  --model-path path/to/model.zip \
  --max-steps 200 \
  --seed 0
```

Add `--deterministic` to evaluate with deterministic policy inference.

### Evaluate Locked Baselines

```bash
python3 -u src/mqt/predictor/rl/experiments/pipeline_evaluation.py \
  --device ibm_boston_156 \
  --figure-of-merit estimated_success_probability \
  --pipelines qiskit_o3 tket_o2 \
  --output baseline_report.json
```

## Full Sweep Usage

Example:

```bash
PYTHONPATH=src python3 -u -m mqt.predictor.rl.experiments.train_eval_sweep \
  --devices ibm_boston_156 iqm_garnet_20 \
  --timesteps 100000 \
  --checkpoint-frequency 2048 \
  --output-dir /path/to/esp_sweep
```

This will:

1. Train all four MDP variants for each selected device.
2. Save intermediate PPO checkpoints during training.
3. Resume from the latest checkpoint if rerun later.
4. Evaluate each completed RL model in stochastic and deterministic mode.
5. Run the two locked baseline pipelines unless you disable them.

Useful options:

- `--skip-pipelines`
  Skip `qiskit_o3` and `tket_o2`.
- `--train-dir`
  Use a custom training-circuit directory.
- `--test-dir`
  Use a custom test-circuit directory.
- `--test-training`
  Use the lightweight training mode.
- `--force`
  Ignore completed markers and rerun everything.
- `--fail-fast`
  Stop on the first failure.
- `--dry-run`
  Print the plan and exit.

## Resume Behavior

Resume is based entirely on the sweep output directory.

To continue an interrupted job:

1. Run the same command again.
2. Reuse the exact same `--output-dir`.
3. Do not pass `--force`.

What the script does on resume:

- if a training run already finished, it skips it
- if a training run was interrupted, it searches `checkpoints/` and resumes from the newest checkpoint
- if an evaluation already finished, it skips it
- if a pipeline result already finished, it skips it

Important detail:

- resume is checkpoint-based, not exact in-memory continuation
- with `--checkpoint-frequency 2048`, the worst-case repeated work is about one checkpoint interval

## SLURM Usage

The repo-root `batch.sh` is the intended entry point for the full job.

It calls `train_eval_sweep.py` once and lets the Python script manage both devices internally.
The batch script exports `PYTHONPATH=$REPO_DIR/src` and runs the sweep as a module.

Default behavior of `batch.sh`:

- devices: `ibm_boston_156 iqm_garnet_20`
- figure of merit: `estimated_success_probability`
- timesteps: `100000`
- checkpoint frequency: `2048`
- pipelines: enabled
- output directory: stable across reruns unless you change it explicitly

Basic submission:

```bash
sbatch batch.sh
```

Override devices and timesteps:

```bash
sbatch --export=MQT_DEVICES="ibm_boston_156 iqm_garnet_20",MQT_TIMESTEPS=100000 batch.sh
```

Disable pipelines:

```bash
sbatch --export=MQT_SKIP_PIPELINES=1 batch.sh
```

Force a specific persistent output directory:

```bash
sbatch --export=MQT_RUN_OUTPUT_DIR=/path/to/shared/esp_sweep batch.sh
```

Useful environment variables for `batch.sh`:

- `MQT_DEVICES`
  Space-separated device list.
- `MQT_TIMESTEPS`
  Training timesteps per RL configuration.
- `MQT_CHECKPOINT_FREQUENCY`
  PPO checkpoint cadence.
- `MQT_SKIP_PIPELINES`
  Set to `1` to skip the pipeline baselines.
- `MQT_OUTPUT_ROOT`
  Root directory for sweep outputs.
- `MQT_RUN_NAME`
  Stable output-directory name below `MQT_OUTPUT_ROOT`.
- `MQT_RUN_OUTPUT_DIR`
  Explicit full output directory. This overrides `MQT_RUN_NAME`.

## Output Layout

The sweep writes one stable directory tree.

Top level:

- `manifest.json`
  Static sweep configuration.
- `progress.json`
  Current completion counts for training, evaluation, and pipelines.
- `sweep.log`
  Human-readable log for live monitoring and debugging.

Per device and MDP:

- `<output-dir>/<device>/rl/<mdp>/status.json`
  Current run status. Check this first while the job is running.
- `<output-dir>/<device>/rl/<mdp>/training.json`
  Final training metadata.
- `<output-dir>/<device>/rl/<mdp>/artifacts/model_<device>_<mdp>.zip`
  Copied final trained model.
- `<output-dir>/<device>/rl/<mdp>/checkpoints/model_checkpoint_*_steps.zip`
  Intermediate PPO checkpoints used for resume.
- `<output-dir>/<device>/rl/<mdp>/evaluation_stochastic.json`
  Full stochastic evaluation result.
- `<output-dir>/<device>/rl/<mdp>/evaluation_deterministic.json`
  Full deterministic evaluation result.

Per pipeline baseline:

- `<output-dir>/<device>/pipelines/<pipeline>/status.json`
  Current baseline status.
- `<output-dir>/<device>/pipelines/<pipeline>/result.json`
  Full baseline result.

## How To Read Results

Start with:

- `progress.json`
  Shows how much of the sweep finished.
- `sweep.log`
  Shows the live execution order, resumes, and failures.
- `<device>/rl/<mdp>/status.json`
  Shows whether a specific RL run is pending, running, completed, interrupted, or failed.

If a training run looks incomplete:

- open `status.json`
- check `completed_timesteps`
- check the newest file in `checkpoints/`

If you want final RL quality for one configuration:

- open `evaluation_stochastic.json`
- open `evaluation_deterministic.json`

Those files contain the full per-circuit results plus aggregate metrics.

If you want the locked baseline comparison:

- open `<device>/pipelines/qiskit_o3/result.json`
- open `<device>/pipelines/tket_o2/result.json`

If something failed:

- check `sweep.log`
- then check the nearest `status.json`
- failed and interrupted phases stay on disk and are reused on the next rerun
