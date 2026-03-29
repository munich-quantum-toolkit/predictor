# RL Workflow

This directory currently exposes four small command-line entry points:

- `data_generation.py`: generate and split RL train/test circuit data from MQT Bench
- `training.py`: train an RL model on an existing training split
- `evaluate.py`: evaluate a trained RL model on an existing test split
- `workflow.py`: run the full generate/train/evaluate pipeline end-to-end

The overlap is intentional:

- the smaller entry points make it easier to rerun only one stage while iterating on experiments
- `workflow.py` keeps the full experiment pipeline reproducible in one command

Typical usage:

1. Generate the train/test split with `python -m mqt.predictor.rl.data_generation`
2. Train with `python -m mqt.predictor.rl.training`
3. Evaluate with `python -m mqt.predictor.rl.evaluate`

Use `python -m mqt.predictor.rl.workflow` when you want the full pipeline in one run.
