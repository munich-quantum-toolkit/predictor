# RL CLI

This directory exposes three small command-line entry points:

- `data_generation.py`: generate and split RL train/test circuit data from MQT Bench
- `training.py`: train an RL model on an existing training split
- `evaluate.py`: evaluate a trained RL model on an existing test split

The split is intentional:

- `data_generation.py` handles dataset preparation separately
- `training.py` stays focused on model training
- `evaluate.py` is a thin CLI wrapper around the evaluation logic in `evaluation.py`

Typical usage:

1. Generate the train/test split with `python -m mqt.predictor.rl.data_generation`
2. Train with `python -m mqt.predictor.rl.training`
3. Evaluate with `python -m mqt.predictor.rl.evaluate`
