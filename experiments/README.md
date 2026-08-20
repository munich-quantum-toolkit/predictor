# Predictor v3 smoke training

This experiment branch combines the current v3 PR stack:

| PR branch                     | Feature                  |
| ----------------------------- | ------------------------ |
| `v3/666-hybrid-mdp`           | v3 MDP policy            |
| `v3/667-stochastic-actions`   | stochastic action trials |
| `v3/674-normalized-features`  | normalized observations  |
| `v3/675-qiskit-passes`        | Qiskit SABRE actions     |
| `v3/670-intermediate-rewards` | intermediate rewards     |

The feature commits are rebased onto the latest `v3/666-hybrid-mdp` tip on
`experiments/v3-small-training`.

## Run and monitor

```console
uv sync
uv run python experiments/train_v3_smoke.py
```

The default run uses a small 8-unit policy head, 2,048 PPO timesteps, 12
circuits with at most five qubits, and three trials for each stochastic action.
Every run has a unique output directory containing:

- `run.json`: exact Git revision, configuration, circuit list, feature surface,
  and current status;
- `monitor.csv`: episode rewards and lengths;
- `tensorboard/`: live training metrics;
- `checkpoints/`: intermediate models;
- `final_model.zip`: the completed model.

Start TensorBoard with the configured output directory:

```console
uv run tensorboard --logdir <output-dir> --host 127.0.0.1 --port 6006
```

For a longer follow-up run, override only the required values, for example:

```console
uv run python experiments/train_v3_smoke.py --timesteps 100000 --checkpoint-every 5000
```
