# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""This module contains the Predictor class, which is used to predict the most suitable compilation pass sequence for a given quantum circuit."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
import torch
from numpy.typing import NDArray
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.utils import set_random_seed
from torch.distributions import Categorical
from torch_geometric.data import Batch, Data

from mqt.predictor.rl.gnn_ppo import create_gnn_policy, train_ppo_with_gnn
from mqt.predictor.rl.helper import GLOBAL_FEATURE_DIM, get_path_trained_model, logger, predicted_action_to_index
from mqt.predictor.rl.predictorenv import PredictorEnv

if TYPE_CHECKING:
    from qiskit import QuantumCircuit
    from qiskit.transpiler import Target
    from stable_baselines3.common.callbacks import BaseCallback

    from mqt.predictor.reward import figure_of_merit
    from mqt.predictor.rl.gnn import SAGEActorCritic


MaskablePPOObservationValue: TypeAlias = NDArray[np.int64] | NDArray[np.float32]
MaskablePPOObservation: TypeAlias = dict[str, MaskablePPOObservationValue]


def _as_maskable_ppo_observation(obs: object) -> MaskablePPOObservation:
    """Normalize flat env observations to the ndarray-only form expected by SB3."""
    if isinstance(obs, Data):
        msg = "MaskablePPO requires flat observations. Construct the predictor with graph=False."
        raise TypeError(msg)
    if not isinstance(obs, dict):
        msg = f"Expected a flat observation dictionary, received {type(obs).__name__}."
        raise TypeError(msg)

    normalized_obs: MaskablePPOObservation = {}
    for key, value in obs.items():
        if not isinstance(key, str):
            msg = f"Expected string observation keys, received {type(key).__name__}."
            raise TypeError(msg)
        if isinstance(value, int):
            normalized_obs[key] = np.asarray(value, dtype=np.int64)
        else:
            normalized_obs[key] = np.asarray(value, dtype=np.float32)
    return normalized_obs


class Predictor:
    """The Predictor class is used to train a reinforcement learning model for a given figure of merit and device such that it acts as a compiler."""

    def __init__(
        self,
        figure_of_merit: figure_of_merit,
        device: Target,
        mdp: str = "paper",
        path_training_circuits: Path | None = None,
        reward_scale: float = 1.0,
        no_effect_penalty: float = -0.001,
        max_episode_steps: int | None = None,
        graph: bool = False,
    ) -> None:
        """Initializes the Predictor object.

        Arguments:
            figure_of_merit: The figure of merit to optimize during compilation.
            device: The target quantum device.
            mdp: The MDP formulation to use ("paper" or "alternative"). Defaults to "paper".
            path_training_circuits: Path to training circuits. Defaults to None.
            reward_scale: Scaling factor for rewards/penalties proportional to fidelity changes.
            no_effect_penalty: Step penalty applied when an action does not change the circuit.
            max_episode_steps: Optional hard cap on environment steps per episode.
            logger_level: Logging level. Defaults to INFO.
            graph: If True, uses a GNN-based policy with PyG graph observations
                   (including circuit-level global features matching the non-GNN policy).
                   If False (default), uses the flat-feature MaskablePPO policy.
        """
        self.graph = graph
        self.env = PredictorEnv(
            reward_function=figure_of_merit,
            device=device,
            path_training_circuits=path_training_circuits,
            reward_scale=reward_scale,
            no_effect_penalty=no_effect_penalty,
            max_episode_steps=max_episode_steps,
            graph=graph,
            mdp=mdp,
        )
        self.device_name = device.description
        self.figure_of_merit = figure_of_merit
        self.gnn_model: SAGEActorCritic | None = None

    def compile_as_predicted(
        self,
        qc: QuantumCircuit | str,
    ) -> tuple[QuantumCircuit, list[str]]:
        """Compiles a given quantum circuit such that the given figure of merit is maximized by using the respectively trained optimized compiler.

        Arguments:
            qc: The quantum circuit to be compiled or the path to a qasm file containing the quantum circuit.

        Returns:
            A tuple containing the compiled quantum circuit and the compilation information. If compilation fails, False is returned.

        Raises:
            RuntimeError: If an error occurs during compilation.
        """
        if not self.graph:
            return self._compile_with_maskable_ppo(qc)
        return self._compile_with_gnn(qc)

    def _compile_with_maskable_ppo(self, qc: QuantumCircuit | str) -> tuple[QuantumCircuit, list[str]]:
        """Compiles the quantum circuit using a trained MaskablePPO model.

        Arguments:
            qc: The quantum circuit to be compiled. It can be a Quantum Circuit object or a string for the path to a qasm file.

        Returns:
            A tuple containing the compiled quantum circuit and a list of the compilation passes used.
        """
        trained_rl_model = load_model("model_" + self.figure_of_merit + "_" + self.device_name)

        obs, _ = self.env.reset(qc, seed=0)

        used_compilation_passes: list[str] = []
        step_records: list[dict] = []
        terminated = False
        truncated = False
        while not (terminated or truncated):
            depth_before = self.env.state.depth()
            gates_before = sum(v for k, v in self.env.state.count_ops().items() if k != "barrier")

            action_masks = get_action_masks(self.env)
            action, _ = trained_rl_model.predict(_as_maskable_ppo_observation(obs), action_masks=action_masks)
            action = predicted_action_to_index(action)
            action_item = self.env.action_set[action]
            used_compilation_passes.append(action_item.name)
            obs, reward, terminated, truncated, _info = self.env.step(action)

            depth_after = self.env.state.depth()
            gates_after = sum(v for k, v in self.env.state.count_ops().items() if k != "barrier")
            step_records.append({
                "step": len(step_records),
                "action": action_item.name,
                "depth_before": depth_before,
                "gates_before": gates_before,
                "depth_after": depth_after,
                "gates_after": gates_after,
                "reward": float(reward),
            })

        log_path = get_path_trained_model() / "actions_maskable_ppo.txt"
        with log_path.open("a", encoding="utf-8") as f:
            circuit_name = self.env.filename or str(qc)
            f.write(f"circuit={circuit_name}\n")
            for r in step_records:
                f.write(
                    f"  step={r['step']} action={r['action']}"
                    f" depth={r['depth_before']}->{r['depth_after']}"
                    f" gates={r['gates_before']}->{r['gates_after']}"
                    f" reward={r['reward']:.6f}\n"
                )
            f.write("\n")

        if not self.env.error_occurred:
            return self.env.export_circuit(), used_compilation_passes

        msg = "Error occurred during compilation."
        raise RuntimeError(msg)

    def _compile_with_gnn(self, qc: QuantumCircuit | str) -> tuple[QuantumCircuit, list[str]]:
        """Compiles the quantum circuit using a trained GNN-based PPO model.

        Arguments:
            qc: The quantum circuit to be compiled. It can be a Quantum Circuit object or a string for the path to a qasm file.

        Returns:
            A tuple containing the compiled quantum circuit and a list of the compilation passes used.

        """
        obs, _ = self.env.reset(qc, seed=0)

        policy = self._load_gnn_model(obs.x.shape[1])  # ty: ignore[unresolved-attribute]
        torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        policy = policy.to(torch_device)
        policy.eval()

        used_passes: list[str] = []
        terminated = False
        truncated = False

        with torch.no_grad():
            while not (terminated or truncated):
                # get the observable in the form of a PyG Batch (with batch size 1) and move to the same device as the model
                batch_obs = Batch.from_data_list([obs]).to(torch_device)  # ty: ignore[invalid-argument-type,unresolved-attribute]
                # evaluate the policy to get action logits and values
                logits, _val = policy(batch_obs)
                # mask the action not allowed by the environment
                mask = torch.tensor(self.env.action_masks(), dtype=torch.bool, device=torch_device)
                logits = logits.masked_fill(~mask.unsqueeze(0), float("-inf"))
                # sample an action from the masked distribution
                dist = Categorical(logits=logits.squeeze(0))
                action = int(dist.sample().item())
                action_name = self.env.action_set[action].name
                # append the action name to the list of used passes and step records for logging
                used_passes.append(action_name)
                # apply the action to the environment and get the new observation, reward, and done flags
                obs, _, terminated, truncated, _ = self.env.step(action)

        if not self.env.error_occurred:
            return self.env.state, used_passes

        msg = "Error occurred during compilation."
        raise RuntimeError(msg)

    def _load_gnn_model(self, node_feature_dim: int) -> SAGEActorCritic:
        """Load a saved GNN model checkpoint.

        Arguments:
            node_feature_dim: The dimension of the node features in the graph observations
        Returns:
            The loaded GNN model.
        """
        model_path = get_path_trained_model() / f"gnn_{self.figure_of_merit}_{self.device_name}.pt"
        if not model_path.is_file():
            msg = (
                f"The GNN RL model 'gnn_{self.figure_of_merit}_{self.device_name}' is not trained yet. "
                "Please train the model before using it."
            )
            raise FileNotFoundError(msg)
        # load the checkpoint and extract the state dict and config
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            cfg = checkpoint.get("config", {})
        else:
            state_dict = checkpoint
            cfg = {}
        # extract model hyperparameters from the config, using defaults if not present
        num_actions = self.env.action_space.n  # ty: ignore[unresolved-attribute]
        hidden_dim = int(cfg.get("hidden_dim", 128))
        num_conv_wo_resnet = int(cfg.get("num_conv_wo_resnet", 2))
        num_resnet_layers = int(cfg.get("num_resnet_layers", 5))
        dropout_p = float(cfg.get("dropout_p", 0.2))
        bidirectional = bool(cfg.get("bidirectional", True))
        global_feature_dim = int(cfg.get("global_feature_dim", 0))

        if "node_feature_dim" in cfg and int(cfg["node_feature_dim"]) != node_feature_dim:
            msg = f"node_feature_dim mismatch: checkpoint={cfg['node_feature_dim']} current={node_feature_dim}"
            raise RuntimeError(msg)
        if "num_actions" in cfg and int(cfg["num_actions"]) != num_actions:
            msg = f"num_actions mismatch: checkpoint={cfg['num_actions']} current={num_actions}"
            raise RuntimeError(msg)

        # create a new model instance with the same hyperparameters and load the state dict
        policy = create_gnn_policy(
            node_feature_dim=node_feature_dim,
            num_actions=num_actions,
            hidden_dim=hidden_dim,
            num_conv_wo_resnet=num_conv_wo_resnet,
            num_resnet_layers=num_resnet_layers,
            dropout_p=dropout_p,
            bidirectional=bidirectional,
            global_feature_dim=global_feature_dim,
        )
        # load the state dict into the model
        policy.load_state_dict(state_dict, strict=True)
        return policy

    def train_model(
        self,
        timesteps: int = 1000,
        verbose: int = 2,
        test: bool = False,
        callback: BaseCallback | None = None,
        checkpoint_directory: Path | None = None,
        checkpoint_frequency: int | None = None,
        resume_from: Path | None = None,
        **kwargs: object,
    ) -> MaskablePPO | None:
        """Trains a model for the given reward function and device.

        Arguments:
            timesteps: Target total training timesteps for MaskablePPO. Ignored in GNN mode.
            verbose: Verbosity level (MaskablePPO only). Defaults to 2.
            test: Use reduced hyperparameters for fast testing. Defaults to False.
            callback: Optional SB3 callback used during training (MaskablePPO only). Defaults to None.
            checkpoint_directory: Optional directory for intermediate checkpoints in both PPO and GNN mode.
            checkpoint_frequency: Optional checkpoint cadence in environment steps for both PPO and GNN mode.
            resume_from: Optional path to a previously saved checkpoint.
            **kwargs: Additional hyperparameters for GNN training:
                - iterations (int): Target total PPO iterations. Defaults to 1000.
                - steps (int): Steps per iteration. Defaults to 2048.
                - num_epochs (int): PPO update epochs. Defaults to 10.
                - minibatch_size (int): Minibatch size. Defaults to 64.
                - hidden_dim (int): GNN hidden dimension. Defaults to 128.
                - num_conv_wo_resnet (int): Non-residual conv layers. Defaults to 2.
                - num_resnet_layers (int): Residual conv layers. Defaults to 5.
                - dropout_p (float): Dropout probability. Defaults to 0.2.
                - bidirectional (bool): Bidirectional message passing. Defaults to True.
                - lr (float): Learning rate for actor/critic heads. Defaults to 3e-4.
                - gnn_lr (float): Learning rate for GNN encoder. Defaults to 1e-4.
        """
        if self.graph:
            self._train_gnn(
                test=test,
                checkpoint_directory=checkpoint_directory,
                checkpoint_frequency=checkpoint_frequency,
                resume_from=resume_from,
                **kwargs,
            )
            return None

        return self._train_maskable_ppo(
            timesteps=timesteps,
            verbose=verbose,
            test=test,
            callback=callback,
            resume_from=resume_from,
        )

    def _train_maskable_ppo(
        self,
        timesteps: int,
        verbose: int,
        test: bool,
        callback: BaseCallback | None = None,
        resume_from: Path | None = None,
    ) -> MaskablePPO:
        """Trains a MaskablePPO model.

        Arguments:
            timesteps: Target total training timesteps.
            verbose: Verbosity level for training.
            test: If True, uses reduced hyperparameters for fast testing.
            callback: Optional SB3 callback used during training.
            resume_from: Optional path to a previously saved PPO checkpoint.
        """
        if test:
            set_random_seed(0)
            n_steps = 32
            n_epochs = 2
            batch_size = 8
            progress_bar = False
        else:
            set_random_seed(0)
            n_steps = 2048
            n_epochs = 10
            batch_size = 64
            progress_bar = True

        logger.debug("Start training for: " + self.figure_of_merit + " on " + self.device_name)
        tensorboard_log = "./model_" + self.figure_of_merit + "_" + self.device_name
        if resume_from is None:
            model = MaskablePPO(
                MaskableMultiInputActorCriticPolicy,
                self.env,
                verbose=verbose,
                tensorboard_log=tensorboard_log,
                gamma=0.98,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
            )
            reset_num_timesteps = True
            remaining_timesteps = timesteps
        else:
            model = MaskablePPO.load(resume_from, env=self.env)
            model.verbose = verbose
            model.tensorboard_log = tensorboard_log
            reset_num_timesteps = False
            remaining_timesteps = max(0, timesteps - int(model.num_timesteps))

        if remaining_timesteps > 0:
            model.learn(
                total_timesteps=remaining_timesteps,
                progress_bar=progress_bar,
                callback=callback,
                reset_num_timesteps=reset_num_timesteps,
            )
        model.save(get_path_trained_model() / ("model_" + self.figure_of_merit + "_" + self.device_name))
        return model

    def _build_gnn_checkpoint_payload(
        self,
        *,
        policy: SAGEActorCritic,
        hidden_dim: int,
        num_conv_wo_resnet: int,
        num_resnet_layers: int,
        dropout_p: float,
        bidirectional: bool,
        node_feature_dim: int,
        num_iterations: int,
        steps_per_iteration: int,
        num_epochs: int,
        minibatch_size: int,
        lr: float,
        gnn_lr: float,
        completed_iterations: int,
        optimizer: torch.optim.Optimizer | None = None,
        shuffle_rng_state: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """Build the serialized checkpoint payload shared by final and intermediate GNN saves."""
        training_state: dict[str, object] = {
            "completed_iterations": completed_iterations,
            "completed_timesteps": completed_iterations * steps_per_iteration,
            "num_iterations": num_iterations,
            "steps_per_iteration": steps_per_iteration,
            "num_epochs": num_epochs,
            "minibatch_size": minibatch_size,
            "lr": lr,
            "gnn_lr": gnn_lr,
            "env_rng_state": self.env.rng.bit_generator.state,
            "torch_rng_state": torch.get_rng_state(),
        }
        if shuffle_rng_state is not None:
            training_state["shuffle_rng_state"] = shuffle_rng_state
        if torch.cuda.is_available():
            training_state["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()

        payload: dict[str, object] = {
            "state_dict": policy.state_dict(),
            "config": {
                "hidden_dim": hidden_dim,
                "num_conv_wo_resnet": num_conv_wo_resnet,
                "num_resnet_layers": num_resnet_layers,
                "dropout_p": dropout_p,
                "bidirectional": bidirectional,
                "node_feature_dim": node_feature_dim,
                "num_actions": self.env.action_space.n,  # ty: ignore[unresolved-attribute]
                "global_feature_dim": GLOBAL_FEATURE_DIM,
            },
            "training_state": training_state,
        }
        if optimizer is not None:
            payload["optimizer_state_dict"] = optimizer.state_dict()
        return payload

    def _train_gnn(
        self,
        test: bool = False,
        checkpoint_directory: Path | None = None,
        checkpoint_frequency: int | None = None,
        resume_from: Path | None = None,
        **kwargs: object,
    ) -> None:
        """Train the GNN model, optionally resuming from a previous checkpoint."""
        if test:
            kwargs.setdefault("iterations", 10)
            kwargs.setdefault("steps", 20)
            kwargs.setdefault("num_epochs", 1)
            kwargs.setdefault("minibatch_size", 32)
            kwargs.setdefault("hidden_dim", 128)
            kwargs.setdefault("num_conv_wo_resnet", 3)
            kwargs.setdefault("num_resnet_layers", 5)

        sample_obs, _ = self.env.reset()
        node_feature_dim = sample_obs.x.shape[1]  # ty: ignore[unresolved-attribute]
        num_actions = self.env.action_space.n  # ty: ignore[unresolved-attribute]

        checkpoint: dict[str, object] = {}
        checkpoint_config: dict[str, object] = {}
        training_state: dict[str, object] = {}
        optimizer_state_dict: dict[str, object] | None = None
        completed_iterations = 0
        shuffle_rng_state: dict[str, object] | None = None

        if resume_from is not None:
            checkpoint = torch.load(resume_from, map_location="cpu", weights_only=False)
            checkpoint_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
            training_state = checkpoint.get("training_state", {}) if isinstance(checkpoint, dict) else {}
            optimizer_state_raw = checkpoint.get("optimizer_state_dict") if isinstance(checkpoint, dict) else None
            optimizer_state_dict = optimizer_state_raw if isinstance(optimizer_state_raw, dict) else None
            completed_iterations = int(training_state.get("completed_iterations", 0))
            shuffle_rng_raw = training_state.get("shuffle_rng_state")
            shuffle_rng_state = shuffle_rng_raw if isinstance(shuffle_rng_raw, dict) else None

        hidden_dim = int(kwargs.get("hidden_dim", checkpoint_config.get("hidden_dim", 128)))  # type: ignore[arg-type]
        num_conv_wo_resnet = int(  # type: ignore[arg-type]
            kwargs.get("num_conv_wo_resnet", checkpoint_config.get("num_conv_wo_resnet", 3))
        )
        num_resnet_layers = int(  # type: ignore[arg-type]
            kwargs.get("num_resnet_layers", checkpoint_config.get("num_resnet_layers", 5))
        )
        dropout_p = float(kwargs.get("dropout_p", checkpoint_config.get("dropout_p", 0.2)))  # type: ignore[arg-type]
        bidirectional = bool(kwargs.get("bidirectional", checkpoint_config.get("bidirectional", True)))
        num_iterations = int(kwargs.get("iterations", training_state.get("num_iterations", 1000)))  # type: ignore[arg-type]
        steps_per_iteration = int(kwargs.get("steps", training_state.get("steps_per_iteration", 2048)))  # type: ignore[arg-type]
        num_epochs = int(kwargs.get("num_epochs", training_state.get("num_epochs", 10)))  # type: ignore[arg-type]
        minibatch_size = int(kwargs.get("minibatch_size", training_state.get("minibatch_size", 64)))  # type: ignore[arg-type]
        lr = float(kwargs.get("lr", training_state.get("lr", 3e-4)))  # type: ignore[arg-type]
        gnn_lr = float(kwargs.get("gnn_lr", training_state.get("gnn_lr", 1e-4)))  # type: ignore[arg-type]

        checkpoint_node_feature_dim = checkpoint_config.get("node_feature_dim")
        if checkpoint_node_feature_dim is not None and int(checkpoint_node_feature_dim) != node_feature_dim:
            msg = f"node_feature_dim mismatch: checkpoint={checkpoint_node_feature_dim} current={node_feature_dim}"
            raise RuntimeError(msg)
        checkpoint_num_actions = checkpoint_config.get("num_actions")
        if checkpoint_num_actions is not None and int(checkpoint_num_actions) != num_actions:
            msg = f"num_actions mismatch: checkpoint={checkpoint_num_actions} current={num_actions}"
            raise RuntimeError(msg)

        if resume_from is None:
            set_random_seed(0)

        policy = create_gnn_policy(
            node_feature_dim=node_feature_dim,
            num_actions=num_actions,
            hidden_dim=hidden_dim,
            num_conv_wo_resnet=num_conv_wo_resnet,
            num_resnet_layers=num_resnet_layers,
            dropout_p=dropout_p,
            bidirectional=bidirectional,
            global_feature_dim=GLOBAL_FEATURE_DIM,
        )

        state_dict = checkpoint.get("state_dict") if isinstance(checkpoint, dict) else None
        if isinstance(state_dict, dict):
            policy.load_state_dict(state_dict, strict=True)

        env_rng_state = training_state.get("env_rng_state")
        if isinstance(env_rng_state, dict):
            self.env.rng.bit_generator.state = env_rng_state
        torch_rng_state = training_state.get("torch_rng_state")
        if isinstance(torch_rng_state, torch.Tensor):
            torch.set_rng_state(torch_rng_state)
        cuda_rng_state_all = training_state.get("torch_cuda_rng_state_all")
        if torch.cuda.is_available() and isinstance(cuda_rng_state_all, list):
            torch.cuda.set_rng_state_all(cuda_rng_state_all)

        next_checkpoint_step: int | None = None
        latest_shuffle_rng_state = shuffle_rng_state
        if checkpoint_directory is not None and checkpoint_frequency is not None and checkpoint_frequency > 0:
            checkpoint_directory.mkdir(parents=True, exist_ok=True)
            completed_steps = completed_iterations * steps_per_iteration
            next_checkpoint_step = ((completed_steps // checkpoint_frequency) + 1) * checkpoint_frequency

        def _save_gnn_checkpoint(
            current_policy: SAGEActorCritic,
            current_optimizer: torch.optim.Optimizer,
            current_completed_iterations: int,
            current_shuffle_rng_state: dict[str, object] | None,
        ) -> None:
            nonlocal latest_shuffle_rng_state
            latest_shuffle_rng_state = current_shuffle_rng_state
            if checkpoint_directory is None or checkpoint_frequency is None or checkpoint_frequency <= 0:
                return

            nonlocal next_checkpoint_step
            assert next_checkpoint_step is not None
            completed_steps = current_completed_iterations * steps_per_iteration
            if completed_steps < next_checkpoint_step:
                return

            payload = self._build_gnn_checkpoint_payload(
                policy=current_policy,
                hidden_dim=hidden_dim,
                num_conv_wo_resnet=num_conv_wo_resnet,
                num_resnet_layers=num_resnet_layers,
                dropout_p=dropout_p,
                bidirectional=bidirectional,
                node_feature_dim=node_feature_dim,
                num_iterations=num_iterations,
                steps_per_iteration=steps_per_iteration,
                num_epochs=num_epochs,
                minibatch_size=minibatch_size,
                lr=lr,
                gnn_lr=gnn_lr,
                completed_iterations=current_completed_iterations,
                optimizer=current_optimizer,
                shuffle_rng_state=current_shuffle_rng_state,
            )
            checkpoint_path = checkpoint_directory / f"model_checkpoint_{completed_steps}_steps.pt"
            torch.save(payload, checkpoint_path)
            next_checkpoint_step = ((completed_steps // checkpoint_frequency) + 1) * checkpoint_frequency

        self.gnn_model, optimizer, completed_iterations = train_ppo_with_gnn(
            env=self.env,
            policy=policy,
            num_iterations=num_iterations,
            steps_per_iteration=steps_per_iteration,
            num_epochs=num_epochs,
            minibatch_size=minibatch_size,
            lr=lr,
            gnn_lr=gnn_lr,
            start_iteration=completed_iterations,
            optimizer_state_dict=optimizer_state_dict,
            shuffle_rng_state=shuffle_rng_state,
            checkpoint_callback=lambda current_policy, current_optimizer, current_completed_iterations, extra: (
                _save_gnn_checkpoint(
                    current_policy,
                    current_optimizer,
                    current_completed_iterations,
                    extra.get("shuffle_rng_state") if isinstance(extra.get("shuffle_rng_state"), dict) else None,
                )
            ),
        )

        model_path = get_path_trained_model() / f"gnn_{self.figure_of_merit}_{self.device_name}.pt"
        final_payload = self._build_gnn_checkpoint_payload(
            policy=self.gnn_model,
            hidden_dim=hidden_dim,
            num_conv_wo_resnet=num_conv_wo_resnet,
            num_resnet_layers=num_resnet_layers,
            dropout_p=dropout_p,
            bidirectional=bidirectional,
            node_feature_dim=node_feature_dim,
            num_iterations=num_iterations,
            steps_per_iteration=steps_per_iteration,
            num_epochs=num_epochs,
            minibatch_size=minibatch_size,
            lr=lr,
            gnn_lr=gnn_lr,
            completed_iterations=completed_iterations,
            optimizer=optimizer,
            shuffle_rng_state=latest_shuffle_rng_state,
        )
        torch.save(final_payload, model_path)


def load_model(model_name: str) -> MaskablePPO:
    """Loads a trained model from the trained model folder.

    Arguments:
        model_name: The name of the model to be loaded.

    Returns:
        The loaded model.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    path = get_path_trained_model()
    if Path(path / (model_name + ".zip")).is_file():
        return MaskablePPO.load(path / (model_name + ".zip"))

    error_msg = f"The RL model '{model_name}' is not trained yet. Please train the model before using it."
    raise FileNotFoundError(error_msg)


def rl_compile(
    qc: QuantumCircuit | str,
    device: Target | None,
    figure_of_merit: figure_of_merit | None = "expected_fidelity",
    predictor_singleton: Predictor | None = None,
    graph: bool = False,
) -> tuple[QuantumCircuit, list[str]]:
    """Compiles a given quantum circuit to a device optimizing for the given figure of merit.

    Arguments:
        qc: The quantum circuit to be compiled. If a string is given, it is assumed to be a path to a qasm file.
        device: The device to compile to.
        figure_of_merit: The figure of merit to be used for compilation. Defaults to "expected_fidelity".
        predictor_singleton: A predictor object that is used for compilation to reduce compilation time when compiling multiple quantum circuits. If None, a new predictor object is created. Defaults to None.
        graph: If True, uses the GNN-based policy. Ignored when ``predictor_singleton`` is provided. Defaults to False.

    Returns:
        A tuple containing the compiled quantum circuit and the compilation information. If compilation fails, False is returned.

    Raises:
        ValueError: If figure_of_merit or device is None and predictor_singleton is also None.
    """
    if predictor_singleton is None:
        if figure_of_merit is None:
            msg = "figure_of_merit must not be None if predictor_singleton is None."
            raise ValueError(msg)
        if device is None:
            msg = "device must not be None if predictor_singleton is None."
            raise ValueError(msg)
        predictor = Predictor(figure_of_merit=figure_of_merit, device=device, graph=graph)
    else:
        predictor = predictor_singleton

    return predictor.compile_as_predicted(qc)
