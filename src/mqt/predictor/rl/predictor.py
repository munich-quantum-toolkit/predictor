"""This module contains the Predictor class, which is used to predict the most suitable compilation pass sequence for a given quantum circuit."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback


from mqt.predictor import reward, rl

if TYPE_CHECKING:
    from qiskit import QuantumCircuit

logger = logging.getLogger("mqt-predictor")
PATH_LENGTH = 260


class OffsetCheckpointCallback(BaseCallback):
    def __init__(self, save_freq, save_path, name_prefix, offset=0, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.offset = offset

    def _on_step(self) -> bool:
        total_steps = self.num_timesteps + self.offset
        if total_steps % self.save_freq == 0:
            path = f"{self.save_path}/{self.name_prefix}_{total_steps}_steps.zip"
            self.model.save(path)
            if self.verbose > 0:
                print(f"✅ Saved checkpoint: {path}")
        return True


class Predictor:
    """The Predictor class is used to train a reinforcement learning model for a given figure of merit and device such that it acts as a compiler."""

    def __init__(
        self, figure_of_merit: reward.figure_of_merit, device_name: str, logger_level: int = logging.INFO
    ) -> None:
        """Initializes the Predictor object."""
        logger.setLevel(logger_level)

        self.env = rl.PredictorEnv(reward_function=figure_of_merit, device_name=device_name)
        self.device_name = device_name
        self.figure_of_merit = figure_of_merit

    def compile_as_predicted(
        self,
        qc: QuantumCircuit,
    ) -> tuple[QuantumCircuit, list[str]]:
        """Compiles a given quantum circuit such that the given figure of merit is maximized by using the respectively trained optimized compiler.

        Arguments:
            qc: The quantum circuit to be compiled or the path to a qasm file containing the quantum circuit.

        Returns:
            A tuple containing the compiled quantum circuit and the compilation information. If compilation fails, False is returned.
        """
        trained_rl_model = rl.helper.load_model("model_" + self.figure_of_merit + "_" + self.device_name)

        obs, _ = self.env.reset(qc, seed=0)

        used_compilation_passes = []
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action_masks = get_action_masks(self.env)
            action, _ = trained_rl_model.predict(obs, action_masks=action_masks)
            action = int(action)
            action_item = self.env.action_set[action]
            used_compilation_passes.append(action_item["name"])
            obs, _reward_val, terminated, truncated, _info = self.env.step(action)

        if not self.env.error_occurred:
            return self.env.state, used_compilation_passes

        msg = "Error occurred during compilation."
        raise RuntimeError(msg)

    def train_model(
        self,
        timesteps: int = 100000,
        model_name: str = "model",
        verbose: int = 2,
        test: bool = False,
        trained: int = 0,
    ) -> None:
        """Train or resume model training with offset checkpointing.

        Arguments:
            timesteps: Total training timesteps desired.
            model_name: Prefix for saved model and logs.
            verbose: Verbosity level for PPO.
            test: If True, uses tiny n_steps for quick test.
            trained: Number of timesteps already trained (for resuming).
        """
        n_steps = 10 if test else 2048
        progress_bar = not test

        log_dir = f"./{model_name}_{self.figure_of_merit}_{self.device_name}"
        ckpt_path = f"./checkpoints/{model_name}_{trained}_steps.zip"

        logger.debug(f"🔁 Checking for checkpoint: {ckpt_path}")

        if os.path.exists(ckpt_path):
            logger.info(f"📦 Loading checkpoint from {ckpt_path}")
            model = MaskablePPO.load(
                ckpt_path,
                env=self.env,
                tensorboard_log=log_dir,
                verbose=verbose,
                device="cuda",  # or "cpu" depending on your setup
            )
        else:
            logger.info(f"🆕 No checkpoint found, starting fresh training")
            model = MaskablePPO(
                MaskableMultiInputActorCriticPolicy,
                self.env,
                verbose=verbose,
                tensorboard_log=log_dir,
                gamma=0.98,
                n_steps=n_steps,
            )

        remaining = timesteps - trained

        callback = OffsetCheckpointCallback(
            save_freq=1000,
            save_path="./checkpoints",
            name_prefix=model_name,
            offset=trained,
            verbose=1,
        )

        tb_log_name = "ppo"
        new_logger = configure(folder=os.path.join(log_dir, tb_log_name), format_strings=["stdout", "tensorboard"])
        model.set_logger(new_logger)
        model.learn(
            total_timesteps=remaining,
            tb_log_name=tb_log_name,
            callback=callback,
            progress_bar=progress_bar,
        )

        model.save(
            rl.helper.get_path_trained_model() / (model_name + "_" + self.figure_of_merit + "_" + self.device_name)
        )
        logger.info("✅ Final model saved.")