# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""GNN-based PPO implementation for training a RL compilation predictor on variable-size circuit graphs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch_geometric.data import Batch

from mqt.predictor.rl.gnn import SAGEActorCritic

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch_geometric.data import Data

    from mqt.predictor.rl.predictorenv import PredictorEnv


@dataclass
class RolloutBuffer:
    """Collect data during rollout and convert to tensors for PPO update. Data collected in lists during rollout, then converted to tensors in finalize() for efficient batch processing during PPO updates.

    Object saved:
        - graphs: list of PyG Data objects (the circuit graph at each step)
        - actions_list: list of ints (action indices taken)
        - log_probs_list: list of floats (log probabilities of taken actions)
        - values_list: list of floats (critic value estimates at each step)
        - rewards_list: list of floats (rewards received at each step)
        - dones_list: list of bools (whether episode ended at each step)
        - masks_list: list of np.ndarray of bools (action masks at each step).
    """

    graphs: list[Data] = field(default_factory=list)
    actions_list: list[int] = field(default_factory=list)
    log_probs_list: list[float] = field(default_factory=list)
    values_list: list[float] = field(default_factory=list)
    rewards_list: list[float] = field(default_factory=list)
    dones_list: list[bool] = field(default_factory=list)
    masks_list: list[np.ndarray] = field(default_factory=list)

    # Finalized tensors for PPO update (initialized in finalize())
    actions: torch.Tensor = field(init=False)
    log_probs: torch.Tensor = field(init=False)
    values: torch.Tensor = field(init=False)
    rewards: torch.Tensor = field(init=False)
    dones: torch.Tensor = field(init=False)
    masks: torch.Tensor = field(init=False)

    def clear(self) -> None:
        """Clear all collected data to start a new rollout."""
        self.graphs = []
        self.actions_list = []
        self.log_probs_list = []
        self.values_list = []
        self.rewards_list = []
        self.dones_list = []
        self.masks_list = []

    def add(
        self,
        graph: Data,
        action: int,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
        mask: np.ndarray | list[bool],
    ) -> None:
        """Add a step to the rollout buffer.

        Args:
            graph: PyG Data object representing the circuit graph at this step
            action: int, index of the action taken
            log_prob: float, log probability of the taken action under the current policy
            value: float, critic value estimate for the current state
            reward: float, reward received after taking the action
            done: bool, whether the episode ended after this step
            mask: np.ndarray or list of bools, action mask for this step (True for valid actions, False for invalid).
        """
        self.graphs.append(graph)
        self.actions_list.append(action)
        self.log_probs_list.append(log_prob)
        self.values_list.append(value)
        self.rewards_list.append(reward)
        self.dones_list.append(done)
        self.masks_list.append(np.asarray(mask, dtype=np.bool_))

    def finalize(self, device: str) -> None:
        """Convert collected lists to tensors for PPO update.

        Args:
            device: 'cuda' or 'cpu' where the tensors should be located.
        """
        self.actions = torch.tensor(self.actions_list, dtype=torch.long, device=device)
        self.log_probs = torch.tensor(self.log_probs_list, dtype=torch.float32, device=device)
        self.values = torch.tensor(self.values_list, dtype=torch.float32, device=device)
        self.rewards = torch.tensor(self.rewards_list, dtype=torch.float32, device=device)
        self.dones = torch.tensor(self.dones_list, dtype=torch.float32, device=device)
        self.masks = torch.tensor(np.stack(self.masks_list), device=device, dtype=torch.bool)


def compute_gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float = 0.98,
    lam: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalized Advantage Estimation (GAE) for the PPO update.

    Args:
        rewards: shape (T,)
        dones: shape (T,)
        values: shape (T,)
        last_value: scalar tensor — V(s_T) for bootstrapping
        gamma: discount factor
        lam: GAE lambda.

    Returns:
        returns: shape (T,)
        advantages: shape (T,)
    """
    # Takes the number of timesteps
    timesteps = rewards.size(0)
    # Initialize tensors for advantages and returns
    advantages = torch.zeros(timesteps, device=rewards.device, dtype=torch.float32)

    gae = 0.0
    # for each timestep t, starting from the end of the trajectory and moving backwards
    for t in reversed(range(timesteps)):
        next_value = last_value if t == timesteps - 1 else values[t + 1]
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * not_done - values[t]
        gae = delta + gamma * lam * not_done * gae
        advantages[t] = gae

    returns = advantages + values
    return returns, advantages


def collect_rollout(
    env: PredictorEnv,
    policy: SAGEActorCritic,
    steps: int,
    device: str,
) -> tuple[RolloutBuffer, torch.Tensor, list[float]]:
    """Collect a rollout trajectory of given number of steps using the current policy.

    Args:
        env: Environment returning PyG Data from reset()/step()
        policy: GNN actor-critic
        steps: Number of steps to collect
        device: 'cuda' or 'cpu'

    Returns:
        buffer: populated RolloutBuffer
        last_value: bootstrap value for GAE
        episode_rewards: total reward accumulated per completed episode
    """
    buffer = RolloutBuffer()
    episode_rewards: list[float] = []
    episode_return: float = 0.0

    obs, _ = env.reset()

    policy.eval()
    with torch.no_grad():
        for _ in range(steps):
            # Convert the current observation (PyG Data) to a batch of size 1 and move to device
            batch = Batch.from_data_list([obs]).to(device)  # ty: ignore[invalid-argument-type,unresolved-attribute]
            # Get action logits and value estimate from the policy
            logits, value = policy(batch)
            # get the mask of valid action from the environment
            mask_cpu = env.action_masks()
            if not any(mask_cpu):
                # If no valid actions, we can't do anything; treat as episode end and reset.
                obs, _ = env.reset()
                episode_return = 0.0
                continue

            mask = torch.as_tensor(mask_cpu, device=device, dtype=torch.bool)
            # Mask invalid actions by setting their logits to -inf, so they have zero probability after softmax.
            # Invert mask because True means valid, but we want to set invalid (False) logits to -inf.
            logits_masked = logits.masked_fill(~mask.unsqueeze(0), float("-inf"))
            # Create a categorical distribution over the masked logits.
            dist = Categorical(logits=logits_masked)
            # Sample an action from the masked distribution and get its log probability.
            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_obs, reward, terminated, truncated, _info = env.step(int(action.item()))
            # Consider episode done if either terminated or truncated is True.
            done = terminated or truncated
            # Update episode return and add transition to buffer.
            episode_return += float(reward)

            buffer.add(
                graph=obs,  # ty: ignore[invalid-argument-type]
                action=int(action.item()),
                log_prob=float(log_prob.item()),
                value=float(value.squeeze(-1).item()),
                reward=float(reward),
                done=bool(done),
                mask=mask_cpu,
            )

            if done:
                episode_rewards.append(episode_return)
                episode_return = 0.0
                obs, _ = env.reset()
            else:
                obs = next_obs

        # Bootstrap value for the current (possibly mid-episode) obs.
        last_batch = Batch.from_data_list([obs]).to(device)  # ty: ignore[invalid-argument-type,unresolved-attribute]
        _, last_value = policy(last_batch)
        last_value = last_value.squeeze(-1).squeeze(0)

    return buffer, last_value, episode_rewards


def ppo_update(
    policy: SAGEActorCritic,
    optimizer: torch.optim.Optimizer,
    buffer: RolloutBuffer,
    last_value: torch.Tensor,
    device: str,
    gamma: float = 0.98,
    lam: float = 0.95,
    clip_range: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    value_clip_range: float = 0.2,
    max_grad_norm: float = 0.5,
    epochs: int = 10,
    minibatch_size: int = 64,
    target_kl: float | None = 0.01,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """PPO update with variable-size graph minibatching.

    Uses value clipping, KL early stopping, and clipped surrogate objective.

    Returns:
        metrics: dict with keys 'mean_kl', 'policy_loss', 'value_loss', 'entropy'
    """
    # If no data was collected, return zero metrics to avoid errors in PPO update.
    if len(buffer.graphs) == 0:
        return {"mean_kl": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    # Convert collected data to tensors for PPO update.
    buffer.finalize(device=device)
    # Compute GAE advantages and returns.
    returns, advantages = compute_gae(
        rewards=buffer.rewards,
        dones=buffer.dones,
        values=buffer.values,
        last_value=last_value,
        gamma=gamma,
        lam=lam,
    )
    # Calculate the standard deviation of advantages for normalization, and handle the case where it's zero or non-finite to avoid division errors.
    adv_std = advantages.std(unbiased=False)
    if not torch.isfinite(adv_std) or adv_std < 1e-8:
        adv_std = torch.tensor(1.0, device=advantages.device)
    # Normalize advantages to have mean 0 and std 1 for more stable PPO updates.
    advantages = (advantages - advantages.mean()) / adv_std

    graphs = buffer.graphs
    actions = buffer.actions
    old_log_probs = buffer.log_probs
    old_values = buffer.values

    num_circuits = len(graphs)
    # Clamp minibatch size to the number of samples so we don't get empty batches.
    effective_mb = min(minibatch_size, num_circuits)
    indices = np.arange(num_circuits)
    shuffle_rng = rng or np.random.default_rng()

    all_kl: list[float] = []
    all_policy_loss: list[float] = []
    all_value_loss: list[float] = []
    all_entropy: list[float] = []
    # train policy for the specified number of epochs, shuffling and creating minibatches each epoch
    policy.train()
    for _epoch in range(epochs):
        shuffle_rng.shuffle(indices)

        for start in range(0, num_circuits, effective_mb):
            mb_idx = indices[start : start + effective_mb]

            mb_graphs = [graphs[i] for i in mb_idx]
            mb_batch = Batch.from_data_list(mb_graphs).to(device)  # ty: ignore[unresolved-attribute]

            mb_actions = actions[mb_idx]
            mb_old_logp = old_log_probs[mb_idx]
            mb_old_values = old_values[mb_idx]
            mb_returns = returns[mb_idx]
            mb_adv = advantages[mb_idx]

            logits, new_values = policy(mb_batch)
            mb_masks = buffer.masks[mb_idx]
            logits = logits.masked_fill(~mb_masks, float("-inf"))
            dist = Categorical(logits=logits)
            new_logp = dist.log_prob(mb_actions)
            entropy = dist.entropy().mean()
            new_values = new_values.squeeze(-1)
            # Compute the PPO clipped surrogate objective for the policy loss.
            logp_diff = torch.clamp(new_logp - mb_old_logp, -20.0, 20.0)
            ratio = torch.exp(logp_diff)
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            value_pred_clipped = mb_old_values + torch.clamp(
                new_values - mb_old_values, -value_clip_range, value_clip_range
            )
            vloss1 = (new_values - mb_returns).pow(2)
            vloss2 = (value_pred_clipped - mb_returns).pow(2)
            value_loss = torch.max(vloss1, vloss2).mean()

            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                kl = (mb_old_logp - new_logp).mean().item()
                all_kl.append(kl)
                all_policy_loss.append(policy_loss.item())
                all_value_loss.append(value_loss.item())
                all_entropy.append(entropy.item())

        mean_kl = float(np.mean(all_kl)) if all_kl else 0.0
        if target_kl is not None and mean_kl > target_kl:
            break

    return {
        "mean_kl": float(np.mean(all_kl)) if all_kl else 0.0,
        "policy_loss": float(np.mean(all_policy_loss)) if all_policy_loss else 0.0,
        "value_loss": float(np.mean(all_value_loss)) if all_value_loss else 0.0,
        "entropy": float(np.mean(all_entropy)) if all_entropy else 0.0,
    }


def train_ppo_with_gnn(
    env: PredictorEnv,
    policy: SAGEActorCritic,
    num_iterations: int = 1000,
    steps_per_iteration: int = 2048,
    num_epochs: int = 10,
    minibatch_size: int = 64,
    lr: float = 3e-4,
    gnn_lr: float = 1e-4,
    gamma: float = 0.98,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    value_clip_range: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    target_kl: float | None = 0.01,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    start_iteration: int = 0,
    optimizer_state_dict: dict[str, Any] | None = None,
    shuffle_rng_state: dict[str, Any] | None = None,
    checkpoint_callback: Callable[[SAGEActorCritic, torch.optim.Optimizer, int, dict[str, Any]], None] | None = None,
) -> tuple[SAGEActorCritic, torch.optim.Optimizer, int]:
    """Train a GNN-PPO agent on variable-size circuit graphs.

    Args:
        env: Environment with graph observations (graph=True).
        policy: SAGEActorCritic model.
        num_iterations: Number of PPO iterations.
        steps_per_iteration: Environment steps per rollout.
        num_epochs: PPO update epochs per rollout.
        minibatch_size: Minibatch size for PPO updates.
        lr: Learning rate for actor/critic heads and trunk.
        gnn_lr: Learning rate for GNN encoder layers.
        gamma: Discount factor (default 0.98, matching non-GNN MaskablePPO).
        gae_lambda: GAE lambda.
        clip_range: PPO clip range.
        value_clip_range: Value function clip range.
        value_coef: Value loss coefficient.
        entropy_coef: Entropy bonus coefficient.
        max_grad_norm: Gradient clipping norm.
        target_kl: KL early stopping threshold (None to disable).
        device: Torch device.
        verbose: Print progress every 10 iterations.
        log_file: Optional path to a CSV file where per-iteration training metrics
                  are appended. Each row contains: iteration, mean_ep_reward,
                  std_ep_reward, num_episodes, policy_loss, value_loss, entropy, mean_kl.

    Returns:
        Trained policy.
    """
    policy = policy.to(device)

    optimizer = torch.optim.Adam([
        {"params": policy.encoder.convs.parameters(), "lr": gnn_lr},
        {"params": policy.encoder.norms.parameters(), "lr": gnn_lr},
        {"params": policy.trunk.parameters(), "lr": lr},
        {"params": policy.actor.parameters(), "lr": lr},
        {"params": policy.critic.parameters(), "lr": lr},
    ])
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)

    shuffle_rng = np.random.default_rng()
    if shuffle_rng_state is not None:
        shuffle_rng.bit_generator.state = shuffle_rng_state

    completed_iterations = start_iteration
    for iteration in range(start_iteration, num_iterations):
        buffer, last_value, _ep_rewards = collect_rollout(
            env=env,
            policy=policy,
            steps=steps_per_iteration,
            device=device,
        )
        # Perform PPO update with the collected rollout data and get training metrics.
        ppo_update(
            policy=policy,
            optimizer=optimizer,
            buffer=buffer,
            last_value=last_value,
            device=device,
            gamma=gamma,
            lam=gae_lambda,
            clip_range=clip_range,
            value_clip_range=value_clip_range,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            max_grad_norm=max_grad_norm,
            epochs=num_epochs,
            minibatch_size=minibatch_size,
            target_kl=target_kl,
            rng=shuffle_rng,
        )
        completed_iterations = iteration + 1
        if checkpoint_callback is not None:
            checkpoint_callback(
                policy,
                optimizer,
                completed_iterations,
                {"shuffle_rng_state": shuffle_rng.bit_generator.state},
            )

    return policy, optimizer, completed_iterations


def create_gnn_policy(
    node_feature_dim: int,
    num_actions: int,
    hidden_dim: int = 128,
    num_conv_wo_resnet: int = 2,
    num_resnet_layers: int = 5,
    dropout_p: float = 0.2,
    bidirectional: bool = True,
    global_feature_dim: int = 0,
) -> SAGEActorCritic:
    """Factory function for creating an SAGEActorCritic policy.

    Args:
        node_feature_dim: Number of input node features.
        num_actions: Number of compilation pass actions (env.action_space.n).
        hidden_dim: Hidden dimension of the GNN layers.
        num_conv_wo_resnet: Number of convolutional layers without residual connection.
        num_resnet_layers: Number of residual convolutional layers.
        dropout_p: Dropout probability.
        bidirectional: Whether to use bidirectional message passing.
        global_feature_dim: Number of flat RL observation features concatenated
        to the graph embedding before the actor/critic heads (0 to disable).

    Returns:
        Initialized SAGEActorCritic.
    """
    return SAGEActorCritic(
        in_feats=node_feature_dim,
        hidden_dim=hidden_dim,
        num_conv_wo_resnet=num_conv_wo_resnet,
        num_resnet_layers=num_resnet_layers,
        num_actions=num_actions,
        dropout_p=dropout_p,
        bidirectional=bidirectional,
        global_feature_dim=global_feature_dim,
    )
