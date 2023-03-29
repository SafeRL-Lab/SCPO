import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Union
import numpy as np
import torch as th
from gym import spaces
from stable_baselines3.common.buffers import BaseBuffer
from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.type_aliases import SafeRolloutBufferSamples
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecNormalize



class SafeRolloutBuffer(BaseBuffer):
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.
    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        cost_space: spaces.Space,
        max_cost: np.ndarray,
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        gamma_s: float = 0.995,
        safety_k: float = 1.,
        reward_bias: float = 0.,
        n_envs: int = 1,
        augmented_state_cumulative_cost_zero: bool = True,
        lagrange_beta: float = 0.
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.gamma_s = gamma_s
        self.reward_bias = reward_bias
        self.safety_k = safety_k
        self.augmented_state_cumulative_cost_zero = augmented_state_cumulative_cost_zero
        self.max_cost = max_cost
        self.lagrange_beta = lagrange_beta
        self.safe_episodes_buffer = None
        self.observations, self.actions, self.rewards, self.advantages = None, None, None, None
        self.returns, self.episode_starts, self.values, self.log_probs = None, None, None, None
        self.generator_ready = False
        self.cost_space = cost_space
        self.cumulative_cost, self.cost = None, None
        self.safety_value = None
        self.safety_advantage, self.safety_return = None, None
        self.reset()

    def reset(self) -> None:
        self.safe_episodes_buffer = []
        self.cumulative_cost = np.zeros((self.buffer_size, self.n_envs) + self.cost_space.shape, dtype=np.float32)
        self.cost = np.zeros((self.buffer_size, self.n_envs) + self.cost_space.shape, dtype=np.float32)
        self.safety_advantage = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.safety_return = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        self.safety_value = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        super().reset()

    def compute_returns_safety_and_advantage(
            self, last_values: th.Tensor,
            last_safety_value : th.Tensor,
            last_costs: np.ndarray,
            dones: np.ndarray
        ) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
        where R is the sum of discounted reward with value bootstrap
        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        """
        # Convert to numpy
        last_values = last_values.clone().cpu().numpy().flatten()
        last_safety_value = last_safety_value.clone().cpu().numpy().flatten()

        last_gae_lam = 0
        episode_terminated = dones

        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
                episode_terminated = dones
                is_episode_safe = (last_costs + self.cumulative_cost[step]) <= self.max_cost
                is_episode_safe = np.all(is_episode_safe, axis=1)
                is_state_safe = np.all(self.cumulative_cost[step] <= self.max_cost, axis=1)
                next_safety_value = last_safety_value
                Q_safety_gamma = is_state_safe * 1.
            else:
                episode_terminated = np.logical_or(self.episode_starts[step+1], episode_terminated)
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
                next_safety_value = self.safety_value[step+1]
                is_next_state_safe = np.all(self.cumulative_cost[step+1] <= self.max_cost, axis=1)
                is_terminal_state = self.episode_starts[step+1]
                is_terminal_state_mask = is_terminal_state.astype(bool)
                is_episode_safe[is_terminal_state_mask] = (np.all(self.cumulative_cost[step] <= self.max_cost, axis=1))[is_terminal_state_mask]
                is_state_safe = np.all(self.cumulative_cost[step] <= self.max_cost, axis=1)

                Q_safety_gamma = ((1-self.gamma_s) * is_next_state_safe + self.gamma_s * Q_safety_gamma) * (1-is_terminal_state) + is_terminal_state * is_state_safe
            Q_safety_gamma_1 = np.where(np.logical_or(episode_terminated, ~is_episode_safe), is_episode_safe.astype(np.float32), is_state_safe)
            safety_advantage = (0.5 * (next_safety_value+Q_safety_gamma) - self.safety_value[step]) * is_state_safe * next_non_terminal
            # safety_advantage = (V_safety_gamma - self.safety_value[step]) * is_state_safe * next_non_terminal
            self.safety_return[step] = Q_safety_gamma

            Q_hat = (0.5 * (self.safety_value[step] + Q_safety_gamma))**self.safety_k
            self.rewards[step] = (self.rewards[step] + self.reward_bias) * Q_hat + self.lagrange_beta * (Q_hat-1) * self.cost[step].sum(axis=1)
            # self.rewards[step] = (self.rewards[step] + self.reward_bias) * Q_hat + 3 * (Q_hat-1) * self.cost[step].sum(axis=1)
            # self.rewards[step] = (self.rewards[step] + self.reward_bias) * Q_hat

            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam

            self.safety_advantage[step] = safety_advantage
            self.advantages[step] = last_gae_lam
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values

    # def add_cumulative_cost_to_state(self):
    #     if self.augmented_state_cumulative_cost_zero:
    #         self.observations = np.concatenate((self.observations, th.zeros(self.observations.shape[0:2] + (1, ))), axis=2)
    #     else:
    #         self.observations = np.concatenate((self.observations, np.clip(self.cumulative_cost/self.max_cost, None, 1.)), axis=2)

    def augment_state(self, state: np.ndarray, cumulative_cost: np.ndarray):
        if self.augmented_state_cumulative_cost_zero:
            return th.cat((th.FloatTensor(state), th.FloatTensor(np.zeros_like(cumulative_cost))), dim=1)
        return th.cat((th.FloatTensor(state), th.FloatTensor(np.clip(cumulative_cost/self.max_cost, None, 1.))), dim=1)

    def get_safe_episode_mean(self):
        return np.mean(self.safe_episodes_buffer)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        cost: np.array,
        cumulative_cost: np.array,
        safety_value: np.array,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
        dones: np.ndarray
    ) -> None:
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs,) + self.obs_shape)

        # Same reshape, for actions
        action = action.reshape((self.n_envs, self.action_dim))
        self.observations[self.pos] = np.array(obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.episode_starts[self.pos] = np.array(episode_start).copy()
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.safety_value[self.pos] = safety_value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.cumulative_cost[self.pos] = cumulative_cost.copy()
        self.cost[self.pos] = cost

        if np.any(dones):
            self.safe_episodes_buffer.extend(np.all(self.cumulative_cost[self.pos][dones] <= self.max_cost, axis=1).tolist())

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size: Optional[int] = None) -> Generator[SafeRolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:

            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "safety_return",
                "safety_advantage",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx: start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> SafeRolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.safety_return[batch_inds].flatten(),
            self.safety_advantage[batch_inds].flatten(),
        )
        return SafeRolloutBufferSamples(*tuple(map(self.to_torch, data)))

