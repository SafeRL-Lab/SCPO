import abc
import json
import os
import warnings
from collections import deque
from copy import deepcopy
from itertools import chain
from typing import Any, Dict, Optional, Type, TypeVar, Union, List, Callable, Tuple

import gym
import numpy as np
import torch as th
from gym import spaces
from torch import nn
from torch.nn import functional as F

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn, obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.safe_ppo.buffer import SafeRolloutBuffer

PPOSelf = TypeVar("PPOSelf", bound="PPO")
x = None

class SafePPO3(OnPolicyAlgorithm):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)

    Paper: https://arxiv.org/abs/1707.06347
    Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param normalize_advantage: Whether to normalize or not the advantage
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        max_cost: np.ndarray = None,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        cost_extractor: Callable[[dict], list] = lambda info: [float("inf")],
        safety_gamma: float = 0.9,
        normalize_safety_advantage: bool = True,
        vf_safety_coef: float = 2,
        n_safety_epochs: int = 10,
        iterate_until_safe: bool = False,
    ):

        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )
        self.n_safety_epochs = n_safety_epochs
        self.vf_safety_coef = vf_safety_coef
        self.safety_gamma = safety_gamma
        self.ep_safety_buffer = deque(maxlen=100)
        self.ep_cumulative_cost_buffer = deque(maxlen=100)
        self.normalize_safety_advantage = normalize_safety_advantage
        self.iterate_until_safe = iterate_until_safe
        self.cost_space = spaces.Space(shape=(1,), dtype=np.float32)
        self.cost_extractor = cost_extractor
        self.max_cost = max_cost
        if self.max_cost is None:
            self.max_cost = np.ones(self.cost_space.shape) * float("inf")

        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        if normalize_advantage:
            assert (
                batch_size > 1
            ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert buffer_size > 1 or (
                not normalize_advantage
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        self.rollout_buffer = SafeRolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            self.cost_space,
            max_cost=self.max_cost,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            safety_gamma=self.safety_gamma
        )

        augmented_observation_space = deepcopy(self.observation_space)
        augmented_observation_space._shape = (self.observation_space.shape[0] + self.cost_space.shape[0], )
        self.policy = self.policy_class(  # pytype:disable=not-instantiable
            #todo handle more cases
            augmented_observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs  # pytype:disable=not-instantiable
        )

        self.policy = self.policy.to(self.device)

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)

        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses, safety_value_losses, safety_losses = [], [], [], []
        clip_fractions = []

        continue_training = True

        # train for n_epochs epochs

        for i in range(self.n_safety_epochs):

            # Do a complete pass on the rollout buffer
            safety_delta_buffer = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                self.policy.optimizer.zero_grad()

                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, safety_values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                safety_values = safety_values.flatten()
                values = values.flatten()
                safety_advantages = rollout_data.safety_advantage

                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_safety_advantage and len(safety_advantages) > 1:
                    safety_advantages = (safety_advantages - safety_advantages.mean()) / (safety_advantages.std() + 1e-8)

                clip_range = self.clip_range(self._current_progress_remaining)
                #ratio = 1 - th.exp(rollout_data.old_log_prob - log_prob)
                prob = th.clamp(th.exp(log_prob), 0.05, None)
                old_prob = th.clamp(th.exp(rollout_data.old_log_prob), 0.05, None)
                ratio = 1 - old_prob/prob
                safety_loss1 = safety_advantages * ratio
                safety_loss2 = safety_advantages * th.clamp(ratio, -clip_range, clip_range)
                safety_loss = -(th.min(safety_loss1, safety_loss2)).mean()

                safety_value_loss = F.mse_loss(rollout_data.safety_return, safety_values)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = safety_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + self.vf_safety_coef * safety_value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    if i == self.n_safety_epochs - 1:
                        safety_value_losses.append(safety_value_loss.item())

                    safety_delta = (ratio * rollout_data.safety_return).mean()
                    safety_delta_buffer.append(safety_delta.item())

                # Optimization step
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                # Logging
                # clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                # clip_fractions.append(clip_fraction)


        safey_delta_approx = np.mean(safety_delta_buffer)
        approx_kl_divs = []
        approx_dl_divs = []

        while safey_delta_approx > 0:
            approx_kl_divs = []
            approx_dl_divs = []
            safety_delta_buffer = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                self.policy.optimizer.zero_grad()

                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, safety_values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                safety_values = safety_values.flatten()
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                safety_advantages = rollout_data.safety_advantage

                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)


                clip_range = self.clip_range(self._current_progress_remaining)
                ratio = 1 - th.exp(rollout_data.old_log_prob - log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, -clip_range, clip_range)
                policy_loss = -(th.min(policy_loss_1, policy_loss_2)).mean()

                #safety_loss = safety_loss or th.FloatTensor([0]).to(self.device)

                # safety critic update

                safety_value_loss = F.mse_loss(rollout_data.safety_return, safety_values)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + self.vf_safety_coef * safety_value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                    safety_delta = (ratio * rollout_data.safety_value).mean()
                    safety_delta_buffer.append(safety_delta.item())

                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

                # Logging
                pg_losses.append(policy_loss.item())
                # clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                # clip_fractions.append(clip_fraction)
                safety_value_losses.append(safety_value_loss.item())
                approx_dl_divs.append((th.exp(log_prob.detach()) - 0.5).mean().item())

            if not continue_training:
                break

            safey_delta_approx = np.mean(safety_delta_buffer)


        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("path", self.logger.dir, exclude="tensorboard")
        self.logger.record("train/safety_delta", np.mean(safety_delta_buffer))

        self.logger.record("train/safety_value_mean", np.mean(self.rollout_buffer.safety_value.flatten()))
        self.logger.record("train/safety_value_std", np.std(self.rollout_buffer.safety_value.flatten()))

        self.logger.record("train/safety_value_loss_mean", np.mean(safety_value_losses))
        self.logger.record("train/safety_loss_mean", np.mean(safety_losses))

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/approx_dl", np.mean(approx_dl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def extract_costs(self, infos: List[dict]) -> np.ndarray:
        return np.array([self.cost_extractor(info) for info in infos])

    def save_hyper_params(self, file_name: Optional[str] = None):
        hyper = {
            "n_safety_epochs": self.n_safety_epochs,
            "n_epochs": self.n_epochs,
            "safety_gamma": self.safety_gamma,
            "normalize_safety_advantage": self.normalize_safety_advantage,
            "vf_safety_coef": self.vf_safety_coef,
            "normalize_advantage": self.normalize_advantage,
            "clip_range": self.clip_range(0),
            "learning_rate": self.learning_rate,
            "ent_coef": self.ent_coef,
            "num_envs": self.env.num_envs,
            "env_class_name": self.env.envs[0].unwrapped.spec.id,
            "gamma": self.gamma,
            "batch_size": self.batch_size,
        }

        file_name = file_name or os.path.basename(self.logger.dir) + ".json"
        file_path = os.path.join("hyper_parameters-PPO2", file_name)
        print(f"saving hyper parameters under {file_path}")
        with open(file_path, 'w') as f:
            json.dump(hyper, f)

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "PPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
        safe_hyper_param_under: Optional[str] = None
    ) -> PPOSelf:

        return super().learn(
            total_timesteps=total_timesteps,
            callback=SaveHyperCallBack(self),
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: SafeRolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()
        last_costs = np.zeros((self.n_envs,) + self.cost_space.shape, dtype=np.float32)
        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                augmented_obs_tensor = th.cat((th.FloatTensor(self._last_obs), th.FloatTensor(last_costs)), dim=1).to(self.device)
                #augmented_obs_tensor = obs_as_tensor(augmented_obs_tensor, self.device)

                actions, values, safe_values, log_probs = self.policy(augmented_obs_tensor)
            actions = actions.cpu().numpy()
            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)
            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    #todo check
                    #terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    terminal_obs = th.FloatTensor(infos[idx]["terminal_observation"])
                    #cumulative_cost_estimate = safe_values[idx].unsqueeze(0)
                    _terminal_obs = terminal_obs.unsqueeze(0).clone()
                    cumulative_cost = rollout_buffer.cumulative_cost[rollout_buffer.pos][idx] + self.extract_costs(infos)[idx]
                    cumulative_cost = th.FloatTensor(cumulative_cost).unsqueeze(0)
                    augmented_terminal_obs = th.cat((_terminal_obs, cumulative_cost), dim=1).to(self.device)
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(augmented_terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(self._last_obs, actions, rewards, last_costs, safe_values, self._last_episode_starts, values, log_probs, dones)
            self._last_obs = new_obs
            self._last_episode_starts = dones
            last_costs = self.extract_costs(infos)
            self._update_info_buffer(infos, dones, self.rollout_buffer.cumulative_cost[self.rollout_buffer.pos-1])


        with th.no_grad():
            # Compute value for the last timestep
            augmented_obs_tensor = th.cat((th.FloatTensor(new_obs), th.FloatTensor(last_costs)), dim=1).to(self.device)
            values = self.policy.predict_values(augmented_obs_tensor)
            safety_values = self.policy.predict_safety_values(augmented_obs_tensor)
        rollout_buffer.compute_returns_safety_and_advantage(last_values=values, last_safety_value=safety_values, last_costs=last_costs, dones=dones)
        rollout_buffer.add_cumulative_cost_to_state()
        callback.on_rollout_end()

        return True


    def _update_info_buffer(self, infos: List[Dict[str, Any]], dones: Optional[np.ndarray] = None,  cumulative_costs: Optional[np.ndarray] = None) -> None:
        """
        Retrieve reward, episode length, episode success and update the buffer
        if using Monitor wrapper or a GoalEnv.

        :param infos: List of additional information about the transition.
        :param dones: Termination signals
        """
        if dones is None:
            dones = np.array([False] * len(infos))
        last_costs = self.extract_costs(infos)

        for idx, info in enumerate(infos):
            maybe_ep_info = info.get("episode")
            maybe_is_success = info.get("is_success")
            if maybe_ep_info is not None:
                self.ep_info_buffer.extend([maybe_ep_info])
            if maybe_is_success is not None and dones[idx]:
                self.ep_success_buffer.append(maybe_is_success)
            if dones[idx] and cumulative_costs is not None:
                self.ep_safety_buffer.append(np.all(cumulative_costs[idx] + last_costs[idx] < self.max_cost))
                self.ep_cumulative_cost_buffer.append(cumulative_costs[idx, 0] + last_costs[idx, 0])


class SaveHyperCallBack(BaseCallback):
    def __init__(self, ppo_safe: SafePPO3):
        super(SaveHyperCallBack, self).__init__()
        self.ppo_safe = ppo_safe

    def _on_step(self, *args):
        return True

    def on_training_start(self, *args):
        self.ppo_safe.save_hyper_params()
