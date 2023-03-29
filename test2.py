import os
import signal
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import sys
# Parallel environments
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.safe_ppo.policies import SafeActorCriticPolicy
from stable_baselines3.safe_ppo.safe_ppo import SafePPO
import gym_safety
import numpy as np

m = SafePPO.load("PPO_71")

max_cost = np.array([0.5], dtype=np.float32)


def cost_extractor(info: dict):
    return list(info["constraint_costs"])


class BasicWrapper(gym.Wrapper):
    def __init__(self, env, cost_extractor, max_cost: np.ndarray):
        super().__init__(env)
        self.cost_extractor = cost_extractor
        self.env = env
        self.max_cost = max_cost

    def step(self, action):
        next_state, reward, done, info = super().step(action)
        cost = cost_extractor(info)
        if np.any(np.array(cost) > self.max_cost):
            done = True
        return next_state, reward, done, info



CartSafe = lambda: BasicWrapper(gym.make('CartSafe-v0'), cost_extractor, max_cost)

env = make_vec_env("CartSafe-v0", n_envs=4)

obs = env.reset()

while True:
    actions = [env.action_space.sample() for _ in range(env.num_envs)]
    obs, rewards, dones, info = env.step(np.array(actions))
    env.render()

