import gym
import pybullet as p
#import safety_gym
import bullet_safety_gym
import bullet_safety_gym.envs.tasks
from controls.keybord_control import KeyboardControl
from stable_baselines3.common.env_util import make_vec_env
import sys
# Parallel environments
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.safe_ppo.policies import SafeActorCriticPolicy
from stable_baselines3.safe_ppo.safe_ppo import SafePPO
import gym_safety
import numpy as np
import torch as th
from utils import env_info

#PPO_724 PPO_1010
model = SafePPO.load("benchmark/CartSafe-v0/seed_1/PPO_1/model.zip", device="cpu")
model.env_id = "CartSafe-v0"
model.cost_extractor = env_info[model.env_id]["cost_extractor"]
env = gym.make(model.env_id)

#not optiomal but respects bounds 1014


def reset_env():
    global obs
    global cumulative_cost
    global reward
    obs = env.reset()
    cumulative_cost = np.zeros(1)
    reward = 0


def toggle_model_use():
    global use_model
    use_model = not use_model
    print(f"using model: {use_model}")


def toggle_reset_on_done():
    global reset_on_done
    reset_on_done = not reset_on_done
    print(f"reset on done: {reset_on_done}")


try:
    env.render()

except Exception as e:
    pass

controller = KeyboardControl(model.env_id, {"r": reset_env, "m": toggle_model_use, "e": toggle_reset_on_done})

obs = env.reset()
cumulative_cost = np.zeros_like(model.max_cost)
zero_cost = np.zeros(1)
reward = 0
use_model = False
reset_on_done = True
done = False

while True:
    done = False
    # augmented_obs = np.append(obs, zero_cost, axis=0)
    augmented_obs = model.rollout_buffer.augment_state(np.expand_dims(obs, 0), np.expand_dims(cumulative_cost, 0))[0].numpy()
    if use_model:
        action, _states = model.predict(augmented_obs, deterministic=False)
    else:
        action = controller.get_action()

    if action is not None:
        obs, rewards, done, info = env.step(action)
        reward += rewards
        cumulative_cost = cumulative_cost + model.extract_costs([info]).squeeze()
        # print(model.extract_costs([info]))
        # action_device = th.from_numpy(action).to(model.device)
        # if info["cost"] != 0.0:
        #     print(info["cost"])
        if np.any(action != 0) or action is not None:
            # print(obs_device)
            obs_device = th.from_numpy(augmented_obs).to(model.device)
            safety = model.policy.predict_safety_values(obs_device.unsqueeze(0)).item()
            value = model.policy.predict_values(obs_device.unsqueeze(0)).item()
            # print((safety, value))
            # print(reward)
            pass
    if done and reset_on_done:
        print((reward, cumulative_cost.item()))
        reset_env()

    env.render()
