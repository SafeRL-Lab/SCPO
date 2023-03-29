import os
import signal
import traceback
#import safety_gym
import gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv, SubprocVecEnv
from stable_baselines3.safe_ppo.policies import SafeActorCriticPolicy
from stable_baselines3.safe_ppo.safe_ppo import SafePPO, CombinedLoss
import gym_safety
import numpy as np
import bullet_safety_gym
from utils import env_info, EarlyTermination, best_hyper

if __name__ == "__main__":

    def signal_handler(signal, frame):
        model.save_in_logger_dir()
        exit()

    signal.signal(signal.SIGINT, signal_handler)

    e = [
        "LunarLander-v2", "CartPole-v1", "CartSafe-v0", "GridNav-v0",
        "SafetyBallCircle-v0", "SafetyBallReach-v0", "SafetyBallGather-v0", "SafetyBallRun-v0",
        "SafetyCarReach-v0",  "SafetyCarCircle-v0", "SafetyCarGather-v0",
        "Safexp-PointGoal1-v0",
        "Safexp-PointButton1-v0"
    ]

    env_name = "SafetyCarCircle-v0"
    env = make_vec_env(env_name, n_envs=4, vec_env_cls=SubprocVecEnv)
    seed = 0
    # del best_hyper[env_name]["num_envs"]
    #
    # model = SafePPO(
    #     SafeActorCriticPolicy,
    #     env,
    #     cost_extractor=env_info[env_name]["cost_extractor"],
    #     max_cost=env_info[env_name]["max_costs"],
    #     verbose=1,
    #     # tensorboard_log=f"benchmark/{env_name}/seed_{seed}",
    #     tensorboard_log=f"experiments/{env_name}",
    #     seed=seed,
    #     **best_hyper[env_name],
    # )

    safety_k = 1.0
    model = SafePPO(
        SafeActorCriticPolicy,
        env,
        env_id=env_name,
        cost_extractor=env_info[env_name]["cost_extractor"],
        verbose=1,
        tensorboard_log=f"experiments/k_{str(safety_k)}runs/{env_name}",
        max_cost=env_info[env_name]["max_costs"],
        learning_rate=2e-4,
        clip_range=0.2,
        max_grad_norm=1.5,
        vf_safety_coef=0.5,
        normalize_safety_advantage=False,
        normalize_advantage=True,
        gamma_s=0.995,
        loss_type=CombinedLoss(safety_advantage_factor=0., ratio_estimator="PPO"),
        n_steps=2048*4,
        n_epochs=5,
        #policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256], s_vf=[256, 256])]),
        #policy_kwargs=dict(net_arch=[dict(pi=[128, 128], vf=[128, 128], s_vf=[128, 128])]),
        ent_coef=0.00,
        batch_size=64,
        augmented_state_cumulative_cost_zero=False,
        safety_k=safety_k,
        reward_bias=0.00,
        lagrange_beta=0.05,
    )

    # model = PPO(ActorCriticPolicy, env, tensorboard_log="v4", verbose=1, learning_rate=3e-4, clip_range=0.2, n_steps=2048, n_epochs=5)
    #model.set_parameters("PPO_5")

    try:
        model.learn(total_timesteps=2500000*5, reset_num_timesteps=True)
    except Exception:
        traceback.print_exc()

    model.save_in_logger_dir()
