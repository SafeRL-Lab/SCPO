import json

import tensorflow as tf
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from os import walk, path
from dataclasses import dataclass
from utils import benchmark_env_ids


def load_json(file_path):
    with open(file_path) as json_file:
        return json.load(json_file)


def load_tb_data(tb_file_path):
    event_acc = EventAccumulator(tb_file_path)
    event_acc.Reload()
    w_times, _, cost_mean = zip(*event_acc.Scalars('rollout/cumulative_cost_mean'))
    _, _, cost_std = zip(*event_acc.Scalars('rollout/cumulative_cost_std'))
    _, _, return_mean = zip(*event_acc.Scalars('rollout/ep_rew_mean'))
    _, _, return_std = zip(*event_acc.Scalars('rollout/ep_rew_std'))
    _, _, safe_episode_mean = zip(*event_acc.Scalars('rollout/safe_episode_mean'))
    return {
        "cost_mean": list(cost_mean),
        "cost_std": list(cost_std),
        "return_mean": list(return_mean),
        "return_std": list(return_std),
        "safe_episode_mean": list(safe_episode_mean),
        "w_times": list(w_times)
    }


def get_tb_file_name(dirpath):
    for (dirpath, dirnames, filenames) in walk(dirpath):
        for filename in filenames:
            if "event.out" in filename:
                return filename
    return None


benchmark_path = ""
experiments = []
for (dirpath, dirnames, filenames) in walk(benchmark_path):
        if "PPO_1" in path.basename(dirpath):
            try:
                tb_path = path.join(dirpath, get_tb_file_name(dirpath))
                config = load_json(path.join(dirpath, "config.json"))
                seed = config["seed"]
                tb_data = load_tb_data(tb_path)
                env_id = config["env_id"]
                algo = config["alg"]
                experiment = {
                    "env_id": env_id,
                    "algorithm": "safe_ppo",
                    "config": config,
                    "tb_data": tb_data
                }
                experiments.append(experiment)
            except Exception as e:
                print(f"failed loading {dirpath}. Error: {e}")

with open("/home/jaafar/Documents/safe_rl/plots/my_experiments.json", "w") as write_file:
    json.dump(experiments, write_file, indent=4)