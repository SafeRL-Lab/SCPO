import json

import tensorflow as tf
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from os import walk, path
from dataclasses import dataclass
import numpy as np


tasks = ["Circle", "Gather", "Reach", "Run"]
agents = ["Ball", "Car"]
algorithms = ["cpo", "lag-trpo", "pcpo", "pdo", "trpo"]
bullet_path = "/home/jaafar/Documents/Data-Bullet-Safety-Gym"


def load_tb_data(tb_file_path):
    event_acc = EventAccumulator(tb_file_path)
    event_acc.Reload()
    w_times, _, cost_mean = zip(*event_acc.Scalars('EpCosts/Mean'))
    _, _, cost_std = zip(*event_acc.Scalars('EpCosts/Std'))
    _, _, return_mean = zip(*event_acc.Scalars('EpRet/Mean'))
    _, _, return_std = zip(*event_acc.Scalars('EpRet/Std'))
    return {
        "cost_mean": list(cost_mean),
        "cost_std": list(cost_std),
        "return_mean": list(return_mean),
        "return_std": list(return_std),
        "w_times": list(w_times)
    }


def get_algo(dir_path):
    for algorithm in algorithms:
        if algorithm in dir_path:
            return algorithm
    raise "unknown algorithm"

@dataclass
class Experiment:
    seed: int
    env_id: str
    algorithm: str
    config: dict
    tb_data: dict


def get_tb_file_name(dirpath):
    for (dirpath, dirnames, filenames) in walk(dirpath):
        if "tb" in dirpath:
            return filenames[0]
    return None

def load_json(file_path):
    with open(file_path) as json_file:
        return json.load(json_file)


def is_target_experiment(dirpath):
    for agent in agents:
        for task in tasks:
            if agent + task in dirpath:
                return True
    return False

experiments = []
for (dirpath, dirnames, filenames) in walk(bullet_path):
        if "seed_" in path.basename(dirpath):
            try:
                if not is_target_experiment(dirpath):
                    continue
                tb_path = path.join(dirpath, "tb", get_tb_file_name(dirpath))
                config = load_json(path.join(dirpath, "config.json"))
                seed = config["seed"]
                tb_data = load_tb_data(tb_path)
                env_id = config["env_id"]
                algo = config["alg"]

                del config["logger_kwargs"]
                del config["seed"]
                experiments.append(Experiment(seed, env_id, algo, config, tb_data))
            except Exception as e:
                print(f"failed loading {dirpath}. Error: {e}")


experiments_dict = list(map(lambda e: e.__dict__, experiments))
with open("/home/jaafar/Documents/safe_rl/plots/pybullet_experiments.json", "w") as write_file:
    json.dump(experiments_dict, write_file, indent=4)

print("yo")