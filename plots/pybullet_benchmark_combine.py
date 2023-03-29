import json
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
from matplotlib import pyplot as plt
from utils import benchmark_env_ids, env_info


@dataclass
class CombinedExperimentData:
    index: int
    env_id: str
    algorithm: str
    data: dict


def load_json(file_path):
    with open(file_path) as json_file:
        return json.load(json_file)

experiments = load_json("pybullet_experiments.json")
experiment_map = defaultdict(list)


def plot_best(best_results, env_id):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    for algorithm, experiment in best_results[env_id].items():
        ax1.plot(experiment.data["return_mean"], label=f"{experiment.algorithm}")
        ax3.plot(experiment.data["cost_mean"], label=f"{experiment.algorithm}")
        ax2.plot(experiment.data["return_std"], label=f"{experiment.algorithm}")
        ax4.plot(experiment.data["cost_std"], label=f"{experiment.algorithm}")

    ax1.set_xlabel("return mean")
    ax3.set_xlabel("cost mean")
    ax2.set_xlabel("return std")
    ax4.set_xlabel("cost std")
    plt.legend()
    plt.show()



def plot(env_id, algorithm, filter=False, index=None):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    for experiment in combined_experiments:
        if env_id in experiment.env_id and experiment.algorithm == algorithm:
            if index is not None and index != experiment.index:
                continue
            approx_cost_mean = np.mean(experiment.data["cost_mean"][-100:])
            if filter and approx_cost_mean > env_info[env_id]["max_costs"][0]:
                continue
            ax1.plot(experiment.data["return_mean"], label=f"{experiment.index}")
            ax1.set_xlabel("return mean")
            ax3.plot(experiment.data["cost_mean"], label=f"{experiment.index}")
            ax3.set_xlabel("cost mean")
            ax2.plot(experiment.data["return_std"], label=f"{experiment.index}")
            ax2.set_xlabel("return std")

            ax4.plot(experiment.data["cost_std"], label=f"{experiment.index}")
            ax4.set_xlabel("cost std")

    plt.legend()
    plt.show()


def combine(tb_datas):
    result = {}
    for key in tb_datas[0].keys():
        stacked_arrays = []
        for tb_data in tb_datas:
            stacked_arrays.append(tb_data[key])
        result[key] = np.mean(stacked_arrays, axis=0).tolist()
    return result


for experiment in experiments:
    config_str = experiment["config"].__str__()
    env_id = experiment["env_id"]
    algorithm = experiment["algorithm"]
    experiment_map[(env_id, algorithm, config_str)].append(experiment["tb_data"])


algorithm_counter = defaultdict(lambda: 0)
combined_experiments = []
for key, tb_datas in experiment_map.items():
    env_id, algorithm, config_str = key
    if "BallCircle" in env_id and algorithm == "cpo":
        print()
    index = algorithm_counter[(env_id, algorithm)]
    algorithm_counter[(env_id, algorithm)] = 1 + index
    combined_data = combine(tb_datas)
    combined_experiments.append(CombinedExperimentData(index, env_id, algorithm, combined_data))



# plot("BallCircle", "cpo")
algs = ["cpo", "lag-trpo", "pcpo", "pdo", "trpo"]
best_return = defaultdict(lambda: -1000.)
best_experiments = defaultdict(dict)
least_cost = defaultdict(lambda: 100000)
least_cost_experiment = {}

for combined_data in combined_experiments:
    if combined_data.env_id not in benchmark_env_ids:
        continue
    approx_cost_mean = np.mean(combined_data.data["cost_mean"][-50:])
    if least_cost[(combined_data.env_id, combined_data.algorithm)] > approx_cost_mean:
        least_cost[(combined_data.env_id, combined_data.algorithm)] = approx_cost_mean
        least_cost_experiment[(combined_data.env_id, combined_data.algorithm)] = combined_data

    if approx_cost_mean > env_info[combined_data.env_id]["max_costs"][0] and combined_data.algorithm != "trpo":
        continue
    approx_return_mean = np.mean(combined_data.data["return_mean"][-50:])
    if approx_return_mean > best_return[(combined_data.env_id, combined_data.algorithm)]:
        best_return[(combined_data.env_id, combined_data.algorithm)] = approx_return_mean
        best_experiments[combined_data.env_id][combined_data.algorithm] = combined_data

for env_id in benchmark_env_ids:
    for algorithm in algs:
        if best_experiments[env_id].get(algorithm) is None:
            best_experiments[env_id][algorithm] = least_cost_experiment[(env_id, algorithm)]

best_experiments_dict = defaultdict(dict)
for env_id in benchmark_env_ids:
    for algorithm in algs:
        best_experiments_dict[env_id][algorithm] = best_experiments[env_id][algorithm].__dict__


with open("/home/jaafar/Documents/safe_rl/plots/pybullet_experiments_combined.json", "w") as write_file:
    json.dump(best_experiments_dict, write_file, indent=4)

