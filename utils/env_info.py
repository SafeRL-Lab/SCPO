import numpy as np

env_info = {
    "SafetyBallCircle-v0": {
        "cost_extractor": lambda info: list([info["cost"]]),
        "max_costs": np.array([25.], dtype=np.float32)
    },
    "SafetyCarCircle-v0": {
        "cost_extractor": lambda info: list([info["cost"]]),
        "max_costs": np.array([25.], dtype=np.float32)
    },
    "CartSafe-v0": {
        "cost_extractor": lambda info: list(info["constraint_costs"]),
        "max_costs": np.array([0.5], dtype=np.float32)
    },
    "SafetyBallReach-v0": {
        "cost_extractor": lambda info: list([info["cost"]]),
        "max_costs": np.array([10.], dtype=np.float32)
    },
    "SafetyCarReach-v0": {
        "cost_extractor": lambda info: list([info["cost"]]),
        "max_costs": np.array([10.], dtype=np.float32)
    },
    "SafetyBallGather-v0": {
        "cost_extractor": lambda info: list([info["cost"]]),
        "max_costs": np.array([0.2], dtype=np.float32)
    },
    "SafetyCarGather-v0": {
        "cost_extractor": lambda info: list([info["cost"]]),
        "max_costs": np.array([0.2], dtype=np.float32)
    },
    "SafetyBallRun-v0": {
        "cost_extractor": lambda info: list([info["cost"]]),
        "max_costs": np.array([25.], dtype=np.float32)
    },
    "Safexp-PointGoal1-v0": {
        "cost_extractor": lambda info: list([info["cost"]]),
        "max_costs": np.array([25.], dtype=np.float32)
    },
    "Safexp-PointButton1-v0": {
        "cost_extractor": lambda info: list([info["cost"]]),
        "max_costs": np.array([25.], dtype=np.float32)
    }
}

benchmark_env_ids = [
    "SafetyBallCircle-v0",
    "SafetyBallReach-v0",
    "SafetyBallGather-v0",
    "SafetyBallRun-v0",
    "SafetyCarCircle-v0",
    "SafetyCarReach-v0",
]
