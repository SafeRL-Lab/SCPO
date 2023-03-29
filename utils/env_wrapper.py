from typing import List, Tuple, Optional

import gym
import numpy as np
import cv2


class VideoRecorder:
    def __init__(self, path: str, fps: int, frame_size: Optional[Tuple[int, int]] = None):
        self.fps = fps
        self.path = path
        self.frame_buffer: List[np.ndarray] = []
        self.frame_size = frame_size

    def add_frame(self, frame: np.ndarray):
        red = frame[:, :, 2].copy()
        blue = frame[:, :, 0].copy()
        frame[:, :, 0] = red
        frame[:, :, 2] = blue
        self.frame_buffer.append(frame)

    def save(self):
        # frame_size = self.frame_buffer[0].shape
        # if len(self.frame_buffer) == 0:
        #     return

        #frame_size = (self.frame_buffer[0].shape[0], self.frame_buffer[0].shape[1])
        self.frame_size = self.frame_size or self.frame_buffer[0].shape[0:2]
        out = cv2.VideoWriter(f"{self.path}.avi", cv2.VideoWriter_fourcc(*'DIVX'), 50, self.frame_size)

        for frame in self.frame_buffer:
            frame = cv2.resize(frame, dsize=self.frame_size, interpolation=cv2.INTER_CUBIC)
            out.write(frame)
        out.release()


class RewardRescale(gym.Wrapper):
    def __init__(self, env, reward_scale: float):
        super().__init__(env)
        self.env = env
        self.reward_scale = reward_scale

    def step(self, action):
        next_state, reward, done, info = super().step(action)
        reward = reward/self.reward_scale
        return next_state, reward, done, info


class EarlyTermination(gym.Wrapper):
    def __init__(self, env, cost_extractor, max_cost: np.ndarray):
        if isinstance(env, str):
            self.env = gym.make(env)
        super().__init__(self.env)
        self.cost_extractor = cost_extractor
        self.max_cost = max_cost
        self.cumulative_cost = np.zeros_like(max_cost)

    def reset(self, **kwargs):
        self.cumulative_cost = 0
        return super().reset(**kwargs)

    def step(self, action):
        next_state, reward, done, info = super().step(action)
        cost = np.array(self.cost_extractor(info))
        self.cumulative_cost = self.cumulative_cost + cost
        if np.any(self.cumulative_cost > self.max_cost):
            done = True
            info['TimeLimit.truncated'] = True
            info["terminal_observation"] = next_state
        return next_state, reward, done, info