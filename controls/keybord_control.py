import queue
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict
import sys
from typing import Callable, Optional, Tuple
from dataclasses import dataclass
from pynput.keyboard import Key, Controller, Listener
import numpy as np
from typing import Dict

if sys.version_info[0:2] >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal


class Movement:
    def __init__(self, type: Literal["arrow", "char"]):
        self.type = type
        self.left = "left" if type == "arrow" else "a"
        self.right = "right" if type == "arrow" else "d"
        self.up = "up" if type == "arrow" else "w"
        self.down = "down" if type == "arrow" else "s"


@dataclass
class KeyState:
    is_pressed: bool = False
    pressed_at_sec: float = 0
    single_press: bool = True


class EnvControl(ABC):
    def get_action(self, key_states: Dict[str, KeyState]):
        pass


class CartControl(EnvControl):
    def get_action(self, key_states: Dict[str, KeyState]):
        if key_states["right"].is_pressed:
            return 1
        elif key_states["left"].is_pressed:
            return 0
        return None


class BallControl(EnvControl):
    def __init__(self, control_type: Literal["arrow", "char"] = "arrow"):
        self.movement = Movement(control_type)

    def get_action(self, key_states: Dict[str, KeyState]):
        current_time = time.time()
        left_motion = BallControl._get_key_intensity(key_states[self.movement.left], current_time)
        right_motion = BallControl._get_key_intensity(key_states[self.movement.right], current_time)
        up_motion = BallControl._get_key_intensity(key_states[self.movement.up], current_time)
        down_motion = BallControl._get_key_intensity(key_states[self.movement.down], current_time)
        action = np.array([right_motion - left_motion, up_motion - down_motion]).clip(-1, 1)

        return action

    @staticmethod
    def _get_key_intensity(key_state: KeyState, current_time: float):
        return key_state.is_pressed * np.clip((current_time - key_state.pressed_at_sec)/5, 0.05, 1.)


class KeyboardControl:
    ENV_TO_CONTROLLER: Dict[Key, EnvControl] = {
        "CartSafe-v0": CartControl(),
        "SafetyBallCircle-v0": BallControl(),
        "SafetyCarCircle-v0": BallControl(),
        "SafetyBallRun-v0": BallControl(),
        "SafetyBallGather-v0": BallControl(),
        "SafetyCarGather-v0": BallControl(),
        "SafetyBallReach-v0": BallControl(),
        "SafetyCarReach-v0": BallControl(),
        "Safexp-PointGoal1-v0": BallControl(),
    }
    SUPPORTED_ENVS = list(ENV_TO_CONTROLLER.keys())

    def __init__(self, env_id, callbacks: Optional[Dict[str, Callable]] = None):
        self.env_id = env_id
        assert env_id in self.SUPPORTED_ENVS, "control not supported for this env"
        self.env_controller = self.ENV_TO_CONTROLLER[env_id]
        self.callbacks = callbacks or {}
        self.key_map: Dict[str, KeyState] = defaultdict(lambda: KeyState())
        self.action = None
        # threading.Thread(target=self._env_controller_th).start()
        self._start()

    def _start(self):
        listener = Listener(on_press=self._on_press, on_release=self._on_release)
        listener.start()

    def _on_press(self, key: Key):
        key_state = self.key_map[self._get_key_name(key)]
        key_state.pressed_at_sec = key_state.pressed_at_sec if key_state.is_pressed else time.time()
        key_state.is_pressed = True
        self._add_action()

    def _on_release(self, key: Key):
        key_name = self._get_key_name(key)
        self.key_map[key_name] = KeyState(False, 0)
        self._add_action()

        if key_name in self.callbacks:
            self.callbacks[key_name]()

    @staticmethod
    def _get_key_code(key: Key) -> int:
        if key.__dict__.get("_value_") is not None:
            key = key.__dict__["_value_"]
        return key.vk

    @staticmethod
    def _get_key_name(key: Key):
        return key.__dict__.get("_name_") or key.__dict__.get("char")

    def get_key_state(self, key_code: str) -> KeyState:
        return self.key_map[key_code]

    def _add_action(self):
        action = self.env_controller.get_action(self.key_map)
        self.action = action

    def get_action(self):
        return self.action
# c = KeyboardControl("CartSafe-v0")
# while True:
#     time.sleep(10)
