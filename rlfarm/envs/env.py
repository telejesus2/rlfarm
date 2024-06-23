from abc import ABC, abstractmethod
from typing import Any, List, Tuple

import numpy as np
import torch

from rlfarm.utils.transition import Transition
from rlfarm.utils.logger import Summary


class ActionSpace(ABC):
    @abstractmethod
    def normalize(self, action: torch.tensor) -> torch.tensor:
        pass

    @abstractmethod
    def sample(self) -> torch.tensor:
        pass

    @abstractmethod
    def get_action_min_max(self, action_min_max: Tuple[np.ndarray] = None) -> Tuple[np.ndarray]:
        pass

    @abstractmethod
    def log_actions(self, actions: torch.tensor, prefix: str) -> List[Summary]:
        pass
    

class Env(ABC):
    def __init__(self):
        self._eval = False

    @property
    def eval(self):
        return self._eval

    @eval.setter
    def eval(self, eval):
        self._eval = eval

    @abstractmethod
    def launch(self) -> None:
        pass

    def close(self) -> None:
        pass

    @abstractmethod
    def reset(self) -> dict:
        pass

    @abstractmethod
    def step(self, action: np.ndarray) -> Transition:
        pass

    @property
    @abstractmethod
    def env(self) -> Any:
        pass

    @property
    @abstractmethod
    def active_task_id(self) -> int:
        pass