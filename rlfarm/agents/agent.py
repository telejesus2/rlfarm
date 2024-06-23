from abc import ABC, abstractmethod
from typing import Any, List

import torch

from rlfarm.utils.logger import Summary
from rlfarm.utils.transition import ActResult


class Agent(ABC):
    @abstractmethod
    def build(self, training: bool, device: torch.device) -> None:
        pass

    @abstractmethod
    def update(self, step: int, sample: dict, warmup: bool = False) -> dict:
        pass

    @abstractmethod
    def act(self, step: int, observation: dict, deterministic: bool,
            explore: bool, track_outputs=False) -> ActResult:
        pass

    def reset(self) -> None:
        pass

    @abstractmethod
    def update_summaries(self, log_scalar_only=True) -> List[Summary]:
        pass

    @abstractmethod
    def act_summaries(self) -> List[Summary]:
        pass

    @abstractmethod
    def load_weights(self, savedir: str) -> None:
        pass

    @abstractmethod
    def save_weights(self, savedir: str) -> None:
        pass