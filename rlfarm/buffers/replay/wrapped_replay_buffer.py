from abc import ABC, abstractmethod
from typing import Any

from torch.utils.data import IterableDataset, DataLoader

from rlfarm.buffers.replay.replay_buffer import ReplayBuffer


class WrappedReplayBuffer(ABC):
    def __init__(self, replay_buffer: ReplayBuffer):
        self._replay_buffer = replay_buffer

    @property
    def replay_buffer(self):
        return self._replay_buffer

    @abstractmethod
    def dataset(self) -> Any:
        pass


class IterableReplayDataset(IterableDataset):
    def __init__(self, replay_buffer: ReplayBuffer):
        self._replay_buffer = replay_buffer

    def _generator(self):
        while True:
            yield self._replay_buffer.sample_transition_batch(pack_in_dict=True)

    def __iter__(self):
        return iter(self._generator())


class IterableReplayBuffer(WrappedReplayBuffer):
    def __init__(self, replay_buffer: ReplayBuffer, num_workers: int = 2):
        super(IterableReplayBuffer, self).__init__(replay_buffer)
        self._num_workers = num_workers

    def dataset(self) -> DataLoader: # TODO: use self._num_workers
        d = IterableReplayDataset(self._replay_buffer)
        # Batch size None disables automatic batching
        return DataLoader(d, batch_size=None, pin_memory=True, prefetch_factor=2,
            # num_workers=8,
            )