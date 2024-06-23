from abc import ABC
from typing import List


class ReplayElement(object):
    def __init__(self, name, shape, type, is_observation=False):
        self.name = name
        self.shape = shape
        self.type = type
        self.is_observation = is_observation


class ReplayBuffer(ABC):
    def replay_capacity(self):
        pass

    def batch_size(self):
        pass

    def get_storage_elements(self) -> List[ReplayElement]:
        pass

    def add(self, action, reward, terminal, timeout, state: dict):
        pass

    def add_final(self, state: dict):
        pass

    def is_empty(self):
        pass

    def is_full(self):
        pass

    def cursor(self):
        pass

    def is_valid_transition(self, index):
        pass

    def sample_index_batch(self, batch_size):
        pass

    def sample_transition_batch(self, batch_size=None, indices=None, pack_in_dict=True):
        pass

    def get_transition_elements(self, batch_size=None):
        pass

    def shutdown(self):
        pass

    def using_disk(self):
        pass