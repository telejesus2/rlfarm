"""An implementation of Prioritized Experience Replay (PER).
This implementation is based on the paper "Prioritized Experience Replay"
by Tom Schaul et al. (2015).
"""
from typing import List
from os.path import join
import pickle

import numpy as np

from rlfarm.buffers.replay.uniform_replay_buffer import UniformReplayBuffer
from rlfarm.buffers.replay.sum_tree import SumTree
from rlfarm.buffers.replay.replay_buffer import ReplayElement
from rlfarm.buffers.replay.utils import invalid_range
from rlfarm.buffers.replay.const import *


class PrioritizedReplayBuffer(UniformReplayBuffer):
    """An out-of-graph Replay Buffer for Prioritized Experience Replay.
    """
    def __init__(self, *args, **kwargs):
        super(PrioritizedReplayBuffer, self).__init__(*args, **kwargs)
        self._sum_tree = SumTree(self._replay_capacity)

    def set_priority(self, indices, priorities):
        """Sets the priority of the given elements according to Schaul et al.
        Args:
          indices: np.array with dtype int32, of indices in range
            [0, replay_capacity).
          priorities: float, the corresponding priorities.
        """
        assert indices.dtype == np.int32, ('Indices must be integers, '
                                           'given: {}'.format(indices.dtype))
        for index, priority in zip(indices, priorities):
            self._sum_tree.set(index, priority)

    def get_priority(self, indices):
        """Fetches the priorities correspond to a batch of memory indices.
        For any memory location not yet used, the corresponding priority is 0.
        Args:
          indices: np.array with dtype int32, of indices in range
            [0, replay_capacity).
        Returns:
          priorities: float, the corresponding priorities.
        """
        assert indices.shape, 'Indices must be an array.'
        assert indices.dtype == np.int32, ('Indices must be int32s, '
                                           'given: {}'.format(indices.dtype))
        batch_size = len(indices)
        priority_batch = np.empty((batch_size), dtype=np.float32)
        for i, memory_index in enumerate(indices):
            priority_batch[i] = self._sum_tree.get(memory_index)
        return priority_batch

    #============================================================================================#
    # STORING
    #============================================================================================#

    def get_storage_elements(self) -> List[ReplayElement]:
        """Returns a default list of elements to be stored in this replay memory.
        """
        storage_elements = super(
            PrioritizedReplayBuffer, self).get_storage_elements()
        storage_elements.append(ReplayElement(PRIORITY, (), np.float32),)

        return storage_elements

    def add(self, action, reward, terminal, timeout, kwargs: dict, priority=None):
        kwargs[PRIORITY] = priority
        super(PrioritizedReplayBuffer, self).add(
            action, reward, terminal, timeout, kwargs)

    def add_final(self, kwargs: dict):
        if self.is_empty() or self._store[TERMINAL][self.cursor() - 1] != 1:
            raise ValueError('The previous transition was not terminal.')

        if (len(kwargs)) != self._num_state_elements:
            raise ValueError('Add expects {} elements, received {}.'.format(
                self._num_state_elements, len(kwargs)))

        for storage_element in self._storage_elements:
            if storage_element.name == TERMINAL:
                # Used to check that user is correctly adding transitions
                kwargs[TERMINAL] = -1
            elif storage_element.name == PRIORITY:
                # 0 priority for final observation.
                kwargs[PRIORITY] = 0.0
            elif storage_element.name not in kwargs:
                kwargs[storage_element.name] = np.empty(
                    storage_element.shape, dtype=storage_element.type)

        self._add(kwargs)

    def _add(self, kwargs: dict):
        """Add to the storage arrays.
        """
        with self._lock:
            cursor = self.cursor()
            priority = kwargs[PRIORITY]
            if priority is None:
                priority = self._sum_tree.max_recorded_priority

            if self._disk_saving:
                self._store[TERMINAL][cursor] = kwargs[TERMINAL]
                with open(join(self._save_dir, '%d.replay' % cursor), 'wb') as f:
                    pickle.dump(kwargs, f)
                # If first add, then pad for correct wrapping
                if self.add_count == 0:
                    self._add_initial_to_disk(kwargs)
            else:
                for name, data in kwargs.items():
                    self._store[name][cursor] = data

            self._sum_tree.set(self.cursor(), priority)
            self.add_count += 1
            self.invalid_range = invalid_range(
                self.cursor(), self._replay_capacity, self._history_len, self._n_steps)

    #============================================================================================#
    # SAMPLING
    #============================================================================================#

    def get_transition_elements(self, batch_size=None):
        """Returns a default list of elements to be returned by sample_transition_batch.
        """
        batch_size = self._batch_size if batch_size is None else batch_size
        parent_transition_type = (
            super(PrioritizedReplayBuffer,
                  self).get_transition_elements(batch_size))
        probablilities_type = [
            ReplayElement(SAMPLING_PROBABILITIES, (batch_size,), np.float32)
        ]
        return parent_transition_type + probablilities_type

    def sample_transition_batch(self, batch_size=None, indices=None,
                                pack_in_dict=True):
        """Returns a batch of transitions with extra storage and the priorities.
        When the transition is terminal next_state_batch has undefined contents.
        Args:
          batch_size: int, number of transitions returned. If None, the default
            batch_size will be used.
          indices: None or list of ints, the indices of every transition in the
            batch. If None, sample the indices uniformly.
        Returns:
          transition_batch: tuple of np.arrays with the shape and type as in
            get_transition_elements().
        """
        transition = super(
            PrioritizedReplayBuffer, self).sample_transition_batch(
            batch_size, indices, pack_in_dict=False)

        transition_elements = self.get_transition_elements(batch_size)
        transition_names = [e.name for e in transition_elements]
        probabilities_index = transition_names.index(SAMPLING_PROBABILITIES)
        indices_index = transition_names.index(INDICES)
        indices = transition[indices_index]
        # The parent returned an empty array for the probabilities. Fill it with the
        # contents of the sum tree.
        transition[probabilities_index][:] = self.get_priority(indices)
        batch_arrays = transition
        if pack_in_dict:
            batch_arrays = self.pack_transition(transition,
                                                transition_elements)
        return batch_arrays

    #============================================================================================#
    # SAMPLING UTILITIES
    #============================================================================================#

    def sample_index_batch(self, batch_size):
        """Returns a batch of valid indices sampled uniformly.
        Args:
          batch_size: int, number of indices returned.
        Returns:
          list of ints, a batch of valid indices sampled uniformly.
        Raises:
          RuntimeError: If the batch was not constructed after maximum number of tries.
        """
        # Sample stratified indices. Some of them might be invalid.
        indices = self._sum_tree.stratified_sample(batch_size)
        allowed_attempts = self._max_sample_attempts
        for i in range(len(indices)):
            if not self.is_valid_transition(indices[i]):
                if allowed_attempts == 0:
                    raise RuntimeError(
                        'Max sample attempts: Tried {} times but only sampled {}'
                        ' valid indices. Batch size is {}'.
                            format(self._max_sample_attempts, i, batch_size))
                index = indices[i]
                while not self.is_valid_transition(
                        index) and allowed_attempts > 0:
                    # If index i is not valid keep sampling others. Note that this
                    # is not stratified.
                    index = self._sum_tree.sample()
                    allowed_attempts -= 1
                indices[i] = index
        return indices