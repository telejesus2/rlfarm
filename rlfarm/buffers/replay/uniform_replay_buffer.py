from typing import List, Type, Dict
import logging
import os
from os.path import join
import math
import pickle
from multiprocessing import Lock

import numpy as np
import torch

from rlfarm.buffers.replay.replay_buffer import ReplayBuffer, ReplayElement
from rlfarm.buffers.replay.utils import invalid_range
from rlfarm.buffers.replay.const import *


class UniformReplayBuffer(ReplayBuffer):
    def __init__(self,
                 state_shape: Dict[str, tuple],
                 state_dtype: Dict[str, Type[np.dtype]],
                 action_shape: tuple = (),
                 action_dtype: Type[np.dtype] = np.float32,
                 reward_shape: tuple = (),
                 reward_dtype: Type[np.dtype] = np.float32,
                 replay_capacity: int = int(1e6),
                 history_len: int = 1,
                 n_steps: List[int] = [1],
                 batch_size: int = 32,
                 gamma: float = 0.99,
                 max_sample_attempts: int = 10000,
                 save_dir: str = None,
                 purge_replay_on_shutdown: bool = True,
                 extra_replay_elements: List[ReplayElement] = [],
                 ):
        self._history_len = history_len
        self._replay_capacity = replay_capacity
        self._batch_size = batch_size
        self._n_steps_list = n_steps
        self._n_steps = max(self._n_steps_list)
        self._gamma = gamma
        self._max_sample_attempts = max_sample_attempts

        # extra attributes
        self._lock = Lock()
        self.add_count = np.array(0) 
        self.invalid_range = np.zeros((self._history_len))
        self._cumulative_discount_vector = np.array(
            [math.pow(self._gamma, n) for n in range(self._n_steps)],
            dtype=np.float32) 

        # saving to disk
        self._disk_saving = save_dir is not None
        self._save_dir = save_dir
        self._purge_replay_on_shutdown = purge_replay_on_shutdown
        if self._disk_saving:
            logging.info('\t saving to disk: %s', self._save_dir)
            os.makedirs(save_dir, exist_ok=True)
        else:
            logging.info('\t saving to RAM')

        # initialize storage
        self._action_shape = action_shape
        self._action_dtype = action_dtype
        self._reward_shape = reward_shape
        self._reward_dtype = reward_dtype
        self._state_shape = state_shape
        self._state_dtype = state_dtype
        self._extra_replay_elements = extra_replay_elements
        self._storage_elements = self.get_storage_elements()
        self._num_storage_elements = len(self._storage_elements)
        self._num_state_elements = len(state_shape)
        self._create_storage()

    @property
    def history_len(self):
        return self._history_len

    @property
    def replay_capacity(self):
        return self._replay_capacity

    @property
    def batch_size(self):
        return self._batch_size

    def is_empty(self):
        """Is the Replay Buffer empty?"""
        return self.add_count == 0

    def is_full(self):
        """Is the Replay Buffer full?"""
        return self.add_count >= self._replay_capacity

    def cursor(self):
        """Index to the location where the next transition will be written."""
        return self.add_count % self._replay_capacity

    #============================================================================================#
    # STORING
    #============================================================================================#

    def get_storage_elements(self) -> List[ReplayElement]:
        """Returns a default list of elements to be stored in this replay memory.
        """
        storage_elements = [
            ReplayElement(ACTION, self._action_shape, self._action_dtype),
            ReplayElement(REWARD, self._reward_shape, self._reward_dtype),
            ReplayElement(TERMINAL, (), np.int8),
            ReplayElement(TIMEOUT, (), np.bool), # TODO (jesus) no algorithm uses this, why store it?
        ]
        for key in self._state_shape.keys():
            storage_elements.append(
                ReplayElement(key, self._state_shape[key], self._state_dtype[key]))
        for element in self._extra_replay_elements:
            storage_elements.append(element)

        return storage_elements

    def _create_storage(self):
        """Creates the numpy arrays used to store transitions.
        TERMINAL storage: [0 ... 0 1 -1 0 ... 0 1 -1 0 ...]
        A terminal flag = 1 can be caused by the environment or timeout 
        """
        self._store = {}
        for storage_element in self._storage_elements:
            array_shape = [self._replay_capacity] + list(storage_element.shape)
            if storage_element.name == TERMINAL:
                self._store[storage_element.name] = np.full(
                    array_shape, -1, dtype=storage_element.type)
            elif not self._disk_saving:
                # If saving to disk, we don't need to store anything else.
                self._store[storage_element.name] = np.empty(
                    array_shape, dtype=storage_element.type)

    def add(self, action, reward, terminal, timeout, kwargs: dict):
        """Adds a transition to the replay memory.
        :param kwargs: contains entries of both state and extra_replay_elements
        """
        if not self.is_empty() and self._store[TERMINAL][self.cursor() - 1] == 1:
            raise ValueError('The previous transition was terminal, but add_final was not called.')

        kwargs[ACTION] = action
        kwargs[REWARD] = reward
        kwargs[TERMINAL] = terminal
        kwargs[TIMEOUT] = timeout

        if (len(kwargs)) != self._num_storage_elements:
            raise ValueError('Add expects {} elements, received {}.'.format(
                self._num_storage_elements, len(kwargs)))

        self._add(kwargs)

    def add_final(self, kwargs: dict):
        """Adds the final state of an episode to the replay memory.
        :param kwargs: contains entries of the state
        """
        if self.is_empty() or self._store[TERMINAL][self.cursor() - 1] != 1:
            raise ValueError('The previous transition was not terminal.')
        
        if (len(kwargs)) != self._num_state_elements:
            raise ValueError('Add expects {} elements, received {}.'.format(
                self._num_state_elements, len(kwargs)))

        for storage_element in self._storage_elements:
            if storage_element.name == TERMINAL:
                # Used to check that user is correctly adding transitions
                kwargs[TERMINAL] = -1
            elif storage_element.name not in kwargs:
                kwargs[storage_element.name] = np.empty(
                    storage_element.shape, dtype=storage_element.type)

        self._add(kwargs)

    def _add(self, kwargs: dict):
        """Add to the storage arrays.
        """
        with self._lock:
            cursor = self.cursor()

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

        transition_elements = [
            ReplayElement(ACTION, (batch_size,) + self._action_shape, self._action_dtype),
            ReplayElement(REWARD, (batch_size, len(self._n_steps_list)) + self._reward_shape, self._reward_dtype),
            ReplayElement(TERMINAL, (batch_size, len(self._n_steps_list)), np.int8),
            ReplayElement(TIMEOUT, (batch_size,), np.bool), # TODO should timeout also have n_steps?
            ReplayElement(INDICES, (batch_size,), np.int32),
            ReplayElement(N_STEPS, (batch_size, len(self._n_steps_list)), np.int16),
        ]
        batch_shape = (batch_size, self._history_len) if self._history_len > 1 else (batch_size,)
        batch_shape_tp1 = (batch_size, self._history_len, len(self._n_steps_list)) if self._history_len > 1 else (batch_size, len(self._n_steps_list))
        for key in self._state_shape.keys():
            transition_elements.append(ReplayElement(
                key, 
                batch_shape + tuple(self._state_shape[key]),
                self._state_dtype[key], is_observation=True))
            transition_elements.append(ReplayElement(
                key + '_tp1', 
                batch_shape_tp1 + tuple(self._state_shape[key]),
                self._state_dtype[key], is_observation=True))
        for element in self._extra_replay_elements:
            transition_elements.append(ReplayElement(
                element.name,
                (batch_size,) + tuple(element.shape),
                element.type))

        return transition_elements

    def pack_transition(self, batch_arrays, transition_elements):
        """Packs the transition into a dict. Each element of the dict is a torch.tensor, 
        except the state and next_state which are dict of tensors.
        """
        transition, state, next_state = {}, {}, {}
        for array, element in zip(batch_arrays, transition_elements):
            if element.is_observation:
                if element.name.endswith('_tp1'):
                    next_state[element.name[:-4]] = torch.tensor(array)
                else:
                    state[element.name] = torch.tensor(array)
            else:
                transition[element.name] = torch.tensor(array)

        transition[STATE] = state
        transition[NEXT_STATE] = next_state
        
        return transition

    def sample_transition_batch(self, batch_size=None, indices=None, pack_in_dict=True):
        """Returns a batch of transitions (including any extra contents).
        If get_transition_elements has been overridden and defines elements not
        stored in self._store, an empty array will be returned and it will be
        left to the child class to fill it.
        When the transition is terminal next_state_batch has undefined contents.
        NOTE: This transition contains the indices of the sampled elements.
        These are only valid during the call to sample_transition_batch,
        i.e. they may be used by subclasses of this replay buffer but may
        point to different data as soon as sampling is done.
        Args:
          batch_size: int, number of transitions returned. If None, the default
            batch_size will be used.
          indices: None or list of ints, the indices of every transition in the
            batch. If None, sample the indices uniformly.
        Returns:
          transition_batch: tuple of np.arrays with the shape and type as in
            get_transition_elements().
        Raises:
          ValueError: If an element to be sampled is missing from the
            replay buffer.
        """
        
        if batch_size is None:
            batch_size = self._batch_size
        with self._lock:
            if indices is None:
                indices = self.sample_index_batch(batch_size)
            else:
                for index in indices:
                    if not self.is_valid_transition(index):
                        raise ValueError('Invalid index %d.' % index)
            assert len(indices) == batch_size

            transition_elements = self.get_transition_elements(batch_size)
            batch_arrays = []
            for element in transition_elements:
                batch_arrays.append(np.empty(element.shape, dtype=element.type))

            for batch_index, state_index in enumerate(indices):
                store = self._store
                if self._disk_saving:
                    store = self._get_store_from_disk(
                        state_index - (self._history_len - 1),
                        state_index + self._n_steps + 1)

                # Fill the contents of each array in the sampled batch.
                terminal_stack = self.get_terminal_stack(state_index)
                assert len(transition_elements) == len(batch_arrays)
                for element_array, element in zip(batch_arrays, transition_elements):
                    if element.is_observation:
                        if not element.name.endswith('_tp1'):
                            element_array[
                                batch_index] = self._get_element_stack(
                                store[element.name],
                                state_index,
                                terminal_stack)
                    elif element.name == INDICES:
                        element_array[batch_index] = state_index
                    elif element.name in store.keys():
                        element_array[batch_index] = (store[element.name][state_index])

                for n_steps_idx, n_steps in enumerate(self._n_steps_list):

                    trajectory_indices = [(state_index + j) % self._replay_capacity
                                        for j in range(n_steps)]
                    trajectory_terminals = self._store[TERMINAL][trajectory_indices]
                    is_terminal_transition = trajectory_terminals.any()
                    if not is_terminal_transition:
                        trajectory_length = n_steps
                    else:
                        # np.argmax of a bool array returns index of the first True.
                        trajectory_length = np.argmax(trajectory_terminals.astype(np.bool), 0) + 1
                    next_state_index = state_index + trajectory_length

                    # store = self._store
                    # if self._disk_saving:
                    #     store = self._get_store_from_disk(
                    #         state_index - (self._history_len - 1),
                    #         next_state_index + 1)

                    trajectory_discount_vector = (
                        self._cumulative_discount_vector[:trajectory_length])
                    trajectory_rewards = self.get_range(store[REWARD],
                                                        state_index,
                                                        next_state_index)
                    terminal_stack_tp1 = self.get_terminal_stack(
                        next_state_index % self._replay_capacity)

                    # Fill the contents of each array in the sampled batch.
                    assert len(transition_elements) == len(batch_arrays)
                    for element_array, element in zip(batch_arrays, transition_elements):
                        if element.is_observation:
                            if element.name.endswith('_tp1'):
                                element_array[
                                    batch_index, n_steps_idx] = self._get_element_stack(
                                    store[element.name[:-4]],
                                    next_state_index % self._replay_capacity,
                                    terminal_stack_tp1)
                        elif element.name == REWARD:
                            # compute discounted sum of rewards in the trajectory.
                            element_array[batch_index, n_steps_idx] = np.sum(
                                trajectory_discount_vector * trajectory_rewards,
                                axis=0)
                        elif element.name == TERMINAL:
                            element_array[batch_index, n_steps_idx] = is_terminal_transition
                        elif element.name == N_STEPS:
                            element_array[batch_index, n_steps_idx] = trajectory_length

        if pack_in_dict:
            batch_arrays = self.pack_transition(
                batch_arrays, transition_elements)
        return batch_arrays

    #============================================================================================#
    # SAMPLING UTILITIES
    #============================================================================================#

    def is_valid_transition(self, index):
        """Checks if the index contains a valid transition.
        Checks for collisions with the end of episodes and the current position
        of the cursor.
        Args:
          index: int, the index to the state in the transition.
        Returns:
          Is the index valid: Boolean.
        """
        # Check the index is in the valid range
        if index < 0 or index >= self._replay_capacity:
            return False
        if not self.is_full():
            # The indices and next_indices must be smaller than the cursor.
            if index >= self.cursor() - self._n_steps:
                return False

        # Skip transitions that straddle the cursor.
        if index in set(self.invalid_range):
            return False

        # Check the index isn't the end of an episode 
        term_stack = self.get_terminal_stack(index)
        if term_stack[-1] == -1:
            return False

        return True

    def get_range(self, array, start_index, end_index):
        """Returns the range of array at the index handling wraparound if necessary.
        Args:
          array: np.array, the array to get the stack from.
          start_index: int, index to the start of the range to be returned. Range
            will wraparound if start_index is smaller than 0.
          end_index: int, exclusive end index. Range will wraparound if end_index
            exceeds replay_capacity.
        Returns:
          np.array, with shape [end_index - start_index, array.shape[1:]].
        """
        assert end_index > start_index, 'end_index must be larger than start_index'
        assert end_index >= 0
        assert start_index < self._replay_capacity
        if not self.is_full():
            assert end_index <= self.cursor(), (
                'Index {} has not been added.'.format(start_index))

        # Fast slice read when there is no wraparound.
        if start_index % self._replay_capacity < end_index % self._replay_capacity:
            return_array = np.array(
                [array[i] for i in range(start_index, end_index)])
        # Slow list read.
        else:
            indices = [(start_index + i) % self._replay_capacity
                       for i in range(end_index - start_index)]
            return_array = np.array([array[i] for i in indices])

        return return_array

    def get_range_stack(self, array, start_index, end_index, terminals=None):
        """Returns the range of array at the index handling wraparound if necessary.
        Args:
          array: np.array, the array to get the stack from.
          start_index: int, index to the start of the range to be returned. Range
            will wraparound if start_index is smaller than 0.
          end_index: int, exclusive end index. Range will wraparound if end_index
            exceeds replay_capacity.
        Returns:
          np.array, with shape [end_index - start_index, array.shape[1:]].
        """
        return_array = np.array(self.get_range(array, start_index, end_index))
        if terminals is None:
            terminals = self.get_range(
                self._store[TERMINAL], start_index, end_index)

        terminals = terminals[:-1]

        # Here we now check if we need to pad the front episodes
        # If any have a terminal of -1, then we have spilled over
        # into the the previous transition
        if np.any(terminals == -1):
            padding_item = return_array[-1]
            _array = list(return_array)[:-1]
            arr_len = len(_array)
            pad_from_now = False
            for i, (ar, term) in enumerate(
                    zip(reversed(_array), reversed(terminals))):
                if term == -1 or pad_from_now:
                    # The first time we see a -1 term, means we have hit the
                    # beginning of this episode, so pad from now.
                    # pad_from_now needed because the next transition (reverse)
                    # will not be a -1 terminal.
                    pad_from_now = True
                    return_array[arr_len - 1 - i] = padding_item
                else:
                    # After we hit out first -1 terminal, we never reassign.
                    padding_item = ar

        return return_array

    def _get_element_stack(self, array, index, terminals=None):
        state = self.get_range_stack(array,
                                     index - self._history_len + 1, index + 1,
                                     terminals=terminals)
        return state

    def get_terminal_stack(self, index):
        return self.get_range(self._store[TERMINAL],
                              index - self._history_len + 1, index + 1)

    def sample_index_batch(self, batch_size):
        """Returns a batch of valid indices sampled uniformly.
        Args:
          batch_size: int, number of indices returned.
        Returns:
          list of ints, a batch of valid indices sampled uniformly.
        Raises:
          RuntimeError: If the batch was not constructed after maximum number of tries.
        """
        if self.is_full():
            # add_count >= self._replay_capacity > self._stack_size
            min_id = (self.cursor() - self._replay_capacity + self._history_len - 1)
            max_id = self.cursor() - self._n_steps
        else:
            min_id = 0
            max_id = self.cursor() - self._n_steps
            if max_id <= min_id:
                raise RuntimeError(
                    'Cannot sample a batch with fewer than stack size '
                    '({}) + n_steps ({}) transitions.'.
                    format(self._history_len, self._n_steps))

        indices = []
        attempt_count = 0
        while (len(indices) < batch_size and
                       attempt_count < self._max_sample_attempts):
            index = np.random.randint(min_id, max_id) % self._replay_capacity
            if self.is_valid_transition(index):
                indices.append(index)
            else:
                attempt_count += 1
        if len(indices) != batch_size:
            raise RuntimeError(
                'Max sample attempts: Tried {} times but only sampled {}'
                ' valid indices. Batch size is {}'.
                    format(self._max_sample_attempts, len(indices), batch_size))

        return indices

    #============================================================================================#
    # DISK
    #============================================================================================#

    def using_disk(self):
        return self._disk_saving

    def _add_initial_to_disk(self ,kwargs: dict):
        for i in range(self._history_len - 1):
            with open(join(self._save_dir, '%d.replay' % (
                    self._replay_capacity - 1 - i)), 'wb') as f:
                pickle.dump(kwargs, f)

    def _get_store_from_disk(self, start_index, end_index):
        """Returns the store at the index handling wraparound if necessary.
        Args:
          start_index: int, index to the start of the range to be returned. Range
            will wraparound if start_index is smaller than 0.
          end_index: int, exclusive end index. Range will wraparound if end_index
            exceeds replay_capacity.
        Returns:
          dict of numpy arrays, each with shape [end_index - start_index, array.shape[1:]].
        """
        assert end_index > start_index, 'end_index must be larger than start_index'
        assert end_index >= 0
        assert start_index < self._replay_capacity
        if not self.is_full():
            assert end_index <= self.cursor(), (
                'Index {} has not been added.'.format(start_index))

        # Here we fake a mini store (buffer)
        store = {storage_element.name: {}
                 for storage_element in self._storage_elements}
        if start_index % self._replay_capacity < end_index % self._replay_capacity:
            for idx in range(start_index, end_index):
                with open(join(self._save_dir, '%d.replay' % idx), 'rb') as f:
                    d = pickle.load(f)
                    for k, v in d.items():
                        store[k][idx] = v
        else:
            for i in range(end_index - start_index):
                idx = (start_index + i) % self._replay_capacity
                with open(join(self._save_dir, '%d.replay' % idx), 'rb') as f:
                    d = pickle.load(f)
                    for k, v in d.items():
                        store[k][idx] = v
        return store

    def shutdown(self):
        if self._purge_replay_on_shutdown:
            # Safely delete replay
            logging.info('Clearing disk replay buffer.')
            for f in [f for f in os.listdir(self._save_dir) if '.replay' in f]:
                os.remove(join(self._save_dir, f))