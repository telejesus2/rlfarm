import logging
import random
from typing import Dict, Tuple, List

import numpy as np
import torch

from rlfarm.envs.env import Env, ActionSpace
from rlfarm.utils.transition import Transition
from rlfarm.utils.logger import Summary, ImageSummary

# for demos
import pickle
from os import listdir
from os.path import join, exists
from natsort import natsorted


class MetaWorldActionSpace(ActionSpace):
    def __init__(self, action_size):
        self.action_size = action_size

        self._min_gripper_action = -1
        self._max_gripper_action = 1
        self._min_action = (self.action_size - 1) * [-1]
        self._max_action = (self.action_size - 1) * [1]


    def normalize(self, action: torch.tensor) -> torch.tensor:
        return action

    def sample(self) -> torch.tensor:
        action = torch.cat(
            [torch.FloatTensor(1).uniform_(min_action, max_action) 
                for (min_action, max_action) in zip(self._min_action, self._max_action)] +
            [torch.FloatTensor(1).uniform_(self._min_gripper_action, self._max_gripper_action)]
        )
        return self.normalize(action.view(1, -1))

    def get_action_min_max(self, action_min_max: Tuple[np.ndarray] = None) -> Tuple[np.ndarray]:
        # make bounds a little bigger
        if action_min_max is None:
            action_min_max = np.array(self._min_action + [self._min_gripper_action]).astype(np.float32), \
                             np.array(self._max_action + [self._max_gripper_action]).astype(np.float32)
        else:
            action_min_max[0][-1] = self._min_gripper_action
            action_min_max[1][-1] = self._max_gripper_action
            action_min_max[0][:-1] -= np.fabs(action_min_max[0][:-1]) * 0.2
            action_min_max[1][:-1] += np.fabs(action_min_max[1][:-1]) * 0.2
        return action_min_max


class MetaWorldEnv(Env):
    def __init__(self,
                 task_name: str,
                 task_class,
                 task_kwargs: dict = {},
                 dataset_root: str = '',
                 headless=True,
                 max_episode_steps=np.inf,
                 stack_vision_channels=True,
                 reward_scale=100.0,
                 state_includes_remaining_time: bool = False,
                 state_includes_previous_action: bool = False,
                 set_done_when_success: bool = False,
                 sparse_reward: bool = False,
    ):
        super(MetaWorldEnv, self).__init__()
        self._stack_vision_channels = stack_vision_channels
        self._task_class = task_class
        self._task_name = task_name
        self._dataset_root = dataset_root
        self._reward_scale = reward_scale
        self._state_includes_remaining_time = state_includes_remaining_time
        self._state_includes_previous_action = state_includes_previous_action
        self._set_done_when_success = set_done_when_success
        self._sparse_reward = sparse_reward
        self._task_kwargs = task_kwargs

        # utilities
        self._i = 0
        self._prev_action = None        
        
        # properties
        self.state_shape = {'low_dim_state': (39,)}
        self.state_dtype = {'low_dim_state': np.float32}
        self.action_shape = (4,)
        self.action_dtype = np.float32
        self.action_space = MetaWorldActionSpace(self.action_shape[0])
        self.max_episode_steps = max_episode_steps
        self.num_demos_per_variation = self._get_num_stored_demos(self._dataset_root, self._task_name)
        self.average_demo_length = None # should be overriden (see buffers.replay.builder.make_buffer())
        logging.info("Number of demos per variation available: %d" % self.num_demos_per_variation)
        logging.info("State shape: %s" % str(self.state_shape))
        
    def launch(self, agent=None):
        self._task = self._task_class()
        self._task._set_task_called = True
        self._task._freeze_rand_vec = False
        self._task.seeded_rand_vec = False
        self._task._partially_observable = False

    @property
    def unwrapped(self):
        return self

    @property
    def history_len(self) -> int:
        return 1

    @property
    def num_variations(self) -> int:
        return 1

    @property
    def env(self):
        return None

    @property
    def active_task_id(self) -> int:
        # returns task index, not the actual id
        return 0

    def render(self, offscreen=False, **kwargs):
        self._task.render(offscreen=offscreen)

    def seed(self, seed):
        # TODO
        pass

    def close(self):
        return self._task.close()

    def step(self, action: np.ndarray) -> Transition:
        obs = self._previous_obs_dict  # in case action fails.
        try:
            obs, reward, terminate, info = self._task.step(action)
            obs = self._extract_ob(obs)
            self._previous_obs_dict = obs
            if self._set_done_when_success and info["success"] == True:
                terminate = True
                reward += 1000 # TODO should remove
            if self._sparse_reward:
                reward = 1 if terminate else 0
        except:
            reward, terminate, info = 0, True, {"failure": True}
        self._i += 1
        self._prev_action = action
        return Transition(obs, reward * self._reward_scale, terminate, info)

    def reset(self):
        logging.info("Episode reset...")
        # reset task
        obs = self._task.reset()
        self._i = 0
        self._prev_action = None
        # reset summaries
        self.episode_summaries = [] # see runners.samplers.rollout_generator.RolloutGenerator

        # track previous obs (in case action fails in step())
        self._previous_obs_dict = self._extract_ob(obs)

        return self._previous_obs_dict

    def _extract_ob(self, ob, t=None, prev_action=None,
                    include_low_dim=True) -> Dict[str, np.ndarray]: 
        new_ob = {}

        # add low dim state
        if include_low_dim:
            key = 'low_dim_state'
            low_dim_state = np.array(ob, dtype=np.float32)
            if self._state_includes_remaining_time:
                tt = 1. - ((self._i if t is None else t) / self.max_episode_steps)
                low_dim_state = np.concatenate([low_dim_state, [tt]]).astype(np.float32)
            if self._state_includes_previous_action:
                pa = self._prev_action if prev_action is None else prev_action
                pa = np.zeros(self.action_shape) if pa is None else pa
                low_dim_state = np.concatenate([low_dim_state, pa]).astype(np.float32)
            new_ob[key] = low_dim_state 

        return new_ob

    def get_demos(self, index=None, random_selection=False, amount=1, variation=None):
        demos = []
        assert random_selection or index is not None

        demos += self._get_stored_demos(
            amount, self._dataset_root, self._task_name,
            random_selection=random_selection, from_episode_number=index)

        return demos

    def extract_demo(self, demo, agent=None):
        "usually called on demos returned by self.get_demos()"

        # extract the actions
        acs = [ob['action'] for ob in demo[1:]]

        # extract the rewards
        rews = [0] * (len(demo) - 2) + [1]
        rews = [r * self._reward_scale for r in rews]

        # extract the states
        obs = [{'low_dim_state': ob['low_dim_state']} for ob in demo]

        sums = []
        return obs, acs, rews, sums

    def _get_num_stored_demos(self, dataset_root: str, task_name: str,):

        task_root = join(dataset_root, task_name)
        if not exists(task_root):
            raise RuntimeError("Can't find the demos for %s at: %s" % (
                task_name, task_root))

        return  len(listdir(task_root))

    def _get_stored_demos(self, 
            amount: int, dataset_root: str, task_name: str,
            random_selection: bool = True, from_episode_number: int = 0):

        task_root = join(dataset_root, task_name)
        if not exists(task_root):
            raise RuntimeError("Can't find the demos for %s at: %s" % (
                task_name, task_root))

        # Sample an amount of examples for the variation of this task
        examples = listdir(task_root)
        if amount == -1:
            amount = len(examples)
        if amount > len(examples):
            raise RuntimeError(
                'You asked for %d examples, but only %d were available.' % (
                    amount, len(examples)))
        if random_selection:
            selected_examples = np.random.choice(examples, amount, replace=False)
        else:
            selected_examples = natsorted(
                examples)[from_episode_number:from_episode_number+amount]

        # Process these examples (e.g. loading observations)
        demos = []
        for example in selected_examples:
            example_path = join(task_root, example)
            with open(join(example_path, "low_dim_obs.pkl"), 'rb') as f:
                obs = pickle.load(f)

            demos.append(obs)

        return demos