import logging
import random
from typing import Dict, Tuple
import os
import copy

import numpy as np
import torch
import torch.nn.functional as F
import scipy.ndimage as ndimage
import pickle
from PIL import Image

from rlfarm.envs.metaworld.metaworld_env import MetaWorldEnv
from rlfarm.utils.transition import Transition
# from rlfarm.vision.observation import Observation as _Observation # TODO uncomment if you want to unpickle without importing rlbench

# same as in RLBench/tools/dataset_generator.py
def save_demo(demo, example_path):

    # Save the low-dimension data
    with open(os.path.join(example_path, 'low_dim_obs.pkl'), 'wb') as f:
        pickle.dump(demo, f)


class RecorderMetaWorldEnv(MetaWorldEnv):
    def __init__(self, *args, **kwargs):
        super(RecorderMetaWorldEnv, self).__init__(*args, **kwargs)

        self.saver = EpisodeSaver(self.max_episode_steps)

    def step(self, action: np.ndarray) -> Transition:
        # self.render()
        obs = self._previous_obs_dict  # in case action fails.
        try:
            obs, reward, terminate, info = self._task.step(action)
            obs = self._extract_ob(obs)
            self._previous_obs_dict = obs
            if info["success"] == True:
                terminate = True
            self.saver.step(obs, action, reward, terminate) # new wrt parent
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
        self.saver.reset(self._previous_obs_dict) # new wrt parent

        return self._previous_obs_dict        


#============================================================================================#
# UTILITIES
#============================================================================================#

def check_and_make(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

#============================================================================================#
# SAVER
#============================================================================================#

class EpisodeSaver(object):
    """
    Save the experience data from a gym env to a file
    and notify the srl server so it learns from the gathered data
    :param name: (str)
    :param max_dist: (float)
    :param state_dim: (int)
    :param globals_: (dict) Environments globals
    :param learn_every: (int)
    :param learn_states: (bool)
    :param path: (str)
    :param relative_pos: (bool)
    """

    def __init__(self, max_episode_len: int, path='data/', name='slide3'):
        super(EpisodeSaver, self).__init__()
        self._max_episode_len = max_episode_len
        self.data_folder = path + name
        self.path = path
        check_and_make(self.data_folder)

        self.actions = []
        self.rewards = []
        self.states = []
        self.episode_step = 0
        self.episode_idx = -1
        self.episode_folder = None     

    def reset(self, observation):
        """
        Called when starting a new episode
        :param observation: 
        :param ground_truth: (numpy array)
        """
        self.episode_idx += 1
        self.actions = []
        self.rewards = []
        self.states = []
        self.episode_step = 0
        self.episode_folder = "record_{:03d}".format(self.episode_idx)
        os.makedirs("{}/{}".format(self.data_folder, self.episode_folder), exist_ok=True)

        self.states.append(observation)

    def step(self, observation, action, reward, done):
        """
        :param observation
        :param action: (int)
        :param reward: (float)
        :param done: (bool) whether the episode is done or not
        :param ground_truth_state: (numpy array)
        """
        self.episode_step += 1
        self.rewards.append(reward)
        self.actions.append(action)
        self.states.append(observation)

        if done or self.episode_step == self._max_episode_len:
            if done: # and len(self.states) < 10: # TODO remove len constraint
                print("SUCCESS")
                for i, obs in enumerate(self.states):
                    if i > 0:
                        obs['action'] = self.actions[i-1]
                        obs['reward'] = self.rewards[i-1]
                save_demo(self.states, os.path.join(self.data_folder, self.episode_folder))