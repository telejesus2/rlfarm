from sre_constants import FAILURE, SUCCESS
from typing import List, Any

import numpy as np

from rlfarm.utils.logger import Summary


# output of envs.env.Env.step()
# see runners.samplers.rollout_generator.RolloutGenerator
class Transition(object):
    def __init__(self, state: dict, reward: float, terminal: bool,
                 info: dict = None, summaries: List[Summary] = None):
        self.state = state
        self.reward = reward
        self.terminal = terminal
        self.info = info or {}
        self.summaries = summaries or []


# see runners.samplers.rollout_generator.RolloutGenerator
# see runners.samplers.sampler.Sampler._update()
# see utils.stat_accumulator
class ReplayTransition(object):
    def __init__(self, state: dict, action: np.ndarray,
                 reward: float, terminal: bool,
                 timeout: bool, error: bool, failure: bool, success: bool,
                 final_state: dict = None,
                 summaries: List[Summary] = None,
                 info: dict = None):
        self.state = state
        self.action = action
        self.reward = reward
        self.terminal = terminal
        self.timeout = timeout # episode ended upon reaching the maximum timesteps
        self.error = error # episode ended because of an error (e.g. IK problem)
        self.failure = failure # transition met the failure conditions of the environment
        self.success = success # transition met the success conditions of the environment
        # final only populated on last timestep
        self.final_state = final_state
        self.summaries = summaries or []
        self.info = info
        self.should_store = True # see rlfarm.runners.samplers.sync_sampler.SyncSampler.sample()
        self.should_log = True # see rlfarm.runners.samplers.sync_sampler.SyncSampler.sample()


# output of agents.agent.Agent.act()
# see runners.samplers.rollout_generator.RolloutGenerator
class ActResult(object):
    def __init__(self, action: Any,
                 state: dict = None,
                 replay_elements: dict = None,
                 info: dict = None):
        self.action = action
        self.state = state or {}
        self.replay_elements = replay_elements or {}
        self.info = info or {}