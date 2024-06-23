from typing import List, Optional

import torch
import numpy as np

from rlfarm.envs.env import Env
from rlfarm.agents.agent import Agent
from rlfarm.buffers.replay.replay_buffer import ReplayBuffer
from rlfarm.runners.samplers.rollout_generator import RolloutGenerator
from rlfarm.utils.stat_accumulator import StatAccumulator

from rlfarm.utils.logger import Summary, ScalarSummary, ImageSummary
from rlfarm.buffers.replay.utils import store_transition, pack_episode
from rlfarm.buffers.replay.const import DEMO

def log_episode(summaries, episode, prefix):
    # assert len(episode[-1].summaries) == 1
    for s in episode[-1].summaries:
        s.name = prefix + "/" + s.name # TODO should not change the original name
        summaries.append(s)
    for k in episode[0].state:
        if 'rgb' in k:
            summaries.extend([
                ImageSummary('%s/%s/first' % (prefix, k), np.transpose(episode[0].state[k], (2, 0, 1))), 
                ImageSummary('%s/%s/last' % (prefix, k), np.transpose(episode[-1].state[k], (2, 0, 1))), 
            ])


class SyncSampler(object):
    def __init__(self,
                 env: Env,
                 replay_buffer: ReplayBuffer,
                 device: torch.device,
                 max_episode_len: int,
                 demo_ratio: int = 0,
                 demo_init_idx: int = 0,
                 stat_accumulator: Optional[StatAccumulator] = None,
                 rollout_generator = RolloutGenerator(),
                 callback = None
    ):
        self._env = env
        self._replay_buffer = replay_buffer
        self._device = device
        self._max_episode_len = max_episode_len
        self._stat_accumulator = stat_accumulator
        self._rollout_generator = (
            RolloutGenerator() if rollout_generator is None
            else rollout_generator)
        self._demo_episodes_idx = demo_init_idx
        self._target_demo_ratio = demo_ratio
        self._callback = callback
        self._demo_transitions_count = None
        self._env_summaries = []

    def start(self):
        self._env.launch()
        self._agent_summaries = []
        self._should_log_act, self._should_log_demo, self._should_log_ep = True, True, True
        self._eval = False
        self._episode, self._episode_cursor = [], 0
        # assumes buffer initialized with demos
        self._demo_transitions_count = self._replay_buffer.add_count.item()

    def stop(self):
        self._env.close()

    def summaries(self, log_scalar_only=True) -> List[Summary]:
        summaries = []
        # env summaries
        if self._stat_accumulator is not None:
            self._env_summaries = self._stat_accumulator.peak()
            summaries.extend(self._env_summaries)
        # agent summaries
        if not log_scalar_only:
            summaries.extend(self._agent_summaries)
            self._agent_summaries[:] = []
            self._should_log_act, self._should_log_demo, self._should_log_ep = True, True, True
        # demo summaries
        summaries.extend(self._get_demo_summaries())
        return summaries

    def sample(self, step: int, agent: Agent, num_samples: int, explore=False):
        for _ in range(num_samples):
            if self._episode_cursor == len(self._episode):
                generator = self._rollout_generator.generator(
                    self._device, self._env, agent, self._max_episode_len,
                    self._replay_buffer.history_len, self._eval, 
                    step_signal_value=step, explore=explore)
                self._episode, self._episode_cursor = [], 0
                for replay_transition in generator:
                    self._episode.append(replay_transition)

                if self._should_log_ep:
                    log_episode(self._agent_summaries, self._episode, 'sampler/episode')
                    self._should_log_ep = False  
                if self._callback is not None:
                    self._callback(
                        self._episode, self._env, agent, self._env_summaries, step, False)

                self._sample_demos(agent, step)

            transition = self._episode[self._episode_cursor]
            self._episode_cursor += 1
            if not self._eval and transition.should_store:
                store_transition(transition, self._replay_buffer)
            if transition.should_log and self._stat_accumulator is not None:
                self._stat_accumulator.step(transition, self._eval)

        # if len(self._agent_summaries) == 0:
        if self._should_log_act:
            for s in agent.act_summaries():
                self._agent_summaries.append(s)
            self._should_log_act = False

    #============================================================================================#
    # DEMOS
    #============================================================================================#

    def _get_demo_summaries(self):
        return [
            ScalarSummary('replay/demo_reuse_ratio',
                self._demo_episodes_idx / max(1, self._env.num_demos_per_variation)),
            ScalarSummary('replay/demo_insert_ratio',
                self._demo_transitions_count / self._replay_buffer.add_count, to_print=True),
        ]

    def _sample_demos(self, agent, step):
        if self._target_demo_ratio > 0:
            while True:
                current_demo_ratio = self._demo_transitions_count / self._replay_buffer.add_count
                if current_demo_ratio >= self._target_demo_ratio:
                    return

                demo_rollout = []
                demos = self._env.get_demos(
                    index = self._demo_episodes_idx % self._env.num_demos_per_variation)

                # store the transitions
                for i, demo in enumerate(demos):
                    obs, acs, rews, sums = self._env.extract_demo(demo)
                    transitions = pack_episode(obs, acs, rews, {DEMO: True}, sums)
                    if self._should_log_demo:
                        log_episode(self._agent_summaries, transitions, 'sampler/demo/' + str(i))
                        if i == len(demos) - 1: self._should_log_demo = False
                    if self._callback is not None:
                        self._callback(
                            transitions, self._env, agent, self._env_summaries, step, True)
                    demo_rollout += transitions
                for transition in demo_rollout:
                    if not self._eval and transition.should_store:
                        store_transition(transition, self._replay_buffer)
                
                self._demo_episodes_idx += 1
                self._demo_transitions_count += len(demo_rollout)