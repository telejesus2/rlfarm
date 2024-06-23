import collections
import logging
import os
import signal
import time
from multiprocessing import Manager, Value
from threading import Thread
from typing import List, Optional

import torch

from rlfarm.utils.logger import Summary, ScalarSummary
from rlfarm.agents.agent import Agent
from rlfarm.buffers.replay.utils import store_transition
from rlfarm.envs.env import Env
from rlfarm.buffers.replay.replay_buffer import ReplayBuffer
from rlfarm.runners.samplers.rollout_generator import RolloutGenerator
from rlfarm.utils.stat_accumulator import StatAccumulator
from rlfarm.runners.samplers._async_sampler import _AsyncSampler


class AsyncSampler(object):
    def __init__(self,
                 seed: int,
                 env: Env,
                 agent: Agent,
                 replay_buffer: ReplayBuffer,
                 device: torch.device,
                 num_train_envs: int,
                 num_train_envs_gpu: int,
                 num_eval_envs: int,
                 episodes: int,
                 max_episode_len: int,
                 demo_ratio: int = 0,
                 demo_init_idx: int = 0,
                 stat_accumulator: Optional[StatAccumulator] = None,
                 rollout_generator: RolloutGenerator = None,
                 callback = None,
                 weightsdir: str = None,
                 max_fails: int = 10,
                 load_weights_freq: int = 1
    ):
        self._env = env
        self._replay_buffer = replay_buffer
        self._device = device
        self._max_episode_len = max_episode_len
        self._stat_accumulator = stat_accumulator
        self._rollout_generator = (
            RolloutGenerator() if rollout_generator is None
            else rollout_generator)
        self._seed = seed
        self._agent = agent
        self._num_train_envs = num_train_envs
        self._num_train_envs_gpu = num_train_envs_gpu
        self._num_eval_envs = num_eval_envs
        self._episodes = episodes
        self._weightsdir = weightsdir
        self._max_fails = max_fails
        self._load_weights_freq = load_weights_freq
        self._manager = Manager()

        # callback
        self._callback = callback
        self._env_summaries = self._manager.list()

        # multi-thread interactions with trainer
        self._p = None
        self._kill_signal = Value('b', 0)
        self.explore_signal = Value('b', 0)
        self._step_signal = Value('i', -1)
        self.target_replay_ratio = None  # Will get overridden later
        self.current_replay_ratio = Value('f', -1)

        # demos
        self._target_demo_ratio = demo_ratio
        self._demo_episodes_idx = Value('i', -1)
        self._demo_episodes_idx.value = demo_init_idx
        self._current_buffer_add_count = Value('i', -1)
        self._demo_transitions_count = Value('i', -1)

    def summaries(self, log_scalar_only=True) -> List[Summary]:
        summaries = []
        # env summaries
        if self._stat_accumulator is not None:
            self._env_summaries = self._stat_accumulator.peak()
            summaries.extend(self._env_summaries)
        # agent summaries
        if not log_scalar_only:
            summaries.extend(self._agent_summaries)
            self._should_log = True
        # demo summaries
        summaries.extend(self._get_demo_summaries())
        return summaries

    def _get_demo_summaries(self):
        return [
            ScalarSummary('replay/demo_reuse_ratio',
                self._demo_episodes_idx.value / max(1, self._env.num_demos_per_variation)),
            ScalarSummary('replay/demo_insert_ratio',
                self._demo_transitions_count.value / self._replay_buffer.add_count, to_print=True),
        ]

    #============================================================================================#
    # MAIN
    #============================================================================================#

    def _update(self):
        self._current_buffer_add_count.value = self._replay_buffer.add_count.item()

        # Move the stored transitions to the replay and accumulate statistics.
        new_transitions = collections.defaultdict(int)
        with self._internal_sampler.write_lock:
            if len(self._internal_sampler.act_summaries) > 0:
                self._agent_summaries = list(self._internal_sampler.act_summaries)
            if self._should_log:
                self._should_log = False
                self._internal_sampler.act_summaries[:] = []
            for name, transition, eval, demo in self._internal_sampler.stored_transitions:
                if not eval and transition.should_store:
                    store_transition(transition, self._replay_buffer)
                new_transitions[name] += 1
                if not demo and transition.should_log and self._stat_accumulator is not None:
                    self._stat_accumulator.step(transition, eval)
            self._internal_sampler.stored_transitions[:] = []  # Clear list
        return new_transitions

    def _run(self, save_load_lock):
        self._internal_sampler = _AsyncSampler(self._seed, self._device, self._manager,
            self._env, self._agent, self._episodes, self._max_episode_len,
            self._replay_buffer.history_len, self._callback,
            self._kill_signal, self._step_signal, self.explore_signal,
            self._rollout_generator, save_load_lock,
            self.current_replay_ratio, self.target_replay_ratio,
            self._target_demo_ratio, self._demo_episodes_idx, 
            self._current_buffer_add_count, self._demo_transitions_count,
            self._env_summaries, self._weightsdir, self._load_weights_freq)
        train_workers = self._internal_sampler.spin_up_workers(
            'train_env', self._num_train_envs, self._num_train_envs_gpu, False)
        eval_workers = self._internal_sampler.spin_up_workers(
            'eval_env', self._num_eval_envs, 0, True)
        workers = train_workers + eval_workers
        no_transitions = {w.name: 0 for w in workers}
        while True:
            for w in workers:
                if w.exitcode is not None:
                    workers.remove(w)
                    if w.exitcode != 0:
                        self._internal_sampler.w_failures[w.name] += 1
                        n_failures = self._internal_sampler.w_failures[w.name]
                        if n_failures > self._max_fails:
                            logging.error('Env %s failed too many times (%d times > %d)' %
                                          (w.name, n_failures, self._max_fails))
                            raise RuntimeError('Too many process failures.')
                        logging.warning('Env %s failed (%d times <= %d). restarting' %
                                        (w.name, n_failures, self._max_fails))
                        w = self._internal_sampler.restart_worker(w.name)
                        workers.append(w)

            if not self._kill_signal.value:
                new_transitions = self._update()
                for w in workers:
                    if new_transitions[w.name] == 0:
                        no_transitions[w.name] += 1
                    else:
                        no_transitions[w.name] = 0
                    if no_transitions[w.name] > 600:  # 5min
                        logging.warning("Env %s hangs, so restarting" % w.name)
                        workers.remove(w)
                        os.kill(w.pid, signal.SIGTERM)
                        w = self._internal_sampler.restart_worker(w.name)
                        workers.append(w)
                        no_transitions[w.name] = 0

            if len(workers) == 0:
                break
            time.sleep(1)

    #============================================================================================#
    # THREADING
    #============================================================================================#

    def start(self, save_load_lock):
        self._agent_summaries = []
        self._should_log = True
        # assumes buffer initialized with demos
        self._demo_transitions_count.value = self._replay_buffer.add_count.item()
        self._current_buffer_add_count.value = self._replay_buffer.add_count.item()
        self._p = Thread(target=self._run, args=(save_load_lock,), daemon=True)
        self._p.name = 'SamplerThread'
        self._p.start()

    def wait(self):
        if self._p.is_alive():
            self._p.join()

    def stop(self):
        if self._p.is_alive():
            self._kill_signal.value = True
            self._p.join()

    def set_step(self, step):
        self._step_signal.value = step