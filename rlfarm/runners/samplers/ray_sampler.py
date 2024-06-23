import collections
import copy
import logging
import os
import signal
import time
import click
import random
from multiprocessing import Value, Manager, Process
from threading import Thread
from typing import List, Optional

import numpy as np
import torch
import ray

from rlfarm.agents.agent import Agent
from rlfarm.utils.logger import Summary, ScalarSummary
from rlfarm.envs.env import Env
from rlfarm.buffers.replay.replay_buffer import ReplayBuffer
from rlfarm.samplers.rollout_generator import RolloutGenerator
from rlfarm.utils.stat_accumulator import StatAccumulator


class RaySampler(object):
    def __init__(self,
                 seed: int,
                 env: Env,
                 agent: Agent,
                 replay_buffer: ReplayBuffer,
                 device: torch.device,
                 num_train_envs: int,
                 num_eval_envs: int,
                 episodes: int,
                 max_episode_len: int,
                 stat_accumulator: Optional[StatAccumulator] = None,
                 weightsdir: str = None):

        self._seed = seed
        self._num_train_envs = num_train_envs
        self._replay_buffer = replay_buffer
        self._device = device
        self._max_episode_len = max_episode_len
        self._stat_accumulator = stat_accumulator
        self._weightsdir = weightsdir

        self._kill_signal = Value('b', 0)
        self._step_signal = Value('i', -1)
        self._previous_loaded_weight_folder = ''

        if not ray.is_initialized():
            ray.init(log_to_driver=True, ignore_reinit_error=True,
                     address='auto', _redis_password='5241590000000000')
        # self._sampler_worker = ray.remote(SamplerWorker)
        self._sampler_worker = ray.remote(num_gpus=0.1)(SamplerWorker)
        self._agent = agent
        self._env = env
        self._workers = collections.defaultdict(None)
        self._workers_started = False

    def start_workers(self):
        if self._workers_started:
            return
        self._workers_started = True
        for worker_id in range(self._num_train_envs):
            self._workers[worker_id] = self._sampler_worker.remote(
                self._device, self._seed, self._max_episode_len,
                self._replay_buffer.history_len,  worker_id,
                self._env, self._agent)

    def shutdown_workers(self):
        """Shuts down the worker."""
        for w in self._workers.values():
            w.shutdown.remote()
        ray.shutdown()

    def _update_workers(self, agent_update, env_update):
        """
        :return: list[ray._raylet.ObjectID]: Remote values of worker ids.
        """
        updating_workers = []
        param_ids = [ray.put(agent_update) for _ in range(self._num_train_envs)]
        env_ids = [ray.put(env_update) for _ in range(self._num_train_envs)]
        for worker_id in range(self._num_train_envs):
            worker = self._workers[worker_id]
            updating_workers.append(
                worker.update.remote(param_ids[worker_id], env_ids[worker_id]))
        return updating_workers

    def obtain_samples(self, itr, num_samples, agent_update, env_update=None):
        active_workers = []
        completed_samples = 0
        batches = []

        # update the policy params of each worker before sampling
        # for the current iteration
        idle_worker_ids = []
        updating_workers = self._update_workers(agent_update, env_update)

        with click.progressbar(length=num_samples, label='Sampling') as pbar:
            while completed_samples < num_samples:
                # if there are workers still being updated, check
                # which ones are still updating and take the workers that
                # are done updating, and start collecting episodes on those
                # workers.
                if updating_workers:
                    updated, updating_workers = ray.wait(updating_workers,
                                                         num_returns=1,
                                                         timeout=0.1)
                    upd = [ray.get(up) for up in updated]
                    idle_worker_ids.extend(upd)

                # if there are idle workers, use them to collect episodes and
                # mark the newly busy workers as active
                while idle_worker_ids:
                    idle_worker_id = idle_worker_ids.pop()
                    worker = self._workers[idle_worker_id]
                    active_workers.append(worker.rollout.remote())

                # check which workers are done/not done collecting a sample
                # if any are done, send them to process the collected
                # episode if they are not, keep checking if they are done
                ready, not_ready = ray.wait(active_workers,
                                            num_returns=1,
                                            timeout=0.001)
                active_workers = not_ready
                for result in ready:
                    ready_worker_id, episode_batch = ray.get(result)
                    idle_worker_ids.append(ready_worker_id)
                    num_returned_samples = len(episode_batch)
                    completed_samples += num_returned_samples
                    batches.append((ready_worker_id, episode_batch))
                    pbar.update(num_returned_samples)

        return batches

    def _update(self):
        agent_update = self._load_model()

        samples = self.obtain_samples(self._step_signal.value, 1000, agent_update)

        # Move the stored transitions to the replay and accumulate statistics.
        new_transitions = [0 for w in self._workers]
        for w_id, batch in samples:
            name, eval = 'train_env', False
            for transition in batch:
                if not eval:
                    kwargs = dict(transition.state)
                    self._replay_buffer.add(
                        np.array(transition.action), transition.reward,
                        transition.terminal,
                        transition.timeout, kwargs)
                    if transition.terminal:
                        self._replay_buffer.add_final(transition.final_state)
                new_transitions[w_id] += 1
                self._stat_accumulator.step(transition, eval)
        return new_transitions

    def _load_model(self):
        if self._weightsdir is None:
            logging.info("'weightsdir' was None, so not loading weights.")
            return
        while True:
            weight_folders = []
            with self._save_load_lock:
                if os.path.exists(self._weightsdir):
                    weight_folders = os.listdir(self._weightsdir)
                if len(weight_folders) > 0:
                    weight_folders = sorted(map(int, weight_folders))
                    # Only load if there has been a new weight saving
                    out = None
                    if self._previous_loaded_weight_folder != weight_folders[-1]:
                        self._previous_loaded_weight_folder = weight_folders[-1]
                        d = os.path.join(self._weightsdir, str(weight_folders[-1]))
                        try:
                            out = torch.load(os.path.join(d, 'pi.pt'), map_location=self._device)
                        except FileNotFoundError:
                            # Rare case when agent hasn't finished writing.
                            time.sleep(1)
                            out = torch.load(os.path.join(d, 'pi.pt'), map_location=self._device)
                    return out
            logging.info('Waiting for weights to become available.')
            time.sleep(1)

    def _run(self, save_load_lock):
        self._save_load_lock = save_load_lock

        self.start_workers()

        no_transitions = [0 for w in self._workers]
        while True:
            if not self._kill_signal.value:
                new_transitions = self._update()
                for w_id in range(len(self._workers)):
                    if new_transitions[w_id] == 0:
                        no_transitions[w_id] += 1
                    else:
                        no_transitions[w_id] = 0
                    if no_transitions[w_id] > 600:  # 5min
                        raise RuntimeError("Env %s hangs, so restarting" % w_id)
            time.sleep(1)

    def summaries(self) -> List[Summary]:
        summaries = []
        summaries.extend(self._stat_accumulator.pop())
        return summaries

    def start(self, save_load_lock):
        self._p = Thread(target=self._run, args=(save_load_lock,), daemon=True)
        self._p.name = 'SamplerThread'
        self._p.start()

    def wait(self):
        if self._p.is_alive():
            self._p.join()

    def stop(self):
        self.shutdown_workers()
        if self._p.is_alive():
            self._kill_signal.value = True
            self._p.join()

    def set_step(self, step):
        self._step_signal.value = step


class SamplerWorker:
    def __init__(self,
                 device,
                 seed,
                 max_episode_len,
                 history_len,
                 worker_number,
                 env, agent):
        print("This actor is allowed to use GPUs {}.".format(ray.get_gpu_ids()))
        self.inner_worker = Worker(
            device, seed, max_episode_len, history_len, worker_number)
        self.worker_id = worker_number
        self.inner_worker.update_env(env)
        self.inner_worker.update_agent(agent)

    def update(self, agent_update, env_update=None):
        self.inner_worker.update_agent(agent_update)
        self.inner_worker.update_env(env_update)
        return self.worker_id

    def rollout(self):
        return (self.worker_id, self.inner_worker.rollout())

    def shutdown(self):
        self.inner_worker.shutdown()


class Worker(object):
    def __init__(self,
                 device,
                 seed,
                 max_episode_len,
                 history_len,
                 worker_number):
        self._device = device
        self._seed = seed
        self._max_episode_len = max_episode_len
        self._history_len = history_len
        self._worker_number = worker_number

        self._agent = None
        self._env = None
        self._rollout_generator = RolloutGenerator()
        self.worker_init()

    def worker_init(self):
        seed = self._seed + self._worker_number
        # self._env.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    def update_agent(self, agent_update):
        if isinstance(agent_update, (dict, tuple, np.ndarray)):
            # self._agent.load_weights(agent_update)
            self._agent._agent._policy_net.load_state_dict(agent_update)
        elif agent_update is not None:
            self._agent = copy.deepcopy(agent_update)
            self._agent.build(training=False, device=self._device)

    def update_env(self, env_update):
        if env_update is not None:
            if self._env is not None: self.shutdown()
            self._env = env_update
            self._env.launch()

    def rollout(self):
        episode_rollout = []
        generator = self._rollout_generator.generator(
            self._device, None, self._env, self._agent,
            self._max_episode_len, self._history_len, eval)
        for replay_transition in generator:
            episode_rollout.append(replay_transition)
        return episode_rollout

    def shutdown(self):
        self._env.close()