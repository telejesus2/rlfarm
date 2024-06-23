import copy
import logging
import multiprocessing
import os
import time
from multiprocessing import Process

import numpy as np
import torch

from rlfarm.agents.agent import Agent
from rlfarm.envs.env import Env
from rlfarm.runners.samplers.rollout_generator import RolloutGenerator
from rlfarm.runners.samplers.sync_sampler import log_episode
from rlfarm.buffers.replay.utils import pack_episode
from rlfarm.buffers.replay.const import DEMO


class _AsyncSampler(object):
    def __init__(self,
                 seed: int,
                 device: torch.device,
                 manager: multiprocessing.Manager,
                 env: Env,
                 agent: Agent,
                 episodes: int,
                 max_episode_len: int,
                 history_len: int,
                 callback,
                 kill_signal,
                 step_signal,
                 explore_signal,
                 rollout_generator: RolloutGenerator(),
                 save_load_lock,
                 current_replay_ratio,
                 target_replay_ratio,
                 target_demo_ratio, 
                 demo_episodes_idx, 
                 current_buffer_add_count, 
                 demo_transitions_count,
                 env_summaries,
                 weightsdir: str = None,
                 load_weights_freq: int=1,
                 ):
        self._seed = seed
        self._device = device
        self._env = env
        self._agent = agent
        self._episodes = episodes
        self._max_episode_len = max_episode_len
        self._history_len = history_len
        self._callback = callback
        self._rollout_generator = rollout_generator
        self._weightsdir = weightsdir
        self._previous_loaded_weight_folder = ''
        self._load_weights_freq = load_weights_freq 

        # multi-process workers
        self.write_lock = manager.Lock()
        self.stored_transitions = manager.list()
        self.act_summaries = manager.list()
        self._kill_signal = kill_signal
        self._step_signal = step_signal
        self._explore_signal = explore_signal
        self._save_load_lock = save_load_lock
        self._current_replay_ratio = current_replay_ratio
        self._target_replay_ratio = target_replay_ratio
        self._w_args = {}
        self.w_failures = {}

        # demo
        self._target_demo_ratio = target_demo_ratio
        self._demo_episodes_idx = demo_episodes_idx
        self._current_buffer_add_count = current_buffer_add_count
        self._demo_transitions_count = demo_transitions_count

        # callback
        self._env_summaries = env_summaries

    def restart_worker(self, name: str):
        w = Process(target=self._run_env, args=self._w_args[name], name=name)
        w.start()
        return w

    def spin_up_workers(self, name: str, num_envs: int, num_envs_gpu: int, eval: bool):
        ws = []
        for i in range(num_envs):
            n = name + str(i)
            self._w_args[n] = (n, eval, i, i>=num_envs_gpu)
            self.w_failures[n] = 0
            w = Process(target=self._run_env, args=self._w_args[n], name=n)
            w.start()
            ws.append(w)
        return ws

    #============================================================================================#
    # CALLED INSIDE EACH WORKER
    #============================================================================================#

    def _load_model(self):
        if self._weightsdir is None:
            logging.info("'weightsdir' was None, so not loading weights.")
            return
        while True:
            weight_folders = []
            with self._save_load_lock:
                if os.path.exists(self._weightsdir):
                    weight_folders = os.listdir(self._weightsdir)
                    weight_folders.remove('old')
                if len(weight_folders) > 0:
                    weight_folders = sorted(map(int, weight_folders))
                    # Only load if there has been a new weight saving
                    if self._previous_loaded_weight_folder != weight_folders[-1]:
                        self._previous_loaded_weight_folder = weight_folders[-1]
                        d = os.path.join(self._weightsdir, str(weight_folders[-1]))
                        try:
                            self._agent.load_weights(d, training=False)
                        except FileNotFoundError:
                            # Rare case when agent hasn't finished writing.
                            time.sleep(1)
                            self._agent.load_weights(d, training=False)
                        logging.info('Agent %s: Loaded weights: %s' % (self._name, d))
                    break
            logging.info('Waiting for weights to become available.')
            time.sleep(1)

    def _run_env(self, name: str, eval: bool, worker_idx: int, cpu: bool):
        logging.basicConfig(
            level=logging.INFO,
            # format='%(threadName)s | %(message)s')
            format='%(asctime)s | %(message)s')
            # format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')

        if cpu: self._device = torch.device("cpu")
        self._name = name
        self._eval = eval
        self._agent = copy.deepcopy(self._agent)
        self._agent.build(training=False, device=self._device)

        # only for the first worker
        sample_demos = worker_idx == 0 and not eval and self._target_demo_ratio > 0
        log_trajectories = worker_idx == 0 and not eval # TODO eval should log as well
        traj_summaries, should_log_demo, should_log_ep = [], True, True
        eval_every = 20 if worker_idx == 0 else self._episodes

        logging.info('%s: Launching env in %s.' % (name, self._device))
        seed = self._seed + worker_idx
        # env.seed(seed)
        # torch.manual_seed(seed)
        # random.seed(seed)
        np.random.seed(seed)

        logging.info('Agent information:')
        logging.info(self._agent)

        env = self._env
        env.eval = eval
        env.launch(self._agent)
        for ep in range(self._episodes):       
            eval = True if ep > 0 and ep % eval_every == 0 else self._eval
            name = self._name.replace('train', 'eval') if eval else self._name
            env.eval = eval

            if ep % self._load_weights_freq == 0:
                logging.debug('Loading weights')
                self._load_model()
            logging.debug('%s: Starting episode %d.' % (name, ep))

            if sample_demos and not eval:
                while True:
                    current_demo_ratio = self._demo_transitions_count.value / self._current_buffer_add_count.value
                    if current_demo_ratio >= self._target_demo_ratio:
                        break

                    demo_rollout = []
                    demos = self._env.get_demos(
                        index = self._demo_episodes_idx.value % self._env.num_demos_per_variation)

                    # store the transitions
                    for i, demo in enumerate(demos):
                        obs, acs, rews, sums = self._env.extract_demo(demo)
                        transitions = pack_episode(obs, acs, rews, {DEMO: True}, sums)
                        if log_trajectories and should_log_demo:
                            log_episode(traj_summaries, transitions, 'sampler/demo/' + str(i))
                            if i == len(demos) - 1: should_log_demo = False
                        if self._callback is not None:
                            self._callback(transitions, env, self._agent, 
                                           self._env_summaries, self._step_signal.value, True)
                        demo_rollout += transitions
                    with self.write_lock:
                        for transition in demo_rollout:
                            self.stored_transitions.append((name, transition, eval, True))

                    self._demo_episodes_idx.value += 1
                    self._demo_transitions_count.value += len(demo_rollout)

            episode_rollout = []
            generator = self._rollout_generator.generator(self._device,
                env, self._agent, self._max_episode_len, self._history_len, eval,
                step_signal=self._step_signal, explore=self._explore_signal.value) 

            try:
                for replay_transition in generator:
                    while True:
                        if self._kill_signal.value:
                            env.close()
                            return
                        if (eval or self._target_replay_ratio is None or
                                self._step_signal.value <= 0 or (
                                        self._current_replay_ratio.value >
                                        self._target_replay_ratio)):
                            break
                        time.sleep(1)
                        logging.debug(
                            'Sampler. Waiting for replay_ratio %f to be more than %f' %
                            (self._current_replay_ratio.value, self._target_replay_ratio))

                    with self.write_lock:
                        if len(self.act_summaries) == 0:
                            # Only store new summaries if the previous ones
                            # have been popped by the main env runner.
                            for s in self._agent.act_summaries():
                                self.act_summaries.append(s)
                    episode_rollout.append(replay_transition)
            except StopIteration as e:
                continue
            except Exception as e:
                env.close()
                raise e

            if log_trajectories and should_log_ep and not eval:
                log_episode(traj_summaries, episode_rollout, 'sampler/episode')
                should_log_ep = False
            if self._callback is not None:
                self._callback(episode_rollout, env, self._agent, 
                               self._env_summaries, self._step_signal.value, False)

            with self.write_lock:
                for transition in episode_rollout:
                    self.stored_transitions.append((name, transition, eval, False))
                if log_trajectories and len(self.act_summaries) == 0 and len(traj_summaries) > 0:
                    self.act_summaries.extend(traj_summaries)
                    traj_summaries, should_log_demo, should_log_ep = [], True, True

        env.close()