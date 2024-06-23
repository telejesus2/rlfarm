import copy
import os
import time
from typing import Optional, List, Union
from multiprocessing import Value

import psutil
import torch

from rlfarm.agents.agent import Agent
from rlfarm.utils.logger import Logger
from rlfarm.runners.samplers.sync_sampler import SyncSampler
from rlfarm.buffers.replay.wrapped_replay_buffer import IterableReplayBuffer
from rlfarm.runners.trainers.trainer import Trainer


class SyncTrainer(Trainer):
    def __init__(self,
                 agent: Agent,
                 sampler: SyncSampler,
                 wrapped_replay_buffer:  Union[IterableReplayBuffer, List[IterableReplayBuffer]],
                 device: torch.device,
                 logger: Logger,
                 kill_signal: Value,
                 iterations: int = int(1e6),
                 transitions_before_train: int = 1000,
                 iterations_before_sample: int = 0,
                 replay_ratio: int = 32,
                 weightsdir: str = './weights',
                 save_freq: int = 100,
    ):
        super(SyncTrainer, self).__init__(
            agent, sampler, wrapped_replay_buffer, device, logger,
            iterations, transitions_before_train, iterations_before_sample,
            weightsdir, save_freq)

        self._kill_signal = kill_signal

        batch_size = sum([r.replay_buffer.batch_size for r in self._wrapped_buffer])
        if batch_size % replay_ratio != 0 and replay_ratio % batch_size != 0:
            raise ValueError("batch_size should be a multiple of replay_ratio, or vice versa.")
        self._learning_freq = batch_size // replay_ratio
        self._sampling_freq = replay_ratio // batch_size

    def start(self):
        self._sampler.start()
        self._agent = copy.deepcopy(self._agent)
        self._agent.build(training=True, device=self._device)

        self._save_model(0)

        datasets = [r.dataset() for r in self._wrapped_buffer]
        datasets_iter = [iter(d) for d in datasets]

        init_replay_size = self._get_sum_add_counts().astype(float)
        batch_size = sum([r.replay_buffer.batch_size for r in self._wrapped_buffer])
        process = psutil.Process(os.getpid())
        num_cpu = psutil.cpu_count()

        # wait for buffers to have enough samples
        self._sampler.sample(0, self._agent, self._transitions_before_train, explore=True)
        assert self._get_sum_add_counts() - init_replay_size >= self._transitions_before_train

        t_start = time.time()
        for i in range(self._iterations):
            if self._kill_signal.value: break

            # should log for i=0 so that logger tracks all summaries during the first iteration
            log_iteration = i % self._logger.log_scalar_frequency == 0

            if log_iteration:
                process.cpu_percent(interval=None)

            # wait for sampler to collect enough samples
            if i >= self._iterations_before_sample:
                if self._learning_freq > 0:
                    self._sampler.sample(i, self._agent, self._learning_freq)
                elif i % self._sampling_freq == 0:
                    self._sampler.sample(i, self._agent, 1)

            t = time.time()
            batch = self._process_batch([next(di) for di in datasets_iter])
            sample_time = time.time() - t

            t = time.time()
            self._step(i, batch)
            step_time = time.time() - t

            if log_iteration:
                self._log_iteration(i, 
                    batch_size, init_replay_size, sample_time, step_time, t_start, process, num_cpu)

            self._logger.end_iteration(i)

            if i > 0 and i % self._save_freq == 0 and self._weightsdir is not None:
                self._save_model(i)

        if self._logger is not None:
            self._logger.close()

        self._stop()