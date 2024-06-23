import copy
import logging
import os
import signal
import sys
import threading
import time
from multiprocessing import Lock
from typing import Optional, List, Union

import numpy as np
import psutil
import torch

from rlfarm.agents.agent import Agent
from rlfarm.utils.logger import Logger
from rlfarm.runners.samplers.async_sampler import AsyncSampler
from rlfarm.buffers.replay.wrapped_replay_buffer import IterableReplayBuffer
from rlfarm.runners.trainers.trainer import Trainer


class AsyncTrainer(Trainer):
    def __init__(self,
                 agent: Agent,
                 sampler: AsyncSampler,
                 wrapped_replay_buffer:  Union[IterableReplayBuffer, List[IterableReplayBuffer]],
                 device: torch.device,
                 logger: Logger,
                 iterations: int = int(1e6),
                 transitions_before_train: int = 1000,
                 iterations_before_sample: int = 0,
                 replay_ratio_min_max: Optional[float] = None,
                 weightsdir: str = './weights',
                 save_freq: int = 100,
    ):
        super(AsyncTrainer, self).__init__(
            agent, sampler, wrapped_replay_buffer, device, logger,
            iterations, transitions_before_train, iterations_before_sample,
            weightsdir, save_freq)

        self._target_replay_ratio = None
        if replay_ratio_min_max is not None:
            if len(replay_ratio_min_max) != 2 or \
                   replay_ratio_min_max[0] < 0  or \
                   replay_ratio_min_max[0] > replay_ratio_min_max[1]:
                raise ValueError(
                    "replay_ratio_min_max must have two positive integers in order.")
            self._target_replay_ratio = replay_ratio_min_max[1]
            sampler.target_replay_ratio = replay_ratio_min_max[0]

    def _signal_handler(self, sig, frame):
        if threading.current_thread().name != 'MainThread':
            return
        logging.info('SIGINT captured. Shutting down. '
                     'This may take a few seconds.')
        self._stop()
        sys.exit(0)

    def start(self):
        signal.signal(signal.SIGINT, self._signal_handler)
        self._save_load_lock = Lock()
        self._sampler.start(self._save_load_lock)
        self._agent = copy.deepcopy(self._agent)
        self._agent.build(training=True, device=self._device)

        # save weights so workers can load
        if self._weightsdir is not None:
            with self._save_load_lock: self._save_model(0)

        datasets = [r.dataset() for r in self._wrapped_buffer]
        datasets_iter = [iter(d) for d in datasets]

        init_replay_size = self._get_sum_add_counts().astype(float)
        batch_size = sum([r.replay_buffer.batch_size for r in self._wrapped_buffer])
        process = psutil.Process(os.getpid())
        num_cpu = psutil.cpu_count()

        # wait for buffers to have enough samples
        self._sampler.explore_signal.value = True 
        while self._get_sum_add_counts() - init_replay_size < self._transitions_before_train:
            time.sleep(1)
            logging.info(
                'Waiting for %d samples before training. Currently have %d.' %
                (self._transitions_before_train, self._get_sum_add_counts() - init_replay_size))
        assert self._get_sum_add_counts() - init_replay_size >= self._transitions_before_train
        self._sampler.explore_signal.value = False
        self._sampler.current_replay_ratio.value = 0 # stop sampling for now

        t_start = time.time()
        for i in range(self._iterations):
            self._sampler.set_step(i)

            log_iteration = i % self._logger.log_scalar_frequency == 0# and i > 0

            if log_iteration:
                process.cpu_percent(interval=None)

            # wait for sampler to collect enough samples
            if i >= self._iterations_before_sample:
                if self._target_replay_ratio is not None:
                    while True:
                        replay_ratio = self._get_sample_to_insert_ratio(i, batch_size, init_replay_size)
                        self._sampler.current_replay_ratio.value = replay_ratio
                        if replay_ratio < self._target_replay_ratio:
                            break
                        time.sleep(1)
                        logging.info(
                            'Trainer. Waiting for replay_ratio %f to be less than %f.' %
                            (replay_ratio, self._target_replay_ratio))
                    del replay_ratio

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
                with self._save_load_lock: self._save_model(i)

        if self._logger is not None:
            self._logger.close()

        self._stop()