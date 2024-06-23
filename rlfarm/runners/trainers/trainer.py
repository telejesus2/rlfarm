import logging
import os
import shutil
from typing import Optional, List, Union
import time

import numpy as np
import torch

from rlfarm.agents.agent import Agent
from rlfarm.utils.logger import Logger, ScalarSummary
from rlfarm.runners.samplers.sync_sampler import SyncSampler
from rlfarm.buffers.replay.wrapped_replay_buffer import IterableReplayBuffer
from rlfarm.buffers.replay.prioritized_replay_buffer import PrioritizedReplayBuffer
from rlfarm.agents.utils import PRIORITIES
from rlfarm.buffers.replay.const import INDICES

NUM_WEIGHTS_TO_KEEP = 50


class Trainer(object):
    def __init__(self,
                 agent: Agent,
                 sampler: SyncSampler,
                 wrapped_replay_buffer:  Union[IterableReplayBuffer, List[IterableReplayBuffer]],
                 device: torch.device,
                 logger: Logger,
                 iterations: int = int(1e6),
                 transitions_before_train: int = 1000,
                 iterations_before_sample: int = 0, # same as warmup_iterations, should be different? TODO
                 weightsdir: str = './weights',
                 save_freq: int = 100,
                 ):
        self._agent = agent
        self._sampler = sampler
        self._iterations = iterations
        self._transitions_before_train = transitions_before_train
        self._iterations_before_sample = iterations_before_sample
        self._device = device
        self._logger = logger
        self._save_freq = save_freq
        self._weightsdir = weightsdir
        os.makedirs(self._weightsdir, exist_ok=True)
        os.makedirs(os.path.join(self._weightsdir, 'old'), exist_ok=True)       
        self._wrapped_buffer = wrapped_replay_buffer if isinstance(
            wrapped_replay_buffer, list) else [wrapped_replay_buffer]

    def _save_model(self, i):
        d = os.path.join(self._weightsdir, str(i))
        os.makedirs(d, exist_ok=True)
        self._agent.save_weights(d)
        # Save some weights
        if i % (100 * self._save_freq) == 0:
            shutil.copytree(d, os.path.join(self._weightsdir, 'old', str(i)))
        # Remove oldest save
        prev_weight = str(i - self._save_freq * NUM_WEIGHTS_TO_KEEP)
        prev_dir = os.path.join(self._weightsdir, prev_weight)
        if os.path.exists(prev_dir):
            shutil.rmtree(prev_dir)

    def _process_batch(self, ds):
        """
        :param ds: list of dicts, each dict entry is a tensor or dict of tensors

        concatenates each tensor and sends them to device
        uses torch.ones(batch_size) as default value for a missing key, 
        i.e. 'sampling probabilites' if not all buffers are prioritized
        """
        out = {}
        keys = set().union(*ds)
        for k in keys:
            if any(isinstance(d.get(k, None), dict) for d in ds):
                out[k] = self._process_batch([d[k] for d in ds])
            else:
                tmp = [d[k] if k in d else torch.ones(self._wrapped_buffer[i].replay_buffer.batch_size)
                    for i, d in enumerate(ds)]
                out[k] = torch.cat(tmp, 0).to(self._device)
        return out

    def _step(self, i, sampled_batch):
        # i > 0 so that logger tracks all summaries during first iteration
        warmup = i > 0 and i < self._iterations_before_sample
        update_dict = self._agent.update(i, sampled_batch, warmup=warmup)
        acc_bs = 0
        for wb in self._wrapped_buffer:
            bs = wb.replay_buffer.batch_size
            if PRIORITIES in update_dict and isinstance(wb.replay_buffer, PrioritizedReplayBuffer):
                wb.replay_buffer.set_priority(
                    sampled_batch[INDICES][acc_bs:acc_bs+bs].cpu().detach().numpy(),
                    update_dict[PRIORITIES][acc_bs:acc_bs+bs])
            acc_bs += bs

    def _stop(self):
        logging.info('Stopping envs ...')
        self._sampler.stop()
        [r.replay_buffer.shutdown() for r in self._wrapped_buffer]

    def _get_add_counts(self):
        return np.array([
            r.replay_buffer.add_count for r in self._wrapped_buffer])

    def _get_sum_add_counts(self):
        return sum([
            r.replay_buffer.add_count for r in self._wrapped_buffer])

    def _get_sample_to_insert_ratio(self, i, batch_size, init_replay_size):
        size_used = batch_size * i
        size_added = self._get_sum_add_counts() - init_replay_size
        replay_ratio = size_used / (size_added + 1e-6)
        return replay_ratio

    def _log_iteration(self, i, batch_size, init_replay_size,
                       sample_time, step_time, t_start, process, num_cpu):
        log_scalar_only = not(i % self._logger.log_array_frequency == 0 and i > 0)

        replay_ratio = self._get_sample_to_insert_ratio(i, batch_size, init_replay_size)
        logging.info('Step %d. Sample time: %s. Step time: %s. Replay ratio: %s.' % (
                        i, sample_time, step_time, replay_ratio))
        agent_summaries = self._agent.update_summaries(log_scalar_only=log_scalar_only)
        env_summaries = self._sampler.summaries(log_scalar_only=log_scalar_only)
        self._logger.add_summaries(i, agent_summaries + env_summaries)

        train_summaries = [
            ScalarSummary('replay/sample_to_insert_ratio', replay_ratio, to_print=True),
            ScalarSummary('replay/update_to_insert_ratio',
                float(i) / float(self._get_sum_add_counts() - init_replay_size + 1e-6)),
            ScalarSummary('monitoring/total_time_in_min', (time.time() - t_start) / 60., to_print=True),
            ScalarSummary('monitoring/sample_time_per_item', sample_time / batch_size),
            ScalarSummary('monitoring/train_time_per_item', step_time / batch_size),
            ScalarSummary('monitoring/memory_gb', process.memory_info().rss * 1e-9),
            ScalarSummary('monitoring/cpu_percent', process.cpu_percent(interval=None) / num_cpu),
        ]
        for r_i, r in enumerate(self._wrapped_buffer):
            train_summaries.extend([
            ScalarSummary('replay%d/replay_ratio' % r_i,
                (r.replay_buffer.batch_size * i) / r.replay_buffer.add_count, to_print=True),
            ScalarSummary('replay%d/add_count' % r_i, r.replay_buffer.add_count),
            ScalarSummary('replay%d/size' % r_i,  r.replay_buffer.replay_capacity
                if r.replay_buffer.is_full() else r.replay_buffer.add_count),
            ])
        self._logger.add_summaries(i, train_summaries)