from typing import List

import numpy as np

from rlfarm.agents.agent import Agent
from rlfarm.buffers.replay.replay_buffer import ReplayBuffer
from rlfarm.buffers.replay.wrapped_replay_buffer import IterableReplayBuffer
from rlfarm.envs.env import Env
from rlfarm.runners.samplers.async_sampler import AsyncSampler
from rlfarm.runners.samplers.sync_sampler import SyncSampler
from rlfarm.runners.samplers.rollout_generator import RolloutGenerator
from rlfarm.runners.trainers.sync_trainer import SyncTrainer
from rlfarm.runners.trainers.async_trainer import AsyncTrainer
from rlfarm.utils.logger import Logger
from rlfarm.utils.stat_accumulator import MultiTaskAccumulator, SingleTaskAccumulator


def make_runners(
    seed: int, env: Env, agent: Agent, train_replays: List[ReplayBuffer], explore_replay: ReplayBuffer,
    callback, logger: Logger, weightsdir: str, device_trainer, device_sampler, kill_signal, config
):
    wrapped_replays = [IterableReplayBuffer(r) for r in train_replays]
    stat_accum = SingleTaskAccumulator(eval_video_fps=30) if env.num_variations == 1 else \
                 MultiTaskAccumulator(env.num_variations, eval_video_fps=30)

    trainer_config = config['trainer']
    buffer_config = config['buffer']
    demo_config = buffer_config['main_buffer']['demos']
    class_ = trainer_config['class']

    if class_ == 'sync':
        sampler = SyncSampler(
            env, explore_replay, device_sampler,
            max_episode_len=env.max_episode_steps,
            stat_accumulator=stat_accum,
            demo_ratio=demo_config['num_ratio'] if buffer_config['main_buffer']['use_demos'] else 0,
            demo_init_idx=demo_config['num_init_per_variation'],
            rollout_generator = RolloutGenerator(track_outputs = 
                'state_encoding' in [buffer_config[key]['class'] for key in buffer_config if 'episode_callback' in key]),
            callback=callback)

        trainer = SyncTrainer(
            agent, sampler, wrapped_replays, device_trainer, logger, kill_signal,
            weightsdir=weightsdir,
            **trainer_config['kwargs'])

    elif class_ == 'async':        
        sampler = AsyncSampler(
            seed, env, agent, explore_replay, device_sampler,
            max_episode_len=env.max_episode_steps,
            num_train_envs=trainer_config['num_train_envs'],
            num_train_envs_gpu=trainer_config['num_train_envs_gpu'],
            num_eval_envs=trainer_config['num_eval_envs'],
            episodes=trainer_config['episodes'] or np.iinfo(np.int64).max,
            stat_accumulator=stat_accum,
            weightsdir=weightsdir,
            demo_ratio=demo_config['num_ratio'] if buffer_config['main_buffer']['use_demos'] else 0,
            demo_init_idx=demo_config['num_init_per_variation'],
            load_weights_freq = trainer_config['load_weights_freq'],
            rollout_generator = RolloutGenerator(track_outputs = 
                'state_encoding' in [buffer_config[key]['class'] for key in buffer_config if 'episode_callback' in key]),
            callback=callback)

        trainer = AsyncTrainer(
            agent, sampler, wrapped_replays, device_trainer, logger,
            weightsdir=weightsdir,
            **trainer_config['kwargs'])

    elif class_ == 'ray':
        raise NotImplementedError()

        sampler = RaySampler(
            seed, env, agent, explore_replay, device_sampler,
            max_episode_len=env.max_episode_steps,
            num_train_envs=trainer_config['num_train_envs'],
            # num_train_envs_gpu=trainer_config['num_train_envs_gpu'],
            num_eval_envs=trainer_config['num_eval_envs'],
            episodes=trainer_config['episodes'] or np.iinfo(np.int64).max,
            stat_accumulator=stat_accum,
            weightsdir=weightsdir,
            demo_ratio=demo_config['num_ratio'] if buffer_config['main_buffer']['use_demos'] else 0,
            demo_init_idx=demo_config['num_init_per_variation'],
            load_weights_freq = trainer_config['load_weights_freq'],
            rollout_generator = RolloutGenerator(track_outputs = 
                'state_encoding' in [buffer_config[key]['class'] for key in buffer_config if 'episode_callback' in key]),
            callback=callback)

        trainer = AsyncTrainer(
            agent, sampler, wrapped_replays, device_trainer, logger,
            weightsdir=weightsdir,
            **trainer_config['kwargs'])

    return trainer, sampler