import os
import pickle
import argparse
import logging
import shutil
import random
import signal
import sys
from multiprocessing import Process, Value

import torch
import numpy as np

from rlfarm.agents.builder import make_agent
from rlfarm.buffers.replay.builder import make_buffer
from rlfarm.agents.preprocess_agent import PreprocessAgent
from rlfarm.envs.builder import make_env
from rlfarm.runners.builder import make_runners
from rlfarm.utils.logger import Logger, load_config


def run_seed(seed, config, logdir, kill_signal=None):
    main_config = config['general']

    if main_config['gpu_trainer'] is not None:
        device_trainer = torch.device("cuda:" + str(main_config['gpu_trainer'])
            if torch.cuda.is_available() else "cpu")
    else:
        device_trainer = torch.device("cpu")          
    print(device_trainer)

    if main_config['gpu_sampler'] is not None:
        device_sampler = torch.device("cuda:" + str(main_config['gpu_sampler'])
            if torch.cuda.is_available() else "cpu")
    else:
        device_sampler = torch.device("cpu")       
    print(device_sampler)

    replaydir = os.path.join(logdir, 'replay')
    weightsdir = os.path.join(logdir, 'weights')

    logger = Logger(
        logdir,
        save_tb=main_config['tensorboard_logging'],
        save_csv=main_config['csv_logging'],
        log_scalar_frequency=main_config['log_freq'],
        log_array_frequency=main_config['log_freq']*100,
        log_console_frequency=main_config['log_freq']*10,
        action_repeat=main_config['action_repeat'],
    )

    # environment
    env = make_env(config)

    # set seed
    env.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    #============================================================================================#
    # BUFFER
    #============================================================================================#

    explore_replay, demo_replay, action_min_max, demo_info, episode_callback = make_buffer(
        replaydir, env, config, device_trainer)
    if demo_replay is None:
        replays = [explore_replay]
        replay_split = [1]
    else:
        replays = [explore_replay, demo_replay]
        replay_split = [0.5, 0.5]

    #============================================================================================#
    # AGENT
    #============================================================================================#

    agent = PreprocessAgent(make_agent(env, action_min_max, config), demo_info)

    logging.info("Action min max: " + str(action_min_max))
    if action_min_max is not None:
        # needed if we want to run the agent again
        os.makedirs(logdir, exist_ok=True)
        with open(os.path.join(logdir, 'action_min_max.pkl'), 'wb') as f:
            pickle.dump(action_min_max, f)

    #============================================================================================#
    # RUNNERS
    #============================================================================================#

    trainer, sampler = make_runners(
        seed, env, agent, replays, explore_replay, episode_callback, logger, weightsdir,
        device_trainer, device_sampler, kill_signal, config
    )

    trainer.start()
    del trainer
    del sampler
    torch.cuda.empty_cache()


def main():
    logging.basicConfig(
        level=logging.INFO,
        # format='%(threadName)s | %(message)s')
        format='%(asctime)s | %(message)s')
        # format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./logs/config.yaml')
    args = parser.parse_args()

    config = load_config(args.config)
    main_config = config['general']
    logdir = os.path.join(main_config['logdir'], main_config['exp_name'])
    os.makedirs(logdir, exist_ok=True)

    synchronous_training = config['trainer']['class'] == 'sync'
    if not synchronous_training and main_config['gpu_sampler'] is not None:
        torch.multiprocessing.set_start_method('spawn')

    # make a copy of the config file
    tmp_config_file = os.path.join(logdir, 'tmp_config.yaml')
    shutil.copy(args.config, tmp_config_file)

    for _ in range(main_config['seeds']):
        seed = len(list(filter(lambda x: 'seed' in x, os.listdir(logdir)))) + 30 # TODO remove the offset
        logging.info('Starting seed %d.' % seed)
        seed_logdir = os.path.join(logdir, 'seed%d' % seed)
        os.makedirs(seed_logdir, exist_ok=True)

        # copy config file
        shutil.copy(tmp_config_file, os.path.join(seed_logdir, 'config.yaml'))

        if synchronous_training:
            kill_signal = Value('b', 0)
            p = Process(target=run_seed, args=(seed, config, seed_logdir, kill_signal))
            def _signal_handler(sig, frame):
                logging.info('SIGINT captured.')
                kill_signal.value = True
                sys.exit(0)
            signal.signal(signal.SIGINT, _signal_handler)
            p.start()
            p.join()
        else:
            run_seed(seed, config, seed_logdir)

if __name__ == '__main__':
    main()