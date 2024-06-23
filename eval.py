import os
import pickle
import argparse
import logging
import time

import torch
import numpy as np

from rlfarm.agents.builder import make_agent
from rlfarm.agents.preprocess_agent import PreprocessAgent
from rlfarm.envs.rlbench.builder import make_env
from rlfarm.utils.logger import Logger, load_config


def eval(agent, env, device, num_episodes=1):
    returns = []
    lengths = []
    success = []
    for _ in range(num_episodes):
        episode_rewards, episode_success = _run_eval_episode(agent, env, device)
        returns.append(sum(episode_rewards))
        lengths.append(len(episode_rewards))
        success.append([episode_success])
    return returns, lengths, success


def _run_eval_episode(agent, env, device):
    state = env.reset()
    agent.reset()
    returns = []
    done = False
    t = 0
    while not done and t <= env.max_episode_steps - 1:
        prepped_data = {k: torch.tensor([v]).to(device) for k, v in state.items()}
        act_result = agent.act(None, prepped_data, deterministic=True)

        action = np.squeeze(act_result.action.cpu().detach().numpy(), 0)
        transition = env.step(action)
        rew = transition.reward
        done = transition.terminal
        state = dict(transition.state)

        time.sleep(0.1)
        returns.append(rew)
        t += 1
    return returns, transition.info.get("success", False)


def main():
    logging.basicConfig(
        level=logging.INFO,
        # format='%(threadName)s | %(message)s')
        format='%(asctime)s | %(message)s')
        # format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='lunar')
    parser.add_argument('--iter', '-i', type=int, default=20000)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--n_experiments', '-e', type=int, default=10)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()

    logdir = os.path.join(args.dir, 'seed%d' % args.seed)
    weightsdir = os.path.join(logdir, 'weights')
    config = load_config(os.path.join(logdir, 'config.yaml'))

    # environment
    config['environment']['kwargs']['headless'] = args.headless
    env = make_env(config, record=args.record)

    # agent
    with open(os.path.join(logdir, 'action_min_max.pkl'), 'rb') as f:
        action_min_max = pickle.load(f)
    agent = PreprocessAgent(make_agent(env, action_min_max, config), True)
    device = torch.device("cuda:0") if args.gpu else torch.device("cpu") 
    agent.build(training=False, device=device)

    # launch env
    env.launch(agent)

    # load weights
    try:
        assert os.path.exists(os.path.join(weightsdir, str(args.iter)))
    except AssertionError:
        weightsdir = os.path.join(weightsdir, 'old')
        assert os.path.exists(os.path.join(weightsdir, str(args.iter)))
    agent.load_weights(os.path.join(weightsdir, str(args.iter)))

    episode_returns, episode_lengths, episode_successes = eval(agent, env, device, args.n_experiments)
    print("---------------------------------------")
    print("Evaluation over %d episodes: average return is %f, average success is %f, average length is %f" %
        (args.n_experiments, np.mean(episode_returns), np.mean(episode_successes), np.mean(episode_lengths)))
    print("---------------------------------------")
    print("---------------------------------------")
    print("Evaluation over %d successful episodes: average return is %f, average success is %f, average length is %f" %
        (np.sum([1 for i in range(len(episode_successes)) if episode_successes[i][0]]), 
         np.mean([episode_returns[i] for i in range(len(episode_successes)) if episode_successes[i][0]]), 
         np.mean([episode_successes[i] for i in range(len(episode_successes)) if episode_successes[i][0]]), 
         np.mean([episode_lengths[i] for i in range(len(episode_successes)) if episode_successes[i][0]])))
    print("---------------------------------------")

    env.close()


if __name__ == '__main__':
    main()