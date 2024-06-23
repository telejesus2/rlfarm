from tkinter import NW
from typing import List
from functools import partial
import logging
import copy

import torch
import numpy as np

from rlfarm.agents.agent import Agent
from rlfarm.utils.transition import ReplayTransition
from rlfarm.envs.env import Env
from rlfarm.networks.builder import make_network
from rlfarm.utils.logger import Summary


#============================================================================================#
# CALLBACK FUNCTIONS
#============================================================================================#


def example_callback(
    transitions: List[ReplayTransition], env: Env, agent: Agent, summaries: List[Summary],
    step: int, demo: bool):
    pass


def state_encoding_callback(
    transitions: List[ReplayTransition], env: Env, agent: Agent, summaries: List[Summary],
    step: int, demo: bool,
    # kwargs
    batch_size=32,
):
    last_transition = transitions[-1]
    assert last_transition.terminal

    if "encoded_state" in last_transition.state:
        for i, transition in enumerate(transitions):
            del transition.state["rgb_state"]
        del last_transition.final_state["rgb_state"]
    else:
        states = torch.cat(
            [torch.FloatTensor(transition.state["rgb_state"][None,...]) for transition in transitions] + 
            [torch.FloatTensor(last_transition.final_state["rgb_state"][None,...])], 0)
        states = (states / 255.) * 2.0 - 1.0
        encoded_states = []

        with torch.no_grad():
            for i in range(0, len(states), batch_size):
                encoded_states += [agent.encoder()(states[i:i+batch_size].to(agent._device)).detach()]
        encoded_states = torch.cat(encoded_states, 0)

        for i, transition in enumerate(transitions):
            transition.state["encoded_state"] = encoded_states[i].cpu().numpy()
            del transition.state["rgb_state"]
        last_transition.final_state["encoded_state"] = encoded_states[-1].cpu().numpy()
        del last_transition.final_state["rgb_state"]


def reward_relabeling_callback(
    transitions: List[ReplayTransition], env: Env, agent: Agent, summaries: List[Summary],
    step: int, demo: bool,
    # kwargs
    strategy="add", include_final=False, bonus=1, steps=0, steps_demo=0,
    schedule_steps=False, schedule_bonus=False,
):
    if steps == "demo": steps = env.average_demo_length or "all"
    if steps_demo == "demo": steps_demo = env.average_demo_length or "all"
    if steps == "all": steps = len(transitions) - 1
    if steps_demo == "all": steps_demo = len(transitions) - 1

    if schedule_steps or schedule_bonus:
        success_rate = 0
        for s in summaries:
            if s.name == "train_envs/success/mean/100":
                success_rate = s.value
                break
        if schedule_steps:
            steps = int(steps * (1 - success_rate))
            steps_demo = int(steps_demo * (1 - success_rate))
        if schedule_bonus:
            bonus = bonus * (1 - success_rate)

    # define bounds
    end = None if include_final else -1
    start = -(steps_demo + 1) if demo else -(steps + 1)
    
    last_transition = transitions[-1]
    assert last_transition.terminal
    info = last_transition.info if last_transition.info is not None else {}
    if not last_transition.timeout and not 'failure' in info:  
        if strategy == "add":
            for transition in transitions[start:end]:
                transition.reward += bonus
        elif strategy == "replace":
            for transition in transitions[start:end]:
                transition.reward = bonus


def goal_relabeling_callback(
    transitions: List[ReplayTransition], env: Env, agent: Agent, summaries: List[Summary],
    step: int, demo: bool,
    # kwargs
    discard_original_trajectory: bool = False,
):
    if demo: # demos should reach the intended goal so no need to relabel
        return

    last_transition = transitions[-1]
    assert last_transition.terminal

    print(last_transition.info.keys())
    if 'reached_goal' in last_transition.info:

        new_task_var = last_transition.info['reached_goal']
        if new_task_var in env._variation_list: # if the reached goal is one of the tasks

            new_transitions = []
            new_task_id = env._variation_list.index(new_task_var) # see rlfarm.envs.rlbench.rlbench_env.active_task_id()

            for transition in transitions:
                new_transition = copy.deepcopy(transition)

                new_transition.should_log = False # fake experience should not count in the logs
                new_transition.info["task_id"] = new_task_id 

                new_transitions.append(new_transition)

                if discard_original_trajectory:
                    transition.should_store = False

            # relabel rewards (TODO we assume sparse reward)    
            last_new_transition = new_transitions[-1]
            last_new_transition.reward = 1 * env._reward_scale

            transitions.extend(new_transitions)
            


#============================================================================================#
# HELPER CLASSES
#============================================================================================#


class _DummyAgent(object):
    def __init__(self, network, device):
        self._encoder = network.to(device)
        self._device = device

    def encoder(self):
        return self._encoder


class _Callback(object):
    def __init__(self, funcs):
        self._funcs = funcs

    def __call__(self, *args):
        return [func(*args) for func in self._funcs]


#============================================================================================#
# BUILDER
#============================================================================================#


def make_callback(config: dict, device: torch.device, state_shape: dict, state_dtype: dict,
                  should_init_agent=False):
    state_shape, state_dtype = state_shape.copy(), state_dtype.copy()

    callbacks, agent = [], None

    i = 1
    while "episode_callback_" + str(i) in config['buffer']:
        tmp_callback, tmp_agent, state_shape, state_dtype = make_one_callback(
            config, device, should_init_agent, state_shape, state_dtype, i)
        callbacks += [tmp_callback]
        if agent is None: agent = tmp_agent
        i += 1

    callback = _Callback(callbacks) if len(callbacks) > 0 else None

    return callback, agent, state_shape, state_dtype 


def make_one_callback(config: dict, device: torch.device, should_init_agent: bool,
                      state_shape: dict, state_dtype: dict, callback_idx
):
    callback_config = config['buffer']['episode_callback_' + str(callback_idx)]
    class_, kwargs = callback_config['class'], callback_config.get('kwargs', {})
    callback, agent = None, None
    logging.info('Callback found: %s' % class_)

    if class_ == 'reward_relabeling':
        assert kwargs['strategy'] in ["add", "replace"]
        assert kwargs['steps'] in ["all", "demo"] or kwargs['steps'] >= 0
        assert kwargs['steps_demo'] in ["all", "demo"] or kwargs['steps_demo'] >= 0
        callback = partial(reward_relabeling_callback, **kwargs)

    elif class_ == 'goal_relabeling':
        callback = partial(goal_relabeling_callback, **kwargs)
        if config['environment']['task']['kwargs']['reward'] != 'sparse':
            raise NotImplementedError()

    elif class_ == 'state_encoding': # assumes rgb_state exists
        # get default network config
        callback_net_path = callback_config['default_network']
        callback_net_config = config.copy()
        for key in callback_net_path.split('.'):
            callback_net_config = callback_net_config[key]
        # update state_shape for buffer
        del state_shape['rgb_state'], state_dtype['rgb_state']
        state_shape['encoded_state'] = (callback_net_config['kwargs']['output_dim'],)
        state_dtype['encoded_state'] = np.float32
        # get dummy agent
        if should_init_agent:
            net = make_network(callback_net_config['class'], callback_net_config['kwargs'])
            agent = _DummyAgent(net, device)
        callback = partial(state_encoding_callback, **kwargs)

    return callback, agent, state_shape, state_dtype