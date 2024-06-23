from multiprocessing import Value

import numpy as np
import torch

from rlfarm.agents.agent import Agent
from rlfarm.envs.env import Env
from rlfarm.utils.transition import ReplayTransition


def take_one_step(
    state, state_history, step,
    device, env, agent, max_episode_len, history_len, eval, training_step,
    explore=False, track_outputs=False,
):
    prepped_data = {k: torch.tensor([v]).to(device) for k, v in state_history.items()}
    act_result = agent.act(
        training_step, prepped_data, deterministic=eval, explore=explore, track_outputs=track_outputs)

    # convert to np if not already
    extra_state = {k: np.array(v.cpu().detach().numpy()) for k, v in act_result.state.items()}
    extra_replay_elements = {k: np.array(v) for k, v in act_result.replay_elements.items()}

    action = np.squeeze(act_result.action.cpu().detach().numpy(), 0)
    transition = env.step(action)
    next_state = dict(transition.state)

    # if last transition, and not terminal, then we timed out
    timeout = False
    if step == max_episode_len - 1:
        timeout = not transition.terminal
        if timeout:
            transition.terminal = True
    
    # if last transition, add summaries
    if transition.terminal:
        transition.summaries += env.episode_summaries

    augmented_state = {}
    augmented_state.update(state)
    augmented_state.update(extra_state)
    augmented_state.update(extra_replay_elements)

    if history_len > 1:
        for k in state_history.keys():
            state_history[k].append(next_state[k])
            state_history[k].pop(0)
    else: 
        state_history = next_state

    transition.info["task_id"] = env.active_task_id 

    replay_transition = ReplayTransition(
        augmented_state, action, transition.reward,
        transition.terminal, timeout, transition.info.get("error", False),
        not transition.info.get("success", True), transition.info.get("success", False),
        summaries=transition.summaries,
        info=transition.info)

    if transition.terminal:
        # TODO (jesus)
        # We follow RLBench, which gives a terminal flag in case of success/failure.
        # If we switch towards MetaWorld the following assert has to change.
        assert sum([replay_transition.timeout, replay_transition.error,
                    replay_transition.failure, replay_transition.success]) == 1
        # if the agent gives us observations then we need to call act
        # one last time (i.e. acting in the terminal state).
        if len(extra_state) > 0:
            prepped_data = {k: torch.tensor([v]).to(device) for k, v in state_history.items()}
            act_result = agent.act(
                training_step, prepped_data, deterministic=eval, explore=explore, track_outputs=track_outputs)
            next_extra_state = {k: np.array(v) for k, v in act_result.state.items()}
            next_state.update(next_extra_state)
        replay_transition.final_state = next_state

    state = dict(transition.state)
    return state, state_history, replay_transition


class RolloutGenerator(object):
    def __init__(self, track_outputs=False):
        self._track_outputs = track_outputs

    def _get_type(self, x):
        if x.dtype == np.float64:
            return np.float32
        return x.dtype

    def generator(self, device: torch.device, env: Env, agent: Agent,
                  max_episode_len: int, history_len: int, eval: bool, 
                  step_signal: Value = None, step_signal_value: int = 0, explore: bool = False):
        training_step = step_signal.value if step_signal is not None else step_signal_value

        state = env.reset()
        agent.reset()
        if history_len > 1:
            state_history = {k: [np.array(v, dtype=self._get_type(v))] * history_len
                for k, v in state.items()}
        else:
            state_history = state

        for step in range(max_episode_len):
            state, state_history, replay_transition = take_one_step(
                state, state_history, step,
                device, env, agent, max_episode_len, history_len, eval, 
                training_step, explore, self._track_outputs)

            yield replay_transition

            if replay_transition.terminal:
                return