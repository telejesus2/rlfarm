import numpy as np

from rlfarm.buffers.replay.replay_buffer import ReplayBuffer
from rlfarm.utils.transition import ReplayTransition


def invalid_range(cursor, replay_capacity, stack_size, n_steps):
    """Returns a array with the indices of cursor-related invalid transitions.
    There are n_steps + stack_size invalid indices:
      - The n_steps indices before the cursor, because we do not have a
        valid N-step transition (including the next state).
      - The stack_size indices on or immediately after the cursor.
    If N = n_steps, K = stack_size, and the cursor is at c, invalid
    indices are:
      c - N, c - N + 1, ..., c, c + 1, ..., c + K - 1.
    It handles special cases in a circular buffer in the beginning and the end.
    Args:
      cursor: int, the position of the cursor.
      replay_capacity: int, the size of the replay memory.
      stack_size: int, the size of the stacks returned by the replay memory.
      n_steps: int, the agent's update horizon.
    Returns:
      np.array of size stack_size with the invalid indices.
    """
    assert cursor < replay_capacity
    return np.array(
        [(cursor - n_steps + i) % replay_capacity
         for i in range(stack_size + n_steps)])


def store_transition(transition: ReplayTransition, replay_buffer: ReplayBuffer):
    kwargs = dict(transition.state)
    replay_buffer.add(
        np.array(transition.action), transition.reward,
        transition.terminal,
        transition.timeout, kwargs)
    if transition.terminal:
        replay_buffer.add_final(transition.final_state)


def pack_episode(obs, acs, rews, extra_ob = {}, summaries = []):
    transitions = []
    # get the transitions
    ob = obs[0]
    for k in range(len(acs)):
        action = acs[k]
        terminal = (k == len(acs) - 1)
        reward = rews[k]
        others = {}
        others.update(extra_ob)
        others.update(ob)
        replay_transition = ReplayTransition(others, action, reward, terminal, False, False, False, False) # TODO everything set to False?
        if not terminal: transitions.append(replay_transition)
        ob = obs[k+1]  # set the next obs
    # final step
    replay_transition.final_state = ob
    replay_transition.summaries = summaries
    transitions.append(replay_transition)
    return transitions