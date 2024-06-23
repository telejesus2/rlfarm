from rlfarm.agents.dqn.dqn import DQN
from rlfarm.functions.q_function import make_q_network


def make_agent(action_min_max, state_shape, action_dim, config):
    # critic
    q_config = config['agent']['critic']
    q_net = make_q_network(
        q_config['class'], 
        q_config['encoder']['class'], q_config['encoder']['kwargs'] or {}, 
        q_config['network']['class'], q_config['network']['kwargs'] or {},
        state_shape, action_dim,
    )

    # sac
    return DQN(
        action_min_max,
        q_net,
        q_config['optimizer']['class'],
        q_config['optimizer']['kwargs'],
        **config['agent']['kwargs'],
    )