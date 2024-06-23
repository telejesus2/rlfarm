from rlfarm.agents.sac.sac import SAC
from rlfarm.functions.q_function import make_q_network
from rlfarm.functions.policy import make_policy_network


def make_agent(action_space, action_min_max, state_shape, action_dim, config):
    # history_len = config['buffer']['main_buffer']['kwargs'].get('history_len', 1) # TODO should use

    # critic
    q_config = config['agent']['critic']
    q_net = make_q_network(
        q_config['class'], 
        q_config['encoder']['class'], q_config['encoder']['kwargs'] or {}, 
        q_config['network']['class'], q_config['network']['kwargs'] or {},
        state_shape, action_dim,
    )

    # actor
    pi_config = config['agent']['actor']
    pi_net = make_policy_network(
        pi_config['class'],
        pi_config['encoder']['class'], pi_config['encoder']['kwargs'] or {}, 
        pi_config['network']['class'], pi_config['network']['kwargs'] or {},
        state_shape, action_dim,
        action_space, action_min_max,
    )

    return SAC(
        action_space,
        action_min_max,
        pi_net,
        q_net,
        pi_config['optimizer']['class'],
        pi_config['optimizer']['kwargs'],
        q_config['optimizer']['class'],
        q_config['optimizer']['kwargs'],
        **config['agent']['kwargs'],
    )