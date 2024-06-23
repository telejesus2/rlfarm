from rlfarm.agents.sacv1sail.sacv1sail import SACv1SAIL
from rlfarm.functions.q_function import make_q_network
from rlfarm.functions.policy import make_policy_network
from rlfarm.functions.value_function import make_value_network


def make_agent(action_space, action_min_max, state_shape, action_dim, config):

    # q
    q_config = config['agent']['q-function']
    q_net = make_q_network(
        q_config['class'], 
        q_config['encoder']['class'], q_config['encoder']['kwargs'] or {}, 
        q_config['network']['class'], q_config['network']['kwargs'] or {},
        state_shape, action_dim,
    )

    # v
    v_config = config['agent']['v-function']
    v_net = make_value_network(
        v_config['encoder']['class'], v_config['encoder']['kwargs'] or {}, 
        v_config['network']['class'], v_config['network']['kwargs'] or {},
        state_shape,
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

    return SACv1SAIL(
        action_space,
        action_min_max,
        pi_net,
        q_net,
        v_net,
        pi_config['optimizer']['class'],
        pi_config['optimizer']['kwargs'],
        q_config['optimizer']['class'],
        q_config['optimizer']['kwargs'],
        v_config['optimizer']['class'],
        v_config['optimizer']['kwargs'],
        **config['agent']['kwargs'],
    )