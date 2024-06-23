from rlfarm.agents.sac.builder import make_agent as make_sac_agent
from rlfarm.agents.sacfd.builder import make_agent as make_sacfd_agent
from rlfarm.agents.sacv1sail.builder import make_agent as make_sacv1sail_agent
from rlfarm.agents.sacae.builder import make_agent as make_sacae_agent
from rlfarm.agents.arm.builder import make_agent as make_arm_agent
from rlfarm.agents.ddpg.builder import make_agent as make_ddpg_agent


def make_agent(env, action_min_max, config): # returns agent, state_shape, state_dtype
    agent_class = config['agent']['class']
    action_space = env.action_space
    state_shape = env.state_shape
    # state_dtype = env.state_dtype
    action_dim = env.action_shape[0]

    if agent_class == 'sac':
        return make_sac_agent(action_space, action_min_max, state_shape, action_dim, config)
    if agent_class == 'sacfd':
        return make_sacfd_agent(action_space, action_min_max, state_shape, action_dim, config)
    if agent_class == 'sacv1-sail':
        return make_sacv1sail_agent(action_space, action_min_max, state_shape, action_dim, config)
    if agent_class == 'sacae':
        return make_sacae_agent(action_space, action_min_max, state_shape, action_dim, config)
    if agent_class == 'arm':
        return make_arm_agent(action_space, action_min_max, state_shape, action_dim, config)
    if agent_class == 'ddpg':
        return make_ddpg_agent(action_space, action_min_max, state_shape, action_dim, config)