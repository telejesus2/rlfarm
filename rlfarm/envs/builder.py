def make_env(config):
    env_class = config['environment']['class']

    if env_class == 'rlbench':
        from rlfarm.envs.rlbench.builder import make_env
        return make_env(config)
    if env_class == 'metaworld':
        from rlfarm.envs.metaworld.builder import make_env
        return make_env(config)