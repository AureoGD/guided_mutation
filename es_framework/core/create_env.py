def create_env(env_config):

    # -----------------------------
    # SIMPLE GYM MODE (DEBUG)
    # -----------------------------
    if isinstance(env_config, dict) and env_config.get("type") == "gym":

        import gymnasium as gym
        from es_framework.core.env_spec import EnvSpec

        env = gym.make(env_config["name"])

        obs, _ = env.reset()

        obs_dim = obs.shape[0]

        if hasattr(env.action_space, "n"):
            act_dim = env.action_space.n
            is_discrete = True
        else:
            act_dim = env.action_space.shape[0]
            is_discrete = False

        env_spec = EnvSpec(obs_dim=obs_dim, act_dim=act_dim, is_discrete=is_discrete)

        return env, env_spec
