import gymnasium as gym

from es_framework.core.env_spec import EnvSpec


def create_env():

    env = gym.make("LunarLander-v3")

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    env_spec = EnvSpec(obs_dim=obs_dim, act_dim=act_dim, is_discrete=True)

    return env, env_spec
