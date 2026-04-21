from dataclasses import dataclass


@dataclass
class EnvSpec:
    obs_dim: int
    act_dim: int
    is_discrete: bool
