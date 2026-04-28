from dataclasses import dataclass, field


@dataclass
class Species:

    id: int

    # core
    optimizer: any
    population: any = None

    # learning
    memory: any = None
    act_density: any = None

    # curriculum
    curriculum: any = None
    scenario_generator: any = None

    # estado
    best_fitness: float = -float("inf")

    # debug
    last_fitness: any = field(default=None)
