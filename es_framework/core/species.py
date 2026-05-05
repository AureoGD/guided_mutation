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
    current_scenarios: any = None
    elite_scenarios: any = None

    # estate
    best_fitness: float = -float("inf")  # best seen so far across all generations, only updates when improved
    last_fitness: any = field(default=None)  #
    success_ratio: float = 0.0  # scallar of the succes ratio of the individual in the current generatio
    elite: any = field(default=None)  # array of elite parameter vectors (genomes) of the current generation
    elite_fitness: any = field(default=None)  # array of the elite fitness of the curret generation
    elite_fitness_combined: any = field(default=None)
    elite_idx: any = field(default=None)  # array of the elite index of the curret generation
