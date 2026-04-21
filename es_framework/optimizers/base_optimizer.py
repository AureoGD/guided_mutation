from abc import ABC, abstractmethod
import numpy as np


class BaseOptimizer(ABC):

    def __init__(self, config: dict):
        self.config = config
        self.elites = None
        self.best_fitness = 0
        self.param_dim = config["num_params"]
        self.population_size = config["pop_size"]

    # ----------------------------------------
    @abstractmethod
    def sample(self) -> np.ndarray:
        """Generate population of candidate solutions"""
        pass

    # ----------------------------------------
    @abstractmethod
    def update(self, fitness: np.ndarray):
        """Update internal distribution"""
        pass

    # ----------------------------------------
    def get_best_params(self) -> np.ndarray:
        return {
            "params": self.elites[0],
            "fitness":
                self.best_fitness  # se você armazenar
        }

    def get_metrics(self) -> dict:
        return {}
