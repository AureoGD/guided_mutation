import numpy as np
from es_framework.optimizers.base_optimizer import BaseOptimizer


class CEMOptimizer(BaseOptimizer):

    def __init__(self, config: dict):
        """
        Cross-Entropy Method Optimizer (decoupled from policy).
        """
        super().__init__(config)

        # -----------------------------
        # Init distribution
        # -----------------------------
        self.mean = np.zeros(self.param_dim, dtype=np.float32)
        self.elites = np.zeros(self.param_dim, dtype=np.float32)
        elite_fraction = config.get('elite_frac', 0.2)
        initial_std_dev = config.get('sigma_init', 0.1)

        # Advanced config
        self.update_rule_type = config.get('update_rule', "standard")
        self.elite_weighting_type = config.get('weighting', "logarithmic")
        self.noise_decay_factor = config.get('sigma_decay', 0.995)
        self.min_std_dev = config.get('min_sigma', 1e-3)
        self.epsilon = config.get('extra_noise_scale', 0.01)

        # -----------------------------
        # State
        # -----------------------------
        self.num_elites = max(1, int(self.population_size * elite_fraction))
        self.std_devs = np.full(self.param_dim, initial_std_dev, dtype=np.float32)

        self.old_mean = np.copy(self.mean)

        self.last_population = None

    # ----------------------------------------
    @property
    def sigma(self):
        return np.mean(self.std_devs)

    # ----------------------------------------
    def sample(self) -> np.ndarray:
        """
        Generate population of candidate solutions
        """
        if self.update_rule_type == "cmaes_type":
            self.old_mean = np.copy(self.mean)

        noise = np.random.randn(self.population_size, self.param_dim).astype(np.float32)
        population = self.mean + (self.std_devs * noise)

        self.last_population = population

        return population

    # ----------------------------------------
    def update(self, fitness: np.ndarray):
        """
        Update distribution based on fitness
        """

        if self.last_population is None:
            raise RuntimeError("sample() must be called before update()")

        # -----------------------------
        # Sort by fitness (descending)
        # -----------------------------
        idx = np.argsort(fitness)[::-1]
        self.elites = self.last_population[idx[:self.num_elites]]

        best_idx = idx[0]
        self.best_fitness = fitness[best_idx]

        # -----------------------------
        # Compute weights
        # -----------------------------
        lambda_ = self._calculate_elite_weights()

        # -----------------------------
        # Update mean
        # -----------------------------
        self.mean = np.average(self.elites, axis=0, weights=lambda_)

        # -----------------------------
        # Update std
        # -----------------------------
        if self.update_rule_type == "cmaes_type":
            diffs = self.elites - self.old_mean
        else:
            diffs = self.elites - self.mean

        variances = np.average(np.square(diffs), axis=0, weights=lambda_)

        self.std_devs = np.sqrt(variances + self.epsilon)

        # Prevent collapse
        self.std_devs = np.maximum(self.std_devs, self.min_std_dev)

        # Decay noise
        self.epsilon *= self.noise_decay_factor
        self.epsilon = max(self.epsilon, 1e-5)

    # ----------------------------------------
    def _calculate_elite_weights(self) -> np.ndarray:

        if self.elite_weighting_type == "logarithmic":
            ranks = np.arange(1, self.num_elites + 1)
            raw = np.log(self.num_elites + 1) - np.log(ranks)

            if np.sum(raw) <= 0:
                return np.full(self.num_elites, 1.0 / self.num_elites, dtype=np.float32)

            return (raw / np.sum(raw)).astype(np.float32)

        else:
            return np.full(self.num_elites, 1.0 / self.num_elites, dtype=np.float32)

    # ----------------------------------------
    def boost_exploration(self, factor=1.5):
        old_sigma = np.mean(self.std_devs)

        self.std_devs = np.clip(self.std_devs * factor, a_min=None, a_max=2.0)
        self.epsilon *= factor

        print(f"[Optimizer] Exploration Boost: {old_sigma:.3f} -> {np.mean(self.std_devs):.3f}")

    def get_metrics(self):
        return {
            "sigma_mean": float(np.mean(self.std_devs)),
            "sigma_min": float(np.min(self.std_devs)),
            "sigma_max": float(np.max(self.std_devs)),
            "epsilon": float(self.epsilon),
            "best_fitness": float(self.best_fitness),
            "mean_shift": float(np.linalg.norm(self.mean - self.old_mean))
        }

    def replace_population(self, new_pop):
        self.last_population = np.array(new_pop)
