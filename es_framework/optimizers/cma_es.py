import numpy as np
from guided_mutation.es_framework.optimizers.base_optimizer import BaseOptimizer


class CMAESOptimizer(BaseOptimizer):

    def __init__(self, config: dict):
        super().__init__(config)

        # -----------------------------
        # Config
        # -----------------------------
        self.population_size = config.get("pop_size", 50)
        self.elite_fraction = config.get("elite_frac", 0.5)
        self.sigma = config.get("sigma_init", 0.5)

        self.param_dim = config["num_params"]
        self.mu = int(self.population_size * self.elite_fraction)

        # -----------------------------
        # State
        # -----------------------------
        self.mean = np.zeros(self.param_dim, dtype=np.float32)
        self.elites = np.zeros(self.param_dim, dtype=np.float32)

        self.C = np.identity(self.param_dim)
        self.B = np.identity(self.param_dim)
        self.D = np.ones(self.param_dim)
        self.inv_sqrt_C = np.identity(self.param_dim)

        self.p_sigma = np.zeros(self.param_dim)
        self.p_c = np.zeros(self.param_dim)

        # Weights
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mu_eff = 1.0 / np.sum(self.weights**2)

        # Learning rates
        self.c_sigma = (self.mu_eff + 2) / (self.param_dim + self.mu_eff + 5)
        self.d_sigma = 1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (self.param_dim + 1)) - 1) + self.c_sigma
        self.c_c = (4 + self.mu_eff / self.param_dim) / (self.param_dim + 4 + 2 * self.mu_eff / self.param_dim)
        self.c1 = 2 / ((self.param_dim + 1.3)**2 + self.mu_eff)
        self.c_mu = min(1 - self.c1, 2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((self.param_dim + 2)**2 + self.mu_eff))

        self.chi_N = np.sqrt(self.param_dim) * (1 - 1 / (4 * self.param_dim) + 1 / (21 * (self.param_dim**2)))

        self.eig_update_counter = 0

        self.last_population = None
        self.last_z = None

    # ----------------------------------------
    def sample(self) -> np.ndarray:
        population = []
        z_vectors = []

        for _ in range(self.population_size):
            z = np.random.randn(self.param_dim).astype(np.float32)
            z_vectors.append(z)

            y = self.B @ (self.D * z)
            x = self.mean + self.sigma * y

            population.append(x)

        self.last_population = np.array(population, dtype=np.float32)
        self.last_z = np.array(z_vectors, dtype=np.float32)

        return self.last_population

    # ----------------------------------------
    def update(self, fitness: np.ndarray):

        if self.last_population is None:
            raise RuntimeError("sample() must be called before update()")

        # Sort descending
        idx = np.argsort(fitness)[::-1]

        self.elites = self.last_population[idx[:self.mu]]
        best_idx = idx[0]
        self.best_fitness = fitness[best_idx]

        elite_z = self.last_z[idx[:self.mu]]

        # -----------------------------
        # Update mean
        # -----------------------------
        self.mean = np.sum(self.weights[:, None] * self.elites, axis=0)

        # -----------------------------
        # z mean
        # -----------------------------
        z_mean = np.sum(self.weights[:, None] * elite_z, axis=0)

        # -----------------------------
        # Update p_sigma
        # -----------------------------
        self.p_sigma = (1 - self.c_sigma) * self.p_sigma + \
            np.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mu_eff) * z_mean

        # -----------------------------
        # Update sigma
        # -----------------------------
        norm_ps = np.linalg.norm(self.p_sigma)
        self.sigma *= np.exp((self.c_sigma / self.d_sigma) * (norm_ps / self.chi_N - 1))

        # -----------------------------
        # Compute y_mean
        # -----------------------------
        y_mean = self.B @ (self.D * z_mean)

        # -----------------------------
        # Update p_c
        # -----------------------------
        threshold = 1.4 + 2 / (self.param_dim + 1)
        expected_norm = np.sqrt(1 - (1 - self.c_sigma)**2)
        expected_norm = max(expected_norm, 1e-8)

        h_sigma = int((norm_ps / expected_norm) < threshold)

        self.p_c = (1 - self.c_c) * self.p_c + \
            h_sigma * np.sqrt(self.c_c * (2 - self.c_c) * self.mu_eff) * y_mean

        # -----------------------------
        # Update covariance
        # -----------------------------
        C_mu = np.zeros((self.param_dim, self.param_dim))

        for i in range(self.mu):
            y_i = self.B @ (self.D * elite_z[i])
            C_mu += self.weights[i] * np.outer(y_i, y_i)

        self.C = (1 - self.c1 - self.c_mu) * self.C + \
            self.c1 * np.outer(self.p_c, self.p_c) + \
            self.c_mu * C_mu

        # -----------------------------
        # Eigen decomposition
        # -----------------------------
        self.eig_update_counter += 1

        if self.eig_update_counter % (self.param_dim // 10 + 1) == 0:
            self.C = (self.C + self.C.T) / 2
            self.D, self.B = np.linalg.eigh(self.C)
            self.D = np.sqrt(np.maximum(self.D, 1e-20))
            self.inv_sqrt_C = self.B @ np.diag(1. / self.D) @ self.B.T

    def get_metrics(self):
        condition_number = np.max(self.D)**2 / np.max([np.min(self.D)**2, 1e-20])

        return {
            "sigma": float(self.sigma),
            "condition_number": float(condition_number),
            "best_fitness": float(self.best_fitness),
            "mu_eff": float(self.mu_eff),
            "p_sigma_norm": float(np.linalg.norm(self.p_sigma))
        }
