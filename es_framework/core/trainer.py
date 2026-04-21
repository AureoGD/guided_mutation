import time
import json
import numpy as np
import random
import multiprocessing as mp

import torch
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

from es_framework.workers.worker_manager import WorkerManager
from es_framework.ml.memory import ReplayBuffer
from es_framework.ml.value_policy.value_function import ValueFunction
from es_framework.models.policy import Policy
from es_framework.ml.dqn_refiner import DQNRefiner

from es_framework.ml.action_density import ActionDensity
import copy


# ----------------------------------------
# JSON SERIALIZATION
# ----------------------------------------
def make_json_serializable(obj):

    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]

    if isinstance(obj, type):
        return obj.__name__

    if hasattr(obj, "__dict__"):
        return {"__class__": obj.__class__.__name__, **{k: make_json_serializable(v) for k, v in obj.__dict__.items()}}
    return str(obj)


# ----------------------------------------
# TRAINER
# ----------------------------------------
class Trainer:

    def __init__(self, config):

        self.config = config

        self.pop_size = config["pop_size"]
        self.species_size = config["species_size"]
        self.num_scenarios = config["num_scenarios"]
        self.max_generations = config["max_generations"]
        self.max_workers = config["max_workers"]
        self.elite_frac = config["elite_frac"]
        self.max_step = config["max_steps"]

        self.optimizer = self._build_optimizer(config)

        elite_size = int(self.elite_frac * self.pop_size)

        memory_size = (elite_size * self.num_scenarios * self.max_step)
        self.memory = ReplayBuffer(capacity=memory_size)

        n_action = self.config["env_spec"].act_dim
        self.act_density = ActionDensity(n_elite=elite_size,
                                         n_actions=n_action,
                                         n_task=self.num_scenarios,
                                         n_steps=self.num_scenarios)

        self.value_function = ValueFunction(config=self.config, device="cpu")
        self.policy = Policy(self.config["env_spec"], self.config["model_config"])

        self.refiner = DQNRefiner(env_fn=self.config["env_fn"],
                                  policy=self.policy,
                                  memory=self.memory,
                                  config=self.config)

    # ----------------------------------------
    def _build_optimizer(self, config):

        if config["optimizer_type"] == "CEM":
            from es_framework.optimizers.cem import CEMOptimizer
            return CEMOptimizer(config)

        elif config["optimizer_type"] == "CMAES":
            from es_framework.optimizers.cma_es import CMAESOptimizer
            return CMAESOptimizer(config)

        else:
            raise NotImplementedError

    # ----------------------------------------
    def _aggregate_fitness(self, results):

        # dicionário: ind_id → lista de rewards
        fitness_dict = {i: [] for i in range(self.pop_size)}

        for _, ind_id, reward, _, _ in results:
            fitness_dict[ind_id].append(reward)

        # média por indivíduo
        fitness = np.array([np.mean(fitness_dict[i]) if fitness_dict[i] else -1e5 for i in range(self.pop_size)])

        return fitness

    # ----------------------------------------
    def _select_elite(self, population, fitness):

        n_elite = max(1, int(len(population) * self.elite_frac))

        idx = np.argsort(fitness)[-n_elite:]

        elite = population[idx]
        elite_fitness = fitness[idx]

        return elite, elite_fitness, idx

    def _generate_scenarios(self):
        return [{"seed": i} for i in range(self.num_scenarios)]

    # ----------------------------------------
    def train(self):

        print("[TRAIN] Starting...")

        # ----------------------------------------
        # 1. CREATE WORKERS AND SAMPLE POPULATION
        # ----------------------------------------
        self.worker_manager = WorkerManager(num_workers=self.max_workers, config=self.config)

        time.sleep(2)

        population = self.optimizer.sample()

        # ----------------------------------------
        # 2. MAIN LOOP
        # ----------------------------------------
        for gen in range(self.max_generations):

            print("=" * 50)
            print(f"GENERATION {gen}".center(50))
            print("=" * 50)

            # ----------------------------------------
            # 2.1.1 GENERATE SCENARIOS (C)
            # ----------------------------------------
            scenarios = self._generate_scenarios()

            # ----------------------------------------
            # 2.1.2 BUILD TASKS AND RUN POPULATION
            # ----------------------------------------
            tasks = []
            task_id = 0

            for ind_id, params in enumerate(population):
                for scenario in scenarios:
                    tasks.append((task_id, ind_id, params, scenario, 0, None))
                    task_id += 1

            self.worker_manager.submit(tasks)
            results = []

            print("POPULATION RUNNING:")
            for result in tqdm(self.worker_manager.collect(len(tasks)), total=len(tasks)):
                results.append(result)

            # ----------------------------------------
            # 2.1.3 ELITE RANKING
            # ----------------------------------------
            fitness = self._aggregate_fitness(results)

            elite, elite_fitness, elite_idx = self._select_elite(population, fitness)

            print(f"Elite fitness: {elite_fitness}")
            # ----------------------------------------
            # 2.2.1 GENERATE NEW SCENARIOS (C')
            # ----------------------------------------
            new_scenarios = self._generate_scenarios()

            # ----------------------------------------
            # 2.2.2 BUILD ELITE TASKS AND RUN
            # ----------------------------------------
            elite_tasks = []
            task_id = 0

            for i, params in enumerate(elite):
                original_id = elite_idx[i]

                for scenario in new_scenarios:
                    elite_tasks.append((task_id, original_id, params, scenario, 0, None))
                    task_id += 1

            self.worker_manager.submit(elite_tasks)
            elite_results = []

            print("ELITE RUNNING:")
            for result in tqdm(self.worker_manager.collect(len(elite_tasks)), total=len(elite_tasks)):
                elite_results.append(result)

            # ----------------------------------------
            # 2.2.3 UPDATE MEMORY
            # ----------------------------------------
            self.act_density.reset_history()
            for _, ind_id, reward, success, trajectory in elite_results:
                self.act_density.attribute_id(ind_id)
                for i, (s, a, r, s_next) in enumerate(trajectory):
                    done = (i == len(trajectory) - 1)
                    self.memory.add(s, a, r, s_next, done)
                    self.act_density.add(ind_id, a)

            # ----------------------------------------
            # 2.2.4 TRAIN VALUE FUNCTION
            # ----------------------------------------

            result = self.value_function.train(self.memory)

            if result is not None:
                train_loss, val_loss = result
                print(f"[TRAIN] V | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
            else:
                print("[TRAIN] V skipped (not enough data)")

            # ----------------------------------------
            # 3.3.1 EXPLORE ELITE WITH REFINER
            # ----------------------------------------

            # refined = []

            # for params in elite:

            #     policy_copy = copy.deepcopy(self.policy)

            #     refiner = DQNRefiner(env_fn=self.config["env_fn"],
            #                          policy=policy_copy,
            #                          memory=self.memory,
            #                          config=self.config)

            #     for scneraio in new_scenarios:
            #         new_params = refiner.refine(params, scenario["seed"])
            #         params = new_params
            #     refined.append(new_params)

            # print(f"[TRAIN] Refined {len(refined)} individuals")

            # ----------------------------------------
            # 2.15 POPULATION UPDATE (SIMPLE VERSION)
            # ----------------------------------------
            self.optimizer.update(fitness)

            population = self.optimizer.sample()

        # ----------------------------------------
        # 3. SHUTDOWN
        # ----------------------------------------
        self.worker_manager.shutdown()

        print("[TRAIN] Done.")
