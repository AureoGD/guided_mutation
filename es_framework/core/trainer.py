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


FLAG_SIMULATE = 0
FLAG_REFINE = 1


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
                    tasks.append((task_id, ind_id, params, scenario, FLAG_SIMULATE, None, None))
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
                    elite_tasks.append((task_id, original_id, params, scenario, FLAG_SIMULATE, None, None))
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
            print("TRAIN V:")
            # result = self.value_function.train(self.memory)

            # if result is not None:
            #     train_loss, val_loss = result
            #     print(f"Train: {train_loss:.4f} | Val: {val_loss:.4f}")
            # else:
            #     print("V skipped (not enough data)")

            # ----------------------------------------
            # 3.3.1 EXPLORE ELITE WITH REFINER
            # ----------------------------------------
            print("PRE-REFINEMENT STATS (Elite in C'):")
            elite_c_prime_fitness = self._aggregate_fitness(elite_results)
            pre_refine_scores = [elite_c_prime_fitness[idx] for idx in elite_idx]
            print(
                f"PRE-REFINE  | Mean: {np.mean(pre_refine_scores):.2f} | Std: {np.std(pre_refine_scores):.2f} | Max: {np.max(pre_refine_scores):.2f}"
            )

            refine_tasks = []
            task_id = 0

            self.config["trained_v_model"] = self.value_function.model.state_dict()

            for i, params in enumerate(elite):
                original_id = elite_idx[i]
                idx_local = self.act_density.ids_map.get(original_id)
                history_list = list(self.act_density.history[idx_local]) if idx_local is not None else []
                memory = self.memory.sample(1000)
                refine_tasks.append((task_id, original_id, params, new_scenarios, FLAG_REFINE, history_list, memory))
                task_id += 1

            self.worker_manager.submit(refine_tasks)

            print(f"REFINING ELITE:")

            # reward_population = []
            # new_pop = copy.deepcopy(population)
            # updated_fitness = fitness.copy()
            # replaced_elite = 0
            # for result in tqdm(self.worker_manager.collect(len(refine_tasks)), total=len(refine_tasks)):
            #     ind_id = result["ind_id"]
            #     post_reward = result["mean_reward"]
            #     pre_reward = fitness[ind_id]
            #     reward_population.append(post_reward)

            #     if post_reward > pre_reward:
            #         new_pop[ind_id] = result["refined_params"]
            #         updated_fitness[ind_id] = post_reward
            #         replaced_elite += 1

            reward_population = []
            results_list = []

            # coleta resultados
            for result in tqdm(self.worker_manager.collect(len(refine_tasks)), total=len(refine_tasks)):
                ind_id = result["ind_id"]
                post_reward = result["mean_reward"]

                reward_population.append(post_reward)

                results_list.append({"ind_id": ind_id, "params": result["refined_params"], "fitness": post_reward})

            # ----------------------------------------
            # competição entre E e E'
            # ----------------------------------------

            candidates = []

            # elite original (E)
            for i, params in enumerate(elite):
                original_id = elite_idx[i]
                candidates.append({
                    "params": params,
                    "fitness": fitness[original_id],
                    "origin": "E",
                    "ind_id": original_id
                })

            # elite refinada (E')
            for res in results_list:
                candidates.append({
                    "params": res["params"],
                    "fitness": res["fitness"],
                    "origin": "E_prime",
                    "ind_id": res["ind_id"]
                })

            # ordena globalmente
            candidates.sort(key=lambda x: x["fitness"], reverse=True)

            # seleciona nova elite
            new_elite_candidates = candidates[:len(elite)]

            # ----------------------------------------
            # Atualiza população APENAS nos índices da elite
            # ----------------------------------------

            new_pop = copy.deepcopy(population)
            updated_fitness = fitness.copy()

            replaced_elite = 0

            for i, selected in enumerate(new_elite_candidates):
                target_idx = elite_idx[i]

                new_pop[target_idx] = selected["params"]
                updated_fitness[target_idx] = selected["fitness"]

                if selected["origin"] == "E_prime":
                    replaced_elite += 1

            print(
                f"POST-REFINE | Mean: {np.mean(reward_population):.2f} | Std: {np.std(reward_population):.2f} | Max: {np.max(reward_population):.2f}"
            )
            print(f"ELITE REFINEMENT: {replaced_elite} ELITE INDIVIDUAL REPLACED")
            num_from_E = sum(1 for c in new_elite_candidates if c["origin"] == "E")
            num_from_Ep = sum(1 for c in new_elite_candidates if c["origin"] == "E_prime")

            print(f"ELITE COMPOSITION - E: {num_from_E} | E': {num_from_Ep}")

            # ----------------------------------------
            # 3.4.1 POPULATION UPDATE (SIMPLE VERSION)
            # ----------------------------------------
            self.optimizer.replace_population(new_pop)
            self.optimizer.update(updated_fitness)
            population = self.optimizer.sample()

        # ----------------------------------------
        # 4. SHUTDOWN
        # ----------------------------------------
        self.worker_manager.shutdown()

        print("[TRAIN] Done.")
