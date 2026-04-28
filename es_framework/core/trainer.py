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
from es_framework.models.checkpoint import ModelCheckpoint
from es_framework.logging.logger import TrainingLogger

from es_framework.ml.action_density import ActionDensity
from es_framework.core.species import Species
from es_framework.core.curriculum import CurriculumManager
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
        self.elite_frac = config["elite_frac"]
        self.num_scenarios = config["num_scenarios"]
        self.max_generations = config["max_generations"]
        self.max_workers = config["max_workers"]

        self.env_config = config["env_config"]
        self.model_config = config["model_config"]
        self.env_spec = config["env_spec"]

        self.elite_size = int(self.pop_size * self.elite_frac)

        # ----------------------------------------
        # SPECIES CONFIG
        # ----------------------------------------
        self.num_species = config.get("species_size", 1)
        self.memory_size = config.get("memory_size", 10000)
        self.top_k = self.config.get("memory_top_k", self.elite_size)
        self.random_k = self.config.get("memory_random_k", self.elite_size)

        # ----------------------------------------
        # SCENARIOS / CURRICULUM (GLOBAL por enquanto)
        # ----------------------------------------
        ScenarioClass = self.config["scenario_generator_class"]
        self.scenario_generator = ScenarioClass()

        self.curriculum = CurriculumManager(self.scenario_generator)

        # ----------------------------------------
        # RUN DIR
        # ----------------------------------------
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        task_name = self.config["job_name"]
        optimizer_name = self.config["optimizer_type"]

        self.run_dir = Path("experiments") / task_name / optimizer_name / timestamp
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # ----------------------------------------
        # LOGGER
        # ----------------------------------------
        self.logger = TrainingLogger(self.run_dir, num_species=self.num_species)

        # ----------------------------------------
        # CHECKPOINT
        # ----------------------------------------
        self.checkpoint = ModelCheckpoint(self.run_dir)

        self.checkpoint.save_model_spec(env_spec=self.env_spec, model_cfg=self.model_config)

        self.best_fitness = -np.inf

        # ----------------------------------------
        # SAVE CONFIG
        # ----------------------------------------
        config_path = self.run_dir / "config.json"
        config_to_save = make_json_serializable(self.config)

        with open(config_path, "w") as f:
            json.dump(config_to_save, f, indent=4)

        # ----------------------------------------
        # BUILD SPECIES
        # ----------------------------------------
        self.species_list = []

        for sp_id in range(self.num_species):

            optimizer = self._build_optimizer(self.config)

            memory = ReplayBuffer(capacity=self.memory_size)

            act_density = ActionDensity(n_elite=self.elite_size,
                                        n_actions=self.env_spec.act_dim,
                                        n_task=self.num_scenarios,
                                        n_steps=self.num_scenarios)

            species = Species(id=sp_id,
                              optimizer=optimizer,
                              memory=memory,
                              act_density=act_density,
                              curriculum=self.curriculum,
                              scenario_generator=self.scenario_generator)

            species.population = optimizer.sample()

            self.species_list.append(species)

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
    def _compute_success_ratio(self, results):
        successes = [success for _, _, _, _, success, _ in results]
        return np.mean(successes)

    # ----------------------------------------
    def _aggregate_fitness(self, results):

        fitness_dict = {}

        for _, _, ind_id, reward, _, _ in results:
            if ind_id not in fitness_dict:
                fitness_dict[ind_id] = []
            fitness_dict[ind_id].append(reward)

        fitness = np.array([np.mean(fitness_dict.get(i, [-1e5])) for i in range(self.pop_size)])

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

    def _train_v_function(self):
        print("TRAIN V:")
        result = self.value_function.train(self.memory)

        if result is not None:
            train_loss, val_loss = result
            print(f"Train: {train_loss:.4f} | Val: {val_loss:.4f}")
        else:
            print("V skipped (not enough data)")

    # ----------------------------------------
    def train(self):

        print("[TRAIN] Starting...")

        # ----------------------------------------
        # WORKERS
        # ----------------------------------------
        self.worker_manager = WorkerManager(num_workers=self.max_workers, config=self.config)
        time.sleep(2)

        # ----------------------------------------
        # MAIN LOOP
        # ----------------------------------------
        for gen in range(self.max_generations):

            gen_start = time.time()

            print("\n" + "=" * 50)
            print(f"{'Gen: ' + str(gen+1) + '/' + str(self.max_generations):^50}")
            print("=" * 50)

            # ----------------------------------------
            # SCENARIOS
            # ----------------------------------------
            scenarios = [self.scenario_generator.sample() for _ in range(self.num_scenarios)]

            # ----------------------------------------
            # 1) EVALUATE ALL SPECIES (C)
            # ----------------------------------------
            tasks = []
            task_id = 0

            for species in self.species_list:

                for ind_id, params in enumerate(species.population):

                    for scenario in scenarios:

                        tasks.append((task_id, species.id, ind_id, params, scenario, FLAG_SIMULATE, None, None))

                        task_id += 1

            self.worker_manager.submit(tasks)

            results = []
            for result in tqdm(self.worker_manager.collect(len(tasks)),
                               total=len(tasks),
                               desc="Evaluating population in C".ljust(35),
                               leave=False):
                results.append(result)

            # ----------------------------------------
            # SPLIT RESULTS BY SPECIES
            # ----------------------------------------
            results_by_species = {sp.id: [] for sp in self.species_list}

            for result in results:
                _, sp_id, ind_id, reward, success, trajectory = result
                results_by_species[sp_id].append(result)

            # ----------------------------------------
            # PROCESS EACH SPECIES
            # ----------------------------------------
            species_metrics = {}
            all_fitness = []
            success_all = []

            for species in self.species_list:

                optimizer = species.optimizer
                memory = species.memory
                act_density = species.act_density

                results_sp = results_by_species[species.id]

                # -------------------------
                # FITNESS
                # -------------------------
                fitness = self._aggregate_fitness(results_sp)
                species.last_fitness = fitness

                all_fitness.extend(fitness)

                success_ratio = self._compute_success_ratio(results_sp)
                success_all.append(success_ratio)

                # -------------------------
                # ELITE
                # -------------------------
                elite, elite_fitness, elite_idx = self._select_elite(species.population, fitness)

                # ----------------------------------------
                # SCENARIOS C'
                # ----------------------------------------
                new_scenarios = [self.scenario_generator.sample() for _ in range(self.num_scenarios)]
                all_scenarios = list({tuple(sorted(s.items())): s for s in (scenarios + new_scenarios)}.values())

                # -------------------------------
                # MEMORY UPDATE (TOP-K + RANDOM)
                # -------------------------------
                act_density.reset_history()

                sorted_idx = np.argsort(fitness)[::-1]

                top_k_idx = sorted_idx[:self.top_k]
                remaining_idx = sorted_idx[self.top_k:]

                if len(remaining_idx) > 0:
                    rand_k_idx = np.random.choice(remaining_idx,
                                                  size=min(self.random_k, len(remaining_idx)),
                                                  replace=False)
                else:
                    rand_k_idx = []

                selected_idx = set(top_k_idx).union(set(rand_k_idx))

                for _, _, ind_id, _, _, trajectory in results_sp:

                    if ind_id not in selected_idx:
                        continue

                    act_density.attribute_id(ind_id)

                    for i, (s, a, r, s_next) in enumerate(trajectory):

                        done = (i == len(trajectory) - 1)

                        memory.add(s, np.array([a, a]), r, s_next, done)

                        act_density.add(ind_id, a)

                # ----------------------------------------
                # REFINEMENT (E → E')
                # ----------------------------------------
                refine_tasks = []
                task_id = 0

                for i, params in enumerate(elite):

                    original_id = elite_idx[i]
                    idx_local = act_density.ids_map.get(original_id)

                    history_list = list(act_density.history[idx_local]) if idx_local is not None else []
                    memory_sample = memory.sample(1000)

                    refine_tasks.append((task_id, species.id, original_id, params, all_scenarios, FLAG_REFINE,
                                         history_list, memory_sample))

                    task_id += 1

                self.worker_manager.submit(refine_tasks)

                reward_population = []
                results_list = []
                metrics_list = []

                for result in tqdm(self.worker_manager.collect(len(refine_tasks)),
                                   total=len(refine_tasks),
                                   desc=f"Refine Species {species.id}".ljust(35),
                                   leave=False):
                    sp_id = result["species_id"]
                    ind_id = result["ind_id"]

                    post_reward = result["mean_reward"]

                    reward_population.append(post_reward)
                    metrics_list.append(result["train_metrics"])

                    results_list.append({"ind_id": ind_id, "params": result["refined_params"], "fitness": post_reward})

                # ----------------------------------------
                # SELECTION (E ∪ E')
                # ----------------------------------------
                candidates = []

                for i, params in enumerate(elite):
                    original_id = elite_idx[i]
                    candidates.append({"params": params, "fitness": fitness[original_id], "origin": "E"})

                for res in results_list:
                    candidates.append({"params": res["params"], "fitness": res["fitness"], "origin": "E'"})

                candidates.sort(key=lambda x: x["fitness"], reverse=True)
                new_elite_candidates = candidates[:len(elite)]

                new_pop = species.population.copy()
                updated_fitness = fitness.copy()

                replaced_elite = 0

                for i, selected in enumerate(new_elite_candidates):
                    target_idx = elite_idx[i]

                    new_pop[target_idx] = selected["params"]
                    updated_fitness[target_idx] = selected["fitness"]

                    if selected["origin"] == "E'":
                        replaced_elite += 1

                # ----------------------------------------
                # UPDATE OPTIMIZER
                # ----------------------------------------
                optimizer.replace_population(new_pop)
                optimizer.update(updated_fitness)

                species.population = optimizer.sample()

                # ----------------------------------------
                # CHECKPOINT (PER SPECIES)
                # ----------------------------------------
                fitness = species.last_fitness
                pop = species.population

                best_idx = np.argmax(fitness)
                best_params_sp = pop[best_idx]
                best_fit_sp = fitness[best_idx]

                # BEST of the specie
                self.checkpoint.save_best(params=best_params_sp, fitness=best_fit_sp, specie=species.id)

                # LAST of specie
                self.checkpoint.save_last(params=best_params_sp, specie=species.id)

                # PERIODIC save for specie
                self.checkpoint.save_periodic(params=best_params_sp,
                                              generation=gen,
                                              interval=self.config.get("checkpoint_freq", 10),
                                              specie=species.id)

                # ----------------------------------------
                # SPECIES METRICS
                # ----------------------------------------
                species_metrics[species.id] = {
                    "fitness_mean": float(np.mean(fitness)),
                    "fitness_std": float(np.std(fitness)),
                    "fitness_max": float(np.max(fitness)),
                    "fitness_min": float(np.min(fitness)),
                    "guided_replace_ratio": replaced_elite / len(elite),
                    "guided_mean_reward": float(np.mean(reward_population)),
                    "guided_std_reward": float(np.std(reward_population))
                }

            # ----------------------------------------
            # GLOBAL METRICS
            # ----------------------------------------
            all_fitness = np.array(all_fitness)
            success_ratio = float(np.mean(success_all))

            gen_time = time.time() - gen_start

            # ----------------------------------------
            # CHECKPOINT GLOBAL
            # ----------------------------------------
            best_value = -np.inf
            best_params = None

            for species in self.species_list:

                fitness = species.last_fitness
                pop = species.population

                idx = np.argmax(fitness)

                if fitness[idx] > best_value:
                    best_value = fitness[idx]
                    best_params = pop[idx]

            # BEST GLOBAL
            self.checkpoint.save_best(params=best_params, fitness=best_value)

            # LAST GLOBAL
            self.checkpoint.save_last(params=best_params)
            # ----------------------------------------
            # LOGGER
            # ----------------------------------------
            extra_metrics = {
                "global": {
                    "success_ratio": success_ratio,
                    "generation_time": gen_time
                },
                "species": species_metrics
            }

            self.logger.log_generation(generation=gen,
                                       fitness=all_fitness,
                                       population=None,
                                       optimizer_metrics={},
                                       extra_metrics=extra_metrics)

            print("-" * 50)
            print("\n")

        # ----------------------------------------
        # SHUTDOWN
        # ----------------------------------------
        self.worker_manager.shutdown()
        self.logger.close()

        print("[TRAIN] Done.")

    def _aggregate_metrics(self, metrics_list):
        keys = metrics_list[0].keys()
        agg = {}

        for k in keys:
            if k == "n_batches":
                continue

            values = np.array([m[k] for m in metrics_list])
            agg[k] = {"mean": values.mean(), "std": values.std()}

        return agg
