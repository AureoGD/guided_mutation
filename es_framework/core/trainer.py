import time
import json
import numpy as np
import random
import multiprocessing as mp

import torch
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

from guided_mutation.es_framework.workers.worker_manager import WorkerManager
from guided_mutation.es_framework.ml.memory import ReplayBuffer
from guided_mutation.es_framework.ml.value_policy.value_function import ValueFunction
from guided_mutation.es_framework.models.policy import Policy
from guided_mutation.es_framework.models.checkpoint import ModelCheckpoint
from guided_mutation.es_framework.logging.logger import TrainingLogger

from guided_mutation.es_framework.ml.action_density import ActionDensity
from guided_mutation.es_framework.core.species import Species
from guided_mutation.es_framework.core.curriculum import CurriculumManager
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

        self.elite_size = max(1, int(self.pop_size * self.elite_frac))

        max_sim_step = config.get("max_steps", 1000)

        # ----------------------------------------
        # SPECIES CONFIG
        # ----------------------------------------
        self.num_species = config.get("species_size", 1)

        self.mem_size = config.get("memory_size", 1e4)
        self.mem_elite_frac = config.get("mem_elite_ratio", 0.5)
        self.mem_alpha = config.get("mem_alpha", 0.3)

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

        ScenarioClass = self.config["scenario_generator_class"]

        self.memory_elite_size = int(self.mem_size * self.mem_elite_frac)

        for sp_id in range(self.num_species):

            scenario_generator = ScenarioClass()
            curriculum = CurriculumManager(scenario_generator)

            optimizer = self._build_optimizer(self.config)

            memory = [ReplayBuffer(capacity=self.memory_elite_size) for _ in range(self.elite_size)]

            act_density = [
                ActionDensity(n_actions=self.env_spec.act_dim, n_task=self.num_scenarios, n_steps=max_sim_step)
                for _ in range(self.elite_size)
            ]

            species = Species(id=sp_id,
                              optimizer=optimizer,
                              memory=memory,
                              act_density=act_density,
                              curriculum=curriculum,
                              scenario_generator=scenario_generator)

            species.population = optimizer.sample()
            self.species_list.append(species)

    # ----------------------------------------
    def _build_optimizer(self, config):

        if config["optimizer_type"] == "CEM":
            from guided_mutation.es_framework.optimizers.cem import CEMOptimizer
            return CEMOptimizer(config)

        elif config["optimizer_type"] == "CMAES":
            from guided_mutation.es_framework.optimizers.cma_es import CMAESOptimizer
            return CMAESOptimizer(config)

        else:
            raise NotImplementedError

    # ----------------------------------------
    def _compute_success_ratio(self, results):
        successes = []
        for result in results:
            successes.append(result["success_percent"])

        return np.mean(successes)

    # ----------------------------------------
    def _aggregate_fitness(self, results):

        fitness_dict = {}
        for result in results:
            ind_id = result["ind_id"]
            ind_reward = result["reward"]
            if ind_id not in fitness_dict:
                fitness_dict[ind_id] = []
            fitness_dict[ind_id].append(ind_reward)

        fitness = np.array([np.mean(fitness_dict.get(i, [-1e4])) for i in range(self.pop_size)])

        return fitness

    def _aggregate_fitness_elite(self, results, elite_idx):
        fitness_dict = {}
        for result in results:
            ind_id = result["ind_id"]
            ind_reward = result["reward"]
            if ind_id not in fitness_dict:
                fitness_dict[ind_id] = []
            fitness_dict[ind_id].append(ind_reward)

        return np.array([np.mean(fitness_dict.get(int(ind_id), [-1e4])) for ind_id in elite_idx])

    # ----------------------------------------
    def _select_elite(self, population, fitness):
        n_elite = max(1, int(self.elite_size))
        idx = np.argsort(fitness)[::-1][:n_elite]
        elite = population[idx]
        elite_fitness = fitness[idx]
        return elite, elite_fitness, idx

    # ----------------------------------------
    def _extract_steps(self, results, ind_id=None):
        steps = []
        for result in results:
            if result.get("status") == "error":
                continue
            if ind_id is not None and result["ind_id"] != ind_id:
                continue
            trajectory = result["extra"]["trajectory"]
            for t, (s, a, r, s_next, done) in enumerate(trajectory):
                steps.append((s, np.array([a, a]), r, s_next, done))
        return steps

    # ----------------------------------------
    def _sample_steps(self, steps, n):
        if len(steps) <= n:
            return steps
        indices = np.random.choice(len(steps), n, replace=False)
        return [steps[j] for j in indices]

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
            # 1) EVALUATE ALL SPECIES (C)
            # ----------------------------------------
            tasks = []
            task_id = 0

            for species in self.species_list:

                # ----------------------------------------
                # SCENARIOS
                # ----------------------------------------
                scenarios = [species.scenario_generator.sample() for _ in range(self.num_scenarios)]
                species.current_scenarios = scenarios

                for ind_id, params in enumerate(species.population):

                    for scenario in scenarios:

                        tasks.append({
                            "task_id": task_id,
                            "species_id": species.id,
                            "ind_id": ind_id,
                            "params": params,
                            "scenario_data": scenario,
                            "flag": FLAG_SIMULATE,
                            "extra": None
                        })

                        task_id += 1

            self.worker_manager.submit(tasks)

            results = []
            for result in tqdm(self.worker_manager.collect(len(tasks)),
                               total=len(tasks),
                               desc="Evaluating population in C".ljust(35),
                               leave=True):
                results.append(result)

            # ----------------------------------------
            # SPLIT C RESULTS BY SPECIES
            # ----------------------------------------
            results_by_species = {sp.id: [] for sp in self.species_list}

            for r in results:
                results_by_species[r["species_id"]].append(r)

            # ----------------------------------------
            # FIND ELITE FROM EACH SPECIES
            # ----------------------------------------

            for species in self.species_list:
                results_sp = results_by_species[species.id]
                species.last_fitness = self._aggregate_fitness(results_sp)
                species.success_ratio = self._compute_success_ratio(results_sp)

                elite, elite_fitness, elite_idx = self._select_elite(species.population, species.last_fitness)

                species.elite = elite
                species.elite_fitness = elite_fitness
                species.elite_idx = elite_idx

            # ----------------------------------------
            # ELITE TASK IN C'
            # ----------------------------------------

            elite_tasks = []
            task_id = 0

            for species in self.species_list:
                scenarios_c_prime = [species.scenario_generator.sample() for _ in range(self.num_scenarios)]
                species.elite_scenarios = scenarios_c_prime

                for i, params in enumerate(species.elite):
                    ind_id = int(species.elite_idx[i])

                    for scenario in scenarios_c_prime:
                        elite_tasks.append({
                            "task_id": task_id,
                            "species_id": species.id,
                            "ind_id": ind_id,
                            "params": params,
                            "scenario_data": scenario,
                            "flag": FLAG_SIMULATE,
                            "extra": None
                        })
                        task_id += 1

            self.worker_manager.submit(elite_tasks)

            elite_results = []

            for result in tqdm(self.worker_manager.collect(len(elite_tasks)),
                               total=len(elite_tasks),
                               desc="Evaluating elite in C'".ljust(35),
                               leave=True):
                elite_results.append(result)

            # ----------------------------------------
            # SPLIT C' RESULTS BY SPECIES
            # ----------------------------------------
            elite_results_by_species = {sp.id: [] for sp in self.species_list}

            for r in elite_results:
                elite_results_by_species[r["species_id"]].append(r)

            # ----------------------------------------
            # POPULATE ELITE MEMORIES
            # ----------------------------------------

            n_self = int((1.0 - self.mem_alpha) * self.memory_elite_size)
            n_other_elite = int((self.mem_alpha / 2.0) * self.memory_elite_size)
            n_random_pop = int((self.mem_alpha / 2.0) * self.memory_elite_size)

            for species in self.species_list:

                all_results_sp = results_by_species[species.id] + elite_results_by_species[species.id]

                elite_ids = set(species.elite_idx.tolist())
                non_elite_results_sp = [r for r in results_by_species[species.id] if r["ind_id"] not in elite_ids]
                non_elite_steps = self._extract_steps(non_elite_results_sp)

                for rank, ind_id in enumerate(species.elite_idx):
                    ind_id = int(ind_id)

                    # 1) Self
                    own_steps = self._sample_steps(self._extract_steps(all_results_sp, ind_id), n_self)
                    species.act_density[rank].reset_history()
                    for (s, a_arr, r, s_next, done) in own_steps:
                        species.act_density[rank].add(a_arr[0])
                        species.memory[rank].add(s, a_arr, r, s_next, done)

                    # 2) Other elites
                    other_ranks = [r for r in range(len(species.elite_idx)) if r != rank]
                    if len(other_ranks) > 0:
                        n_per_other = max(1, n_other_elite // len(other_ranks))
                        for other_rank in other_ranks:
                            other_mem = species.memory[other_rank]
                            if len(other_mem) == 0:
                                continue
                            s_b, a_b, r_b, sn_b, d_b = other_mem.sample(min(n_per_other, len(other_mem)))
                            for j in range(len(s_b)):
                                species.memory[rank].add(s_b[j], a_b[j], r_b[j], sn_b[j], d_b[j])

                    # 3) Random non-elite
                    for (s, a_arr, r, s_next, done) in self._sample_steps(non_elite_steps, n_random_pop):
                        species.memory[rank].add(s, a_arr, r, s_next, done)

                elite_results_sp = [r for r in all_results_sp if r["ind_id"] in elite_ids]
                species.elite_fitness_combined = self._aggregate_fitness_elite(elite_results_sp, species.elite_idx)

            # ----------------------------------------
            # CREAT THE MUTATION TASK
            # ----------------------------------------
            mutation_tasks = []
            task_id = 0

            for species in self.species_list:
                all_scenarios = species.current_scenarios + species.elite_scenarios

                for i, params in enumerate(species.elite):
                    ind_id = int(species.elite_idx[i])
                    extra = {
                        "memory": species.memory[i],
                        "act_density": species.act_density[i].history.copy(),
                        "st_bt": self.config.get("st_bt_flag", False)
                    }
                    mutation_tasks.append({
                        "task_id": task_id,
                        "species_id": species.id,
                        "ind_id": ind_id,
                        "params": params,
                        "scenario_data": all_scenarios,
                        "flag": FLAG_REFINE,
                        "extra": extra
                    })
                    task_id += 1

            self.worker_manager.submit(mutation_tasks)

            mutation_results = []

            for result in tqdm(self.worker_manager.collect(len(mutation_tasks)),
                               total=len(mutation_tasks),
                               desc="Guided mutation in E".ljust(35),
                               leave=True):
                mutation_results.append(result)

            species_metrics = {}

            for species in self.species_list:
                replaced_elite = 0
                refined_lookup = {}
                for r in mutation_results:
                    if r["species_id"] == species.id:
                        refined_lookup[r["ind_id"]] = r

                elite_original_fitness = []
                elite_muted_fitness = []
                for i, params in enumerate(species.elite):
                    ind_id = int(species.elite_idx[i])
                    original_fitness = species.elite_fitness_combined[i]

                    elite_original_fitness.append(original_fitness)

                    refined = refined_lookup.get(ind_id)

                    if refined is None or refined["status"] == "timeout_error":
                        continue

                    if refined["reward"] > original_fitness:
                        species.elite[i] = refined["extra"]["refined_params"]
                        species.elite_fitness[i] = refined["reward"]
                        replaced_elite += 1

                        elite_muted_fitness.append(refined["reward"])

                new_pop = species.population.copy()
                updated_fitness = species.last_fitness.copy()

                for i in range(len(species.elite)):
                    ind_id = int(species.elite_idx[i])
                    new_pop[ind_id] = species.elite[i]
                    updated_fitness[ind_id] = species.elite_fitness[i]

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
                    "fitness_mean": float(np.mean(species.last_fitness)),
                    "fitness_std": float(np.std(species.last_fitness)),
                    "fitness_max": float(np.max(species.last_fitness)),
                    "fitness_min": float(np.min(species.last_fitness)),
                    "guided_replace_ratio": replaced_elite / len(species.elite),
                    "elite_o_mean_reward": float(np.mean(elite_original_fitness)),
                    "elite_o_std_reward": float(np.std(elite_original_fitness)),
                    "elite_m_mean_reward": float(np.mean(elite_muted_fitness)) if elite_muted_fitness else 0.0,
                    "elite_m_std_reward": float(np.std(elite_muted_fitness)) if elite_muted_fitness else 0.0,
                    "success_ratio": float(species.success_ratio),
                    "curriculum_difficulty": species.scenario_generator.get_current_difficult()
                }

                # ----------------------------------------
                # SPECIES UPDATE
                # ----------------------------------------

                species.optimizer.replace_population(new_pop)
                species.optimizer.update(updated_fitness)
                species.population = species.optimizer.sample()

                # curriculum update
                species.curriculum.update(species.success_ratio)

            # ----------------------------------------
            # GLOBAL METRICS
            # ----------------------------------------
            # all_fitness = np.array(all_fitness)
            # success_ratio = float(np.mean(success_all))

            gen_time = time.time() - gen_start

            # ----------------------------------------
            # CHECKPOINT GLOBAL
            # ----------------------------------------
            best_value = -np.inf
            best_params = None

            all_fitness = []
            all_success = []

            for species in self.species_list:

                all_fitness.extend(species.last_fitness)  # extend, not append
                all_success.append(species.success_ratio)

                fitness = species.last_fitness
                pop = species.population

                idx = np.argmax(fitness)

                if fitness[idx] > best_value:
                    best_value = fitness[idx]
                    best_params = pop[idx]

            all_fitness = np.array(all_fitness)
            success_ratio = float(np.mean(all_success))

            self.checkpoint.save_best(params=best_params, fitness=best_value)

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

            self.logger.log_generation(
                generation=gen,
                fitness=all_fitness,  # full array
                population=None,
                optimizer_metrics={},
                extra_metrics=extra_metrics)
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
