import json
import time
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
from pathlib import Path

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


class TrainingLogger:

    def __init__(self, run_dir: str, num_species: int = 1, top_k_species: int = 3):

        self.run_dir = Path(run_dir)

        # -----------------------------
        # Species config
        # -----------------------------
        self.num_species = num_species
        self.top_k_species = top_k_species

        # -----------------------------
        # Directories
        # -----------------------------
        self.log_dir = self.run_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.tb_dir = self.log_dir / "tb"
        self.tb_dir.mkdir(parents=True, exist_ok=True)

        # -----------------------------
        # Timing
        # -----------------------------
        self.start_time = time.time()
        self.last_gen_time = self.start_time

        # -----------------------------
        # TensorBoard
        # -----------------------------
        self.tb_writer = None
        if SummaryWriter:
            self.tb_writer = SummaryWriter(log_dir=str(self.tb_dir))

        # -----------------------------
        # JSON log
        # -----------------------------
        self.log_path = self.log_dir / "training_log.jsonl"

        # -----------------------------
        # State
        # -----------------------------
        self.global_best = -np.inf

    # ----------------------------------------
    def log_generation(
        self,
        generation: int,
        fitness: np.ndarray,
        population: Optional[np.ndarray],
        optimizer_metrics: Dict[str, float],
        extra_metrics: Optional[Dict[str, Any]] = None,
    ):

        now = time.time()

        duration = now - self.last_gen_time
        total_time = now - self.start_time
        self.last_gen_time = now

        # ----------------------------------------
        # GLOBAL FITNESS
        # ----------------------------------------
        stats = {
            "generation": generation,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "duration": float(duration),
            "total_time": float(total_time),
            "fitness_mean": float(np.mean(fitness)),
            "fitness_max": float(np.max(fitness)),
            "fitness_min": float(np.min(fitness)),
            "fitness_std": float(np.std(fitness)),
        }

        # Global best
        self.global_best = max(self.global_best, stats["fitness_max"])
        stats["fitness_global_best"] = float(self.global_best)

        # ----------------------------------------
        # POPULATION DIVERSITY
        # ----------------------------------------
        if population is not None:
            try:
                pop_std = np.mean(np.std(population, axis=0))
                stats["param_std"] = float(pop_std)
            except Exception:
                pass

        # ----------------------------------------
        # OPTIMIZER METRICS
        # ----------------------------------------
        if optimizer_metrics:
            stats.update(optimizer_metrics)

        # ----------------------------------------
        # EXTRA METRICS
        # ----------------------------------------
        species_metrics = None

        if extra_metrics:

            # separa global vs species
            if "species" in extra_metrics:
                species_metrics = extra_metrics["species"]

            if "global" in extra_metrics:
                stats.update(extra_metrics["global"])
            else:
                stats.update(extra_metrics)

        # ----------------------------------------
        # SAVE JSON
        # ----------------------------------------
        with open(self.log_path, "a") as f:
            f.write(json.dumps(stats) + "\n")

        # ----------------------------------------
        # TENSORBOARD (GLOBAL)
        # ----------------------------------------
        if self.tb_writer:
            for key, value in stats.items():
                self._log_value(key, value, generation)

        # ----------------------------------------
        # TENSORBOARD (SPECIES)
        # ----------------------------------------
        if self.tb_writer and species_metrics and self.num_species > 1:

            # ordena espécies por fitness médio
            species_sorted = sorted(species_metrics.items(), key=lambda x: x[1].get("fitness_mean", -1e9), reverse=True)

            top_species = species_sorted[:self.top_k_species]

            for sp_id, metrics in top_species:

                for key, value in metrics.items():

                    tag = f"Species/{sp_id}/{self._format_tag(key)}"

                    if isinstance(value, (int, float)):
                        self.tb_writer.add_scalar(tag, float(value), generation)

                    elif isinstance(value, dict) and "mean" in value:
                        mean = float(value["mean"])
                        std = float(value.get("std", 0.0))

                        self.tb_writer.add_scalar(f"{tag}/mean", mean, generation)

                        if std > 1e-8:
                            self.tb_writer.add_scalar(f"{tag}/std", std, generation)
                            self.tb_writer.add_scalar(f"{tag}/mean+std", mean + std, generation)
                            self.tb_writer.add_scalar(f"{tag}/mean-std", mean - std, generation)

        # ----------------------------------------
        # CONSOLE
        # ----------------------------------------
        print(f"[GEN {generation:04d}] | "
              f"Mean: {stats['fitness_mean']:.3f} | "
              f"Max: {stats['fitness_max']:.3f} | "
              f"Std: {stats['fitness_std']:.3f} | "
              f"Time: {duration:.2f}s")

    # ----------------------------------------
    def _log_value(self, key, value, generation):

        try:
            if isinstance(value, (int, float)):
                tag = self._format_tag(key)
                self.tb_writer.add_scalar(tag, float(value), generation)

            elif isinstance(value, dict) and "mean" in value:
                mean = float(value["mean"])
                std = float(value.get("std", 0.0))

                base_tag = self._format_tag(key)

                self.tb_writer.add_scalar(f"{base_tag}/mean", mean, generation)

                if std > 1e-8:
                    self.tb_writer.add_scalar(f"{base_tag}/std", std, generation)
                    self.tb_writer.add_scalar(f"{base_tag}/mean+std", mean + std, generation)
                    self.tb_writer.add_scalar(f"{base_tag}/mean-std", mean - std, generation)

        except Exception:
            pass

    # ----------------------------------------
    def _format_tag(self, key: str) -> str:

        if key.startswith("fitness_"):
            return f"Fitness/{key.replace('fitness_', '')}"

        elif key == "param_std":
            return "Population/std"

        elif "loss" in key or "td_error" in key:
            return f"RL/{key}"

        elif "q_" in key:
            return f"RL/{key}"

        elif "exploration" in key or "better_ratio" in key:
            return f"Exploration/{key}"

        elif "advantage" in key:
            return f"RL/{key}"

        elif "replace" in key or "elite_from" in key or "guided" in key:
            return f"GuidedMutation/{key}"

        elif "success" in key:
            return f"Task/{key}"

        elif "time" in key or "duration" in key:
            return f"Timing/{key}"

        elif "sigma" in key or "step" in key:
            return f"CEM/{key}"

        else:
            return f"Metrics/{key}"

    # ----------------------------------------
    def close(self):

        if self.tb_writer:
            self.tb_writer.close()

        print(f"\n[Logger] Results saved at: {self.run_dir}")
