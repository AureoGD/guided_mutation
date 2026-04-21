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

    def __init__(self, run_dir: str):
        """
        Args:
            run_dir: diretório da run (definido pelo Trainer)
        """

        self.run_dir = Path(run_dir)

        # -----------------------------
        # Create directories
        # -----------------------------
        self.log_dir = self.run_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.tb_dir = self.log_dir / "tb"
        self.tb_dir.mkdir(parents=True, exist_ok=True)

        # -----------------------------
        # Time tracking
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
        extra_metrics: Optional[Dict[str, float]] = None,
    ):

        now = time.time()

        duration = now - self.last_gen_time
        total_time = now - self.start_time
        self.last_gen_time = now

        # ----------------------------------------
        # FITNESS STATS
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
        if extra_metrics:
            stats.update(extra_metrics)

        # ----------------------------------------
        # SAVE JSONL
        # ----------------------------------------
        with open(self.log_path, "a") as f:
            f.write(json.dumps(stats) + "\n")

        # ----------------------------------------
        # TENSORBOARD
        # ----------------------------------------
        if self.tb_writer:
            for k, v in stats.items():
                try:
                    val = float(v)

                    if "fitness" in k:
                        tag = f"Fitness/{k.replace('fitness_', '')}"

                    elif k == "param_std":
                        tag = "Population/std"

                    elif "sigma" in k or "step" in k:
                        tag = f"Optimizer/{k}"

                    elif "time" in k or "duration" in k:
                        tag = f"Timing/{k}"

                    elif "num_" in k:
                        tag = f"System/{k}"

                    elif k in ["difficulty", "success_rate", "max_stage_reached"]:
                        tag = f"Curriculum/{k}"

                    else:
                        tag = f"Metrics/{k}"

                    self.tb_writer.add_scalar(tag, val, generation)

                except (TypeError, ValueError):
                    continue

        # ----------------------------------------
        # CONSOLE
        # ----------------------------------------
        print(f"[GEN {generation:04d}] | "
              f"Mean: {stats['fitness_mean']:.3f} | "
              f"Max: {stats['fitness_max']:.3f} | "
              f"Std: {stats['fitness_std']:.3f} | "
              f"Time: {duration:.2f}s")

    # ----------------------------------------
    def close(self):
        if self.tb_writer:
            self.tb_writer.close()

        print(f"\n[Logger] Results saved at: {self.run_dir}")
