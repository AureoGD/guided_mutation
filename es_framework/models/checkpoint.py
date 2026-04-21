import torch
import numpy as np
from pathlib import Path


class ModelCheckpoint:

    def __init__(self, run_dir):

        self.run_dir = Path(run_dir)

        self.model_dir = self.run_dir / "models"
        self.ckpt_dir = self.model_dir / "checkpoints"

        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.best_fitness = -float("inf")
        self.spec_saved = False

    # ----------------------------------------
    def save_model_spec(self, env_spec, model_cfg):
        """
        Salva a estrutura do modelo (uma única vez)
        """

        if self.spec_saved:
            return

        spec = {
            "env_spec": {
                "obs_dim": env_spec.obs_dim,
                "act_dim": env_spec.act_dim,
                "is_discrete": env_spec.is_discrete,
            },
            "model_cfg": model_cfg,
        }

        path = self.model_dir / "model_spec.pt"
        torch.save(spec, path)

        self.spec_saved = True

        print("[CHECKPOINT] Model spec saved")

    # ----------------------------------------
    def update(self, optimizer, generation):

        best = optimizer.get_best_params()

        if best is None:
            return

        best_params = best["params"]
        best_fitness = best["fitness"]

        if best_fitness > self.best_fitness:

            self.best_fitness = best_fitness

            path = self.model_dir / "best_params.npy"
            np.save(path, best_params)

            print(f"[CHECKPOINT] New best | Fitness: {best_fitness:.4f}")

    # ----------------------------------------
    def save_last(self, optimizer):

        best = optimizer.get_best_params()

        if best is None:
            return

        path = self.model_dir / "last_params.npy"
        np.save(path, best["params"])

    # ----------------------------------------
    def save_periodic(self, optimizer, generation, interval=50):

        if generation % interval != 0:
            return

        best = optimizer.get_best_params()

        if best is None:
            return

        path = self.ckpt_dir / f"gen_{generation:05d}.npy"
        np.save(path, best["params"])
