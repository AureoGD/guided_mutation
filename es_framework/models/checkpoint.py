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
    def save_best(self, params, fitness, specie=None):
        """
        Salva o melhor indivíduo (global ou por espécie)
        """

        if specie is None:
            # -----------------------------
            # GLOBAL
            # -----------------------------
            if fitness > self.best_fitness:

                self.best_fitness = fitness

                path = self.model_dir / "best_params.npy"
                np.save(path, params)

                print(f"[CHECKPOINT] New GLOBAL best | Fitness: {fitness:.4f}")

        else:
            # -----------------------------
            # POR ESPÉCIE
            # -----------------------------
            if not hasattr(self, "best_fitness_species"):
                self.best_fitness_species = {}

            prev_best = self.best_fitness_species.get(specie, -np.inf)

            if fitness > prev_best:

                self.best_fitness_species[specie] = fitness

                path = self.model_dir / f"best_params_sp_{specie}.npy"
                np.save(path, params)

                print(f"[CHECKPOINT] New BEST | Species {specie} | Fitness: {fitness:.4f}")

    def save_last(self, params, specie=None):
        """
        Salva o último indivíduo (global ou por espécie)
        """

        if specie is None:
            path = self.model_dir / "last_params.npy"
        else:
            path = self.model_dir / f"last_params_sp_{specie}.npy"

        np.save(path, params)

    # ----------------------------------------
    def save_periodic(self, params, generation, interval=50, specie=None):

        if generation % interval != 0:
            return

        if specie is None:
            path = self.ckpt_dir / f"gen_{generation:05d}.npy"
        else:
            path = self.ckpt_dir / f"spe_{specie}_gen_{generation:05d}.npy"

        np.save(path, params)
