import os

# ----------------------------------------
# FORCE CPU (evitar warning CUDA)
# ----------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch

torch.cuda.is_available = lambda: False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.enabled = False

# ----------------------------------------
# IMPORTS
# ----------------------------------------
import numpy as np
import multiprocessing as mp

from es_framework.core.trainer import Trainer
from es_framework.models.policy import Policy
from env.simple_env import create_env

# ----------------------------------------
# CONFIG
# ----------------------------------------
config = {

    # -----------------------------
    # Experiment
    # -----------------------------
    "optimizer_type": "CEM",
    "job_name": "lunarlander_debug",

    # -----------------------------
    # Training
    # -----------------------------
    "pop_size": 80,
    "species_size": 1,
    "num_scenarios": 5,
    "max_generations": 100,
    "max_workers": 10,
    "max_steps": 1000,

    # -----------------------------
    # ES params
    # -----------------------------
    "sigma_init": 0.05,
    "sigma_decay": 0.995,
    "elite_frac": 0.2,

    # -----------------------------
    # RL params
    # -----------------------------
    "rl_steps": 200,
    "batch_size": 128,
    "gamma": 0.99,
    "epsilon": 0.2,

    # -----------------------------
    # V-guided exploration
    # -----------------------------
    "delta_v": 20.0,
    "epsilon_boost": 1.3,
    "epsilon_max": 0.6,

    # -----------------------------
    # Model
    # -----------------------------
    "model_config": {
        "layers": [
            {
                "units": 32,
                "activation": "tanh"
            },
            {
                "units": 32,
                "activation": "tanh"
            },
        ]
    },
    "v_policy": {
        "window_size": 10,
        "batch_size": 256
    },

    # placeholders
    "env_config": None,
    "scenario_generator_class": None,
}

# ----------------------------------------
# MAIN
# ----------------------------------------
if __name__ == "__main__":

    mp.set_start_method("spawn", force=True)

    print("[INIT] Creating LunarLander env...")

    # ----------------------------------------
    # CREATE ENV (REAL)
    # ----------------------------------------
    env, env_spec = create_env()

    config["env_spec"] = env_spec
    config["env_fn"] = create_env

    print(f"[INIT] Obs dim: {env_spec.obs_dim}")
    print(f"[INIT] Act dim: {env_spec.act_dim}")

    # ----------------------------------------
    # BUILD POLICY
    # ----------------------------------------
    policy = Policy(env_spec, config["model_config"])
    num_params = policy.num_parameters()

    config["num_params"] = num_params

    print(f"[INIT] Number of parameters: {num_params}")

    # ----------------------------------------
    # TRAIN
    # ----------------------------------------
    trainer = Trainer(config)
    trainer.train()
