import numpy as np
from es_framework.models.policy import Policy
from env.norm_state import normalize_state


def step_env(env, action):
    """
    Compatível com Gymnasium e custom env
    """

    out = env.step(action)

    # Gymnasium
    if len(out) == 5:
        next_state, reward, terminated, truncated, _ = out
        done = terminated or truncated

    # Custom env
    else:
        next_state, reward, done, _ = out

    return next_state, reward, done


def reset_env(env, scenario):
    """
    Compatível com Gymnasium e custom env
    """

    out, _ = env.reset(seed=scenario)

    return out


# ----------------------------------------
# WORKER LOOP
# ----------------------------------------
def worker_loop(worker_id, task_queue, result_queue, config, start_event):

    print(f"[Worker {worker_id}] Starting...")

    # ----------------------------------------
    # ENV + POLICY INIT
    # ----------------------------------------
    env_fn = config["env_fn"]
    env_spec = config["env_spec"]
    model_config = config["model_config"]

    env, _ = env_fn()

    policy = Policy(env_spec, model_config)

    start_event.wait()

    # ----------------------------------------

    while True:

        task = task_queue.get()

        if task is None:
            print(f"[Worker {worker_id}] Shutting down.")
            break

        task_id, ind_id, params, scenario, flag, extra = task

        # ----------------------------------------
        # LOAD PARAMETERS
        # ----------------------------------------
        policy.set_parameters(params)

        # ----------------------------------------
        # RUN EPISODE
        # ----------------------------------------
        state = reset_env(env, scenario["seed"])
        st = normalize_state(state)
        trajectory = []
        total_reward = 0.0
        success = 0

        max_steps = config.get("max_steps", 1000)

        for step in range(max_steps):

            action, _ = policy.predict(st)

            next_state, reward, done = step_env(env, action)
            st_new = normalize_state(next_state)
            trajectory.append((st, action, reward, st_new))

            total_reward += reward

            st = st_new

            if done:
                break

        result_queue.put((task_id, ind_id, np.float32(total_reward), success, trajectory))
