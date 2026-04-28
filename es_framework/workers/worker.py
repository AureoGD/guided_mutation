import numpy as np
from es_framework.models.policy import Policy
from env.norm_state import normalize_state
from es_framework.ml.dqn_refiner import DQNRefiner
from es_framework.ml.memory import ReplayBuffer

FLAG_SIMULATE = 0
FLAG_REFINE = 1


def step_env(env, action):
    out = env.step(action)

    if len(out) == 5:
        next_state, reward, terminated, truncated, info = out
        done = terminated or truncated

    else:
        next_state, reward, done, info = out

    return next_state, reward, done, info


def reset_env(env, scenario):
    out, _ = env.reset(seed=scenario)

    return out


def worker_loop(worker_id, task_queue, result_queue, config, start_event):
    print(f"[Worker {worker_id}] Starting...")
    env_fn = config["env_fn"]
    env, _ = env_fn()
    policy = Policy(config["env_spec"], config["model_config"])

    refiner = None

    start_event.wait()

    while True:
        task = task_queue.get()
        if task is None:
            break

        task_id, species_id, ind_id, params, scenario_data, flag, extra, memory = task

        if flag == FLAG_SIMULATE:
            policy.set_parameters(params)
            state = reset_env(env, scenario_data["seed"])
            st = normalize_state(state)
            trajectory = []
            total_reward = 0.0

            for step in range(config.get("max_steps", 1000)):
                action, _ = policy.predict(st)
                next_state, reward, done, info = step_env(env, action)
                st_new = normalize_state(next_state)
                trajectory.append((st, action, reward, st_new))
                total_reward += reward
                st = st_new
                if done:
                    break

            # sucess_flag = float(info["sucess_flag"])
            sucess_flag = 0

            result_queue.put((task_id, species_id, ind_id, np.float32(total_reward), sucess_flag, trajectory))

        elif flag == FLAG_REFINE:
            if refiner is None:
                n_cenarios = len(scenario_data)
                local_mem = ReplayBuffer(capacity=config.get("rl_steps", 200))
                refiner = DQNRefiner(env_fn, policy, local_mem, n_cenarios, config)

            refiner.add_memory(memory)

            if "trained_v_model" in config:
                refiner.v_network.load_state_dict(config["trained_v_model"])

            refiner.sync_with_elite(params)
            refiner.set_initial_history(extra)
            cumulative_reward = 0
            sucess_info_list = []
            for scenario in scenario_data:
                params, reward, sucess_flag = refiner.refine(params, scenario["seed"])
                sucess_info_list.append(sucess_flag)
                cumulative_reward += reward

            meam_reward = cumulative_reward / len(scenario_data)

            result_queue.put({
                "type": "refined_result",
                "species_id": species_id,
                "ind_id": ind_id,
                "refined_params": params,
                "mean_reward": meam_reward,
                "memory_chunk": refiner.memory.get_all(),
                "train_metrics": refiner.train_metrics
            })

            # refiner.memory.buffer.clear()
