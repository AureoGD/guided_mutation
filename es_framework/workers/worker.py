import time
import threading
import numpy as np
from guided_mutation.es_framework.models.policy import Policy
from guided_mutation.es_framework.ml.guided_qv import GuidedQV
from guided_mutation.es_framework.ml.memory import ReplayBuffer

FLAG_SIMULATE = 0
FLAG_REFINE = 1


def _step_env(env, action):
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    return reward, next_state, done, info


def _unpack_scenarios_data(scenario_data):
    seed = scenario_data.get("seed", 44)
    reset_options = scenario_data.get("config", None)
    return seed, reset_options


def _heartbeat_fn(heartbeat, stop_event, interval=2.0):
    while not stop_event.is_set():
        heartbeat[0] = time.time()
        stop_event.wait(interval)


def _unpac_info_for_hb(heartbeat, info):
    heartbeat[1] = info["sim_step"]
    heartbeat[2] = info["env_id"]


def worker_loop(worker_id, task_queue, result_queue, config, heartbeat, start_event):

    env_config = config["env_config"]
    env_fn = config["env_fn"]
    env, _ = env_fn(env_config, worker_id)
    policy = Policy(config["env_spec"], config["model_config"])
    success_fcn = config.get("success_criterion", None)
    max_sim_step = config.get("max_steps", 1000)

    # reset env for test
    try:
        _, info = env.reset()
        print(f"Env {worker_id} successfully created")
    except Exception as e:
        print(f"[Worker {worker_id}] Failed to create env: {type(e).__name__}: {e}")

    gm = None

    # Start event
    start_event.wait()

    # Heart-beat inicialization:
    stop_hb = threading.Event()
    hb_thread = threading.Thread(target=_heartbeat_fn, args=(heartbeat, stop_hb), daemon=True)
    hb_thread.start()

    while True:

        task = task_queue.get()

        if task is None:
            stop_hb.set()
            break

        task_id = task["task_id"]
        species_id = task["species_id"]
        ind_id = task["ind_id"]
        params = task["params"]
        scenario_data = task["scenario_data"]
        flag = task["flag"]
        extra_in = task["extra"]

        # --------------------------------------------------------
        # FLAG_SIMULATE
        # --------------------------------------------------------

        total_reward = 0.0
        success_percent = 0.0

        try:

            if flag == FLAG_SIMULATE:
                # set policy weights
                policy.set_parameters(params)

                # get scenario info and reset
                seed, reset_options = _unpack_scenarios_data(scenario_data)

                state, _ = env.reset(seed=seed, options=reset_options)

                trajectory = []

                for step in range(max_sim_step):
                    action, _ = policy.predict(state)
                    reward, next_state, done, info = _step_env(env, action)
                    trajectory.append((state, action, reward, next_state))
                    total_reward += reward

                    if step % 10 == 0:
                        _unpac_info_for_hb(heartbeat, info)

                    if done:
                        break
                    state = next_state

                if success_fcn is not None:
                    success_percent = success_fcn(info=info, reward=total_reward)

                extra_out = {"trajectory": trajectory}

            # --------------------------------------------------------
            # FLAG_REFINE
            # --------------------------------------------------------
            elif flag == FLAG_REFINE:
                if gm is None:
                    n_scenarios = len(scenario_data)
                    local_mem = ReplayBuffer(capacity=config.get("rl_steps", 200))
                    gm = GuidedQV(policy, local_mem, n_scenarios)

                elite_mem = extra_in["memory"]
                elite_act_density = extra_in["act_density"]
                st_bt_flag = extra_in["st_bt"]

                gm.add_batch(elite_mem)
                gm.set_initial_history(elite_act_density)
                gm.sync_with_elite(params)

                success_list = []

                for scenario in scenario_data:

                    seed, reset_options = _unpack_scenarios_data(scenario)
                    state, _ = env.reset(seed=seed, options=reset_options)

                    for step in range(max_sim_step):

                        action = gm.select_action(state)

                        gm.update_act_density(action[0])

                        reward, next_state, done, info = _step_env(env, action[0])
                        mem = [state, action, reward, next_state, done]
                        gm.add_transition(mem)

                        total_reward += reward

                        if step % 10 == 0:
                            _unpac_info_for_hb(heartbeat, info)

                        if st_bt_flag:
                            gm.train_step()

                        if done:
                            break

                        state = next_state

                    if not st_bt_flag:
                        gm.train_batch()

                    if success_fcn is not None:
                        success_list.append(success_fcn(info=info, reward=total_reward))

                success_percent = np.sum(success_list) / n_scenarios
                total_reward = total_reward / n_scenarios

                extra_out = {
                    "refined_params": gm.get_parameters(),
                    "train_metrics": gm.train_metrics,
                }

            result_queue.put({
                "type": "simulation_result" if flag == FLAG_SIMULATE else "refined_result",
                "task_id": task_id,
                "species_id": species_id,
                "ind_id": ind_id,
                "reward": np.float32(total_reward),
                "success_percent": success_percent,
                "status": "success",
                "extra": extra_out
            })

        except Exception as e:
            print(f"[Worker {worker_id}] Task {task_id} failed: {type(e).__name__}: {e}")
            result_queue.put({
                "type": "simulation_result" if flag == FLAG_SIMULATE else "refined_result",
                "task_id": task_id,
                "species_id": species_id,
                "ind_id": ind_id,
                "reward": np.float32(-1e4),
                "success_percent": 0.0,
                "status": "error",
                "extra": {}
            })
