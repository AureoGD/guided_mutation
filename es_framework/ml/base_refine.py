from es_framework.ml.value_policy.model import ValueCNN
from collections import deque
import torch
import numpy as np
from env.norm_state import normalize_state


class BaseRefiner:

    def __init__(self, env_fn, policy, memory, config, device='cpu'):

        self.env_fn = env_fn
        self.policy = policy
        self.memory = memory
        self.config = config

        self.max_steps = config.get("rl_steps", 200)

        env_spec = config["env_spec"]

        v_spec = config["v_policy"]

        self.window_size = v_spec.get("window_size", 5)

        input_dim = env_spec.obs_dim + 1
        self.v_network = ValueCNN(obs_dim=input_dim, window_size=self.window_size).to(device)

        self.v_network.device = device
        self.trajectory_window = deque(maxlen=self.window_size)
        self.reward_window = deque(maxlen=self.window_size)
        self.current_delta_reward = 0
        self.v_network_value = 0

        self.eps = 1e-6

    def update_v_model(self, trained_model):
        self.v_network.load_state_dict(trained_model.state_dict())
        self.v_network.eval()

    # ----------------------------------------
    # MAIN ENTRY POINT
    # ----------------------------------------
    def refine(self, initial_params, scenario):

        # ----------------------------------------
        # 1. LOAD PARAMETERS (ES → RL)
        # ----------------------------------------
        self.policy.set_parameters(initial_params)

        # ----------------------------------------
        # 2. CREATE ENV
        # ----------------------------------------
        env, _ = self.env_fn()
        state, _ = env.reset(seed=scenario)
        trajectory_window = []
        # ----------------------------------------
        # 3. INTERACTION LOOP
        # ----------------------------------------
        st = normalize_state(state)
        for step in range(self.max_steps):

            # ação definida pelo algoritmo filho
            action = self.select_action(st)

            # step no ambiente
            next_state, reward, terminated, truncated, _ = env.step(action)

            new_st = normalize_state(next_state)

            done = terminated or truncated

            # ----------------------------------------
            # STORE TRANSITION (shared buffer)
            # ----------------------------------------
            self.memory.add(st, action, reward, new_st, done)

            # ----------------------------------------
            # V NETWORK PREDICT
            # ----------------------------------------
            current_step_data = np.concatenate([st, np.array([action])])

            self.trajectory_window.append(current_step_data)
            self.reward_window.append(reward)

            if len(self.trajectory_window) == self.window_size:
                window_np = np.array(self.trajectory_window)

                window_tensor = torch.FloatTensor(window_np).unsqueeze(0).to(self.v_network.device)
                with torch.no_grad():
                    self.v_network_value = self.v_network(window_tensor).item()

                # self.current_delta_reward = (self.reward_window[-1] - self.reward_window[0]) / (self.reward_window[0] +
                #                                                                                 self.eps)

                diff_real = self.reward_window[-1] - self.reward_window[0]
                self.current_delta_reward = np.sign(diff_real) * np.log1p(np.abs(diff_real))
            # ----------------------------------------
            # LEARNING STEP (delegado)
            # ----------------------------------------
            self.train_step()

            # próximo estado
            st = new_st

        # ----------------------------------------
        # 4. RETURN UPDATED PARAMETERS
        # ----------------------------------------
        return self.policy.get_parameters()

    # ----------------------------------------
    # METHODS TO BE IMPLEMENTED BY CHILD
    # ----------------------------------------
    def select_action(self, state):
        raise NotImplementedError

    def train_step(self):
        raise NotImplementedError
