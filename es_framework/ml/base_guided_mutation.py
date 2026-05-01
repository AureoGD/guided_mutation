from guided_mutation.es_framework.ml.value_policy.model import ValueCNN
from collections import deque
import torch
import numpy as np
from guided_mutation.env.norm_state import normalize_state


class GuidedRefiner:

    def __init__(self, config, policy, memory):
        self.config = config
        self.policy = policy
        self.memory = memory

        self.train_metrics = {}

    def get_parameters(self):
        return self.policy.get_parameters()

    def add_batch(self, batch):
        for s, a, r, sn, d in zip(*batch):
            self.memory.add(s, a, r, sn, d)

    def add_transition(self, mem):
        state = mem[0]
        action = mem[1]
        reward = mem[2]
        new_state = mem[3]
        done = mem[4]
        self.memory.add(state, action, reward, new_state, done)

    def select_action(self, state):
        raise NotImplementedError

    def train_step(self):
        pass

    def train_batch(self):
        pass

    def predcit_delta_reward(self):
        # uncertainty-driven exploration
        # current_step_data = np.concatenate([st, np.array([a_taken])])

        # self.trajectory_window.append(current_step_data)
        # self.reward_window.append(reward)

        # if len(self.trajectory_window) == self.window_size:
        #     window_np = np.array(self.trajectory_window)

        #     window_tensor = torch.FloatTensor(window_np).unsqueeze(0).to(self.v_network.device)
        #     with torch.no_grad():
        #         self.v_network_value = self.v_network(window_tensor).item()

        #     # self.current_delta_reward = (self.reward_window[-1] - self.reward_window[0]) / (self.reward_window[0] +
        #     #                                                                                 self.eps)

        #     diff_real = self.reward_window[-1] - self.reward_window[0]
        #     self.current_delta_reward = np.sign(diff_real) * np.log1p(np.abs(diff_real))
        #     e_window.append(self.current_delta_reward - self.v_network_value)
        pass
