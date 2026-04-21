import numpy as np
import torch
import torch.nn as nn

from es_framework.models.policy import Policy
from es_framework.ml.base_refine import BaseRefiner


class DQNRefiner(BaseRefiner):

    def __init__(self, env_fn, policy, memory, config):
        super().__init__(env_fn, policy, memory, config)

        # Parâmetros de RL vindos do config
        self.gamma = config.get("gamma", 0.99)
        self.batch_size = config.get("batch_size", 32)
        self.lr = config.get("lr", 1e-3)
        self.tau = config.get("tau", 0.005)
        self.device = config.get("device", "cpu")
        self.min_eps = 0.1
        self.max_eps = 0.5

        self.v_threshold = self.config.get("v_error_threshold", 0.1)

        self.q_online = Policy(config["env_spec"], config["model_config"]).to(self.device)
        self.q_target = Policy(config["env_spec"], config["model_config"]).to(self.device)

        # Inicializa target com os mesmos pesos da online
        self.q_target.load_state_dict(self.q_online.state_dict())

        self.optimizer = torch.optim.Adam(self.q_online.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def sync_with_elite(self, elite_params):

        self.q_online.set_parameters(elite_params)
        self.q_target.load_state_dict(self.q_online.state_dict())

        self.optimizer = torch.optim.Adam(self.q_online.parameters(), lr=self.lr)

    def get_refined_params(self):

        return self.q_online.get_parameters()

    # ----------------------------------------
    # ACTION SELECTION (ε-greedy)
    # ----------------------------------------
    def select_action(self, state):

        if np.random.rand() < self.epsilon:
            return np.random.randint(self.q_online.act_dim)

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.q_online(state)

        return torch.argmax(q_values, dim=1).item()

    # ----------------------------------------
    # TRAIN STEP (DQN)
    # ----------------------------------------
    def train_step(self):

        if len(self.memory) < self.batch_size:
            return

        s, a, r, s_next, done = self.memory.sample(self.batch_size)

        self.adapt_exploration(s, r)

        s = torch.FloatTensor(s).to(self.device)
        a = torch.LongTensor(a).unsqueeze(1).to(self.device)
        r = torch.FloatTensor(r).to(self.device)
        s_next = torch.FloatTensor(s_next).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        q_sa = self.q_online(s).gather(1, a).squeeze()

        with torch.no_grad():
            q_next_max = self.q_target(s_next).max(1)[0]
            target = r + self.gamma * q_next_max * (1 - done)
        loss = self.criterion(q_sa, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        for target_param, online_param in zip(self.q_target.parameters(), self.q_online.parameters()):
            target_param.data.copy_(self.tau * online_param.data + (1.0 - self.tau) * target_param.data)

    # ----------------------------------------
    # OPTIONAL: epsilon decay (futuro)
    # ----------------------------------------
    def decay_epsilon(self, factor=0.995):
        self.epsilon = max(self.min_eps, self.epsilon * factor)

    def adapt_exploration(self):
        error = abs(self.current_delta_reward - self.v_network_value)

        if error <= self.v_threshold:
            self.epsilon = min(self.epsilon * 1.05, self.max_eps)
        else:
            # Diminui exploração (Foca no que a V ainda não entende)
            self.epsilon = max(self.epsilon * 0.99, self.min_eps)
