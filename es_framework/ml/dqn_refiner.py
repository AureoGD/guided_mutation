import numpy as np
import torch
import torch.nn as nn

from es_framework.models.policy import Policy
from es_framework.ml.base_refine import BaseRefiner
from es_framework.ml.action_density import ActionDensity


class DQNRefiner(BaseRefiner):

    def __init__(self, env_fn, policy, memory, n_cenarios, config):
        super().__init__(env_fn, policy, memory, config)

        # Parâmetros de RL vindos do config
        self.gamma = config.get("gamma", 0.99)
        self.batch_size = config.get("batch_size", 128)
        self.lr = config.get("lr", 1e-3)
        self.tau = config.get("tau", 0.005)
        self.device = config.get("device", "cpu")
        self.min_eps = 0.1
        self.max_eps = 0.5
        self.epsilon = 0.1

        self.v_threshold = self.config.get("v_error_threshold", 0.1)

        self.q_online = Policy(config["env_spec"], config["model_config"]).to(self.device)
        self.q_target = Policy(config["env_spec"], config["model_config"]).to(self.device)

        # Inicializa target com os mesmos pesos da online
        self.q_target.load_state_dict(self.q_online.state_dict())

        self.optimizer = torch.optim.Adam(self.q_online.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

        self.local_density = ActionDensity(n_elite=1,
                                           n_task=n_cenarios,
                                           n_steps=self.max_steps,
                                           n_actions=self.n_action)
        self.local_density.attribute_id(ind_id=0)

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

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.q_online(state_tensor)

        a_policy = torch.argmax(q_values, dim=1).item()

        if np.random.rand() < self.epsilon:
            probs = self.get_refinement_distribution()
            a_taken = np.random.choice(self.q_online.act_dim, p=probs)
        else:
            a_taken = a_policy

        return a_taken, a_policy

    # ----------------------------------------
    # TRAIN STEP (DQN)
    # ----------------------------------------
    def train_step(self):

        if len(self.memory) < self.batch_size:
            return

        s, a, r, s_next, done = self.memory.sample(self.batch_size)

        # self.adapt_exploration()

        self.decay_epsilon()

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
            self.epsilon = min(self.epsilon * 1.005, self.max_eps)
        else:
            self.epsilon = max(self.epsilon * 0.99, self.min_eps)

    def set_initial_history(self, history_list):
        self.local_density.history[0].clear()
        if history_list:
            self.local_density.history[0].extend(list(history_list))

    def update_action_online(self, action):
        self.local_density.add(id_elite=0, action=action)

    def get_refinement_distribution(self):
        return self.local_density.get_distribution(id_elite=0)

    def add_memory(self, batch):
        for s, a, r, sn, d in zip(*batch):
            self.memory.add(s, a, r, sn, d)

    def train_batch(self, epochs=10):

        if len(self.memory) < self.batch_size:
            return

        all_s, all_a, all_r, all_sn, all_d = self.memory.sample(len(self.memory))

        dataset_size = len(all_s)
        indices = np.arange(dataset_size)

        lambda_rank = 0.1  # peso da loss de comparação (ajustável)

        for epoch in range(epochs):

            np.random.shuffle(indices)

            for start_idx in range(0, dataset_size, self.batch_size):

                batch_indices = indices[start_idx:start_idx + self.batch_size]

                # ----------------------------------------
                # Batch
                # ----------------------------------------
                s = torch.FloatTensor(all_s[batch_indices]).to(self.device)

                a = torch.LongTensor(all_a[batch_indices]).to(self.device)
                a_taken = a[:, 0].unsqueeze(1)
                a_policy = a[:, 1].unsqueeze(1)

                r = torch.FloatTensor(all_r[batch_indices]).to(self.device)
                s_next = torch.FloatTensor(all_sn[batch_indices]).to(self.device)
                done = torch.FloatTensor(all_d[batch_indices]).to(self.device)

                # ----------------------------------------
                # Q(s,a_taken)
                # ----------------------------------------
                q_values = self.q_online(s)
                q_taken = q_values.gather(1, a_taken).squeeze()

                # ----------------------------------------
                # Target Bellman
                # ----------------------------------------
                with torch.no_grad():
                    q_next_max = self.q_target(s_next).max(1)[0]
                    target = r + self.gamma * q_next_max * (1 - done)

                loss_dqn = self.criterion(q_taken, target)

                # ----------------------------------------
                # Ranking loss (SÓ quando houve exploração)
                # ----------------------------------------
                q_policy = q_values.gather(1, a_policy).squeeze()

                # máscara: só quando ações são diferentes
                mask = (a_taken.squeeze() != a_policy.squeeze()).float()

                # comparação usando target (mais estável)
                rank_term = torch.relu(q_policy - target)

                # aplica máscara
                if mask.sum() > 0:
                    loss_rank = (rank_term * mask).sum() / mask.sum()
                else:
                    loss_rank = torch.tensor(0.0, device=self.device)

                # ----------------------------------------
                # Loss final
                # ----------------------------------------
                # loss = loss_dqn + lambda_rank * loss_rank
                loss = loss_dqn

                # ----------------------------------------
                # Otimização
                # ----------------------------------------
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # ----------------------------------------
        # Sync target network
        # ----------------------------------------
        self.q_target.load_state_dict(self.q_online.state_dict())

        # Soft update da Target Network (opcional por step ou por época)
        # for target_param, online_param in zip(self.q_target.parameters(), self.q_online.parameters()):
        #     target_param.data.copy_(self.tau * online_param.data + (1.0 - self.tau) * target_param.data)
