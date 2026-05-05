import torch
import numpy as np
import torch.nn as nn

from guided_mutation.es_framework.models.policy import Policy
from guided_mutation.es_framework.ml.base_guided_mutation import GuidedRefiner
from guided_mutation.es_framework.ml.action_density import ActionDensity
from guided_mutation.es_framework.ml.memory import ReplayBuffer


class GuidedQV(GuidedRefiner):

    def __init__(self, config, n_scenarios):
        super().__init__(config)

        # training variables:
        self.lambda_cf = 0.1

        self.gamma = config.get("gamma", 0.99)
        self.batch_size = config.get("batch_size", 128)
        self.lr = config.get("lr", 1e-3)
        self.tau = config.get("tau", 0.005)  # for online train (to do latter)
        self.min_eps = config.get("min_eps", 0.1)
        self.max_eps = config.get("max_eps", 0.5)
        self.epsilon = config.get("epsilon", 0.25)
        self.device = config.get("device", "cpu")

        mem_total_size = config.get("memory_size", 1e4)
        alpha = config.get("mem_elite_ratio", 0.5)

        # Initialize the Q networks
        self.q_online = Policy(config["env_spec"], config["model_config"]).to(self.device)
        self.q_target = Policy(config["env_spec"], config["model_config"]).to(self.device)

        self.q_target.load_state_dict(self.q_online.state_dict())

        self.optimizer = torch.optim.Adam(self.q_online.parameters(), lr=self.lr)

        self.criterion = nn.MSELoss()

        # memories inicialization

        self.n_elite = int(mem_total_size * alpha)
        self.n_local = mem_total_size - self.n_elite

        self.elite_mem = ReplayBuffer(capacity=self.n_elite)
        self.local_mem = ReplayBuffer(capacity=self.n_local)

        # Create the local density method

        max_sim_step = config.get("max_steps", 1000)

        self.local_act_density = ActionDensity(
            n_actions=self.q_online.act_dim,
            n_task=n_scenarios,
            n_steps=max_sim_step,
        )

        self.train_metrics = {
            "loss_dqn": 0.0,
            "loss_rank": 0.0,
            "td_error": 0.0,
            "exploration_ratio": 0.0,
            "better_ratio": 0.0,
            "mean_advantage": 0.0,
            "q_mean": 0.0,
            "q_std": 0.0,
            "n_batches": 0
        }

    def sync_with_elite(self, elite_params):

        self.q_online.set_parameters(elite_params)
        self.q_target.load_state_dict(self.q_online.state_dict())

        self.optimizer = torch.optim.Adam(self.q_online.parameters(), lr=self.lr)

    def select_action(self, state):

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.q_online(state_tensor)

        a_policy = torch.argmax(q_values, dim=1).item()

        if np.random.rand() < self.epsilon:
            probs = self.local_act_density.get_distribution()
            a_taken = np.random.choice(self.q_online.act_dim, p=probs)
        else:
            a_taken = a_policy

        return np.array([a_taken, a_policy])

    def update_act_density(self, action):
        self.local_act_density.add(action=action)

    def train_batch(self):

        # ----------------------------------------
        # Check minimum data
        # ----------------------------------------
        if len(self.local_mem) < self.batch_size:
            return

        # ----------------------------------------
        # Sample 50/50 from both memories
        # ----------------------------------------

        n_elite = min(self.n_elite, len(self.elite_mem))
        n_local = min(self.n_local, len(self.local_mem))

        s_e, a_e, r_e, sn_e, d_e = self.elite_mem.sample(n_elite)
        s_l, a_l, r_l, sn_l, d_l = self.local_mem.sample(n_local)

        all_s = np.concatenate([s_e, s_l], axis=0)
        all_a = np.concatenate([a_e, a_l], axis=0)
        all_r = np.concatenate([r_e, r_l], axis=0)
        all_sn = np.concatenate([sn_e, sn_l], axis=0)
        all_d = np.concatenate([d_e, d_l], axis=0)

        # ----------------------------------------
        # Reset metrics
        # ----------------------------------------
        self.train_metrics = {
            "loss_dqn": 0.0,
            "loss_rank": 0.0,
            "td_error": 0.0,
            "exploration_ratio": 0.0,
            "better_ratio": 0.0,
            "mean_advantage": 0.0,
            "q_mean": 0.0,
            "q_std": 0.0,
            "n_batches": 0
        }

        # ----------------------------------------
        # Chunk loop
        # ----------------------------------------
        dataset_size = len(all_s)
        indices = np.arange(dataset_size)
        np.random.shuffle(indices)

        N = self.config.get("n_chunks", 4)
        chunks = np.array_split(indices, N)

        for chunk in chunks:

            for start_idx in range(0, len(chunk), self.batch_size):

                batch_indices = chunk[start_idx:start_idx + self.batch_size]

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
                # Forward
                # ----------------------------------------
                q_values = self.q_online(s)
                q_taken = q_values.gather(1, a_taken).squeeze(1)
                q_policy = q_values.gather(1, a_policy).squeeze(1)

                # ----------------------------------------
                # Target
                # ----------------------------------------
                with torch.no_grad():
                    q_next_max = self.q_target(s_next).max(1)[0]
                    target = r + self.gamma * q_next_max * (1 - done)

                # ----------------------------------------
                # L_dqn
                # ----------------------------------------
                loss_dqn = self.criterion(q_taken, target)

                # ----------------------------------------
                # L_cf
                # ----------------------------------------
                mask = (a_taken.squeeze() != a_policy.squeeze()).float()
                rank_term = torch.relu(q_policy - target)
                loss_cf = (rank_term * mask).sum() / (mask.sum() + 1e-8)

                loss = loss_dqn + self.lambda_cf * loss_cf

                # ----------------------------------------
                # Backprop
                # ----------------------------------------
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # ----------------------------------------
                # Metrics
                # ----------------------------------------
                with torch.no_grad():
                    td_error = torch.abs(q_taken - target).mean().item()
                    exploration_ratio = (a_taken.squeeze() != a_policy.squeeze()).float().mean().item()
                    better_ratio = (target > q_policy).float().mean().item()
                    advantage = (target - q_policy).mean().item()
                    q_mean = q_values.mean().item()
                    q_std = q_values.std().item()

                self.train_metrics["loss_dqn"] += loss_dqn.item()
                self.train_metrics["loss_rank"] += loss_cf.item()
                self.train_metrics["td_error"] += td_error
                self.train_metrics["exploration_ratio"] += exploration_ratio
                self.train_metrics["better_ratio"] += better_ratio
                self.train_metrics["mean_advantage"] += advantage
                self.train_metrics["q_mean"] += q_mean
                self.train_metrics["q_std"] += q_std
                self.train_metrics["n_batches"] += 1

            self.q_target.load_state_dict(self.q_online.state_dict())

        # ----------------------------------------
        # Normalize metrics
        # ----------------------------------------
        n = self.train_metrics["n_batches"]
        if n > 0:
            for k in self.train_metrics:
                if k != "n_batches":
                    self.train_metrics[k] /= n

    def set_initial_history(self, history_list):
        self.local_act_density.history.extend(history_list)
