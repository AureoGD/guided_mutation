import torch
import numpy as np
import torch.nn as nn

from guided_mutation.es_framework.models.policy import Policy
from guided_mutation.es_framework.ml.base_guided_mutation import GuidedRefiner
from guided_mutation.es_framework.ml.action_density import ActionDensity


class GuidedQV(GuidedRefiner):

    def __init__(self, config, policy, memory, n_scenarios):
        super().__init__(config, policy, memory)

        self.q_online = Policy(config["env_spec"], config["model_config"]).to(self.device)
        self.q_target = Policy(config["env_spec"], config["model_config"]).to(self.device)

        # Create the local density method
        self.local_act_density = ActionDensity(n_elite=1,
                                               n_task=n_scenarios,
                                               n_steps=self.max_steps,
                                               n_actions=self.n_action)

        self.local_act_density.attribute_id(ind_id=0)

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
            probs = self.get_refinement_distribution()
            a_taken = np.random.choice(self.q_online.act_dim, p=probs)
        else:
            a_taken = a_policy

        return np.array([a_taken, a_policy])

    def update_act_density(self, action):
        self.local_act_density.add(id_elite=0, action=action)
