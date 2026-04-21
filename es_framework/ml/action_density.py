import numpy as np
from collections import deque, Counter


class ActionDensity:

    def __init__(self, n_elite, n_task, n_steps, n_actions):
        self.n_individuos = n_elite
        self.n_actions = n_actions
        self.max_size = n_task * n_steps
        self.history = [deque(maxlen=self.max_size) for _ in range(n_elite)]
        self.ids_map = {}

    def add(self, id_elite, action):
        # Corrigido para self.ids_map e self.history
        idx = self.ids_map.get(id_elite)
        if idx is not None:
            self.history[idx].append(action)

    def reset_history(self):
        for h in self.history:
            h.clear()
        self.ids_map.clear()

    def attribute_id(self, ind_id):
        if ind_id not in self.ids_map:
            novo_indice = len(self.ids_map)
            if novo_indice < self.n_individuos:
                self.ids_map[ind_id] = novo_indice
            else:
                self.ids_map[ind_id] = novo_indice % self.n_individuos

    def get_distribution(self, id_elite, epsilon=1e-6):
        # Corrigido para self.ids_map
        idx = self.ids_map.get(id_elite)

        if idx is None or len(self.history[idx]) == 0:
            return np.ones(self.n_actions) / self.n_actions

        history = self.history[idx]

        count = Counter(history)
        dist = np.array([count[i] for i in range(self.n_actions)], dtype=float)

        v_max = dist.max()
        v_min = dist.min()

        # Equação Linear: (Max + Min + Epsilon) - V_i
        k = v_max + v_min + epsilon
        num = k - dist

        den = num.sum()
        probs = num / den

        return probs
