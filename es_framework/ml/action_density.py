import numpy as np
from collections import deque, Counter


class ActionDensity:

    def __init__(self, n_task, n_steps, n_actions):
        self.n_actions = n_actions
        self.max_size = n_task * n_steps
        self.history = deque(maxlen=self.max_size)

    def add(self, action):
        self.history.append(action)

    def reset_history(self):
        self.history.clear()

    def get_distribution(self, epsilon=1e-6):

        history = self.history

        count = Counter(history)
        dist = np.array([count[i] for i in range(self.n_actions)], dtype=float)

        v_max = dist.max()
        v_min = dist.min()

        k = v_max + v_min + epsilon
        num = k - dist

        den = num.sum()
        probs = num / den

        return probs
