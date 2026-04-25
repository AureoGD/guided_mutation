import numpy as np


class ReplayBuffer:

    def __init__(self, capacity):

        self.capacity = capacity
        self.buffer = []
        self.position = 0

    # ----------------------------------------
    # ADD SINGLE TRANSITION
    # ----------------------------------------
    def add(self, s, a, r, s_next, done):

        data = (s, a, r, s_next, done)

        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.position] = data

        self.position = (self.position + 1) % self.capacity

    # ----------------------------------------
    # SAMPLE BATCH
    # ----------------------------------------

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self.buffer))
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]

        s, a, r, s_next, done = zip(*batch)
        return (np.array(s, dtype=np.float32), np.array(a, dtype=np.int64), np.array(r, dtype=np.float32),
                np.array(s_next, dtype=np.float32), np.array(done, dtype=np.float32))

    def get_all(self):
        return list(self.buffer)

    # ----------------------------------------
    def __len__(self):
        return len(self.buffer)
