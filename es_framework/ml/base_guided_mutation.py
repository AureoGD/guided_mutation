class GuidedRefiner:

    def __init__(self, config):
        self.config = config
        self.train_metrics = {}

    def get_parameters(self, nn):
        return nn.get_parameters()

    def add_batch(self, memory, batch):
        for s, a, r, sn, d in zip(*batch):
            memory.add(s, a, r, sn, d)

    def add_transition(self, memomry, mem):
        state = mem[0]
        action = mem[1]
        reward = mem[2]
        new_state = mem[3]
        done = mem[4]
        memomry.add(state, action, reward, new_state, done)

    def select_action(self, state):
        raise NotImplementedError

    def train_step(self):
        pass

    def train_batch(self):
        pass
