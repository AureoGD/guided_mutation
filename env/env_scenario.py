from env.base_task_scenario import BaseTaskScenario
import random


class TaskScenario(BaseTaskScenario):

    def __init__(self, current_difficulty=1, config=None):
        self.current_difficulty = current_difficulty
        self.max_difficulty = 10

    def sample(self):
        return {"seed": random.randint(a=0, b=10000)}

    def apply(self, env):
        pass
