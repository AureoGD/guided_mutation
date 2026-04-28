import numpy as np

class CurriculumManager:

    def __init__(self, scenario, window=20, threshold=0.9):
        self.scenario = scenario
        self.window = window
        self.threshold = threshold
        self.history = []

    def update(self, success_rate):

        self.history.append(success_rate)

        if len(self.history) > self.window:
            self.history.pop(0)

        if len(self.history) < self.window:
            return

        avg = np.mean(self.history)

        if avg > self.threshold:
            if self.scenario.current_difficulty < self.scenario.max_difficulty:
                self.scenario.current_difficulty += 1
                print(f"[Curriculum] Difficulty -> {self.scenario.current_difficulty}")

            self.history.clear()