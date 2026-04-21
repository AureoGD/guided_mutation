import numpy as np
from collections import deque
from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from es_framework.optimizers.cem import CEMOptimizer
    from environment.learning_phases import LearningPhases


class CurriculumManager:

    def __init__(
            self,
            phases: 'LearningPhases' = None,  # Make optional to prevent import errors if None passed
            plateau_patience: int = 10,
            consistency_threshold: float = 0.8,
            target_task_idx: float = 4.0):

        self.phases = phases
        self.plateau_window_size = plateau_patience
        self.consistency_threshold = consistency_threshold
        self.target_task_idx = target_task_idx

        # Difficulty State
        self.current_stage_idx = 0

        # Performance Tracking
        self.score_history = deque(maxlen=self.plateau_window_size)
        self.consecutive_successes = 0
        self.current_max_stage = 0

    def get_difficulty(self) -> float:
        """
        Returns a float representing the current difficulty level.
        Used by the worker to set the physics or reward scaling.
        """
        return float(self.current_stage_idx)

    def get_reset_conditions(self, n_scenarios: int = 10) -> List[Tuple]:
        """
        Generates a FIXED set of scenarios that the ENTIRE population will face.
        Returns: List of (q0, v0, b0)
        """
        scenarios, _ = self.phases.get_initial_conditions(num_conditions=n_scenarios,
                                                          max_difficulty_id=self.current_stage_idx)
        return scenarios

    def update(self, success_rate: float, optimizer: 'CEMOptimizer') -> bool:
        """
        Updates the curriculum based on the population's success rate.
        Returns True if the stage changed (to trigger a checkpoint save).
        """
        self.score_history.append(success_rate)
        avg_success = np.mean(self.score_history)

        print(f"   >>> [Curriculum] Stage: {self.current_stage_idx} | "
              f"Success Rate: {success_rate:.2f} | Avg: {avg_success:.2f}")

        if avg_success >= self.consistency_threshold:
            if self.current_stage_idx < self.target_task_idx:
                self.current_stage_idx += 1
                self.consecutive_successes = 0
                self.score_history.clear()  # Reset history for the new harder stage

                print(f"   >>> [Curriculum] LEVEL UP! Advancing to Stage {self.current_stage_idx}")

                # Optional: Boost exploration when entering a new stage
                if hasattr(optimizer, 'boost_exploration'):
                    optimizer.boost_exploration(factor=1.5)

                return True  # Stage changed

        elif avg_success < 0.1 and self.current_stage_idx > 0:
            # self.current_stage_idx -= 1
            # self.score_history.clear()
            # print(f"   >>> [Curriculum] REGRADING... Back to Stage {self.current_stage_idx}")
            pass

        return False
