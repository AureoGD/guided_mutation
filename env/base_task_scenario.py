from abc import ABC, abstractmethod


class BaseTaskScenario(ABC):

    @abstractmethod
    def sample(self):
        """Return a new scenario instance"""
        pass

    @abstractmethod
    def apply(self, env):
        """Apply scenario to environment (reset conditions)"""
        pass
