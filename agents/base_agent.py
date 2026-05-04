from abc import ABC, abstractmethod

class BaseAgent(ABC):
    
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def act(self, observation, action_mask):
        """
        Given the current observation and legal actions,
        return an action (0=fold, 1=call, 2=raise).
        """
        pass

    def reset(self):
        """
        Called at the start of each new game.
        Override if your agent needs to reset internal state.
        """
        pass
    
    def record_action(self, action):
        pass  # Override if agent tracks action history

    def __str__(self):
        return self.name