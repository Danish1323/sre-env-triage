from abc import ABC, abstractmethod
from typing import Any

from models import SreDecisionAction

class Agent(ABC):
    """Base interface for all SRE environment agents."""
    
    @abstractmethod
    def act(self, obs: Any) -> SreDecisionAction:
        """
        Given the current observation, return a structured action.
        
        Args:
            obs: The current environment observation (usually SreDecisionObservation).
            
        Returns:
            A parsed SreDecisionAction to be passed to env.step().
        """
        pass
