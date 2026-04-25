import random
from typing import Any

from agents.base import Agent
from models import VALID_ACTIONS, SreDecisionAction

class RandomAgent(Agent):
    """Agent that chooses a random valid action."""
    
    def act(self, obs: Any) -> SreDecisionAction:
        # Just pick a random action from the list of valid actions
        action_name = random.choice(VALID_ACTIONS)
        return SreDecisionAction(
            action_name=action_name,
            rationale="Randomly selected action."
        )
