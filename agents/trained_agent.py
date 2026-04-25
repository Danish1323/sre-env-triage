import logging
from typing import Any, List

from agents.base import Agent
from llm.openrouter_client import OpenRouterLLMClient
from llm.prompts import SYSTEM_PROMPT, build_user_prompt, parse_action
from models import SreDecisionAction

logger = logging.getLogger(__name__)

class TrainedAgent(Agent):
    """
    Agent that uses the LLM (OpenRouter) to decide actions.
    This acts as the 'trained' model in our pipeline.
    """
    
    def __init__(self):
        self.client = OpenRouterLLMClient()
        self.history: List[dict] = []
        
    def act(self, obs: Any) -> SreDecisionAction:
        obs_dict = obs.model_dump()
        
        # We need to maintain history across steps for the prompt
        # We extract last action and feedback if not step 0
        if obs_dict.get("time_step", 0) > 0 and obs_dict.get("last_action"):
            self.history.append({
                "action": obs_dict.get("last_action"),
                "feedback": obs_dict.get("action_feedback", "")
            })
            
        # If reset (step 0), clear history
        if obs_dict.get("time_step", 0) == 0:
            self.history = []

        user_msg = build_user_prompt(obs_dict, self.history)

        try:
            llm_response = self.client.chat_complete(
                system_prompt=SYSTEM_PROMPT,
                user_message=user_msg,
                temperature=0.1,
                max_tokens=256,
            )
            action_name = parse_action(llm_response)
        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            action_name = "inspect_logs"
            llm_response = f"Failed to call LLM: {exc}"

        return SreDecisionAction(
            action_name=action_name,
            rationale=llm_response
        )
