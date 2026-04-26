import sys
import os
import logging
import traceback

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from models import SreDecisionAction
from server.sre_decision_env_environment import SreDecisionEnvironment

def test_env_interface():
    env = SreDecisionEnvironment()
    print("Testing reset()...")
    reset_res = env.reset()
    
    assert isinstance(reset_res, dict), "Reset must return a dict"
    assert "observation" in reset_res, "Reset missing 'observation'"
    assert "reward" in reset_res, "Reset missing 'reward'"
    assert "done" in reset_res, "Reset missing 'done'"
    assert "info" in reset_res, "Reset missing 'info'"
    assert "reward" not in reset_res["observation"], "Observation must not contain embedded 'reward'"
    assert "done" not in reset_res["observation"], "Observation must not contain embedded 'done'"
    
    print("Reset OK:", reset_res)
    
    print("\nTesting step()...")
    action = SreDecisionAction(action_name="inspect_logs", rationale="test")
    step_res = env.step(action)
    
    assert isinstance(step_res, dict), "Step must return a dict"
    assert "observation" in step_res, "Step missing 'observation'"
    assert "reward" in step_res, "Step missing 'reward'"
    assert "done" in step_res, "Step missing 'done'"
    assert "info" in step_res, "Step missing 'info'"
    assert "reward" not in step_res["observation"], "Observation must not contain embedded 'reward'"
    assert "done" not in step_res["observation"], "Observation must not contain embedded 'done'"
    assert isinstance(step_res["reward"], float), f"Reward must be float, got {type(step_res['reward'])}"
    assert isinstance(step_res["done"], bool), f"Done must be bool, got {type(step_res['done'])}"
    
    print("Step OK:", step_res)
    print("\nAll interface assertions passed! PPO compatibility guaranteed.")

if __name__ == "__main__":
    test_env_interface()
