"""
Minimal contract tests for the OpenEnv step/reset interface.

Validates:
- reset() returns the canonical OpenEnv dict shape.
- step() with a valid action returns the canonical OpenEnv dict shape.
- step() with an invalid action applies a penalty (low reward).

Run::

    python -m pytest tests/test_env_contract.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server.sre_decision_env_environment import SreDecisionEnvironment
from models import SreDecisionAction, SreDecisionObservation


class TestResetContract:
    """reset() must return the canonical OpenEnv step-result dict."""

    def test_reset_returns_dict(self):
        env = SreDecisionEnvironment()
        result = env.reset()
        assert isinstance(result, dict), "reset() must return a dict"

    def test_reset_has_observation_key(self):
        env = SreDecisionEnvironment()
        result = env.reset()
        assert "observation" in result, "reset() result must contain 'observation'"

    def test_reset_observation_is_dict(self):
        env = SreDecisionEnvironment()
        result = env.reset()
        assert isinstance(result["observation"], dict)

    def test_reset_observation_fields(self):
        env = SreDecisionEnvironment()
        result = env.reset()
        obs = result["observation"]
        # Must include all SreDecisionObservation fields
        for field in SreDecisionObservation.model_fields:
            assert field in obs, f"observation missing field '{field}'"

    def test_reset_done_is_false(self):
        env = SreDecisionEnvironment()
        result = env.reset()
        assert result["done"] is False

    def test_reset_observation_has_no_reward_or_done(self):
        env = SreDecisionEnvironment()
        result = env.reset()
        obs = result["observation"]
        assert "reward" not in obs, "observation must not contain embedded 'reward'"
        assert "done" not in obs, "observation must not contain embedded 'done'"


class TestStepContract:
    """step() must return the canonical OpenEnv step-result dict."""

    def test_step_returns_dict(self):
        env = SreDecisionEnvironment()
        env.reset()
        result = env.step(SreDecisionAction(action_name="inspect_logs"))
        assert isinstance(result, dict), "step() must return a dict"

    def test_step_has_required_keys(self):
        env = SreDecisionEnvironment()
        env.reset()
        result = env.step(SreDecisionAction(action_name="inspect_logs"))
        for key in ("observation", "reward", "done", "info"):
            assert key in result, f"step() result missing key '{key}'"

    def test_step_observation_fields(self):
        env = SreDecisionEnvironment()
        env.reset()
        result = env.step(SreDecisionAction(action_name="inspect_logs"))
        obs = result["observation"]
        for field in SreDecisionObservation.model_fields:
            assert field in obs, f"step observation missing field '{field}'"

    def test_step_reward_is_float(self):
        env = SreDecisionEnvironment()
        env.reset()
        result = env.step(SreDecisionAction(action_name="inspect_logs"))
        assert isinstance(result["reward"], float)

    def test_step_done_is_bool(self):
        env = SreDecisionEnvironment()
        env.reset()
        result = env.step(SreDecisionAction(action_name="inspect_logs"))
        assert isinstance(result["done"], bool)

    def test_step_time_step_increments(self):
        env = SreDecisionEnvironment()
        env.reset()
        result = env.step(SreDecisionAction(action_name="inspect_logs"))
        assert result["observation"]["time_step"] == 1

    def test_step_observation_has_no_reward_or_done(self):
        env = SreDecisionEnvironment()
        env.reset()
        result = env.step(SreDecisionAction(action_name="inspect_logs"))
        obs = result["observation"]
        assert "reward" not in obs
        assert "done" not in obs

    def test_valid_action_with_rationale(self):
        env = SreDecisionEnvironment()
        env.reset()
        action = SreDecisionAction(action_name="inspect_metrics", rationale="Check CPU load")
        result = env.step(action)
        assert isinstance(result, dict)
        assert result["reward"] > 0


class TestInvalidActionPenalty:
    """An invalid action_name must receive a penalty (low reward)."""

    def test_invalid_action_gives_low_reward(self):
        env = SreDecisionEnvironment()
        env.reset()
        # "scale_up" was removed from VALID_ACTIONS — it is now invalid
        result = env.step(SreDecisionAction(action_name="scale_up"))
        assert result["reward"] < 0.3, (
            f"Invalid action should receive a low reward, got {result['reward']}"
        )

    def test_invalid_action_episode_continues(self):
        env = SreDecisionEnvironment()
        env.reset()
        result = env.step(SreDecisionAction(action_name="totally_fake_action"))
        # Episode should not immediately end on an invalid action (unless max steps)
        assert isinstance(result["done"], bool)
