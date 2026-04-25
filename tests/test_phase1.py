"""
Unit tests for Phase 1 — environment, sensors, actions, rewards.

Run::

    cd sre_decision_env
    python -m pytest tests/ -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from server.sensors import ROOT_CAUSES, get_logs_signal, get_observer_signal
from server.rewards import compute_step_reward, compute_terminal_reward
from server.sre_decision_env_environment import SreDecisionEnvironment, MAX_STEPS
from models import VALID_ACTIONS, SreDecisionAction, SreDecisionObservation


# ---------------------------------------------------------------------------
# Sensor tests
# ---------------------------------------------------------------------------

class TestSensors:
    def test_logs_signal_keys(self):
        for rc in ROOT_CAUSES:
            sig = get_logs_signal(rc)
            assert "latency_spike" in sig
            assert "error_rate" in sig
            assert "log_anomaly_score" in sig

    def test_logs_signal_types(self):
        for rc in ROOT_CAUSES:
            sig = get_logs_signal(rc)
            assert isinstance(sig["latency_spike"], bool)
            assert 0.0 <= sig["error_rate"] <= 1.0
            assert 0.0 <= sig["log_anomaly_score"] <= 1.0

    def test_observer_signal_keys(self):
        for rc in ROOT_CAUSES:
            sig = get_observer_signal(rc)
            assert "server_b_health" in sig
            assert "cpu_usage" in sig
            assert "memory_usage" in sig

    def test_observer_health_values(self):
        valid_health = {"healthy", "degraded", "down"}
        for rc in ROOT_CAUSES:
            sig = get_observer_signal(rc)
            assert sig["server_b_health"] in valid_health

    def test_no_issue_low_error_rate(self):
        """no_issue should produce low error rates on average."""
        rates = [get_logs_signal("no_issue")["error_rate"] for _ in range(50)]
        assert sum(rates) / len(rates) < 0.15


# ---------------------------------------------------------------------------
# Reward tests
# ---------------------------------------------------------------------------

class TestRewards:
    def test_invalid_action_penalty(self):
        r = compute_step_reward("bad_action", "no_issue", 1, MAX_STEPS, is_valid=False)
        assert r < 0

    def test_correct_remediation_positive(self):
        r = compute_step_reward("restart_service", "server_A_failure", 1, MAX_STEPS, is_valid=True)
        assert r > 0

    def test_harmful_action_penalty(self):
        r = compute_step_reward("resolve_incident", "server_A_failure", 1, MAX_STEPS, is_valid=True)
        assert r < 0

    def test_diagnostic_action_small_positive(self):
        r = compute_step_reward("inspect_logs", "no_issue", 1, MAX_STEPS, is_valid=True)
        assert r > 0

    def test_terminal_timeout_penalty(self):
        r = compute_terminal_reward(False, "no_issue", None, MAX_STEPS, MAX_STEPS)
        assert r < 0


# ---------------------------------------------------------------------------
# Environment tests
# ---------------------------------------------------------------------------

class TestEnvironment:
    def test_reset_returns_observation(self):
        env = SreDecisionEnvironment()
        obs = env.reset()
        assert isinstance(obs, SreDecisionObservation)

    def test_hidden_state_not_in_obs(self):
        env = SreDecisionEnvironment()
        obs = env.reset()
        obs_dict = obs.model_dump()
        assert "root_cause" not in obs_dict
        assert "_root_cause" not in str(obs_dict)

    def test_step_increments_counter(self):
        env = SreDecisionEnvironment()
        env.reset()
        obs = env.step(SreDecisionAction(action_name="inspect_logs"))
        assert obs.time_step == 1

    def test_valid_actions_in_obs(self):
        env = SreDecisionEnvironment()
        obs = env.reset()
        for action in VALID_ACTIONS:
            assert action in obs.available_actions

    def test_episode_terminates_on_resolve(self):
        env = SreDecisionEnvironment()
        env.reset()
        obs = env.step(SreDecisionAction(action_name="resolve_incident"))
        assert obs.done is True or obs.incident_resolved is True

    def test_episode_terminates_on_max_steps(self):
        env = SreDecisionEnvironment()
        env.reset()
        obs = None
        for _ in range(MAX_STEPS):
            obs = env.step(SreDecisionAction(action_name="inspect_logs"))
        assert obs.done is True

    def test_step_without_reset_raises(self):
        env = SreDecisionEnvironment()
        with pytest.raises(RuntimeError):
            env.step(SreDecisionAction(action_name="inspect_logs"))

    def test_all_root_causes_covered(self):
        """Each root cause should be reachable."""
        env = SreDecisionEnvironment()
        seen = set()
        for _ in range(100):
            env.reset()
            seen.add(env._root_cause)
        assert seen == set(ROOT_CAUSES)


# ---------------------------------------------------------------------------
# Action model tests
# ---------------------------------------------------------------------------

class TestActionModel:
    def test_valid_action_names(self):
        for name in VALID_ACTIONS:
            a = SreDecisionAction(action_name=name)
            assert a.action_name == name

    def test_action_with_rationale(self):
        a = SreDecisionAction(action_name="inspect_logs", rationale="I see a spike")
        assert a.rationale == "I see a spike"
