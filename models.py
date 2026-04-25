"""
Data models for the SRE Decision Environment.

Defines the structured Action and Observation types used by the
orchestrator agent when interacting with the OpenEnv-compatible server.

Phase 1 — Dec-POMDP scaffold.
"""

from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------

VALID_ACTIONS: List[str] = [
    "inspect_logs",
    "inspect_metrics",
    "check_deploy_history",
    "declare_severity_low",
    "declare_severity_high",
    "restart_service",
    "rollback_service",
    "resolve_incident",
]


class SreDecisionAction(Action):
    """
    A single action taken by the orchestrator agent.

    The ``action_name`` must be one of the entries in VALID_ACTIONS.
    The optional ``rationale`` field carries the agent's chain-of-thought
    so it can be logged and evaluated without affecting the environment.
    """

    action_name: str = Field(
        ...,
        description=(
            "Name of the action to execute. Must be one of: "
            + ", ".join(VALID_ACTIONS)
        ),
    )
    rationale: Optional[str] = Field(
        default=None,
        description="Agent's chain-of-thought reasoning (optional, not used by env).",
    )


# ---------------------------------------------------------------------------
# Observation space
# ---------------------------------------------------------------------------


class LogsSignal(Observation):
    """Noisy signal from the logs sensor module."""

    latency_spike: bool = Field(default=False, description="Was a latency spike detected?")
    error_rate: float = Field(default=0.0, description="Observed error rate (0-1 scale).")
    log_anomaly_score: float = Field(
        default=0.0, description="Anomaly score from log pattern analysis (0-1 scale)."
    )


class ObserverSignal(Observation):
    """Noisy signal from the observer sensor module."""

    server_b_health: str = Field(
        default="healthy",
        description="Observed health of server B: 'healthy', 'degraded', or 'down'.",
    )
    cpu_usage: float = Field(default=0.0, description="Observed CPU usage (0-1 scale).")
    memory_usage: float = Field(default=0.0, description="Observed memory usage (0-1 scale).")


class SreDecisionObservation(Observation):
    """
    Full observation bundle delivered to the orchestrator agent each step.

    Contains signals from all sensor modules plus episode metadata.
    The hidden root cause is NEVER included here.
    """

    logs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Noisy log sensor signals (latency_spike, error_rate, log_anomaly_score).",
    )
    observer: Dict[str, Any] = Field(
        default_factory=dict,
        description="Noisy observer sensor signals (server_b_health, cpu_usage, memory_usage).",
    )
    time_step: int = Field(default=0, description="Current step index within the episode.")
    last_action: Optional[str] = Field(
        default=None, description="The action taken on the previous step."
    )
    action_feedback: Optional[str] = Field(
        default=None, description="Human-readable feedback about the last action's effect."
    )
    incident_resolved: bool = Field(
        default=False, description="True when the episode has been terminated by the agent."
    )
    available_actions: List[str] = Field(
        default_factory=lambda: VALID_ACTIONS,
        description="The complete list of valid action names the agent may choose.",
    )
