"""
Reward function for the SRE Decision Environment.

Design principles (Phase 1):
- Reward correct diagnosis/resolution heavily.
- Penalise harmful or irrelevant actions mildly.
- Penalise excessive steps to encourage efficiency.
- Penalise invalid action names.

Phase 1 — Dec-POMDP scaffold.
"""

from typing import Optional


# ---------------------------------------------------------------------------
# Action classification tables
# ---------------------------------------------------------------------------

# Actions that gather information — no direct state change
DIAGNOSTIC_ACTIONS = {
    "inspect_logs",
    "inspect_metrics",
    "check_deploy_history",
    "declare_severity_low",
    "declare_severity_high",
}

# Actions that mutate the system — carry risk if wrong
REMEDIATION_ACTIONS = {
    "restart_service",
    "rollback_service",
    "resolve_incident",
}

# Correct remediation per root cause
CORRECT_REMEDIATION: dict = {
    "server_A_failure": {"restart_service", "rollback_service"},
    "memory_leak": {"restart_service"},
    "network_latency": {"resolve_incident"},   # infra issue, escalate/resolve
    "transient_spike": {"resolve_incident"},   # wait and close
    "no_issue": {"resolve_incident"},
}

# Harmful remediations per root cause (wrong action on a real problem)
HARMFUL_REMEDIATION: dict = {
    "server_A_failure": {"resolve_incident"},  # closing a real outage is bad
    "memory_leak": {"rollback_service"},        # rollback doesn't fix a leak
    "network_latency": {"restart_service", "rollback_service"},
    "transient_spike": {"restart_service", "rollback_service"},
    "no_issue": {"restart_service", "rollback_service"},
}


# ---------------------------------------------------------------------------
# Step reward
# ---------------------------------------------------------------------------

def compute_step_reward(
    action_name: str,
    root_cause: str,
    step: int,
    max_steps: int,
    is_valid: bool,
) -> float:
    """
    Compute the immediate reward for a single environment step.

    Args:
        action_name:  The action chosen by the agent.
        root_cause:   The hidden root cause (known only to the env).
        step:         Current step number (0-indexed).
        max_steps:    Episode step limit.
        is_valid:     Whether the action_name is in the valid action set.

    Returns:
        A float reward value.
    """
    if not is_valid:
        return -0.5  # penalise invalid action names

    # Small step cost to encourage efficiency
    step_penalty = -0.05

    if action_name in DIAGNOSTIC_ACTIONS:
        return step_penalty + 0.10   # small positive: info gathering is good

    if action_name in REMEDIATION_ACTIONS:
        correct = CORRECT_REMEDIATION.get(root_cause, set())
        harmful = HARMFUL_REMEDIATION.get(root_cause, set())

        if action_name in correct:
            # Bonus for resolving quickly
            efficiency_bonus = max(0.0, (max_steps - step) / max_steps) * 0.5
            return 1.0 + efficiency_bonus

        if action_name in harmful:
            return -1.0  # harmful action penalty

        return step_penalty + 0.0   # neutral remediation (not wrong, not right)

    return step_penalty


# ---------------------------------------------------------------------------
# Terminal reward
# ---------------------------------------------------------------------------

def compute_terminal_reward(
    resolved: bool,
    root_cause: str,
    last_action: Optional[str],
    steps_taken: int,
    max_steps: int,
) -> float:
    """
    Compute an additional reward at episode end.

    Args:
        resolved:     Whether the agent issued a terminal action.
        root_cause:   Hidden root cause.
        last_action:  Last action taken.
        steps_taken:  Total steps consumed.
        max_steps:    Episode step limit.

    Returns:
        Terminal reward float.
    """
    if not resolved:
        return -0.5   # ran out of steps without resolving

    correct = CORRECT_REMEDIATION.get(root_cause, set())
    if last_action in correct:
        return 0.0    # already got step reward for correct action

    return -0.3       # resolved but with wrong action (better than timeout)
