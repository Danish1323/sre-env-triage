"""
Orchestrator agent prompt builder.

Builds the system and user prompts for the orchestrator LLM.
The agent receives a structured text rendering of the current
SreDecisionObservation and must output a valid action name.

Phase 1 — single-agent orchestrator with chain-of-thought.
"""

from typing import Any, Dict, List

SYSTEM_PROMPT = """\
You are an expert SRE (Site Reliability Engineer) incident response agent.

Your job is to diagnose and resolve production incidents by observing partial, noisy signals
from the environment and choosing the best action.

## Rules
1. You CANNOT see the hidden root cause directly. You must infer it from signals.
2. Each step you must choose EXACTLY ONE action from the list below.
3. Think step-by-step before choosing. Output ONLY the action name on the final line.
4. Prefer diagnostic actions early in the episode. Use remediation actions only when confident.
5. Once you call `resolve_incident`, the episode ends.

## Available actions
- inspect_logs           → re-read the latest log sensor signals
- inspect_metrics        → re-read the latest observer/metrics signals
- check_deploy_history   → review recent deployments (no new signal generated)
- declare_severity_low   → tag the incident as P3/P4 (low severity)
- declare_severity_high  → tag the incident as P1/P2 (high severity)
- restart_service        → restart the primary affected service
- rollback_service       → roll back to the last stable deployment
- resolve_incident       → close the incident and end the episode

## Output format
Reason: <one-sentence reasoning>
Action: <exact action name>
"""


def build_user_prompt(
    obs: Dict[str, Any],
    history: List[Dict[str, str]],
) -> str:
    """
    Build the user-turn message from the current observation dict.

    Args:
        obs:     Dict representation of SreDecisionObservation.
        history: List of previous {action, feedback} dicts this episode.

    Returns:
        Formatted user prompt string.
    """
    logs = obs.get("logs", {})
    observer = obs.get("observer", {})
    step = obs.get("time_step", 0)
    last_action = obs.get("last_action") or "none"
    feedback = obs.get("action_feedback", "")

    lines = [
        f"=== Step {step} ===",
        "",
        "## Logs Signal",
        f"  latency_spike      : {logs.get('latency_spike', 'N/A')}",
        f"  error_rate         : {logs.get('error_rate', 'N/A')}",
        f"  log_anomaly_score  : {logs.get('log_anomaly_score', 'N/A')}",
        "",
        "## Observer Signal",
        f"  server_b_health    : {observer.get('server_b_health', 'N/A')}",
        f"  cpu_usage          : {observer.get('cpu_usage', 'N/A')}",
        f"  memory_usage       : {observer.get('memory_usage', 'N/A')}",
        "",
        f"Last action taken  : {last_action}",
        f"Environment feedback: {feedback}",
    ]

    if history:
        lines += ["", "## Action history this episode"]
        for i, h in enumerate(history[-5:], 1):  # show last 5
            lines.append(f"  {i}. {h['action']} → {h['feedback']}")

    lines += [
        "",
        "Based on the above signals, choose the single best action.",
        "Remember: output exactly:",
        "Reason: <short reasoning>",
        "Action: <action_name>",
    ]

    return "\n".join(lines)


def parse_action(llm_response: str) -> str:
    """
    Parse the LLM's response to extract a valid action name.

    Searches for a line starting with 'Action:' and returns the value.
    Falls back to scanning for known action keywords if parsing fails.

    Args:
        llm_response: Raw text output from the LLM.

    Returns:
        Extracted action name string (may still be invalid — env handles that).
    """
    from sre_decision_env.models import VALID_ACTIONS  # lazy import avoids circulars

    for line in llm_response.splitlines():
        if line.lower().startswith("action:"):
            candidate = line.split(":", 1)[1].strip().lower().replace(" ", "_")
            return candidate

    # Fallback: scan for known action names anywhere in the response
    lower = llm_response.lower()
    for action in VALID_ACTIONS:
        if action in lower:
            return action

    return "inspect_logs"  # safe default
