"""
Sensor modules for the SRE Decision Environment.

These are NOT independent RL agents — they are deterministic-but-noisy
functions that convert the hidden root cause into partial observations
delivered to the orchestrator.

Phase 1 — Dec-POMDP scaffold.
"""

import random
from typing import Dict, Any

# ---------------------------------------------------------------------------
# Root cause registry
# ---------------------------------------------------------------------------

ROOT_CAUSES = [
    "server_A_failure",
    "memory_leak",
    "network_latency",
    "transient_spike",
    "no_issue",
]


# ---------------------------------------------------------------------------
# Logs sensor
# ---------------------------------------------------------------------------

def get_logs_signal(root_cause: str) -> Dict[str, Any]:
    """
    Convert the hidden root cause into a noisy logs observation.

    The same root cause does NOT always produce the same signal —
    stochasticity is intentional to force the agent to reason under
    uncertainty rather than memorise a lookup table.

    Args:
        root_cause: One of the ROOT_CAUSES strings.

    Returns:
        Dict with keys: latency_spike, error_rate, log_anomaly_score.
    """
    r = random.random()  # noise factor in [0, 1)

    if root_cause == "server_A_failure":
        return {
            "latency_spike": r > 0.15,          # high chance
            "error_rate": round(0.55 + r * 0.35, 3),
            "log_anomaly_score": round(0.70 + r * 0.25, 3),
        }
    elif root_cause == "memory_leak":
        return {
            "latency_spike": r > 0.40,
            "error_rate": round(0.20 + r * 0.30, 3),
            "log_anomaly_score": round(0.50 + r * 0.30, 3),
        }
    elif root_cause == "network_latency":
        return {
            "latency_spike": r > 0.20,
            "error_rate": round(0.10 + r * 0.25, 3),
            "log_anomaly_score": round(0.30 + r * 0.35, 3),
        }
    elif root_cause == "transient_spike":
        # Misleading: sometimes looks like a real problem, sometimes benign
        return {
            "latency_spike": r > 0.50,
            "error_rate": round(0.05 + r * 0.20, 3),
            "log_anomaly_score": round(0.10 + r * 0.40, 3),
        }
    else:  # no_issue
        return {
            "latency_spike": r > 0.90,          # rare false-positive
            "error_rate": round(r * 0.05, 3),
            "log_anomaly_score": round(r * 0.10, 3),
        }


# ---------------------------------------------------------------------------
# Observer sensor
# ---------------------------------------------------------------------------

def get_observer_signal(root_cause: str) -> Dict[str, Any]:
    """
    Convert the hidden root cause into a noisy observer observation.

    Args:
        root_cause: One of the ROOT_CAUSES strings.

    Returns:
        Dict with keys: server_b_health, cpu_usage, memory_usage.
    """
    r = random.random()

    if root_cause == "server_A_failure":
        health = random.choice(["degraded", "down", "down"])  # weighted toward down
        return {
            "server_b_health": health,
            "cpu_usage": round(0.60 + r * 0.35, 3),
            "memory_usage": round(0.40 + r * 0.40, 3),
        }
    elif root_cause == "memory_leak":
        health = random.choice(["healthy", "degraded", "degraded"])
        return {
            "server_b_health": health,
            "cpu_usage": round(0.30 + r * 0.30, 3),
            "memory_usage": round(0.70 + r * 0.28, 3),
        }
    elif root_cause == "network_latency":
        health = random.choice(["healthy", "degraded"])
        return {
            "server_b_health": health,
            "cpu_usage": round(0.20 + r * 0.30, 3),
            "memory_usage": round(0.20 + r * 0.30, 3),
        }
    elif root_cause == "transient_spike":
        health = random.choice(["healthy", "healthy", "degraded"])
        return {
            "server_b_health": health,
            "cpu_usage": round(0.40 + r * 0.30, 3),
            "memory_usage": round(0.30 + r * 0.25, 3),
        }
    else:  # no_issue
        return {
            "server_b_health": "healthy",
            "cpu_usage": round(r * 0.20, 3),
            "memory_usage": round(r * 0.20, 3),
        }
