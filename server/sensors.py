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
            "latency_spike": random.gauss(0.8, 0.2) > 0.5,
            "error_rate": max(0.0, min(1.0, random.gauss(0.7, 0.15))),
            "log_anomaly_score": max(0.0, min(1.0, random.gauss(0.8, 0.1))),
            "five_xx_error_rate": max(0.0, min(1.0, random.gauss(0.6, 0.2))),
        }
    elif root_cause == "memory_leak":
        return {
            "latency_spike": random.gauss(0.6, 0.3) > 0.5,
            "error_rate": max(0.0, min(1.0, random.gauss(0.3, 0.15))),
            "log_anomaly_score": max(0.0, min(1.0, random.gauss(0.6, 0.2))),
            "five_xx_error_rate": max(0.0, min(1.0, random.gauss(0.1, 0.1))),
        }
    elif root_cause == "network_latency":
        return {
            "latency_spike": random.gauss(0.9, 0.1) > 0.5,
            "error_rate": max(0.0, min(1.0, random.gauss(0.2, 0.1))),
            "log_anomaly_score": max(0.0, min(1.0, random.gauss(0.4, 0.15))),
            "five_xx_error_rate": max(0.0, min(1.0, random.gauss(0.25, 0.15))),
        }
    elif root_cause == "transient_spike":
        # Misleading: sometimes looks like a real problem, sometimes benign
        return {
            "latency_spike": random.gauss(0.5, 0.4) > 0.5,
            "error_rate": max(0.0, min(1.0, random.gauss(0.15, 0.2))),
            "log_anomaly_score": max(0.0, min(1.0, random.gauss(0.3, 0.25))),
            "five_xx_error_rate": max(0.0, min(1.0, random.gauss(0.05, 0.1))),
        }
    else:  # no_issue
        return {
            "latency_spike": random.gauss(0.1, 0.1) > 0.5,
            "error_rate": max(0.0, min(1.0, random.gauss(0.02, 0.02))),
            "log_anomaly_score": max(0.0, min(1.0, random.gauss(0.05, 0.05))),
            "five_xx_error_rate": max(0.0, min(1.0, random.gauss(0.01, 0.01))),
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
            "cpu_usage": max(0.0, min(1.0, random.gauss(0.8, 0.15))),
            "memory_usage": max(0.0, min(1.0, random.gauss(0.6, 0.2))),
            "db_connections": int(max(0, random.gauss(50, 20))),
        }
    elif root_cause == "memory_leak":
        health = random.choice(["healthy", "degraded", "degraded"])
        return {
            "server_b_health": health,
            "cpu_usage": max(0.0, min(1.0, random.gauss(0.4, 0.15))),
            "memory_usage": max(0.0, min(1.0, random.gauss(0.9, 0.05))),
            "db_connections": int(max(0, random.gauss(150, 30))),
        }
    elif root_cause == "network_latency":
        health = random.choice(["healthy", "degraded"])
        return {
            "server_b_health": health,
            "cpu_usage": max(0.0, min(1.0, random.gauss(0.3, 0.1))),
            "memory_usage": max(0.0, min(1.0, random.gauss(0.3, 0.1))),
            "db_connections": int(max(0, random.gauss(800, 200))), # Spike in db conns
        }
    elif root_cause == "transient_spike":
        health = random.choice(["healthy", "healthy", "degraded"])
        return {
            "server_b_health": health,
            "cpu_usage": max(0.0, min(1.0, random.gauss(0.6, 0.2))),
            "memory_usage": max(0.0, min(1.0, random.gauss(0.4, 0.15))),
            "db_connections": int(max(0, random.gauss(300, 100))),
        }
    else:  # no_issue
        return {
            "server_b_health": "healthy",
            "cpu_usage": max(0.0, min(1.0, random.gauss(0.1, 0.05))),
            "memory_usage": max(0.0, min(1.0, random.gauss(0.15, 0.05))),
            "db_connections": int(max(0, random.gauss(100, 10))),
        }
