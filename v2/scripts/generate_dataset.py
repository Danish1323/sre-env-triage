"""
Expert Dataset Generator for SRE Multi-Agent System
====================================================
Generates training data for 4 specialist agents + 1 lead manager.
Format: JSONL with (prompt, chosen_response, reward) for GRPO/PPO RL training via Unsloth.

Agents:
  - log_investigator   (1000 samples)
  - metric_analyst     (1000 samples)
  - incident_commander (1000 samples)
  - infra_executor     (1000 samples)
  - lead_manager       (5000 samples - knows everything)
"""

import json
import random
import sys
from pathlib import Path
from typing import Dict, Tuple, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# ── Scenario Primitives (mirror of v2/env/scenarios.py) ──────────────────────

SERVICES = ["api_gateway", "auth_service", "user_db", "frontend_service", "product_db", "cache_service"]

ROOT_CAUSES = ["cpu_saturation", "memory_leak", "db_connection_leak", "cascading_failure"]

FIX_MAP = {
    "cpu_saturation":    {"fix": "scale",   "target": "api_gateway"},
    "memory_leak":       {"fix": "restart", "target": "auth_service"},
    "db_connection_leak":{"fix": "restart", "target": "user_db"},
    "cascading_failure": {"fix": "restart", "target": "cache_service"},
}

def generate_scenario(difficulty: str = "medium") -> Dict:
    """Generate a complete, deterministic scenario dict."""
    rc = random.choice(ROOT_CAUSES)
    target = FIX_MAP[rc]["target"]
    fix_type = FIX_MAP[rc]["fix"]

    logs = {svc: "INFO: Service running normally." for svc in SERVICES}
    metrics = {
        svc: {
            "cpu": round(random.uniform(0.1, 0.4), 3),
            "latency": round(random.uniform(0.01, 0.05), 3),
            "error_rate": 0.0
        }
        for svc in SERVICES
    }

    # Apply failure signatures
    if rc == "db_connection_leak":
        metrics[target]["latency"] = 1.5
        metrics[target]["error_rate"] = 0.8
        logs[target] = "ERROR: connection pool exhausted. Waiting for connection..."
        if difficulty in ["medium", "hard"]:
            metrics["auth_service"]["latency"] = round(random.uniform(1.0, 1.4), 3)
            logs["auth_service"] = "WARN: Timeout connecting to user_db"

    elif rc == "memory_leak":
        metrics[target]["cpu"] = 0.95
        logs[target] = "FATAL: OutOfMemoryError in auth_service"

    elif rc == "cascading_failure":
        metrics[target]["error_rate"] = 1.0
        logs[target] = "ERROR: Cache eviction failed"
        metrics["frontend_service"]["latency"] = round(random.uniform(1.8, 2.5), 3)
        logs["frontend_service"] = "ERROR: Cache miss spike, database overwhelmed"
        metrics["product_db"]["cpu"] = round(random.uniform(0.9, 0.99), 3)

    elif rc == "cpu_saturation":
        metrics[target]["cpu"] = round(random.uniform(0.95, 0.99), 3)
        metrics[target]["latency"] = round(random.uniform(0.6, 1.0), 3)
        logs[target] = "WARN: Thread starvation detected"

    return {
        "root_cause": rc,
        "target_service": target,
        "fix_type": fix_type,
        "logs": logs,
        "metrics": metrics,
        "difficulty": difficulty
    }


# ── Per-Agent System Prompts ──────────────────────────────────────────────────

LOG_INVESTIGATOR_SYSTEM = """You are the Log Investigator in an SRE incident response team.
Your ONLY job is to query service logs and report anomalies.

Available services: [api_gateway, auth_service, user_db, frontend_service, product_db, cache_service]
Severity levels: INFO (normal) | WARN (degraded) | ERROR (failing) | FATAL (critical)

Rules:
- Query EVERY service before sharing findings.
- ONLY report what the logs actually say. NEVER invent log entries.
- Share your findings on the message bus using share_info.
- If a log says "INFO: Service running normally.", report it as healthy.
- Focus on ERROR and FATAL entries as primary indicators."""

METRIC_ANALYST_SYSTEM = """You are the Metric Analyst in an SRE incident response team.
Your ONLY job is to query service metrics and identify bottlenecks.

Available services: [api_gateway, auth_service, user_db, frontend_service, product_db, cache_service]
Thresholds:
  cpu > 0.85 = HIGH LOAD
  latency > 0.5 = HIGH LATENCY
  error_rate > 0.1 = ELEVATED ERRORS

Rules:
- Query metrics for ALL services.
- Report ONLY what the metrics tool returns. NEVER invent numbers.
- Share anomalies via share_info with specific values.
- A cpu of 0.95 on auth_service with FATAL log = memory_leak."""

INCIDENT_COMMANDER_SYSTEM = """You are the Incident Commander in an SRE incident response team.
Your job is to analyze team findings and determine the EXACT root cause.

Valid root cause IDs (use EXACTLY these strings):
  - cpu_saturation      → api_gateway CPU > 0.85, WARN: Thread starvation
  - memory_leak         → auth_service CPU > 0.85, FATAL: OutOfMemoryError
  - db_connection_leak  → user_db latency > 1.0, ERROR: connection pool exhausted
  - cascading_failure   → cache_service error_rate = 1.0, frontend + product_db degraded

Root cause → Correct fix mapping:
  cpu_saturation    → scale  api_gateway
  memory_leak       → restart auth_service
  db_connection_leak → restart user_db
  cascading_failure  → restart cache_service

Rules:
- Call propose_hypothesis with the exact root cause ID string.
- Call resolve_incident with the exact root cause ID string.
- NEVER use descriptive names like "high cpu" or "timeout" — use the ID."""

INFRA_EXECUTOR_SYSTEM = """You are the Infra Executor in an SRE incident response team.
Your ONLY job is to apply the correct remediation fix.

Fix catalog (MEMORIZE THIS):
  cpu_saturation    → execute_fix(service_name="api_gateway",   fix_type="scale")
  memory_leak       → execute_fix(service_name="auth_service",  fix_type="restart")
  db_connection_leak → execute_fix(service_name="user_db",       fix_type="restart")
  cascading_failure  → execute_fix(service_name="cache_service", fix_type="restart")

Rules:
- Read the hypothesis from the Incident Commander.
- Apply EXACTLY the fix from the catalog above.
- Do NOT apply "scale" for a leak or "restart" for CPU issues.
- After executing, share the result via share_info."""

LEAD_MANAGER_SYSTEM = """You are the Lead SRE Manager. You oversee the full incident response pipeline.
You understand every role: log investigation, metric analysis, triage, and remediation.

Environment Catalog:
  Services: [api_gateway, auth_service, user_db, frontend_service, product_db, cache_service]
  Root Causes: [cpu_saturation, memory_leak, db_connection_leak, cascading_failure]
  Fix Map:
    cpu_saturation    → scale   api_gateway
    memory_leak       → restart auth_service
    db_connection_leak → restart user_db
    cascading_failure  → restart cache_service

Log Signatures:
  WARN: Thread starvation      → cpu_saturation on api_gateway
  FATAL: OutOfMemoryError      → memory_leak on auth_service
  ERROR: connection pool       → db_connection_leak on user_db
  ERROR: Cache eviction failed → cascading_failure on cache_service

Metric Signatures:
  api_gateway   cpu > 0.85  AND latency > 0.5 → cpu_saturation
  auth_service  cpu > 0.85                     → memory_leak
  user_db       latency > 1.0, error_rate > 0.5 → db_connection_leak
  cache_service error_rate = 1.0               → cascading_failure

You ALWAYS follow this pipeline:
  1. Query logs for all services → identify anomalies
  2. Query metrics for all services → confirm with numbers
  3. Propose the EXACT root cause ID
  4. Execute the correct fix
  5. Resolve the incident with the EXACT root cause ID"""


# ── Sample Builders ──────────────────────────────────────────────────────────

def build_log_investigator_sample(scenario: Dict) -> Dict:
    """A sample where the Log Investigator queries all logs and shares findings."""
    logs = scenario["logs"]
    rc = scenario["root_cause"]
    target = scenario["target_service"]

    # Build the "observation" that would be shown to this agent
    user_prompt = (
        "INCIDENT ACTIVE. Query all service logs and report every anomaly to the team.\n"
        "Services to check: api_gateway, auth_service, user_db, frontend_service, product_db, cache_service"
    )

    # Simulate what good tool calls look like
    tool_calls = []
    for svc in SERVICES:
        tool_calls.append(f'Action: query_logs\nAction Input: {{"service_name": "{svc}"}}\nObservation: Logs for {svc}: {logs[svc]}')

    anomalies = [f"{svc}: {logs[svc]}" for svc in SERVICES if logs[svc] != "INFO: Service running normally."]
    healthy = [svc for svc in SERVICES if logs[svc] == "INFO: Service running normally."]

    summary = f"Anomalies detected: {'; '.join(anomalies) if anomalies else 'None'}. Healthy: {', '.join(healthy)}."

    tool_calls.append(
        f'Action: share_info\nAction Input: {{"message": "{summary}"}}\nObservation: Message shared.'
    )

    response = "\n\n".join(tool_calls) + f"\n\nThought: I now know the final answer.\nFinal Answer: {summary}"

    # Reward: +1.0 for querying all 6 services + finding anomaly, +0.2 per correct share
    reward = 1.0 + (0.2 if anomalies else 0.0)

    return {
        "agent": "log_investigator",
        "root_cause": rc,
        "target_service": target,
        "messages": [
            {"role": "system", "content": LOG_INVESTIGATOR_SYSTEM},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": response}
        ],
        "reward": reward,
        "metadata": {
            "anomalies_found": anomalies,
            "difficulty": scenario["difficulty"]
        }
    }


def build_metric_analyst_sample(scenario: Dict) -> Dict:
    """A sample where the Metric Analyst queries all metrics and flags anomalies."""
    metrics = scenario["metrics"]
    rc = scenario["root_cause"]
    target = scenario["target_service"]

    user_prompt = (
        "INCIDENT ACTIVE. Query metrics for all services and flag any CPU > 0.85, "
        "Latency > 0.5, or Error Rate > 0.1. Share your findings."
    )

    tool_calls = []
    for svc in SERVICES:
        m = metrics[svc]
        tool_calls.append(
            f'Action: query_metrics\nAction Input: {{"service_name": "{svc}"}}\n'
            f'Observation: Metrics for {svc}: {json.dumps(m)}'
        )

    anomalies = []
    for svc, m in metrics.items():
        flags = []
        if m["cpu"] > 0.85:
            flags.append(f"HIGH CPU ({m['cpu']})")
        if m["latency"] > 0.5:
            flags.append(f"HIGH LATENCY ({m['latency']}s)")
        if m["error_rate"] > 0.1:
            flags.append(f"HIGH ERROR RATE ({m['error_rate']})")
        if flags:
            anomalies.append(f"{svc}: {', '.join(flags)}")

    summary = f"Metric anomalies: {'; '.join(anomalies) if anomalies else 'All services nominal'}."

    tool_calls.append(
        f'Action: share_info\nAction Input: {{"message": "{summary}"}}\nObservation: Message shared.'
    )

    response = "\n\n".join(tool_calls) + f"\n\nThought: I now know the final answer.\nFinal Answer: {summary}"

    reward = 1.0 + (0.2 if anomalies else 0.0)

    return {
        "agent": "metric_analyst",
        "root_cause": rc,
        "target_service": target,
        "messages": [
            {"role": "system", "content": METRIC_ANALYST_SYSTEM},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": response}
        ],
        "reward": reward,
        "metadata": {
            "anomalies_found": anomalies,
            "difficulty": scenario["difficulty"]
        }
    }


def build_incident_commander_sample(scenario: Dict) -> Dict:
    """A sample where the Commander reads the bus, proposes, and resolves."""
    rc = scenario["root_cause"]
    target = scenario["target_service"]
    fix = scenario["fix_type"]
    logs = scenario["logs"]
    metrics = scenario["metrics"]

    # Simulate what the team reported on the bus
    log_anomalies = [f"{svc}: {logs[svc]}" for svc in SERVICES if logs[svc] != "INFO: Service running normally."]
    metric_anomalies = []
    for svc, m in metrics.items():
        if m["cpu"] > 0.85 or m["latency"] > 0.5 or m["error_rate"] > 0.1:
            metric_anomalies.append(f"{svc} cpu={m['cpu']} lat={m['latency']} err={m['error_rate']}")

    user_prompt = (
        f"The team has reported the following on the message bus:\n"
        f"LOG INVESTIGATOR: {'; '.join(log_anomalies) if log_anomalies else 'All logs normal.'}\n"
        f"METRIC ANALYST: {'; '.join(metric_anomalies) if metric_anomalies else 'All metrics normal.'}\n\n"
        f"Propose the correct root cause hypothesis and resolve the incident."
    )

    response = (
        f"Thought: Based on the log anomaly '{log_anomalies[0] if log_anomalies else 'none'}' "
        f"and metric signature '{metric_anomalies[0] if metric_anomalies else 'none'}', "
        f"the root cause is clearly {rc}.\n\n"
        f'Action: propose_hypothesis\nAction Input: {{"hypothesis": "{rc}"}}\n'
        f"Observation: Hypothesis proposed: {rc}\n\n"
        f"Thought: The hypothesis is confirmed. I will now resolve the incident.\n\n"
        f'Action: resolve_incident\nAction Input: {{"root_cause_id": "{rc}"}}\n'
        f"Observation: Incident resolved with root cause: {rc}\n\n"
        f"Thought: I now know the final answer.\nFinal Answer: Incident resolved. Root cause: {rc}. "
        f"Recommended fix: {fix} on {target}."
    )

    reward = 1.5  # +1.0 for correct hypothesis + 0.5 for correct resolution

    return {
        "agent": "incident_commander",
        "root_cause": rc,
        "target_service": target,
        "messages": [
            {"role": "system", "content": INCIDENT_COMMANDER_SYSTEM},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": response}
        ],
        "reward": reward,
        "metadata": {
            "difficulty": scenario["difficulty"]
        }
    }


def build_infra_executor_sample(scenario: Dict) -> Dict:
    """A sample where the Executor gets the hypothesis and applies the correct fix."""
    rc = scenario["root_cause"]
    target = scenario["target_service"]
    fix = scenario["fix_type"]

    user_prompt = (
        f"The Incident Commander has identified the root cause as: {rc}\n"
        f"Apply the correct remediation fix according to the fix catalog."
    )

    response = (
        f"Thought: The root cause is {rc}. According to the fix catalog:\n"
        f"  {rc} → execute_fix(service_name='{target}', fix_type='{fix}')\n"
        f"I will apply this fix now.\n\n"
        f'Action: execute_fix\nAction Input: {{"service_name": "{target}", "fix_type": "{fix}"}}\n'
        f"Observation: Remediation Result: Executed {fix} on {target}\n\n"
        f'Action: share_info\nAction Input: {{"message": "Applied {fix} to {target}. Fix executed successfully."}}\n'
        f"Observation: Message shared.\n\n"
        f"Thought: I now know the final answer.\n"
        f"Final Answer: Applied {fix} to {target} to resolve {rc}. Result: Success."
    )

    reward = 1.0  # +1.0 for correct fix on correct service

    return {
        "agent": "infra_executor",
        "root_cause": rc,
        "target_service": target,
        "messages": [
            {"role": "system", "content": INFRA_EXECUTOR_SYSTEM},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": response}
        ],
        "reward": reward,
        "metadata": {
            "fix_applied": fix,
            "service": target,
            "difficulty": scenario["difficulty"]
        }
    }


def build_lead_manager_sample(scenario: Dict) -> Dict:
    """A sample where the Lead Manager runs the full E2E pipeline."""
    rc = scenario["root_cause"]
    target = scenario["target_service"]
    fix = scenario["fix_type"]
    logs = scenario["logs"]
    metrics = scenario["metrics"]

    user_prompt = (
        "INCIDENT ACTIVE. Run the full SRE response pipeline:\n"
        "1. Query all logs and metrics\n"
        "2. Identify the root cause (use EXACT ID)\n"
        "3. Execute the correct fix\n"
        "4. Resolve the incident"
    )

    # Build the full E2E chain-of-thought
    log_steps = "\n\n".join([
        f'Action: query_logs\nAction Input: {{"service_name": "{svc}"}}\nObservation: Logs for {svc}: {logs[svc]}'
        for svc in SERVICES
    ])
    metric_steps = "\n\n".join([
        f'Action: query_metrics\nAction Input: {{"service_name": "{svc}"}}\nObservation: Metrics for {svc}: {json.dumps(metrics[svc])}'
        for svc in SERVICES
    ])

    anomaly_log = [f"{svc}: {logs[svc]}" for svc in SERVICES if logs[svc] != "INFO: Service running normally."]
    anomaly_metric = [
        f"{svc} cpu={metrics[svc]['cpu']} lat={metrics[svc]['latency']}"
        for svc in SERVICES if metrics[svc]["cpu"] > 0.85 or metrics[svc]["latency"] > 0.5 or metrics[svc]["error_rate"] > 0.1
    ]

    response = (
        f"Thought: I'll start by querying all service logs.\n\n"
        f"{log_steps}\n\n"
        f"Thought: Now querying all service metrics.\n\n"
        f"{metric_steps}\n\n"
        f"Thought: Analysis:\n"
        f"  Log anomalies: {'; '.join(anomaly_log) if anomaly_log else 'None'}\n"
        f"  Metric anomalies: {'; '.join(anomaly_metric) if anomaly_metric else 'None'}\n"
        f"  Root cause pattern matches: {rc}\n\n"
        f'Action: propose_hypothesis\nAction Input: {{"hypothesis": "{rc}"}}\nObservation: Hypothesis proposed: {rc}\n\n'
        f'Action: execute_fix\nAction Input: {{"service_name": "{target}", "fix_type": "{fix}"}}\nObservation: Remediation Result: Executed {fix} on {target}\n\n'
        f'Action: resolve_incident\nAction Input: {{"root_cause_id": "{rc}"}}\nObservation: Incident resolved with root cause: {rc}\n\n'
        f"Thought: I now know the final answer.\n"
        f"Final Answer: Incident RESOLVED. Root cause: {rc}. Fix applied: {fix} on {target}."
    )

    reward = 2.0  # Full reward for full E2E success

    return {
        "agent": "lead_manager",
        "root_cause": rc,
        "target_service": target,
        "messages": [
            {"role": "system", "content": LEAD_MANAGER_SYSTEM},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": response}
        ],
        "reward": reward,
        "metadata": {
            "difficulty": scenario["difficulty"],
            "total_tools_called": 16,  # 6 logs + 6 metrics + propose + fix + resolve + share
            "is_full_pipeline": True
        }
    }


# ── Main Generator ────────────────────────────────────────────────────────────

def generate_dataset(
    output_dir: str = "data/training",
    specialist_samples: int = 1000,
    manager_samples: int = 5000,
    seed: int = 42
):
    random.seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    difficulties = ["easy", "medium", "medium", "hard"]  # Weighted toward medium

    agent_configs = [
        ("log_investigator",   specialist_samples, build_log_investigator_sample),
        ("metric_analyst",     specialist_samples, build_metric_analyst_sample),
        ("incident_commander", specialist_samples, build_incident_commander_sample),
        ("infra_executor",     specialist_samples, build_infra_executor_sample),
        ("lead_manager",       manager_samples,    build_lead_manager_sample),
    ]

    total_stats = {}

    for agent_name, n_samples, builder_fn in agent_configs:
        print(f"\n[+] Generating {n_samples} samples for: {agent_name}")
        samples = []

        rc_counts = {rc: 0 for rc in ROOT_CAUSES}

        for i in range(n_samples):
            difficulty = random.choice(difficulties)
            scenario = generate_scenario(difficulty)
            sample = builder_fn(scenario)
            samples.append(sample)
            rc_counts[scenario["root_cause"]] += 1

        # Save as JSONL
        out_file = output_path / f"{agent_name}_train.jsonl"
        with open(out_file, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

        # Also save Unsloth-compatible format (flat prompt/chosen/reward)
        unsloth_file = output_path / f"{agent_name}_unsloth.jsonl"
        with open(unsloth_file, "w") as f:
            for s in samples:
                unsloth_sample = {
                    "prompt": s["messages"][:-1],   # system + user
                    "chosen": s["messages"][-1]["content"],  # assistant response
                    "reward": s["reward"],
                    "agent": s["agent"],
                    "root_cause": s["root_cause"]
                }
                f.write(json.dumps(unsloth_sample) + "\n")

        total_stats[agent_name] = {
            "total": n_samples,
            "root_cause_distribution": rc_counts,
            "avg_reward": sum(s["reward"] for s in samples) / len(samples),
            "files": [str(out_file), str(unsloth_file)]
        }
        print(f"    ✓ Saved {n_samples} samples to {out_file}")
        print(f"    ✓ Unsloth format saved to {unsloth_file}")
        print(f"    Distribution: {rc_counts}")

    # Save stats file
    stats_file = output_path / "dataset_stats.json"
    with open(stats_file, "w") as f:
        json.dump(total_stats, f, indent=2)

    print(f"\n{'='*60}")
    print(f"✅ Dataset generation complete!")
    print(f"   Total samples: {sum(v['total'] for v in total_stats.values())}")
    print(f"   Output directory: {output_path.resolve()}")
    print(f"   Stats: {stats_file}")


if __name__ == "__main__":
    generate_dataset(
        output_dir="data/training",
        specialist_samples=1000,
        manager_samples=5000,
        seed=42
    )
