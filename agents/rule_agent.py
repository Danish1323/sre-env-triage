from typing import Any

from agents.base import Agent
from models import SreDecisionAction

class RuleBasedAgent(Agent):
    """Agent that uses simple heuristics to solve the incident."""
    
    def act(self, obs: Any) -> SreDecisionAction:
        d = obs.model_dump()
        logs = d.get("logs", {})
        observer = d.get("observer", {})
        
        # If we just took a remediation step and episode isn't over, resolve it
        last_action = d.get("last_action", "")
        if last_action in ["restart_service", "rollback_service"] and not obs.incident_resolved:
            return SreDecisionAction(action_name="resolve_incident", rationale="Remediation applied, resolving.")
        
        if last_action == "declare_severity_high" and not obs.incident_resolved:
            return SreDecisionAction(action_name="rollback_service", rationale="Declared high severity, rolling back.")

        # Heuristic 1: Latency Spike + High CPU -> Server A failure or load
        latency = logs.get("latency_spike", False)
        cpu = observer.get("cpu_usage", 0.0)
        db_conn = observer.get("db_connections", 0)
        
        if latency and cpu > 0.6:
            return SreDecisionAction(action_name="restart_service", rationale="High CPU + Latency -> Restart service")
            
        # Heuristic 2: Memory leak -> High memory
        mem = observer.get("memory_usage", 0.0)
        if mem > 0.7:
            return SreDecisionAction(action_name="restart_service", rationale="High memory -> Restart service")
            
        # Heuristic 3: Network Latency + DB Spike
        if latency and db_conn > 500:
            return SreDecisionAction(action_name="rollback_service", rationale="DB spike + latency -> Rollback service")
            
        # Default diagnostic
        if last_action == "inspect_logs":
            return SreDecisionAction(action_name="inspect_metrics", rationale="Already checked logs, checking metrics.")
        elif last_action == "inspect_metrics":
            return SreDecisionAction(action_name="resolve_incident", rationale="Nothing obvious found, closing as transient.")
            
        return SreDecisionAction(action_name="inspect_logs", rationale="Starting investigation with logs.")
