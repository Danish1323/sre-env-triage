import os
from typing import Optional, List, Literal
from crewai.tools import tool
from v2.env.sre_env import SreDecisionEnvV2
from v2.agents.action_space import InvestigatorAction, AnalystAction, CoordinatorAction, ExecutorAction

# Global reference to the environment
_env_instance: Optional[SreDecisionEnvV2] = None

def set_env(env: SreDecisionEnvV2):
    global _env_instance
    _env_instance = env

@tool("query_logs")
def query_logs(service_name: str) -> str:
    """
    Query logs for a specific service.
    Valid services: [api_gateway, auth_service, user_db, payment_service, cache_service]
    """
    if _env_instance is None:
        return "Error: Environment not initialized."
    
    action = InvestigatorAction(action_type="query_logs", target_service=service_name)
    obs, rewards, dones, info = _env_instance.step({"investigator": action})
    
    # Extract logs from investigator observation
    logs = obs["investigator"]["logs"].get(service_name, "No logs found.")
    return f"Logs for {service_name}: {logs}"

@tool("query_metrics")
def query_metrics(service_name: str) -> str:
    """
    Query metrics (CPU, Latency, Error Rate) for a service.
    Valid services: [api_gateway, auth_service, user_db, payment_service, cache_service]
    """
    if _env_instance is None:
        return "Error: Environment not initialized."
    
    action = AnalystAction(action_type="query_metrics", target_service=service_name)
    obs, rewards, dones, info = _env_instance.step({"analyst": action})
    
    # Extract metrics from analyst observation
    metrics = obs["analyst"]["metrics"].get(service_name, "No metrics found.")
    return f"Metrics for {service_name}: {metrics}"

@tool("share_info")
def share_info(message: str) -> str:
    """Share findings with the team via the message bus."""
    if _env_instance is None:
        return "Error: Environment not initialized."
    
    # We use investigator as a generic sender for this tool
    action = InvestigatorAction(action_type="share_info", message=message)
    _env_instance.step({"investigator": action})
    return f"Message shared on bus: {message}"

@tool("execute_fix")
def execute_fix(service_name: str, fix_type: Literal["restart", "rollback", "scale"]) -> str:
    """
    Execute a remediation fix.
    Use 'scale' for cpu_saturation, 'restart' for memory_leak or db_connection_leak.
    """
    if _env_instance is None:
        return "Error: Environment not initialized."
    
    action = ExecutorAction(action_type="execute_fix", target_service=service_name, fix_type=fix_type)
    obs, rewards, dones, info = _env_instance.step({"executor": action})
    
    feedback = obs["executor"].get("action_feedback", "Fix executed.")
    return f"Remediation Result: {feedback}"

@tool("propose_hypothesis")
def propose_hypothesis(hypothesis: str) -> str:
    """Propose a root cause hypothesis to the team."""
    if _env_instance is None:
        return "Error: Environment not initialized."
    
    action = CoordinatorAction(action_type="propose_hypothesis", hypothesis=hypothesis)
    _env_instance.step({"coordinator": action})
    return f"Hypothesis proposed: {hypothesis}"

@tool("resolve_incident")
def resolve_incident(root_cause_id: str) -> str:
    """
    Finalize the incident resolution with the EXACT root cause ID.
    IDs: [cpu_saturation, memory_leak, db_connection_leak, cascading_failure]
    """
    if _env_instance is None:
        return "Error: Environment not initialized."
    
    action = CoordinatorAction(action_type="finalize", root_cause_guess=root_cause_id)
    _env_instance.step({"coordinator": action})
    return f"Incident resolution attempted with ID: {root_cause_id}. Check final state for success."
