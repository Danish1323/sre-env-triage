import os
from crewai import Agent, Task, Crew, Process, LLM
from v2.agents.tools import (
    query_logs, query_metrics, share_info, 
    execute_fix, propose_hypothesis, resolve_incident
)

# 1. Setup LLM
llm = LLM(
    model="openrouter/openai/gpt-oss-120b",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    temperature=0.0 # Strict for technical tasks
)

# 2. SRE SERVICE CATALOG (The "Ground Truth" for the model)
CATALOG = """
SERVICES: [api_gateway, auth_service, user_db, payment_service, cache_service]
ROOT_CAUSES: [cpu_saturation, memory_leak, db_connection_leak, cascading_failure]
FIX_TYPES: 
 - 'scale' (use for cpu_saturation)
 - 'restart' (use for memory_leak or db_connection_leak)
"""

# 3. Define Agents with the Catalog
log_specialist = Agent(
    role="Log Investigator",
    goal=f"Extract logs for the catalog services: {CATALOG}",
    backstory="You are a senior SRE. You provide raw log data. You only look at services listed in the catalog.",
    tools=[query_logs, share_info],
    llm=llm,
    max_iter=10,
    verbose=True
)

metric_specialist = Agent(
    role="Metric Analyst",
    goal=f"Extract metrics for the catalog services: {CATALOG}",
    backstory="You are a performance expert. You provide raw metrics. You only look at services listed in the catalog.",
    tools=[query_metrics, share_info],
    llm=llm,
    max_iter=10,
    verbose=True
)

infra_specialist = Agent(
    role="Infra Executor",
    goal=f"Apply fixes using the catalog definitions: {CATALOG}",
    backstory="You execute remediation. You MUST use 'scale' for CPU and 'restart' for Leaks.",
    tools=[execute_fix, share_info],
    llm=llm,
    max_iter=10,
    verbose=True
)

incident_manager = Agent(
    role="Incident Commander",
    goal=f"Identify the EXACT root cause from the catalog: {CATALOG}",
    backstory="You are the lead orchestrator. You MUST use the EXACT root cause strings from the catalog when calling resolve_incident.",
    tools=[propose_hypothesis, resolve_incident, share_info],
    llm=llm,
    max_iter=10,
    verbose=True
)

# 4. Define Tasks
def create_sre_crew():
    log_task = Task(
        description="Query logs for all catalog services. Share findings.",
        expected_output="Log anomalies reported.",
        agent=log_specialist
    )

    metric_task = Task(
        description="Query metrics for all catalog services. Share findings.",
        expected_output="Metric anomalies reported.",
        agent=metric_specialist
    )

    triage_task = Task(
        description=f"Based on reports, pick the EXACT root cause from: [cpu_saturation, memory_leak, db_connection_leak, cascading_failure].",
        expected_output="A specific root cause hypothesis.",
        agent=incident_manager,
        context=[log_task, metric_task]
    )

    fix_task = Task(
        description="Execute the correct fix type for the chosen hypothesis.",
        expected_output="Remediation status.",
        agent=infra_specialist,
        context=[triage_task]
    )

    res_task = Task(
        description="Verify and resolve using the EXACT root cause string.",
        expected_output="Resolution status.",
        agent=incident_manager,
        context=[fix_task]
    )

    return Crew(
        agents=[log_specialist, metric_specialist, infra_specialist, incident_manager],
        tasks=[log_task, metric_task, triage_task, fix_task, res_task],
        process=Process.sequential,
        verbose=True
    )
