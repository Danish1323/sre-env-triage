from typing import List, Dict, Any, Optional
from pydantic import BaseModel

# Shared Message Model
class Message(BaseModel):
    sender: str
    content: str
    step: int

# 1. Investigator Observation
class InvestigatorObservation(BaseModel):
    logs: Dict[str, str] = {} # e.g. {"auth_service": "ERROR: Connection timeout..."}
    message_bus: List[Message] = []

# 2. Analyst Observation
class AnalystObservation(BaseModel):
    metrics: Dict[str, Dict[str, float]] = {} # e.g. {"auth_service": {"cpu": 0.95, "latency": 1.2}}
    message_bus: List[Message] = []

# 3. Coordinator Observation
class CoordinatorObservation(BaseModel):
    message_bus: List[Message] = []
    active_hypotheses: List[str] = []

# 4. Executor Observation
class ExecutorObservation(BaseModel):
    message_bus: List[Message] = []
    action_feedback: Optional[str] = None # e.g. "Restarted auth_service successfully"
