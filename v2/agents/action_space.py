from typing import Literal, Optional
from pydantic import BaseModel, Field

# 1. Investigator Actions
class InvestigatorAction(BaseModel):
    action_type: Literal["query_logs", "share_info", "idle"] = Field(..., description="Action type")
    target_service: Optional[str] = Field(None, description="Service to query logs for")
    message: Optional[str] = Field(None, description="Message to share on bus")

# 2. Analyst Actions
class AnalystAction(BaseModel):
    action_type: Literal["query_metrics", "share_info", "idle"] = Field(..., description="Action type")
    target_service: Optional[str] = Field(None, description="Service to query metrics for")
    message: Optional[str] = Field(None, description="Message to share on bus")

# 3. Coordinator Actions
class CoordinatorAction(BaseModel):
    action_type: Literal["propose_hypothesis", "finalize", "idle"] = Field(..., description="Action type")
    hypothesis: Optional[str] = Field(None, description="A hypothesis to share with the team")
    root_cause_guess: Optional[str] = Field(None, description="Final decision on root cause to end episode")

# 4. Executor Actions
class ExecutorAction(BaseModel):
    action_type: Literal["execute_fix", "idle"] = Field(..., description="Action type")
    fix_type: Optional[Literal["restart", "rollback", "scale"]] = Field(None, description="Type of fix")
    target_service: Optional[str] = Field(None, description="Service to apply fix to")
