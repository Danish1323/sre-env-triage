from typing import Dict, List, Optional
from v2.env.communication import MessageBus
from v2.env.scenarios import ScenarioGenerator

class GlobalState:
    def __init__(self):
        self.message_bus = MessageBus()
        self.scenario_gen = ScenarioGenerator()
        
        # Hidden State
        self.root_cause: str = ""
        self.target_service: str = ""
        self.full_logs: Dict[str, str] = {}
        self.full_metrics: Dict[str, Dict[str, float]] = {}
        
        # Agent specific tracked state
        self.queried_logs_history: List[str] = []
        self.queried_metrics_history: List[str] = []
        self.active_hypotheses: List[str] = []
        self.action_feedback: Optional[str] = None
        self.is_resolved: bool = False
        
        self.steps = 0
        self.max_steps = 10

    def reset(self, difficulty="medium"):
        self.message_bus.reset()
        rc, target, logs, metrics = self.scenario_gen.generate(difficulty)
        self.root_cause = rc
        self.target_service = target
        self.full_logs = logs
        self.full_metrics = metrics
        
        self.queried_logs_history = []
        self.queried_metrics_history = []
        self.active_hypotheses = []
        self.action_feedback = None
        self.is_resolved = False
        self.steps = 0
