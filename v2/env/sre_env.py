from typing import Dict, Any, Tuple
import yaml

from v2.agents.action_space import InvestigatorAction, AnalystAction, CoordinatorAction, ExecutorAction
from v2.agents.observation_space import InvestigatorObservation, AnalystObservation, CoordinatorObservation, ExecutorObservation
from v2.env.state import GlobalState
from v2.env.reward import RewardSystem

class SreDecisionEnvV2:
    def __init__(self, config_path="v2/config/env_config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.state = GlobalState()
        self.reward_system = RewardSystem(self.config)

    def reset(self, difficulty="medium") -> Dict[str, Any]:
        self.state.reset(difficulty=difficulty)
        return self._get_observations()

    def _get_observations(self) -> Dict[str, Any]:
        messages = self.state.message_bus.get_messages()
        
        # Investigator Obs
        inv_logs = {svc: self.state.full_logs[svc] for svc in self.state.queried_logs_history if svc in self.state.full_logs}
        inv_obs = InvestigatorObservation(logs=inv_logs, message_bus=messages).model_dump()
        
        # Analyst Obs
        ana_metrics = {svc: self.state.full_metrics[svc] for svc in self.state.queried_metrics_history if svc in self.state.full_metrics}
        ana_obs = AnalystObservation(metrics=ana_metrics, message_bus=messages).model_dump()
        
        # Coordinator Obs
        coord_obs = CoordinatorObservation(message_bus=messages, active_hypotheses=self.state.active_hypotheses).model_dump()
        
        # Executor Obs
        exec_obs = ExecutorObservation(message_bus=messages, action_feedback=self.state.action_feedback).model_dump()
        
        return {
            "investigator": inv_obs,
            "analyst": ana_obs,
            "coordinator": coord_obs,
            "executor": exec_obs
        }

    def step(self, actions: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, bool], Dict[str, Any]]:
        self.state.steps += 1
        self.state.action_feedback = None # Reset transient feedback
        
        # 1. Process Actions (Update hidden state)
        
        # Investigator
        inv_a = actions.get("investigator")
        if inv_a:
            if inv_a.action_type == "query_logs" and inv_a.target_service:
                if inv_a.target_service not in self.state.queried_logs_history:
                    self.state.queried_logs_history.append(inv_a.target_service)
            elif inv_a.action_type == "share_info" and inv_a.message:
                self.state.message_bus.add_message("investigator", inv_a.message)
                
        # Analyst
        ana_a = actions.get("analyst")
        if ana_a:
            if ana_a.action_type == "query_metrics" and ana_a.target_service:
                if ana_a.target_service not in self.state.queried_metrics_history:
                    self.state.queried_metrics_history.append(ana_a.target_service)
            elif ana_a.action_type == "share_info" and ana_a.message:
                self.state.message_bus.add_message("analyst", ana_a.message)
                
        # Coordinator
        coord_a = actions.get("coordinator")
        if coord_a:
            if coord_a.action_type == "propose_hypothesis" and coord_a.hypothesis:
                self.state.active_hypotheses.append(coord_a.hypothesis)
                self.state.message_bus.add_message("coordinator", f"Hypothesis: {coord_a.hypothesis}")
                
        # Executor
        exec_a = actions.get("executor")
        if exec_a:
            if exec_a.action_type == "execute_fix" and exec_a.target_service and exec_a.fix_type:
                self.state.message_bus.add_message("executor", f"Executed {exec_a.fix_type} on {exec_a.target_service}")
                self.state.action_feedback = f"Executed {exec_a.fix_type} on {exec_a.target_service}"

        # 2. Calculate Rewards
        rewards = self.reward_system.calculate(self.state, actions)
        
        # 3. Advance state
        self.state.message_bus.step()
        
        # 4. Determine Done
        dones = {}
        is_done = self.state.is_resolved or self.state.steps >= self.state.max_steps
        for ag in ["investigator", "analyst", "coordinator", "executor"]:
            dones[ag] = is_done
            
        return self._get_observations(), rewards, dones, {"root_cause": self.state.root_cause}
