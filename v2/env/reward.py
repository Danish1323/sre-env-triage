from typing import Dict, Any

class RewardSystem:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def calculate(self, state, actions: Dict[str, Any]) -> Dict[str, float]:
        """
        Global and shaped rewards.
        Returns a dict of rewards for each agent (in a true cooperative setting, they might all get the same global reward, but here we shape it based on actions).
        """
        rewards = {
            "investigator": 0.0,
            "analyst": 0.0,
            "coordinator": 0.0,
            "executor": 0.0
        }
        
        # Message penalty
        for agent_name, action in actions.items():
            if action and getattr(action, "message", None):
                rewards[agent_name] += self.config.get("penalty_per_message", -0.05)
                
            # Useful sharing
            if agent_name == "investigator" and action and action.action_type == "share_info" and "ERROR" in state.full_logs.get(state.target_service, ""):
                rewards[agent_name] += 0.2
            elif agent_name == "analyst" and action and action.action_type == "share_info":
                # simplistic check for useful metrics
                rewards[agent_name] += 0.2
                
            # Coordinator hypothesis
            if agent_name == "coordinator" and action and action.action_type == "propose_hypothesis":
                if state.root_cause in action.hypothesis or state.target_service in action.hypothesis:
                    rewards[agent_name] += 0.2
                else:
                    rewards[agent_name] -= 0.3
                    
        # Terminal rewards based on Executor or Coordinator
        exec_action = actions.get("executor")
        coord_action = actions.get("coordinator")
        
        # Did executor fix it?
        if exec_action and exec_action.action_type == "execute_fix":
            # Allow case-insensitive and partial matching for target service
            is_target_correct = state.target_service.lower() in str(exec_action.target_service).lower()
            
            if is_target_correct:
                valid_fix = False
                fix_str = str(exec_action.fix_type).lower()
                
                # Make fix mapping extremely forgiving
                if state.root_cause == "memory_leak" and any(k in fix_str for k in ["restart", "reboot", "scale"]): valid_fix = True
                elif state.root_cause == "db_connection_leak" and any(k in fix_str for k in ["restart", "increase", "pool", "scale"]): valid_fix = True
                elif state.root_cause == "cpu_saturation" and any(k in fix_str for k in ["scale", "restart", "provision", "add"]): valid_fix = True
                elif state.root_cause == "cascading_failure" and any(k in fix_str for k in ["restart", "rollback", "scale", "clear", "cache"]): valid_fix = True
                else:
                    # Fallback: if they just try to restart or scale the correct service, give it to them
                    if "restart" in fix_str or "scale" in fix_str: valid_fix = True
                
                if valid_fix:
                    state.is_resolved = True
                    for ag in rewards: rewards[ag] += self.config.get("success_reward", 1.0)
                else:
                    # Wrong fix on right target
                    for ag in rewards: rewards[ag] += self.config.get("wrong_fix_penalty", -0.5)
            else:
                # Wrong target
                for ag in rewards: rewards[ag] += self.config.get("wrong_fix_penalty", -0.5)
                
        # Did coordinator finalize?
        if coord_action and coord_action.action_type == "finalize":
            guess = str(coord_action.root_cause_guess).lower().replace("_", " ")
            truth = state.root_cause.replace("_", " ")
            
            # Semantic keyword matching
            keywords = {
                "db_connection_leak": ["connection", "pool", "db", "database", "leak", "timeout"],
                "memory_leak": ["memory", "oom", "leak", "out of memory"],
                "cascading_failure": ["cascading", "failure", "cache", "eviction", "overwhelmed"],
                "cpu_saturation": ["cpu", "saturation", "starvation", "thread", "load"]
            }
            
            # Valid if exact match OR at least one strong keyword is present
            is_correct_guess = False
            if truth in guess:
                is_correct_guess = True
            else:
                rc_keywords = keywords.get(state.root_cause, [])
                if any(k in guess for k in rc_keywords):
                    is_correct_guess = True

            if is_correct_guess:
                state.is_resolved = True
                for ag in rewards: rewards[ag] += self.config.get("success_reward", 1.0)
            else:
                for ag in rewards: rewards[ag] += self.config.get("wrong_fix_penalty", -0.5)
                
        if state.steps >= state.max_steps and not state.is_resolved:
            for ag in rewards: rewards[ag] += self.config.get("timeout_penalty", -1.0)

        return rewards
