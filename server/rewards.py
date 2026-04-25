"""
Distance-Aware Reward function for the SRE Decision Environment.
Produces a continuous, sigmoid-squashed reward bounded between (0.01, 0.99).
Encourages correct sequencing (diagnose -> act) and penalizes repetition.
"""

import math
from typing import List, Optional

# ---------------------------------------------------------------------------
# Action -> Root Cause Mapping
# ---------------------------------------------------------------------------

# Maps action names to a list of root causes they directly address or gather info for
ACTION_EFFECT_MAP = {
    "inspect_logs": ["any"],
    "inspect_metrics": ["any"],
    "check_deploy_history": ["deploy_issue"],

    "restart_service": ["server_A_failure", "memory_leak"],
    "rollback_service": ["deploy_issue", "server_A_failure"], # adding server failure here as fallback
    "scale_up": ["high_load", "memory_leak"],

    "declare_severity_low": ["any"],
    "declare_severity_high": ["any"],

    "resolve_incident": ["network_latency", "transient_spike", "no_issue"]
}


def _clip(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

# ---------------------------------------------------------------------------
# Distance-Aware Reward
# ---------------------------------------------------------------------------

def compute_step_reward(
    action_name: str,
    root_cause: str,
    step: int,
    max_steps: int,
    is_valid: bool,
    history: List[str],
    diagnosed: bool,
    resolved: bool
) -> tuple[float, dict]:
    """
    Compute distance-aware reward.
    """
    if not is_valid:
        # Heavily penalize invalid actions
        return 0.05, {"invalid_action": True, "action_score": -2.0, "final_reward": 0.05}

    breakdown = {}
    
    # 1. Base Action Score
    if action_name == "resolve_incident" and root_cause in ACTION_EFFECT_MAP["resolve_incident"]:
        action_score = 1.0
    elif action_name in ["restart_service", "rollback_service", "scale_up"] and root_cause in ACTION_EFFECT_MAP.get(action_name, []):
        action_score = 1.0 # Directly solves
    elif action_name in ["inspect_logs", "inspect_metrics"] or "any" in ACTION_EFFECT_MAP.get(action_name, []):
        action_score = 0.4 # Diagnostic
    elif root_cause in ACTION_EFFECT_MAP.get(action_name, []):
        action_score = 0.6 # Relevant but maybe not perfect fix
    else:
        action_score = -0.5 # Irrelevant action -> shifted negative for squashing
        
    breakdown["action_score"] = action_score
    
    # 2. Sequential Logic (Diagnosis Bonus / Penalty)
    diagnosis_bonus = 0.0
    if action_name in ["restart_service", "rollback_service", "scale_up", "resolve_incident"]:
        if not diagnosed:
            diagnosis_bonus = -1.0 # Blind guess penalty
        else:
            diagnosis_bonus = 0.5  # Properly diagnosed before acting
    elif action_name in ["inspect_logs", "inspect_metrics"] and not diagnosed:
        diagnosis_bonus = 0.5 # Rewarding first diagnosis
        
    breakdown["diagnosis_bonus"] = diagnosis_bonus

    # 3. Irrelevance / Repetition Penalty
    penalty = 0.0
    if action_score < 0:
        penalty += 0.5 # Irrelated
        
    if len(history) > 0 and action_name == history[-1]:
        penalty += 1.0 # Repeated action
        
    breakdown["penalty"] = penalty

    # 4. Timing Reward
    timing_bonus = 0.0
    if action_score == 1.0:
        if step < max_steps / 2:
            timing_bonus = 0.5
        else:
            timing_bonus = -0.2
            
    breakdown["timing_bonus"] = timing_bonus
    
    # 5. Terminal Reward
    terminal_bonus = 0.0
    if resolved:
        terminal_bonus = 1.5

    # 6. Final Formula
    w1, w2, w3, w4 = 1.5, 1.0, 1.0, 1.0
    
    raw_reward = (
        (w1 * action_score) + 
        (w2 * diagnosis_bonus) + 
        (w3 * timing_bonus) - 
        (w4 * penalty) + 
        terminal_bonus
    )
    
    breakdown["raw_reward"] = round(raw_reward, 3)
    
    final_reward = _clip(_sigmoid(raw_reward), 0.01, 0.99)
    breakdown["final_reward"] = round(final_reward, 3)

    return final_reward, breakdown
