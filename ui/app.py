"""
Phase 2 Gradio UI — SRE Decision Environment (Debug & RL Friendly).

Changes vs Phase 1:
- Debug mode reveals hidden root cause
- Layout mapped to RL flow: State/Observation -> Decision -> Outcome
- Raw observation JSON view
- Dedicated Agent Reasoning UI component
- Structured DataFrame history for easy debugging
"""

import os
import sys
from pathlib import Path
from typing import Optional
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

from server.sre_decision_env_environment import SreDecisionEnvironment, MAX_STEPS
from models import VALID_ACTIONS, SreDecisionAction
from llm.openrouter_client import OpenRouterLLMClient
from llm.prompts import SYSTEM_PROMPT, build_user_prompt, parse_action

# ── Action grouping for display ──────────────────────────────────────────────
DIAGNOSTIC = [
    "inspect_logs",
    "inspect_metrics",
    "check_deploy_history",
    "declare_severity_low",
    "declare_severity_high",
]
REMEDIATION = [
    "restart_service",
    "rollback_service",
    "resolve_incident",
]

ACTION_LABELS = {a: f"🔍 {a}" for a in DIAGNOSTIC}
ACTION_LABELS.update({a: f"⚡ {a}" for a in REMEDIATION})

# ── Global state ──────────────────────────────────────────────────────────────
env = SreDecisionEnvironment()
llm_client: Optional[OpenRouterLLMClient] = None
_history = []          # list of {step, action, reward, feedback, reasoning}
_current_obs = None
_episode_total_reward = 0.0
_debug_mode = False


def _try_init_llm() -> str:
    global llm_client
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key or key == "your_openrouter_key_here":
        return "⚠️  OPENROUTER_API_KEY not set — Manual mode only."
    try:
        llm_client = OpenRouterLLMClient()
        return "✅  LLM ready  (meta-llama/llama-3.2-3b-instruct via OpenRouter)"
    except Exception as exc:
        return f"❌  LLM init failed: {exc}"


# ── Reward explanation ────────────────────────────────────────────────────────
def _explain_reward(reward: float, action_name: str) -> str:
    if reward >= 0.75:
        return f"✅ **Correct fix!** `{action_name}` is extremely close to the root cause."
    if reward >= 0.60:
        return f"🔍 **Useful Info.** `{action_name}` gathered highly relevant signals or was a sequential diagnosis."
    if reward >= 0.40:
        return f"↔️ **Sub-optimal.** `{action_name}` was generic or slightly off-target."
    if reward >= 0.20:
        return f"❌ **Harmful/Wrong!** `{action_name}` exacerbated the problem or was an un-diagnosed blind guess."
    return f"⚠️ **Invalid/Repeated.** Action '{action_name}' is penalized heavily."


# ── Observation display ───────────────────────────────────────────────────────
def _obs_md(obs, debug: bool) -> str:
    if obs is None:
        return "_Click **🔄 Reset** to start a new incident._"
    d = obs.model_dump()
    logs = d.get("logs", {})
    obs_sig = d.get("observer", {})

    root_line = ""
    if debug and hasattr(env, "_root_cause") and env._root_cause:
        root_line = f"\n> 🔓 **[DEBUG] Hidden root cause: `{env._root_cause}`**\n"

    lines = [
        f"### Step {d.get('time_step', 0)} / {MAX_STEPS}",
        root_line,
        "| Signal | Value |",
        "|--------|-------|",
        f"| 🕐 Latency spike | `{logs.get('latency_spike', 'N/A')}` |",
        f"| ❗ Error rate | `{logs.get('error_rate', 'N/A')}` |",
        f"| 📈 5xx Error rate | `{logs.get('five_xx_error_rate', 'N/A')}` |",
        f"| 🔬 Anomaly score | `{logs.get('log_anomaly_score', 'N/A')}` |",
        f"| 🖥️ Server B health | `{obs_sig.get('server_b_health', 'N/A')}` |",
        f"| ⚙️ CPU usage | `{obs_sig.get('cpu_usage', 'N/A')}` |",
        f"| 💾 Memory usage | `{obs_sig.get('memory_usage', 'N/A')}` |",
        f"| 🔌 DB Connections | `{obs_sig.get('db_connections', 'N/A')}` |",
    ]
    if d.get("incident_resolved"):
        lines.append("\n---\n🏁 **Episode complete. Click Reset to start again.**")
    return "\n".join(lines)

def _history_data() -> list[list]:
    if not _history:
        return []
    
    rows = []
    for h in _history:
        a = h["action"]
        r = h["reward"]
        sign = f"+{r:.2f}" if r >= 0 else f"{r:.2f}"
        rows.append([
            h["step"],
            f"`{a}`",
            sign,
            h.get("feedback", ""),
            h.get("reasoning", "N/A")[:100] + "..." if h.get("reasoning") else "N/A"
        ])
    return rows


# ── Event handlers ────────────────────────────────────────────────────────────
def _blank_outputs():
    return (
        _obs_md(None, _debug_mode),
        {},
        "0.00",
        "—",
        {},
        "⚠️ Click Reset first!",
        "N/A",
        _history_data(),
        gr.update(interactive=False),
        gr.update(interactive=False)
    )

def _get_outputs(obs, action_name, reward, reasoning="Manual action"):
    sign = f"+{reward:.2f}" if reward >= 0 else f"{reward:.2f}"
    explanation = _explain_reward(reward, action_name)
    breakdown = obs.metadata.get("reward_breakdown", {})
    
    feedback = f"{explanation}\n\n**Environment Feedback:**\n> {obs.action_feedback}"
    if obs.incident_resolved:
        feedback += f"\n\n🏁 **Episode complete. Total reward: {_episode_total_reward:.2f}**"

    return (
        _obs_md(obs, _debug_mode),
        obs.model_dump(),
        f"{_episode_total_reward:.2f}",
        sign,
        breakdown,
        feedback,
        reasoning,
        _history_data(),
        gr.update(interactive=False),
        gr.update(interactive=False)
    )


def reset_episode(debug):
    global _current_obs, _history, _episode_total_reward, _debug_mode
    _debug_mode = bool(debug)
    _current_obs = env.reset()
    _history = []
    _episode_total_reward = 0.0
    return (
        _obs_md(_current_obs, _debug_mode),
        _current_obs.model_dump(),
        "0.00",
        "—",
        {},
        "🟢 New incident started. Investigate before you remediate!",
        "N/A",
        _history_data(),
        gr.update(interactive=True),
        gr.update(interactive=True)
    )


def manual_action(action_display_name: str, debug: bool):
    global _current_obs, _history, _episode_total_reward, _debug_mode
    _debug_mode = bool(debug)

    if _current_obs is None: return _blank_outputs()
    if _current_obs.incident_resolved: return _get_outputs(_current_obs, _history[-1]["action"] if _history else "N/A", 0)

    action_name = action_display_name.replace("🔍 ", "").replace("⚡ ", "").strip()
    action = SreDecisionAction(action_name=action_name)
    
    _current_obs = env.step(action)
    reward = _current_obs.reward or 0.0
    _episode_total_reward += reward
    
    _history.append({
        "step": _current_obs.metadata.get("step", 0),
        "action": action_name, 
        "reward": reward, 
        "feedback": _current_obs.action_feedback or "",
        "reasoning": "Manual Action"
    })

    return _get_outputs(_current_obs, action_name, reward, "Manual execution. No LLM reasoning.")


def agent_action(debug: bool):
    global _current_obs, _history, _episode_total_reward, _debug_mode
    _debug_mode = bool(debug)

    if llm_client is None:
        outs = list(_blank_outputs())
        outs[5] = "❌ LLM not available. Set OPENROUTER_API_KEY in your .env file."
        return tuple(outs)

    if _current_obs is None: return _blank_outputs()
    if _current_obs.incident_resolved: return _get_outputs(_current_obs, _history[-1]["action"] if _history else "N/A", 0)

    obs_dict = _current_obs.model_dump()
    user_msg = build_user_prompt(obs_dict, _history)

    try:
        llm_response = llm_client.chat_complete(
            system_prompt=SYSTEM_PROMPT,
            user_message=user_msg,
            temperature=0.1,
            max_tokens=256,
        )
        action_name = parse_action(llm_response)
    except Exception as exc:
        action_name = "inspect_logs"
        llm_response = f"[LLM error: {exc}] Defaulting to inspect_logs."

    action = SreDecisionAction(action_name=action_name)
    _current_obs = env.step(action)
    reward = _current_obs.reward or 0.0
    _episode_total_reward += reward
    
    _history.append({
        "step": _current_obs.metadata.get("step", 0),
        "action": action_name, 
        "reward": reward, 
        "feedback": _current_obs.action_feedback or "",
        "reasoning": llm_response
    })

    return _get_outputs(_current_obs, action_name, reward, llm_response)


# ── Build UI ──────────────────────────────────────────────────────────────────
def build_ui() -> gr.Blocks:
    llm_status = _try_init_llm()
    labeled_actions = [ACTION_LABELS[a] for a in VALID_ACTIONS]

    with gr.Blocks(
        title="SRE Decision Env — Phase 2"
    ) as demo:

        gr.Markdown("# 🚨 SRE Decision Environment — Phase 2")
        gr.Markdown(
            "**Diagnose the hidden root cause from noisy partial signals. "
            "Use diagnostic actions to gather clues, then pick the right fix.**"
        )
        
        with gr.Row():
            gr.Markdown(f"**LLM Status:** {llm_status}")
            debug_toggle = gr.Checkbox(label="🔓 Debug mode (reveal hidden state)", value=False)
            reset_btn = gr.Button("🔄 Reset Episode", variant="primary")

        with gr.Row():
            # ── Column 1: State & Observation ──────────────────────────────────
            with gr.Column(scale=1, variant="panel"):
                gr.Markdown("### 1. State & Observation")
                obs_display = gr.Markdown(value="_Click Reset to start._")
                gr.Markdown("**Raw JSON (for RL tracing):**")
                obs_json = gr.JSON(value={}, label="Observation JSON")

            # ── Column 2: Action & Decision ────────────────────────────────────
            with gr.Column(scale=1, variant="panel"):
                gr.Markdown("### 2. Action & Decision")
                
                gr.Markdown("#### Manual Action")
                action_dropdown = gr.Dropdown(
                    choices=labeled_actions,
                    value=labeled_actions[0],   # 🔍 inspect_logs
                    label="Choose Action",
                )
                manual_btn = gr.Button("▶ Execute Manual Action")

                gr.Markdown("#### Agent Action")
                agent_btn = gr.Button("🤖 Let Agent Decide", variant="secondary")
                
                gr.Markdown("#### Agent Reasoning")
                agent_reasoning_md = gr.Markdown(value="_No agent reasoning yet._")

            # ── Column 3: Outcome & Reward ─────────────────────────────────────
            with gr.Column(scale=1, variant="panel"):
                gr.Markdown("### 3. Outcome & Reward")
                feedback_md = gr.Markdown(value="_No feedback yet._")
                
                with gr.Row():
                    reward_step  = gr.Textbox(label="Step Reward", value="—", interactive=False)
                    reward_total = gr.Textbox(label="Total Reward", value="0.00", interactive=False)
                    
                gr.Markdown("**Reward Breakdown:**")
                reward_breakdown_json = gr.JSON(value={}, label="Breakdown")

        # ── History Table ──────────────────────────────────────────────────
        gr.Markdown("### Episode History")
        history_df = gr.Dataframe(
            headers=["Step", "Action", "Reward", "Feedback", "Rationale Preview"],
            datatype=["number", "markdown", "str", "str", "str"],
            interactive=False,
            row_count=5,
            wrap=True
        )

        # ── Wire events ──────────────────────────────────────────────────────
        outs = [
            obs_display, 
            obs_json, 
            reward_total, 
            reward_step, 
            reward_breakdown_json, 
            feedback_md, 
            agent_reasoning_md, 
            history_df,
            manual_btn,
            agent_btn
        ]

        reset_btn.click(fn=reset_episode, inputs=[debug_toggle], outputs=outs)
        manual_btn.click(fn=manual_action, inputs=[action_dropdown, debug_toggle], outputs=outs)
        agent_btn.click(fn=agent_action, inputs=[debug_toggle], outputs=outs)

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _host = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    _port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    ui = build_ui()
    
    import gradio as gr
    ui.launch(server_name=_host, server_port=_port, share=False, theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate"))
