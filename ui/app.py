"""
Phase 1 Gradio UI — SRE Decision Environment (improved).

Changes vs v1:
- Debug mode toggle reveals the hidden root cause
- Actions are grouped: Diagnostic vs Remediation
- Per-step reward explanation shown
- Full agent LLM reasoning displayed
- Episode history table with rewards per step
"""

import os
import sys
from pathlib import Path
from typing import Optional

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
_history = []          # list of {action, reward, feedback}
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
    if reward >= 1.0:
        return f"✅ **Correct fix!** `{action_name}` resolved the root cause."
    if reward == -1.0:
        return f"❌ **Wrong fix!** `{action_name}` was harmful for this root cause."
    if reward > 0:
        return f"🔍 **Info gathered.** `{action_name}` is a safe diagnostic action (+{reward:.2f})."
    if reward < -0.3:
        return f"⚠️ **Penalty.** Check the action name or the episode timed out."
    return f"↔️  Neutral. `{action_name}` had no significant effect."


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
        f"| 🔬 Anomaly score | `{logs.get('log_anomaly_score', 'N/A')}` |",
        f"| 🖥️ Server B health | `{obs_sig.get('server_b_health', 'N/A')}` |",
        f"| ⚙️ CPU usage | `{obs_sig.get('cpu_usage', 'N/A')}` |",
        f"| 💾 Memory usage | `{obs_sig.get('memory_usage', 'N/A')}` |",
        "",
        f"**Feedback:** {d.get('action_feedback', '')}",
    ]
    if d.get("incident_resolved"):
        lines.append("\n---\n🏁 **Episode complete. Click Reset to start again.**")
    return "\n".join(lines)


def _history_md() -> str:
    if not _history:
        return "_No actions yet._"
    rows = ["| # | Action | Reward | Type |",
            "|---|--------|--------|------|"]
    for i, h in enumerate(_history, 1):
        a = h["action"]
        r = h["reward"]
        tag = "🔍 Diagnostic" if a in DIAGNOSTIC else "⚡ Remediation"
        sign = f"+{r:.2f}" if r >= 0 else f"{r:.2f}"
        rows.append(f"| {i} | `{a}` | **{sign}** | {tag} |")
    return "\n".join(rows)


# ── Event handlers ────────────────────────────────────────────────────────────
def reset_episode(debug):
    global _current_obs, _history, _episode_total_reward, _debug_mode
    _debug_mode = bool(debug)
    _current_obs = env.reset()
    _history = []
    _episode_total_reward = 0.0
    return (
        _obs_md(_current_obs, _debug_mode),
        _history_md(),
        "0.00",
        "—",
        "🟢 New incident started. Investigate before you remediate!",
    )


def manual_action(action_display_name: str, debug: bool):
    global _current_obs, _history, _episode_total_reward, _debug_mode
    _debug_mode = bool(debug)

    if _current_obs is None:
        return _obs_md(None, _debug_mode), _history_md(), "0.00", "—", "⚠️ Click Reset first!"
    if _current_obs.incident_resolved:
        return _obs_md(_current_obs, _debug_mode), _history_md(), f"{_episode_total_reward:.2f}", "—", "Episode over — click Reset."

    # Strip the emoji prefix we added for display
    action_name = action_display_name.replace("🔍 ", "").replace("⚡ ", "").strip()

    action = SreDecisionAction(action_name=action_name)
    _current_obs = env.step(action)
    reward = _current_obs.reward or 0.0
    _episode_total_reward += reward
    _history.append({"action": action_name, "reward": reward, "feedback": _current_obs.action_feedback or ""})

    sign = f"+{reward:.2f}" if reward >= 0 else f"{reward:.2f}"
    explanation = _explain_reward(reward, action_name)
    status = f"{explanation}\n\n**Action sent:** `{action_name}`"
    if _current_obs.incident_resolved:
        status += f"\n\n🏁 Episode complete. Total reward: **{_episode_total_reward:.2f}**"

    return (
        _obs_md(_current_obs, _debug_mode),
        _history_md(),
        f"{_episode_total_reward:.2f}",
        sign,
        status,
    )


def agent_action(debug: bool):
    global _current_obs, _history, _episode_total_reward, _debug_mode
    _debug_mode = bool(debug)

    if llm_client is None:
        return _obs_md(_current_obs, _debug_mode), _history_md(), f"{_episode_total_reward:.2f}", "—", \
               "❌ LLM not available. Set OPENROUTER_API_KEY in your .env file."
    if _current_obs is None:
        return _obs_md(None, _debug_mode), _history_md(), "0.00", "—", "⚠️ Click Reset first!"
    if _current_obs.incident_resolved:
        return _obs_md(_current_obs, _debug_mode), _history_md(), f"{_episode_total_reward:.2f}", "—", "Episode over — click Reset."

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
    _history.append({"action": action_name, "reward": reward, "feedback": _current_obs.action_feedback or ""})

    sign = f"+{reward:.2f}" if reward >= 0 else f"{reward:.2f}"
    explanation = _explain_reward(reward, action_name)

    status = (
        f"🤖 **Agent picked:** `{action_name}`\n\n"
        f"{explanation}\n\n"
        f"---\n**Full agent reasoning:**\n```\n{llm_response[:600]}\n```"
    )
    if _current_obs.incident_resolved:
        status += f"\n\n🏁 Episode complete. Total reward: **{_episode_total_reward:.2f}**"

    return (
        _obs_md(_current_obs, _debug_mode),
        _history_md(),
        f"{_episode_total_reward:.2f}",
        sign,
        status,
    )


# ── Build UI ──────────────────────────────────────────────────────────────────
def build_ui() -> gr.Blocks:
    llm_status = _try_init_llm()
    labeled_actions = [ACTION_LABELS[a] for a in VALID_ACTIONS]

    with gr.Blocks(
        title="SRE Decision Env — Phase 1",
        theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate"),
    ) as demo:

        gr.Markdown("# 🚨 SRE Decision Environment — Phase 1")
        gr.Markdown(
            "**Diagnose the hidden root cause from noisy partial signals. "
            "Use diagnostic actions to gather clues, then pick the right fix.**"
        )
        gr.Markdown(f"**LLM:** {llm_status}")

        gr.Markdown(
            "> 🔍 **Diagnostic actions** (inspect_logs, inspect_metrics, etc.) → always safe, small +reward  \n"
            "> ⚡ **Remediation actions** (restart, rollback, resolve) → big +reward if correct, **-1 if wrong**"
        )

        with gr.Row():
            # ── Left: observation ───────────────────────────────────────────
            with gr.Column(scale=2):
                obs_display = gr.Markdown(value="_Click Reset to start._")
                with gr.Row():
                    reward_total = gr.Textbox(label="Total Reward", value="0.00", interactive=False)
                    reward_step  = gr.Textbox(label="Last Step Reward", value="—", interactive=False)

            # ── Right: controls ─────────────────────────────────────────────
            with gr.Column(scale=1):
                debug_toggle = gr.Checkbox(label="🔓 Debug mode (show root cause)", value=False)
                reset_btn = gr.Button("🔄 Reset / New Incident", variant="primary")

                gr.Markdown("**Manual action:**")
                action_dropdown = gr.Dropdown(
                    choices=labeled_actions,
                    value=labeled_actions[0],   # 🔍 inspect_logs
                    label="Choose Action",
                )
                manual_btn = gr.Button("▶ Execute Action")

                gr.Markdown("**LLM agent:**")
                agent_btn = gr.Button("🤖 Agent Step", variant="secondary")

        status_box = gr.Textbox(label="Step Result / Agent Reasoning", lines=8, interactive=False)

        gr.Markdown("### Episode History")
        history_display = gr.Markdown("_No actions yet._")

        # ── Wire events ──────────────────────────────────────────────────────
        outs = [obs_display, history_display, reward_total, reward_step, status_box]

        reset_btn.click(fn=reset_episode, inputs=[debug_toggle], outputs=outs)
        manual_btn.click(fn=manual_action, inputs=[action_dropdown, debug_toggle], outputs=outs)
        agent_btn.click(fn=agent_action, inputs=[debug_toggle], outputs=outs)

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _host = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    _port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    ui = build_ui()
    ui.launch(server_name=_host, server_port=_port, share=False)
