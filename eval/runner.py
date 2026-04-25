"""
Phase 1 benchmark runner.

Runs the orchestrator LLM agent against the benchmark scenario suite
and collects per-episode metrics. Results are saved to assets/ for plotting.

Usage::

    cd sre_decision_env
    python -m eval.runner --phase phase1

Phase 1 — baseline evaluation.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Allow running as a script from the sre_decision_env directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

from server.sensors import ROOT_CAUSES
from server.sre_decision_env_environment import SreDecisionEnvironment
from models import SreDecisionAction
from llm.openrouter_client import OpenRouterLLMClient
from llm.prompts import SYSTEM_PROMPT, build_user_prompt, parse_action

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
BENCHMARK_PATH_P1 = Path(__file__).resolve().parents[1] / "scenarios" / "benchmark_cases.json"
BENCHMARK_PATH_P2 = Path(__file__).resolve().parents[1] / "scenarios" / "phase2_benchmark_cases.json"
ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"
ASSETS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Single episode runner
# ---------------------------------------------------------------------------

def run_episode(
    env: SreDecisionEnvironment,
    client: OpenRouterLLMClient,
    root_cause_override: Optional[str] = None,
    verbose: bool = False,
) -> Dict:
    """
    Run one complete episode and return metrics.

    Args:
        env:                   The environment instance.
        client:                LLM client for action generation.
        root_cause_override:   Force a specific root cause (for benchmark).
        verbose:               Print step-by-step details.

    Returns:
        Episode metric dict.
    """
    # Patch root cause for deterministic benchmark scenarios
    obs = env.reset()
    if root_cause_override:
        env._root_cause = root_cause_override  # benchmark override (internal)

    history: List[Dict] = []
    total_reward = 0.0
    steps = 0
    resolved = False

    while True:
        obs_dict = obs.model_dump()
        user_msg = build_user_prompt(obs_dict, history)

        try:
            llm_response = client.chat_complete(
                system_prompt=SYSTEM_PROMPT,
                user_message=user_msg,
                temperature=0.1,
                max_tokens=256,
            )
        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            llm_response = "Action: inspect_logs"

        action_name = parse_action(llm_response)
        if verbose:
            logger.info("Step %d | action=%s", steps + 1, action_name)

        action = SreDecisionAction(action_name=action_name, rationale=llm_response)
        obs = env.step(action)

        reward = obs.reward or 0.0
        total_reward += reward
        steps += 1
        history.append({"action": action_name, "feedback": obs.action_feedback or ""})

        if obs.done or obs.incident_resolved:
            resolved = action_name == "resolve_incident"
            break

    return {
        "root_cause": env._root_cause,
        "steps": steps,
        "total_reward": round(total_reward, 4),
        "resolved": resolved,
        "last_action": action_name,
        "reward_breakdown": obs.metadata.get("reward_breakdown", {})
    }


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(phase: str = "phase1", verbose: bool = False) -> None:
    """Run all benchmark cases and save results + plots."""
    benchmark_file = BENCHMARK_PATH_P2 if phase == "phase2" else BENCHMARK_PATH_P1
    with open(benchmark_file) as f:
        bench = json.load(f)

    cases = bench["cases"]
    logger.info("Running %d benchmark cases — phase=%s", len(cases), phase)

    client = OpenRouterLLMClient()
    env = SreDecisionEnvironment()

    results = []
    for case in cases:
        logger.info("▶ %s  [%s]", case["id"], case["label"])
        start = time.time()
        metrics = run_episode(
            env=env,
            client=client,
            root_cause_override=case["root_cause"],
            verbose=verbose,
        )
        metrics["id"] = case["id"]
        metrics["label"] = case["label"]
        metrics["elapsed_s"] = round(time.time() - start, 2)
        results.append(metrics)
        logger.info(
            "   reward=%.3f  steps=%d  resolved=%s",
            metrics["total_reward"],
            metrics["steps"],
            metrics["resolved"],
        )

    # Save raw results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = ASSETS_DIR / f"{phase}_results_{ts}.json"
    with open(results_path, "w") as f:
        json.dump({"phase": phase, "timestamp": ts, "results": results}, f, indent=2)
    logger.info("Results saved → %s", results_path)

    # Summary stats
    rewards = [r["total_reward"] for r in results]
    success = [r for r in results if r["resolved"]]
    logger.info(
        "=== %s SUMMARY ===  avg_reward=%.3f  success_rate=%.0f%%  n=%d",
        phase.upper(),
        sum(rewards) / len(rewards),
        100 * len(success) / len(results),
        len(results),
    )

    # Plot
    _plot_results(results, phase, ts)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_results(results: List[Dict], phase: str, ts: str) -> None:
    """Save a simple reward bar chart for the benchmark run."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_theme(style="darkgrid")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"SRE Decision Env — {phase.upper()} Baseline", fontsize=14)

        ids = [r["id"] for r in results]
        rewards = [r["total_reward"] for r in results]
        steps = [r["steps"] for r in results]
        colors = ["#2ecc71" if r["resolved"] else "#e74c3c" for r in results]

        # Reward bar chart
        axes[0].bar(ids, rewards, color=colors)
        axes[0].set_title("Total Reward per Scenario")
        axes[0].set_xlabel("Scenario ID")
        axes[0].set_ylabel("Total Reward")
        axes[0].tick_params(axis="x", rotation=45)
        axes[0].legend(
            handles=[
                plt.Rectangle((0, 0), 1, 1, color="#2ecc71", label="Resolved"),
                plt.Rectangle((0, 0), 1, 1, color="#e74c3c", label="Failed"),
            ]
        )

        # Steps bar chart
        axes[1].bar(ids, steps, color="#3498db")
        axes[1].set_title("Steps Taken per Scenario")
        axes[1].set_xlabel("Scenario ID")
        axes[1].set_ylabel("Steps")
        axes[1].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plot_path = ASSETS_DIR / f"{phase}_benchmark_{ts}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Plot saved → %s", plot_path)
    except Exception as exc:
        logger.warning("Plotting failed (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SRE Decision Env — Benchmark Runner")
    parser.add_argument("--phase", default="phase1", help="Phase label for output files")
    parser.add_argument("--verbose", action="store_true", help="Print step details")
    args = parser.parse_args()

    run_benchmark(phase=args.phase, verbose=args.verbose)
