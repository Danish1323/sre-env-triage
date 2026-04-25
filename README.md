# sre-decision-env

A multi-agent SRE incident-response environment built on the [OpenEnv](https://openenv.ai/) framework. This project simulates a real incident workflow as a **Dec-POMDP** (Decentralized Partially Observable Markov Decision Process):

- A hidden root cause exists.
- Noisy sensors expose incomplete clues.
- An orchestrator agent reasons over those clues to act.

---

## Phase 1 Overview

Phase 1 provides the foundational environment:
- **Hidden Root Causes**: e.g., `server_A_failure`, `memory_leak`, `network_latency`, `transient_spike`.
- **Sensors**: Noisy "logs" and "observer" modules generate stochastic, partial signals.
- **Orchestrator Backend**: Powered by `meta-llama/llama-3.2-3b-instruct` via the OpenRouter API.
- **UI**: A Gradio web UI to step through the environment manually or using the agent.
- **Evaluation**: A benchmark suite (`scenarios/benchmark_cases.json`) to establish a baseline performance.

---

## Quick Start

### Prerequisites
Make sure you have `uv` installed for dependency management, or use Docker.
Set your OpenRouter API key in a `.env` file at the root of the project:
```bash
OPENROUTER_API_KEY=your_key_here
```

### Running the Gradio UI

You can interact with the environment manually or click the "Agent Step" button to let the LLM take actions.

```bash
# Run the UI locally
uv run python -m ui.app
```
The UI will start at `http://0.0.0.0:7860`.

### Running the Benchmark

You can run the Phase 1 baseline evaluation. This will play through the scenarios defined in `scenarios/benchmark_cases.json` and generate performance charts in the `assets/` directory.

```bash
# Run the evaluation suite
uv run python -m eval.runner --phase phase1 --verbose
```

---

## Project Structure

- `server/`: Contains the core OpenEnv implementation (`sre_decision_env_environment.py`), reward functions, and noisy sensor logic.
- `llm/`: Client interfaces for model inference (e.g., OpenRouter).
- `models.py`: Defines the `SreDecisionAction` and `SreDecisionObservation` schemas.
- `ui/`: Gradio application for visual interaction.
- `eval/`: Automated runner and plotting scripts for benchmark evaluation.
- `scenarios/`: Benchmark test cases defining specific hidden root causes.
