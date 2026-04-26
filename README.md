# sre-decision-env

A multi-agent SRE incident-response environment built on the [OpenEnv](https://openenv.ai/) framework. This project simulates a real incident workflow as a **Dec-POMDP** (Decentralized Partially Observable Markov Decision Process):

- A hidden root cause exists.
- Noisy sensors expose incomplete clues.
- An orchestrator agent reasons over those clues to act.

---

## Phase 3 Overview & Evaluation

The environment natively supports Reinforcement Learning and Language Model fine-tuning through strict formatting and distance-aware reward scaling (bounded `0.01` to `0.99`). 

We establish continuous metric reporting to track agent learning over time.

### Training Curves (Untrained Baseline vs Rule-based Baseline)

The following curves show the initial performance of the untrained Random Agent against the structured Rule-Based baseline over 100 episodes. 

![Total Reward over Episodes](./reports/plots/reward_vs_episode_curve.png)
*Figure 1: Total reward accumulated per episode. The random agent struggles due to harsh penalties on blind guessing, while the rule agent scores consistently higher.*

![Mean Time To Resolution](./reports/plots/MTTR_vs_episode_curve.png)
*Figure 2: Steps taken to resolve the incident. Lower is better. The random baseline thrashes around randomly, taking far longer than the targeted rule-based actions.*

![Efficiency over Episodes](./reports/plots/efficiency_vs_episode_curve.png)
*Figure 3: Action efficiency (ratio of highly correct diagnostic/remediation actions vs total actions). Random hovers around zero, showing no logical progression.*

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

You can generate evaluation data and Hugging Face SFT-ready textual data (`data/*_hf_dataset.txt`) by running the evaluator.

```bash
# Evaluate an agent (random, rule, trained)
uv run python -m eval.evaluate --agent random --num_episodes 100
```

---

## Project Structure

- `server/`: Contains the core OpenEnv implementation (`sre_decision_env_environment.py`), strictly bounded distance-aware `rewards.py`, and noisy `sensors.py`.
- `llm/`: Client interfaces for model inference (e.g., OpenRouter).
- `models.py`: Defines the `SreDecisionAction` and `SreDecisionObservation` schemas.
- `ui/`: Gradio application for visual interaction and step-by-step debugging.
- `eval/`: Automated RL evaluation pipeline (`evaluate.py`) that generates HF datasets, CSV metrics, and training curves.
- `data/`: Exported episode metrics and raw instruction/response textual datasets.
- `reports/plots/`: Auto-generated training comparison curves.
