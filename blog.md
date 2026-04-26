# Building the Ultimate AI Site Reliability Engineer: How We Fine-Tuned Llama 3.2 to Acheive a 100% Triage Success Rate

> *When systems go down, every second counts. What if your first responder wasn't human?*

Site Reliability Engineering (SRE) is notoriously difficult to automate. When an incident occurs, responders have to dig through noisy logs, correlate disparate metrics across microservices, form hypotheses, and execute remediation actions—all under extreme pressure. 

But what if we could train a Large Language Model to do exactly that? 

Today, we're thrilled to introduce our **Autonomous SRE Agent**, built on **Llama 3.2 3B**, trained to navigate complex incident response scenarios with incredible precision.

---

## 🛑 The Challenge: SRE is a Dec-POMDP 

To an AI, a microservice architecture is essentially a Decentralized Partially Observable Markov Decision Process (Dec-POMDP). 
- **The True State is Hidden:** A root cause (like a memory leak) exists somewhere.
- **Sensors are Noisy:** Logs and metrics provide incomplete, sometimes misleading clues.
- **Actions have Consequences:** Restarting the wrong service might cause a cascading failure, while querying too many logs wastes precious time.

Standard LLMs struggle in these environments. They hallucinate non-existent services ("I will restart *Service A*"), take actions out of order, or blindly guess the root cause. We needed a model that could *reason* systematically.

---

## 🧠 The Solution: SFT meets GRPO 

To build an agent capable of real-world triage, we combined the lightweight power of Meta's **Llama 3.2 3B Instruct** with the lightning-fast training capabilities of **Unsloth**. 

Our training pipeline utilized a state-of-the-art two-phase approach:

### Phase 1: Supervised Fine-Tuning (SFT)
We started by teaching the model the "grammar" of SRE. Using a curated dataset of thousands of incident response workflows, we taught the model how to declare actions using strict formatting (e.g., `Action: execute_fix\nAction Input: {"service_name": "auth_service"}`). This established a baseline competency in reading logs and metrics.

### Phase 2: Reinforcement Learning via GRPO
SFT isn't enough to prevent hallucinations. For that, we turned to **Group Relative Policy Optimization (GRPO)**. We designed a custom, harsh reward function:
- **+0.5** for resolving the incident correctly.
- **+0.3** for identifying the exact root cause.
- **-0.2** for hallucinating an unknown service.
- **-0.3** for applying the wrong fix.

By penalizing blind guessing and rewarding deliberate, logical deduction, the model learned to behave like a seasoned on-call engineer.

---

## 📈 The Results: 100% Success on Baseline Scenarios

The results speak for themselves. We evaluated the fine-tuned agent against four critical baseline scenarios in our `sre-env-triage` simulator:

| Scenario | Root Cause Found? | Fix Applied? | Service Targeted? | Overall Success |
| :--- | :---: | :---: | :---: | :---: |
| **Memory Leak** | ✅ | ✅ | ✅ | **100%** |
| **CPU Saturation** | ✅ | ✅ | ✅ | **100%** |
| **DB Connection Leak** | ✅ | ✅ | ✅ | **100%** |
| **Cascading Failure** | ✅ | ✅ | ✅ | **100%** |

Unlike the baseline models that thrashed around executing random actions, our fine-tuned agent zeroed in on the root cause and applied the exact remediation required—every single time.

---

## 🚀 Try It Yourself

The best part? Because we used Unsloth and 4-bit quantization, this entire model can be run locally or trained on a single free T4 GPU via Google Colab. 

We've open-sourced the model weights and the training dataset. You can pull the agent right now from Hugging Face:

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="princeuser/llama-3.2-3b-sre-agent",
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
```

**[🔗 Check out the Model on Hugging Face](https://huggingface.co/princeuser/llama-3.2-3b-sre-agent)**

Are you ready to add an AI to your on-call rotation? Let us know what you think in the comments!
