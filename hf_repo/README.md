---
language:
- en
license: llama3.2
base_model: unsloth/Llama-3.2-3B-Instruct
tags:
- sre
- agent
- unsloth
- reinforcement-learning
- grpo
- rlhf
- autonomous-agents
datasets:
- danish1423/sre-agent-training-data
---

# Llama 3.2 3B SRE Agent

This is a fine-tuned version of **Llama 3.2 3B Instruct** specifically trained to act as an autonomous Site Reliability Engineering (SRE) agent. It is designed to navigate the [SRE Decision Environment](https://github.com/danish1423/sre-env-triage), a Dec-POMDP simulator for incident response.

The model was trained using **Unsloth** with a two-phase approach:
1. **SFT (Supervised Fine-Tuning):** Trained on an expert-curated dataset of SRE incident response workflows.
2. **GRPO (Group Relative Policy Optimization):** Refined using reinforcement learning based on a custom reward function that penalizes hallucinations and rewards correct root cause identification, valid service targeting, and appropriate fix execution.

## 🚀 Usage

You can load this model efficiently using `unsloth` for inference. Since this repository contains LoRA adapters, they will be seamlessly merged into the base model.

### Installation

```bash
pip install unsloth
```

### Loading the Model

```python
from unsloth import FastLanguageModel
import torch

# Load the model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="princeuser/llama-3.2-3b-sre-agent",
    max_seq_length=2048,
    dtype=None,          # Auto-detects bfloat16/float16
    load_in_4bit=True,   # Optimizes for consumer GPUs like T4
)

# Switch to inference mode
FastLanguageModel.for_inference(model)

# Define your incident scenario
system_prompt = """You are the Lead SRE Manager.
Available services: [api_gateway, auth_service, user_db, frontend_service, product_db, cache_service]
Root Causes: [cpu_saturation, memory_leak, db_connection_leak, cascading_failure]
Fix Map: cpu_saturation→scale api_gateway | memory_leak→restart auth_service | db_connection_leak→restart user_db | cascading_failure→restart cache_service"""

scenario = """INCIDENT ACTIVE.
Logs show: auth_service → FATAL: OutOfMemoryError in auth_service
Metrics show: auth_service cpu=0.95, latency=0.03, error_rate=0.0
All other services are running normally.
What is the root cause and how do you fix it?"""

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user",   "content": scenario},
]

# Apply chat template
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# Generate response
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.1,  # Low temperature for deterministic actions
        do_sample=True,
    )

response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(response)
```

## 📊 Evaluation & Capabilities

The model was rigorously tested against standard failure scenarios defined in the `sre-env-triage` project:

| Scenario | Root Cause | Fix Action | Service Target | Success Rate |
| :--- | :--- | :--- | :--- | :--- |
| **Memory Leak** | ✅ Correct | ✅ Correct | ✅ Correct | 100% |
| **CPU Saturation** | ✅ Correct | ✅ Correct | ✅ Correct | 100% |
| **DB Connection Leak** | ✅ Correct | ✅ Correct | ✅ Correct | 100% |
| **Cascading Failure** | ✅ Correct | ✅ Correct | ✅ Correct | 100% |

### Strengths
- **Format Adherence:** Strictly follows required action outputs (e.g., `Action: execute_fix\nAction Input: {"service_name": "auth_service"}`).
- **Reduced Hallucinations:** Thanks to GRPO penalization, it rarely targets non-existent services (e.g., avoids hallucinating "Service A" or "service_x").

## 🛠️ Training Details

- **Framework:** Unsloth & TRL
- **Hardware:** 1x T4 GPU (16GB VRAM)
- **Quantization:** 4-bit (bitsandbytes)
- **LoRA Rank:** r=16
- **Phases:** SFT followed by GRPO with a custom strict-penalty reward function.
- **Dataset:** [danish1423/sre-agent-training-data](https://huggingface.co/datasets/danish1423/sre-agent-training-data)

## License

This model follows the `llama3.2` license. Please ensure compliance with Meta's acceptable use policy.
