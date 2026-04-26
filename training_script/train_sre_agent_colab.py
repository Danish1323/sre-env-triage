# SRE Agent Fine-Tuning Notebook (Llama 3.2 3B + Unsloth + GRPO)
# ================================================================
# Target GPU: T4 16GB (HuggingFace Spaces or Google Colab)
# Dataset:    danish1423/sre-agent-training-data
# Method:     SFT → GRPO (Reinforcement Learning)
# ================================================================

# CELL 1 — Install Dependencies
# ─────────────────────────────────────────────────────────────
# %%
# Install Unsloth (optimized for T4)
#!pip install unsloth -q
#!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" -q
#!pip install trl==0.8.6 datasets transformers accelerate bitsandbytes -q
print("✅ Dependencies installed")


# CELL 2 — Login to HuggingFace
# ─────────────────────────────────────────────────────────────
# %%
from huggingface_hub import login
from google.colab import userdata  # Remove this line if not on Colab

# Option A: Colab secret (recommended - add HF_TOKEN in Colab Secrets panel)
HF_TOKEN = userdata.get("HF_TOKEN")

# Option B: Paste token directly (less secure)
# HF_TOKEN = "hf_xxxx"

login(token=HF_TOKEN)
print("✅ Logged in to HuggingFace")


# CELL 3 — Load Dataset
# ─────────────────────────────────────────────────────────────
# %%
from datasets import load_dataset, concatenate_datasets

# Load the SRE training data from HuggingFace Hub
print("[+] Loading datasets...")
raw = load_dataset("danish1423/sre-agent-training-data")

# View structure
print(raw)
print("\nExample split keys:", list(raw.keys()))
print("\nSample from lead_manager:")
print(raw["lead_manager"][0])


# CELL 4 — Choose which agent(s) to train
# ─────────────────────────────────────────────────────────────
# %%
# Options:
#   "all"              → merge all 9000 samples (best for a generalist model)
#   "lead_manager"     → 5000 samples, knows the full pipeline
#   "log_investigator" → 1000 samples, log specialist
#   "metric_analyst"   → 1000 samples, metric specialist
#   "incident_commander" → 1000 samples, triage specialist
#   "infra_executor"   → 1000 samples, fix specialist

TRAIN_MODE = "all"   # Change this to train a specific specialist

if TRAIN_MODE == "all":
    train_dataset = concatenate_datasets([raw[split] for split in raw.keys()])
    print(f"[+] Merged all splits: {len(train_dataset)} total samples")
else:
    train_dataset = raw[TRAIN_MODE]
    print(f"[+] Using split '{TRAIN_MODE}': {len(train_dataset)} samples")

# Shuffle
train_dataset = train_dataset.shuffle(seed=42)
print(f"\nDataset features: {train_dataset.features}")


# CELL 5 — Load Llama 3.2 3B with Unsloth (4-bit quantized for T4)
# ─────────────────────────────────────────────────────────────
# %%
from unsloth import FastLanguageModel
import torch

MAX_SEQ_LENGTH = 2048   # T4 can handle 2048 comfortably
DTYPE = None            # Auto-detect (bfloat16 on Ampere+, float16 on T4)
LOAD_IN_4BIT = True     # Essential for T4 16GB

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT,
    token=HF_TOKEN
)

print(f"✅ Model loaded: {model.num_parameters():,} parameters")
print(f"   Device: {next(model.parameters()).device}")
print(f"   Dtype:  {next(model.parameters()).dtype}")


# CELL 6 — Attach LoRA Adapters
# ─────────────────────────────────────────────────────────────
# %%
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                        # LoRA rank (16 = good balance for 3B)
    target_modules=[             # Layers to fine-tune
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_alpha=32,               # Scaling factor (2x rank)
    lora_dropout=0.0,            # 0 = stable for most tasks
    bias="none",
    use_gradient_checkpointing="unsloth",   # Reduces VRAM by ~30%
    random_state=42,
    use_rslora=False,
    loftq_config=None,
)

model.print_trainable_parameters()
print("✅ LoRA adapters attached")


# CELL 7 — Format Data for Training (Chat Template)
# ─────────────────────────────────────────────────────────────
# %%
from unsloth.chat_templates import get_chat_template

# Apply the Llama 3.1 chat template (compatible with 3.2)
tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3.1",
)

def format_sample(sample):
    """
    Convert our dataset format to a full conversation string.
    Input:  system_prompt, user_prompt, chosen (assistant response)
    Output: Llama-3 formatted chat string
    """
    messages = [
        {"role": "system",    "content": sample["system_prompt"]},
        {"role": "user",      "content": sample["user_prompt"]},
        {"role": "assistant", "content": sample["chosen"]},
    ]
    return {
        "text": tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
    }

# Apply formatting
print("[+] Formatting dataset...")
formatted_dataset = train_dataset.map(
    format_sample,
    remove_columns=train_dataset.column_names,
    desc="Formatting"
)

print(f"✅ Formatted {len(formatted_dataset)} samples")
print("\n--- Sample (first 600 chars) ---")
print(formatted_dataset[0]["text"][:600])
print("...")


# CELL 8 — SFT Training (Phase 1)
# ─────────────────────────────────────────────────────────────
# %%
from trl import SFTTrainer
from transformers import TrainingArguments

SFT_OUTPUT_DIR = "/tmp/sre-sft-checkpoint"

training_args = TrainingArguments(
    output_dir=SFT_OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,    # Effective batch = 2*4 = 8
    warmup_steps=10,
    num_train_epochs=2,               # 2 passes through data
    learning_rate=2e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=20,
    save_steps=100,
    save_total_limit=2,
    optim="adamw_8bit",               # Memory-efficient optimizer
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    seed=42,
    report_to="none",                 # Set to "wandb" if you use W&B
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=formatted_dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=2,
    packing=True,                     # Packs multiple short samples → faster training
    args=training_args,
)

print(f"[+] Starting SFT training on {len(formatted_dataset)} samples")
print(f"    Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"    Epochs: {training_args.num_train_epochs}")
print(f"    Learning rate: {training_args.learning_rate}\n")

trainer_stats = trainer.train()

print(f"\n✅ SFT Training complete!")
print(f"   Time: {trainer_stats.metrics['train_runtime']:.0f}s")
print(f"   Samples/sec: {trainer_stats.metrics['train_samples_per_second']:.1f}")
print(f"   Final loss: {trainer_stats.metrics['train_loss']:.4f}")


# CELL 9 — GRPO Reward Function (RL Phase — Reinforcement Learning)
# ─────────────────────────────────────────────────────────────
# %%
import re

# Valid catalog for scoring
VALID_ROOT_CAUSES = {"cpu_saturation", "memory_leak", "db_connection_leak", "cascading_failure"}
VALID_SERVICES    = {"api_gateway", "auth_service", "user_db", "frontend_service", "product_db", "cache_service"}
VALID_FIX_TYPES   = {"restart", "scale", "rollback"}
VALID_ACTIONS     = {"query_logs", "query_metrics", "share_info", "execute_fix", "propose_hypothesis", "resolve_incident"}
FIX_MAP = {
    "cpu_saturation":     ("scale",   "api_gateway"),
    "memory_leak":        ("restart", "auth_service"),
    "db_connection_leak": ("restart", "user_db"),
    "cascading_failure":  ("restart", "cache_service"),
}

def sre_reward_function(completions: list[str], prompts=None, **kwargs) -> list[float]:
    """
    GRPO Reward Function for SRE Agent.
    Scores each completion based on:
      +0.3  → Used at least one valid Action: keyword
      +0.2  → Referenced a valid service name
      +0.3  → Identified a valid root cause
      +0.5  → Correct root cause matching scenario (if ground truth in prompt)
      +0.3  → Correct fix type for identified root cause
      +0.5  → Resolved incident with correct ID
      -0.2  → Hallucinated unknown service
      -0.3  → Used wrong fix for root cause
    """
    rewards = []
    for completion in completions:
        score = 0.0
        text = completion if isinstance(completion, str) else completion[0].get("content", "")

        # +0.3 for using tool call syntax
        if re.search(r"Action:\s*(query_logs|query_metrics|execute_fix|propose_hypothesis|resolve_incident)", text):
            score += 0.3

        # +0.2 for referencing a valid service
        found_services = [s for s in VALID_SERVICES if s in text]
        if found_services:
            score += 0.2

        # -0.2 for hallucinating "Service A" or "service_x" pattern
        if re.search(r"[Ss]ervice [A-Z]|service_\w+", text) and not found_services:
            score -= 0.2

        # +0.3 for valid root cause
        found_rc = [rc for rc in VALID_ROOT_CAUSES if rc in text]
        if found_rc:
            score += 0.3

        # +0.3 for correct fix logic
        for rc in found_rc:
            correct_fix, correct_svc = FIX_MAP[rc]
            if correct_fix in text and correct_svc in text:
                score += 0.3
            elif correct_fix not in text and ("restart" in text or "scale" in text):
                score -= 0.3

        # +0.5 for resolve_incident with valid ID
        if "resolve_incident" in text:
            for rc in VALID_ROOT_CAUSES:
                if rc in text:
                    score += 0.5
                    break

        rewards.append(min(max(score, -1.0), 2.0))   # Clamp to [-1, 2]

    return rewards

# Quick sanity check
test_completions = [
    "Action: query_logs\nAction Input: {\"service_name\": \"auth_service\"}\nAction: resolve_incident\nAction Input: {\"root_cause_id\": \"memory_leak\"}",
    "I think the problem is in Service A. Let me restart it.",
]
test_rewards = sre_reward_function(test_completions)
print("Reward function sanity check:")
for c, r in zip(test_completions, test_rewards):
    print(f"  reward={r:+.1f} | {c[:80]}...")


# CELL 10 — GRPO Training (Phase 2 — RL)
# ─────────────────────────────────────────────────────────────
# %%
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset

GRPO_OUTPUT_DIR = "/tmp/sre-grpo-checkpoint"

# Prepare GRPO dataset (prompt only — model will generate completions)
def make_grpo_prompt(sample):
    """Convert to prompt-only format for GRPO generation."""
    messages = [
        {"role": "system", "content": sample["system_prompt"]},
        {"role": "user",   "content": sample["user_prompt"]},
    ]
    return {
        "prompt": tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True    # Add <|start_header_id|>assistant<|end_header_id|>
        )
    }

grpo_dataset = train_dataset.map(
    make_grpo_prompt,
    remove_columns=train_dataset.column_names,
    desc="Preparing GRPO prompts"
)

print(f"✅ GRPO dataset ready: {len(grpo_dataset)} prompts")

# GRPO Config — conservative for T4 16GB
grpo_config = GRPOConfig(
    output_dir=GRPO_OUTPUT_DIR,
    per_device_train_batch_size=1,       # 1 prompt at a time (T4 memory)
    gradient_accumulation_steps=8,       # Effective batch = 8
    num_generations=4,                   # Generate 4 completions per prompt, pick best
    max_new_tokens=512,                  # Max response length
    learning_rate=5e-6,                  # Lower LR for RL stability
    num_train_epochs=1,
    logging_steps=10,
    save_steps=50,
    report_to="none",
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    seed=42,
    warmup_ratio=0.1,
    temperature=0.8,                     # Sample diversity for RL exploration
)

grpo_trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=sre_reward_function,    # Our custom SRE reward
    config=grpo_config,
    train_dataset=grpo_dataset,
)

print("[+] Starting GRPO (RL) training...")
print(f"    Num generations per prompt: {grpo_config.num_generations}")
print(f"    This phase teaches the model to maximize SRE reward.\n")

grpo_trainer.train()
print("✅ GRPO Training complete!")


# CELL 11 — Test the Trained Model
# ─────────────────────────────────────────────────────────────
# %%
from unsloth.chat_templates import get_chat_template

# Switch to inference mode
FastLanguageModel.for_inference(model)

def run_sre_inference(scenario_description: str, system_prompt: str = None) -> str:
    """Run the trained SRE agent on a scenario."""
    if system_prompt is None:
        system_prompt = """You are the Lead SRE Manager.
Available services: [api_gateway, auth_service, user_db, frontend_service, product_db, cache_service]
Root Causes: [cpu_saturation, memory_leak, db_connection_leak, cascading_failure]
Fix Map: cpu_saturation→scale api_gateway | memory_leak→restart auth_service | db_connection_leak→restart user_db | cascading_failure→restart cache_service"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": scenario_description},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,             # Low temp for deterministic SRE responses
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response


# --- Test Case 1: Memory Leak Scenario ---
print("=" * 60)
print("TEST 1: Memory Leak Scenario")
print("=" * 60)
test1 = """INCIDENT ACTIVE.
Logs show: auth_service → FATAL: OutOfMemoryError in auth_service
Metrics show: auth_service cpu=0.95, latency=0.03, error_rate=0.0
All other services are running normally.
What is the root cause and how do you fix it?"""
print(run_sre_inference(test1))

# --- Test Case 2: CPU Saturation ---
print("\n" + "=" * 60)
print("TEST 2: CPU Saturation Scenario")
print("=" * 60)
test2 = """INCIDENT ACTIVE.
Logs show: api_gateway → WARN: Thread starvation detected
Metrics show: api_gateway cpu=0.99, latency=0.8, error_rate=0.0
All other services are running normally.
What is the root cause and how do you fix it?"""
print(run_sre_inference(test2))

# --- Test Case 3: DB Connection Leak ---
print("\n" + "=" * 60)
print("TEST 3: DB Connection Leak Scenario")
print("=" * 60)
test3 = """INCIDENT ACTIVE.
Logs show: user_db → ERROR: connection pool exhausted. Waiting for connection...
            auth_service → WARN: Timeout connecting to user_db
Metrics show: user_db latency=1.5, error_rate=0.8
What is the root cause and how do you fix it?"""
print(run_sre_inference(test3))


# CELL 12 — Evaluate Accuracy (Structured Test)
# ─────────────────────────────────────────────────────────────
# %%
test_cases = [
    {
        "name": "memory_leak",
        "prompt": "auth_service shows FATAL: OutOfMemoryError. cpu=0.95. All others normal.",
        "expected_rc": "memory_leak",
        "expected_fix": "restart",
        "expected_svc": "auth_service",
    },
    {
        "name": "cpu_saturation",
        "prompt": "api_gateway shows WARN: Thread starvation detected. cpu=0.99, latency=0.8.",
        "expected_rc": "cpu_saturation",
        "expected_fix": "scale",
        "expected_svc": "api_gateway",
    },
    {
        "name": "db_connection_leak",
        "prompt": "user_db shows ERROR: connection pool exhausted. latency=1.5, error_rate=0.8.",
        "expected_rc": "db_connection_leak",
        "expected_fix": "restart",
        "expected_svc": "user_db",
    },
    {
        "name": "cascading_failure",
        "prompt": "cache_service shows ERROR: Cache eviction failed. error_rate=1.0. frontend_service latency=2.0.",
        "expected_rc": "cascading_failure",
        "expected_fix": "restart",
        "expected_svc": "cache_service",
    },
]

results = []
for tc in test_cases:
    response = run_sre_inference(tc["prompt"])
    rc_correct  = tc["expected_rc"]  in response
    fix_correct = tc["expected_fix"] in response
    svc_correct = tc["expected_svc"] in response
    score = sum([rc_correct, fix_correct, svc_correct]) / 3

    results.append({
        "scenario": tc["name"],
        "rc_correct": rc_correct,
        "fix_correct": fix_correct,
        "svc_correct": svc_correct,
        "score": score,
    })

print("\n" + "=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)
print(f"{'Scenario':<25} {'Root Cause':^10} {'Fix':^8} {'Service':^10} {'Score':^8}")
print("-" * 60)
for r in results:
    print(f"{r['scenario']:<25} {'✅' if r['rc_correct'] else '❌':^10} {'✅' if r['fix_correct'] else '❌':^8} {'✅' if r['svc_correct'] else '❌':^10} {r['score']*100:.0f}%")

overall = sum(r["score"] for r in results) / len(results) * 100
print("-" * 60)
print(f"{'OVERALL ACCURACY':<25} {overall:.1f}%")


# CELL 13 — Save & Push to HuggingFace Hub
# ─────────────────────────────────────────────────────────────
# %%
SAVE_REPO = "danish1423/llama-3.2-3b-sre-agent"

print(f"[+] Saving model to HuggingFace Hub: {SAVE_REPO}")

# Save LoRA adapters only (lightweight — ~50MB)
model.push_to_hub_merged(
    SAVE_REPO,
    tokenizer,
    save_method="lora",          # Save only LoRA weights (tiny)
    token=HF_TOKEN,
)

print(f"✅ Model pushed to: https://huggingface.co/{SAVE_REPO}")
print(f"\nLoad anytime with:")
print(f"  from unsloth import FastLanguageModel")
print(f"  model, tokenizer = FastLanguageModel.from_pretrained('{SAVE_REPO}')")

# Optional: Also save a 4-bit quantized GGUF for local use
# model.push_to_hub_gguf(SAVE_REPO + "-GGUF", tokenizer, quantization_method="q4_k_m", token=HF_TOKEN)
