"""
Push SRE Training Datasets to HuggingFace Hub
=============================================
Uploads all 5 agent datasets as a single HuggingFace Dataset repo.

Usage:
    HF_TOKEN=<your_token> uv run python v2/scripts/push_to_hf.py --repo <username>/<repo-name>
    
    or set token with:
    uv run huggingface_hub login
    then:
    uv run python v2/scripts/push_to_hf.py --repo <username>/<repo-name>
"""

import argparse
import json
import os
import sys
from pathlib import Path

from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, login

# ── Parse Args ────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument(
    "--repo",
    type=str,
    default="Danish1323/sre-agent-training-data",
    help="HuggingFace repo ID (e.g. your-username/sre-agent-training-data)"
)
parser.add_argument(
    "--private",
    action="store_true",
    default=False,
    help="Make the dataset repo private"
)
args = parser.parse_args()

# ── Auth ──────────────────────────────────────────────────────────────────────

token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
if not token:
    print("\nNo HF_TOKEN found in environment.")
    print("Please run: uv run huggingface-cli login")
    print("Or set the environment variable: export HF_TOKEN=hf_xxxx\n")
    token = input("Paste your HuggingFace token here (or press Enter to skip): ").strip()
    if not token:
        print("No token provided. Exiting.")
        sys.exit(1)

login(token=token)

# ── Load JSONL files ──────────────────────────────────────────────────────────

TRAINING_DIR = Path("data/training")

AGENT_FILES = {
    "log_investigator":   TRAINING_DIR / "log_investigator_unsloth.jsonl",
    "metric_analyst":     TRAINING_DIR / "metric_analyst_unsloth.jsonl",
    "incident_commander": TRAINING_DIR / "incident_commander_unsloth.jsonl",
    "infra_executor":     TRAINING_DIR / "infra_executor_unsloth.jsonl",
    "lead_manager":       TRAINING_DIR / "lead_manager_unsloth.jsonl",
}

def load_jsonl(path: Path):
    records = []
    with open(path, "r") as f:
        for line in f:
            record = json.loads(line)
            # Flatten messages for HF dataset compatibility
            record["system_prompt"] = record["prompt"][0]["content"]
            record["user_prompt"]   = record["prompt"][1]["content"]
            del record["prompt"]
            records.append(record)
    return records

print(f"\n[+] Loading datasets from {TRAINING_DIR}...")

splits = {}
for agent_name, filepath in AGENT_FILES.items():
    if not filepath.exists():
        print(f"  ✗ Missing: {filepath}")
        continue
    records = load_jsonl(filepath)
    splits[agent_name] = Dataset.from_list(records)
    print(f"  ✓ {agent_name}: {len(records)} samples loaded")

# ── Create DatasetDict ────────────────────────────────────────────────────────

dataset_dict = DatasetDict(splits)

print(f"\n[+] DatasetDict created:")
print(dataset_dict)

# ── Push to Hub ───────────────────────────────────────────────────────────────

print(f"\n[+] Pushing to HuggingFace Hub: {args.repo}")
print(f"    Private: {args.private}")

dataset_dict.push_to_hub(
    repo_id=args.repo,
    private=args.private,
    token=token,
    commit_message="Add SRE multi-agent training dataset (9000 expert trajectories)"
)

print(f"\n✅ Upload complete!")
print(f"   View at: https://huggingface.co/datasets/{args.repo}")
