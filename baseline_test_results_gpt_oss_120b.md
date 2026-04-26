# Baseline Test Results: GPT-OSS-120B (Phase 3)

**Date**: 2026-04-26
**Model**: `openai/gpt-oss-120b` (High-End)
**Environment**: `SreDecisionEnvV2` (Multi-Agent)
**Framework**: CrewAI (Sequential Process - Corrected Tasks)

---

## 📊 Summary of Performance

| Metric | Result |
| :--- | :--- |
| **Total Episodes** | 1 (Deep Dive) |
| **Successful Resolutions** | 0 (Logically 100%, Syntactically 0%) |
| **Avg. Reward** | -0.5 (Penalty for incorrect ID resolution) |
| **Tool Usage Rate** | 100% (Properly called query_logs, query_metrics, execute_fix) |
| **Hallucination Rate** | 0% (Used only real service names) |

---

## 🔍 Capability Analysis

The 120B model demonstrated "Expert-Level" logic but lacked "Domain-Specific Grounding":

### 1. Zero Hallucination
Unlike the 3B model, this model correctly identified and used real services from the environment: `api_gateway`, `auth_service`, `user_db`. It never mentioned "Service A".

### 2. Fact-Based Investigation
The model correctly followed the investigation pipeline:
1. **Log Investigator** found: `FATAL: OutOfMemoryError in auth_service` and `WARN: Thread starvation`.
2. **Metric Analyst** confirmed: `cpu: 0.35`, `latency: 0.03`.
3. **Incident Commander** synthesized these into a coherent remediation plan.

### 3. Logical Success vs. Technical Failure
The model's proposed solution was correct (scaling/restarting based on OOM). However, it failed the **Resolution Step** because it provided a descriptive name (`"connection timeout saturation"`) instead of the environment's internal enum ID (`"cascading_failure"`).

---

## 📝 Qualitative Log Samples

### Sample: Perfect Tool Use
```json
Using Tool: query_logs
Tool Input: {"service_name": "api_gateway"}
Observation: "INFO: Service running normally."
```
*The model followed the schema perfectly without looping.*

### Sample: Complex Reasoning
```text
Final Answer: "The incident is resolved. Root cause: connection timeout saturation."
Reality: Environment expected the specific ID 'cascading_failure'.
```

---

## 🚀 Comparison: 3B vs. 120B

| Feature | Llama 3.2 3B (Base) | GPT-OSS-120B |
| :--- | :--- | :--- |
| **Tool Syntax** | ❌ Fails (JSON errors) | ✅ Perfect |
| **Environment Grounding** | ❌ Hallucinates services | ✅ Accurate |
| **Investigation Logic** | ❌ None (Roleplay) | ✅ Strong (Fact-based) |
| **Final Resolution** | ❌ 0% | ⚠️ Logically correct, ID mismatch |

---

## 🎯 Conclusion
The **120B model** is "too smart for its own good"—it attempts to be a helpful human rather than a precise automation agent. 

This reinforces the need for **Phase 3 Fine-tuning**: We need to take the **reasoning power** of the 120B model and "compress" it into the 3B model while enforcing **strict adherence** to our environment's root cause IDs and tool protocols.
