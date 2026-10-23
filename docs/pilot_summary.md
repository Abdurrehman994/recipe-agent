# Pilot run: 10 development scenarios across all configurations

Date: 2026-10-23. Model: `openai/gpt-oss-120b` on Groq (free tier), reasoning effort medium,
JSON-schema output. Scenarios: `dev_001`–`dev_010` (39 user turns). The 70 held-out scenarios
were not used.

Purpose, as requested: verify the evaluation pipeline end to end and measure actual model calls
and token usage, to size the final evaluation.

## 1. Pipeline check

Every stage ran without errors on the pilot scenarios: constraint extraction (rule-based and
LLM), dialogue state, the five system configurations, logging, claim verification and metric
computation. 70 unit tests pass.

| System | Action acc. | Constraint satisfaction | Silent relaxations | Infeasible recall | Tool calls / turn |
|---|---|---|---|---|---|
| RAG baseline (transcript retrieval) | 0.897 | 0.214 | 35 of 39 | 0.00 | 1.0 |
| RAG + structured memory | 0.897 | 0.949 | 2 | 0.00 | 1.1 |
| Full agent | 1.000 | 1.000 | 0 | 1.00 | 6.4 |
| Agent without memory | 0.846 | 0.384 | 22 | 0.00 | 5.9 |
| Agent without validation | 0.949 | 0.234 | 32 | 0.00 | 1.0 |

Numbers are identical with the rule-based and the LLM extractor, because both reach 1.000
extraction F1 on these development scenarios (both were tuned on the development split; the
held-out split will separate them). Unsupported factual claims: 0 of 2,085 checked, as expected
for explanations generated from tool output. The pilot confirms the memory and validation
ablations behave as intended: removing memory drops constraint satisfaction from 1.00 to 0.38,
removing validation drops it to 0.23.

## 2. Measured model usage

The LLM is used for one thing: converting each user turn into structured constraint updates.
Retrieval, validation, the decision policy and the explanations are deterministic code, so they
make no model calls. One extraction call per user turn.

| Quantity | Measured |
|---|---|
| Calls per user turn | 1 |
| Prompt tokens per call | 1,695 (system prompt ≈ 1,100, state + utterance ≈ 100, schema overhead) |
| Completion tokens per call | 267 (JSON output ≈ 40, the rest is the model's reasoning) |
| Total tokens per call | 1,962 |
| Pilot total, 39 turns, fresh calls | 76,538 tokens |
| Wall time per call | ≈ 2 s pacing + 1–3 s latency |

Configurations that keep the dialogue state (RAG + memory, full agent, agent without
validation) send identical prompts, so one extraction pass serves all three through the on-disk
response cache. The memory ablation sends different prompts (state rebuilt from the current turn)
and needs its own pass. The RAG baseline makes no model calls.

## 3. Resource estimate for the final evaluation

70 held-out scenarios, 267 user turns, plus one verification pass on the 30 development
scenarios (108 turns) to check that nothing changed between pilot and final run.

| Run | Calls | Tokens (at 1,962 per call) |
|---|---|---|
| Held-out, memory configurations (shared) | 267 | 0.52 M |
| Held-out, memory ablation | 267 | 0.52 M |
| Development verification pass (both) | 216 | 0.42 M |
| Contingency for one re-run after a bug fix (50 %) | 375 | 0.74 M |
| **Total** | **≈ 1,125** | **≈ 2.2 M** |

On Groq's free tier (200,000 tokens per day for this model) that is about 11 days of quota.
On Lehmus AI (GPT OSS 120B is in its catalogue) it is a few hours of unattended running at
one call every 2–3 seconds, or minutes if the endpoint allows parallel requests. No GPU or
other CSC compute is needed: the pipeline is CPU-only, the recipe database is 11 MB, and a full
configuration run over all 100 scenarios takes under a minute excluding model calls.

If the optional extension is included (an LLM-written explanation checked by the reviewer, one
extra call per recommend turn per configuration), add roughly 5 × 267 calls ≈ 1,335 calls and
≈ 2.7 M tokens.

## 4. Requirements for the final run

- Lehmus AI API key and endpoint URL (OpenAI-compatible). JSON-schema output support should be
  confirmed; if unavailable, the extractor already validates every response against the schema
  and drops invalid items, so plain JSON mode suffices.
- Same model as the pilot, `openai/gpt-oss-120b`, so the final run replicates the pilot setup.
- Approximately 2.2 M tokens (3.5 M with the optional extension).
