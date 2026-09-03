# Production LLM Provider Options for the Chat Agent

**Date:** 2026-08-01
**Context:** `reference-app/backend/llm.py` / `agent.py` implement an OpenAI-compatible, swappable LLM layer that defaults to local Ollama (`llama3.2:3b`) in dev. The chat agent calls tools (`get_instruments`, `get_stem_activity`, `get_bpm_key`) and depends on reliable structured tool-call responses. Hard requirement for the hosted prod option: **no credit card at signup for the free tier** (must be impossible to accidentally bill), OpenAI-compatible chat-completions API, and reliable function/tool calling. Baseline to beat: local Ollama `llama3.2:3b`, 0% hallucination rate across 30 grounding questions, ~2.9s/turn, ~1.3 tool calls/turn.

This is research only — nothing was installed, no provider chosen, no code touched. Every claim below is tied to a primary source (provider docs/pricing pages, fetched directly, August 2026) or explicitly flagged as third-party/unverified.

---

## Executive summary

| Provider | No-card free tier? | OpenAI-compatible tool calling? | Verdict |
|---|---|---|---|
| **Cerebras** (cloud.cerebras.ai) | **No — changed since July.** Docs now say new accounts get $5 in free credits "after adding a verified payment method," and API/Playground access stays inactive until a card is added. **This fails the hard requirement as of today.** | Yes — OpenAI-compatible `tools`/`tool_choice`, strict mode, parallel tool calls by default. Only 3 models on the platform now (`gpt-oss-120b` production; `gemma-4-31b`, `zai-glm-4.7` preview) — the `llama-3.3-70b` / `qwen3` models many 2026 blog posts still cite for Cerebras have been deprecated/removed. | Disqualified today unless the repo owner is fine with adding a card that never gets charged (their own docs say it won't charge unless you buy more credits, but this contradicts the "impossible to accidentally bill" framing). |
| **Groq** (console.groq.com) | Groq's own docs never mention a payment method anywhere in the free-tier/rate-limits/quickstart pages, and no card-requirement language could be found in official docs — consistent with widespread 2026 third-party reporting of no-card signup. **Not explicitly stated by Groq itself**, so treat as "very likely true, not directly confirmed in Groq's own copy." | Yes — explicit per-model tool-use support table; `llama-3.1-8b-instant` and `llama-3.3-70b-versatile` (the models closest to the local baseline) support local tools, remote MCP tools, parallel tool calling, and JSON mode. GPT-OSS 20B/120B support tools but **not parallel tool calls**. | Strongest candidate on paper: real free tier, documented tool-calling matrix, 131K context on all production models, published per-model rate limits. |
| **OpenRouter** free (`:free`) models | Yes — usable at $0 balance, no card required for base free use (card only needed if buying credits to raise the daily cap). | Depends entirely on the underlying routed model; not documented as a blanket guarantee. | Viable but thin and volatile — see below. |
| **Together AI** | **No** — the earlier $25/$100 no-card signup credit was discontinued (per docs and multiple sources); the platform now requires a minimum $5 credit purchase to use it at all. | N/A (requires payment) | Disqualified. |
| **Fireworks AI** | Partial — $1 in free credits usable without a card, but capped at 10 RPM without one; a card is required to reach normal (6,000 RPM) throughput, and the $1 credit is a one-time trial, not a renewing free tier. | Documented, OpenAI-compatible. | Disqualified as a *sustained* no-card free tier — it's a trial credit, not a tier. |

**The single most important finding in this pass:** Cerebras's own docs, as of the page's last-modified timestamp (2026-07-23), now gate the "Free Trial" tier behind a verified payment method. This is a change from what the prior (2026-07-22) research pass recorded ("1M free tokens/day, no card") and from what most third-party 2026 blog posts still repeat. If the no-card requirement is truly hard, **Cerebras is out** under its current terms, leaving Groq (and, more thinly, OpenRouter's free models) as the no-card options actually worth testing.

---

## 1. Cerebras (cloud.cerebras.ai)

Source: [inference-docs.cerebras.ai/support/rate-limits](https://inference-docs.cerebras.ai/support/rate-limits) (page `dateModified: 2026-07-23T21:44:07.320Z`, fetched directly — the rendered page is a Mintlify/Next.js SPA, so the numbers below come from the embedded JSON/React props in the raw HTML, not a scraped summary), [inference-docs.cerebras.ai/models/overview](https://inference-docs.cerebras.ai/models/overview), [inference-docs.cerebras.ai/capabilities/tool-use](https://inference-docs.cerebras.ai/capabilities/tool-use), [www.cerebras.ai/pricing](https://www.cerebras.ai/pricing).

### Credit card / free-tier gate

Verbatim from the docs' own FAQ on the rate-limits page:

> "New accounts receive **$5 in free credits** after adding a verified payment method. These credits expire 30 days after they're granted and can be used across all public models. There is no charge until you choose to purchase additional credits. **If you skip adding a payment method at sign-up, Playground and API access remain inactive until you do.**"

The marketing pricing page ([cerebras.ai/pricing](https://www.cerebras.ai/pricing)) is vaguer — it just says "Free Trial: Get started with $5 in free credits after making an account" — and doesn't itself mention a card. The docs page is more authoritative and more recent, and it is unambiguous: **no payment method, no API access.** This directly conflicts with the "no credit card required" framing repeated across nearly every third-party 2026 blog post found in this pass (freellm.net, pricepertoken.com, getaiperks.com, tokenmix.ai, adam.holter.com) and with the prior (2026-07-22) research note in this repo. Either Cerebras changed policy recently, or those write-ups (and the earlier research pass) were wrong/stale even at the time. Given the direct docs quote, **treat "no card required" as false for Cerebras today** and re-verify at implementation time in case it flips back.

### Free-tier rate limits (as documented, per model)

| Model ID | Tier | RPM | TPM | TPH | TPD |
|---|---|---|---|---|---|
| `gpt-oss-120b` | Free Trial | 5 | 30K | 1M | 1M |
| `zai-glm-4.7` | Free Trial | 5 | 30K | 1M | 1M |
| `gemma-4-31b` | Free Trial | 5 | 30K | 1M | 1M (image limits: 2/request, 4 MB payload) |

At 5 RPM, the free tier is materially tighter than what most 2026 blogs cite (many still quote "30 RPM, 1M tokens/day" — that appears to describe an older tier configuration; the current docs table says 5 RPM). The **Developer (Pay-as-you-go)** tier — unlocked by *any* credit purchase — jumps to no hourly/daily caps and much higher RPM/TPM (e.g., `gpt-oss-120b`: 1K RPM / 1M TPM).

### Model menu (current, per official Model Catalog)

Cerebras's hosted lineup has shrunk to three models total as of this fetch:

- **Production:** `gpt-oss-120b` (120B, OpenAI's open-weight model, ~3,000 tok/s on Cerebras hardware)
- **Preview** (eval-only, can be discontinued without notice): `gemma-4-31b` (~1,850 tok/s), `zai-glm-4.7` (355B, ~1,000 tok/s)

Notably absent: `llama-3.3-70b`, `qwen3-32b`, `qwen3-235b`, `llama-4-scout` — all still referenced by name in 2026 third-party "free tier" write-ups, meaning several of those posts describe a lineup that's since been deprecated. The rate-limits page itself references a pending deprecation "by August 17, 2026," consistent with active churn.

### Context length

Not stated as a per-tier column in the official rate-limits or model-overview tables fetched. Third-party sources (not independently confirmed against a Cerebras doc page) claim an **8,192-token cap on the free tier** with higher context (up to 64K/131K) on paid tiers for some models — **treat this specific number as unverified** until confirmed directly in-account (the docs note "check the Limits section within your account" for exact current numbers).

### Tool/function calling

Documented at [inference-docs.cerebras.ai/capabilities/tool-use](https://inference-docs.cerebras.ai/capabilities/tool-use):

- OpenAI-compatible format: tools defined as `{"type": "function", ...}` with JSON-schema parameters, matching the OpenAI Chat Completions convention the agent already targets.
- **Strict mode** (`strict: true` in the function object) for exact schema-conforming arguments (requires `additionalProperties: false` on all objects).
- **Parallel tool calling** enabled by default, controllable via `parallel_tool_calls`.
- Multi-turn tool-call workflows are supported.
- The docs only name `gpt-oss-120b` and `zai-glm-4.7` in the tool-use examples — consistent with the shrunk model list above.

No caveats about small-model tool-calling fragility are stated in Cerebras's own docs (they simply don't discuss reliability at that level of detail).

---

## 2. Groq (console.groq.com)

Sources: [console.groq.com/docs/rate-limits](https://console.groq.com/docs/rate-limits), [console.groq.com/docs/models](https://console.groq.com/docs/models), [console.groq.com/docs/tool-use](https://console.groq.com/docs/tool-use) — all fetched directly.

### Credit card / free-tier gate

No mention of "credit card" or "payment method" anywhere in the rate-limits, models, or quickstart doc pages fetched for this report. The docs simply present a **"Free Plan Limits" / "Developer Plan Limits"** toggle on the same rate-limit table, implying the Free Plan is usable standalone. This is consistent with widespread 2026 third-party reporting ("sign up at console.groq.com with an email or Google account and you're making API calls within minutes... no credits system, no per-token charge on the free tier"). **Caveat:** this is an absence-of-evidence argument (no card language found), not a direct "no card required" statement from Groq itself — worth a 30-second manual signup check before relying on it for a public deployment.

### Free-tier rate limits (official, per model — `console.groq.com/docs/rate-limits`, "Free Plan Limits")

| Model ID | RPM | RPD | TPM | TPD |
|---|---|---|---|---|
| `llama-3.1-8b-instant` | 30 | 14,400 | 6,000 | 500,000 |
| `llama-3.3-70b-versatile` | 30 | 1,000 | 12,000 | 100,000 |
| `openai/gpt-oss-120b` | 30 | 1,000 | 8,000 | 200,000 |
| `openai/gpt-oss-20b` | 30 | 1,000 | 8,000 | 200,000 |
| `qwen/qwen3.6-27b` | 30 | 1,000 | 8,000 | 200,000 |
| `groq/compound` / `groq/compound-mini` | 30 | 250 | 70,000 | — |
| `whisper-large-v3` (ASR, not relevant here) | 20 | 2,000 | — | — |

`llama-3.1-8b-instant` — the model closest in size/class to the local baseline (`llama3.2:3b`) — has by far the most generous free-tier daily request budget (14.4K/day vs. 1K/day for the 70B and gpt-oss models), because it's cheap to serve. At ~1.3 tool calls/turn and a 2.9s/turn baseline, 14.4K RPD is roughly 10,000+ conversational turns/day of headroom — comfortably enough for a public demo app.

### Model menu & context length (official, `console.groq.com/docs/models`)

| Model | Context window | Max completion tokens | Speed (Groq's own figure) |
|---|---|---|---|
| Llama 3.1 8B (`llama-3.1-8b-instant`) | 131,072 | 131,072 | ~560 tok/s |
| Llama 3.3 70B (`llama-3.3-70b-versatile`) | 131,072 | 32,768 | ~280 tok/s |
| GPT-OSS 120B (`openai/gpt-oss-120b`) | 131,072 | 65,536 | ~500 tok/s |
| GPT-OSS 20B (`openai/gpt-oss-20b`) | 131,072 | 65,536 | ~1,000 tok/s |
| Groq Compound / Compound Mini | 131,072 | 8,192 | ~450 tok/s |

All production models share a 131K context window — comfortably larger than whatever the chat agent currently needs, and much larger than Cerebras's reported (unverified) 8K free-tier cap.

### Tool/function calling

Documented at [console.groq.com/docs/tool-use](https://console.groq.com/docs/tool-use), with an explicit per-model capability table:

- **Full support** (local tools + remote MCP tools + parallel tool calling + JSON mode): `llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, `qwen/qwen3.6-27b`, `minimaxai/minimax-m2.7`.
- **Partial support**: `openai/gpt-oss-20b` / `openai/gpt-oss-120b` — local + remote tool use + built-in tools, but **no parallel tool calling**.
- **Compound models** — built-in tools only, server-side execution, not general-purpose function calling.
- No explicit reliability caveats are stated in Groq's own docs for smaller models — see §4 for GitHub-sourced reliability signal instead.

---

## 3. Other no-card candidates checked

### OpenRouter (free `:free` models)

Source: [openrouter.ai/docs/api-reference/limits](https://openrouter.ai/docs/api-reference/limits) plus third-party corroboration.

- Sign-up: email only, $0 balance usable, no card required for the base free tier.
- Rate limits: **20 RPM always**; **50 requests/day** under $10 lifetime credits purchased, rising to **1,000 requests/day** once $10+ has been purchased (i.e., the generous daily cap requires having put a card in at some point — the *base* no-card tier is only 50 req/day).
- Model lineup: **volatile**. Third-party tracking reports 15 free model IDs live around July 27, 2026, down from 20 nine days earlier, after Meta Llama and Qwen free variants were delisted entirely. This is not a stable menu to build a product on — a model your agent depends on can disappear with no announcement window comparable to Groq/Cerebras's "preview model" framing.
- Tool calling: no blanket documentation found; support is inherited per-model from whichever backend OpenRouter is routing to, so it would need to be checked per specific `:free` model ID at implementation time, and re-checked whenever the free lineup rotates.
- **Verdict:** technically satisfies the no-card requirement, but at 50 req/day (before any spend) and a rotating model list, it's a much weaker production floor than Groq's free tier.

### Together AI

- Multiple 2026 sources conflict, but the more recent and more specific ones say the earlier $25 (previously $5, briefly $100 per some posts) no-card signup credit was **retired around July 2025**, and Together's own docs now state there's no free trial — a **minimum $5 credit purchase** is required to use the platform at all.
- **Verdict: disqualified** — fails the no-card requirement outright under current terms.

### Fireworks AI

- New accounts get **$1 in free credits**, usable without adding a card, but the request rate without a card is capped at **10 RPM**; adding a card raises the ceiling to a flat 6,000 RPM and unlocks the normal spend-tier ladder ($50/mo → $500/mo → ...).
- This is a one-time trial credit (~1M tokens against a 70B-class model per some estimates), not a renewing free allotment — it will run out, at which point the app either needs a card or stops working.
- **Verdict: disqualified as a sustained free tier** — it's a trial, and the no-card path is rate-limited to the point of being unusable for a public demo (10 RPM ≈ one request every 6 seconds, shared across all users).

No other genuinely free, no-card, OpenAI-compatible hosted option was found in this pass beyond the ones above (Cerebras, Groq, OpenRouter free models, Together, Fireworks were the candidates named in the research brief, and none of the search results surfaced a materially different additional option as of August 2026).

---

## 4. Tool-calling reliability with small/medium open models — what's actually documented vs. unknown

This is the weakest-evidenced part of the research, exactly as flagged in the brief. What's findable:

- **Cerebras:** no GitHub issues or provider caveats were found specifically about tool-calling reliability on `gpt-oss-120b` (the only production, free-tier-eligible model with documented tool support) — likely because Cerebras's current free-tier lineup is narrow and comparatively new. **This is an unknown, not a "no problems reported" result** — the absence of issues may just reflect low sample size / low community usage of Cerebras's tool-calling path.
- **Groq:** several GitHub issues surfaced describing friction with Llama 3.1/3.3 tool calling through Groq specifically:
  - [BerriAI/litellm#5195](https://github.com/BerriAI/litellm/issues/5195) — "Tool calling is not working with `groq/llama-3.1-70b-versatile`."
  - [microsoft/autogen#3217](https://github.com/microsoft/autogen/issues/3217) — `BadRequestError: Failed to call a function. Please adjust your prompt` when using Llama 3.1 via Groq in AutoGen; the same code worked against GPT-4o.
  - A Hugging Face discussion on the (separate, fine-tuned) `Groq/Llama-3-Groq-8B-Tool-Use` model notes missing `<tool_call>` tags in output — a formatting-fragility failure mode in the same family.
  - General community sentiment in these threads: tool calls "work but the model is fragile compared to Claude 3.5 Sonnet," with issues traced to prompt formatting and framework-specific request shaping rather than a fundamental Groq API defect.
  - None of these issues are recent (2026) confirmations against the *current* `llama-3.1-8b-instant` / `llama-3.3-70b-versatile` tool-use path documented today, and none map directly onto this app's exact 3-tool, single-turn-ish usage pattern (`get_instruments`, `get_stem_activity`, `get_bpm_key`).
- **Bottom line:** this is genuinely under-documented for the specific case that matters here (an 8B-class model doing 1-3 simple, well-typed tool calls per turn against a small fixed tool set). Neither provider's own docs discuss reliability at that granularity, and the GitHub signal that exists is old, framework-specific, and about different (larger or differently fine-tuned) models. **This needs the same kind of local eval harness already built for the Ollama baseline (0% hallucination, 30 grounding questions) run against whichever hosted model is chosen, before trusting it in production.**

---

## 5. Latency expectations

| Provider / model | Published throughput | Source | Confidence |
|---|---|---|---|
| Groq `llama-3.1-8b-instant` | ~560 tok/s output | [console.groq.com/docs/models](https://console.groq.com/docs/models) | Provider-published, current |
| Groq `llama-3.3-70b-versatile` | ~280 tok/s output | Same | Provider-published, current |
| Groq `gpt-oss-120b` / `gpt-oss-20b` | ~500 / ~1,000 tok/s | Same | Provider-published, current |
| Cerebras `gpt-oss-120b` | ~3,000 tok/s output | [inference-docs.cerebras.ai/models/overview](https://inference-docs.cerebras.ai/models/overview) | Provider-published, current |
| Cerebras `gemma-4-31b` / `zai-glm-4.7` | ~1,850 / ~1,000 tok/s | Same | Provider-published, current |
| Groq time-to-first-token | ~0.22s (Llama 2 70B, older public benchmark) | Third-party (ArtificialAnalysis-referenced), **not from current Groq docs and not for a current free-tier model** | **Unverified for the models in scope today** |
| Cerebras time-to-first-token | Not found in official docs for any current model | — | **Unverified** |

Neither provider's own docs publish a clean "time to first token" or "total completion time" figure for the exact models on their free tiers today — both companies market **output tokens/sec** (post-first-token throughput), not TTFT, as their headline number. Reasoning from the throughput numbers: at 500-1,000+ tok/s, a ~200-400 token tool-calling response would generate in well under a second on either provider, versus the ~2.9s/turn Ollama baseline (which includes local model load/compute on much weaker hardware). In practice, for this app's 1.3-tool-calls/turn pattern, **total turn latency will likely be dominated by the number of round trips to the provider (network + queueing) rather than raw generation speed** — every extra tool-call round trip adds one full network hop. This is a reasoned extrapolation, not a cited number, and should be measured directly (e.g., a simple curl-timed script hitting each provider's chat-completions endpoint with the app's actual tool schema) before assuming it beats the local baseline in wall-clock terms.

---

## Unknowns needing local verification

1. **Does Cerebras's free tier actually require a card today, or is the docs FAQ describing a different flow than what a fresh signup experiences?** The quote is unambiguous, but it's worth a 5-minute manual signup attempt (from a throwaway email) to confirm before ruling Cerebras out entirely, since it directly overturns both the July 22 research note and most third-party write-ups.
2. **Does Groq's signup flow ever ask for a card?** No card language appears in Groq's docs, but that's an absence-of-evidence finding, not a direct confirmation — a manual signup check is cheap and worth doing before relying on it for a public deployment.
3. **Cerebras's actual free-tier context length** — official docs don't state it in the tables fetched; only third-party sources claim 8,192 tokens. Needs confirmation from the account's own Limits page after signup.
4. **Tool-calling reliability at the exact usage pattern this app needs** (3 simple, typed tools; ~1.3 calls/turn; grounding-question style prompts) — not documented by either provider at this granularity, and the GitHub issues found are old and about different frameworks/models. This is the single biggest open risk and should be closed by re-running the existing 30-question eval harness against both Groq `llama-3.1-8b-instant` and Cerebras `gpt-oss-120b` (if the card question resolves favorably) before picking one.
5. **Real end-to-end turn latency** (not just generation throughput) for this app's specific request shape, including the 1.3-tool-calls/turn overhead — needs direct measurement, not extrapolation from published tok/s figures.
6. **OpenRouter free-model churn** — if OpenRouter is considered as a fallback/secondary option, whatever specific `:free` model is picked today may not exist in a month; this would need a monitoring/fallback strategy baked in, not a one-time choice.
7. **Whether Cerebras's and Groq's rate limits (5 RPM for Cerebras free tier; 1K RPD for Groq's 70B-class models) are sufficient for actual public traffic patterns**, once the app has more than a handful of concurrent users — both are fine for a solo-demo scale, but the ceilings differ a lot by model within each provider and should be checked against expected traffic before launch.

---

## Open decision

No provider is recommended here. The facts above (especially the Cerebras card-requirement change, which most existing write-ups — including this repo's own July 22 note — don't reflect) are meant to inform, not make, the call. Choosing between Groq, a possible-with-caveats Cerebras, and OpenRouter's free tier as a fallback is deferred to a follow-up ticket, ideally paired with running the existing local eval harness against the top 1-2 candidates before committing.
