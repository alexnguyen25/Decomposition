Type: grilling
Status: resolved (2026-09-03)

## Question

Ticket 03's research narrowed the field: Cerebras is disqualified (now
requires a card), Together and Fireworks are disqualified (same). The live
options are **Groq** (strongest on paper — no card language, full tool
calling, 30 RPM/500K TPD on `llama-3.1-8b-instant`) and **OpenRouter free
models** (genuinely no-card but thin at 50 req/day, lineup churns monthly).

Needs a real decision on:
- Groq as primary, OpenRouter as an unused fallback — or does the fact that
  OpenRouter's free lineup changes monthly make it worth wiring up as an
  actual automatic fallback in `llm.py`/`agent.py` (the swappable layer
  already supports swapping `LLM_BASE_URL`/`LLM_MODEL` via env config, so a
  fallback isn't free but isn't a big lift either)?
- Whether to re-run the eval harness (the same one that measured 0%
  hallucination rate on local Ollama llama3.2:3b) against Groq's
  `llama-3.1-8b-instant` before committing — ticket 03 flagged hosted
  tool-calling reliability at 8B scale as genuinely unverified for this
  app's specific 3-tool pattern.
- Rate-limit math: does Groq's 30 RPM / 500K TPD comfortably cover expected
  recruiter-screening traffic (low volume, bursty) alongside the chat
  agent's ~1.3 average tool calls per turn, or does it need the same kind
  of per-IP cooldown the app already has for uploads?

## Resolution (2026-09-03)

Groq `llama-3.1-8b-instant`, no automatic fallback. Groq allows 14,400
req/day vs OpenRouter's 50 at a zero balance, and an untested fallback
path is worse than none. OpenRouter stays a documented two-env-var swap.
Note: Groq limits are per-organization, so a shared key shares the quota.
