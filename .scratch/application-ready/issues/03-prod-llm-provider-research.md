Type: research
Status: resolved

## Question

The reference-app's LLM layer (`reference-app/backend/llm.py`,
`agent.py`) is OpenAI-compatible and swappable: Ollama for local dev,
documented options for prod are Cerebras (1M free tokens/day, no card) and
Groq. That documentation is from the July 22 research pass — re-verify
current (as of August 2026) facts before Alex picks one for the deployed
chat agent + description feature:

- Current free-tier limits, rate limits, and model menu for Cerebras and
  Groq (the specific instruct models available, context length, tool-calling
  support — the chat agent needs function calling).
- Any other no-card, genuinely-free OpenAI-compatible host worth
  considering that's emerged since July (e.g. OpenRouter free models,
  Together, Fireworks free tier).
- Latency expectations for a public demo (the chat agent currently averages
  ~2.9s/turn on local Ollama llama3.2:3b — what's realistic on each hosted
  option, and does the model choice affect tool-calling reliability the way
  it did locally, per the eval harness's 0% hallucination rate on
  llama3.2:3b).
- Whether the description feature's grounding contract and the chat agent's
  post-hoc grounding check both need the response to reliably use
  structured JSON / tool-calling — confirm the candidate providers support
  this the same way Ollama does.

Deliverable: a short findings report (docs/research/), no provider chosen
yet — that choice is Alex's, informed by these facts, likely as a quick
follow-up decision once this lands.

## Research notes

- **Cerebras** — free-tier lineup has shrunk to 3 models (`gpt-oss-120b` prod;
  `gemma-4-31b`, `zai-glm-4.7` preview; the `llama-3.3-70b`/`qwen3` models many
  blogs still cite are gone). Rate limits per model: 5 RPM / 30K TPM / 1M
  TPD. Tool calling is OpenAI-compatible (strict mode, parallel calls) on
  `gpt-oss-120b`/`zai-glm-4.7`. **Important change since July: Cerebras's own
  docs now say a verified payment method is required at signup** ("Playground
  and API access remain inactive until you do") — this contradicts the July 22
  note and most third-party write-ups, and likely disqualifies Cerebras under
  the no-card requirement as of today.
- **Groq** — no card-requirement language found anywhere in Groq's own docs
  (free vs. paid tiers are just a limits toggle on the same page). Free-tier
  limits per model, e.g. `llama-3.1-8b-instant`: 30 RPM / 14.4K RPD / 6K TPM /
  500K TPD, 131K context. Tool calling: full support (parallel calls, MCP
  remote tools, JSON mode) on `llama-3.1-8b-instant`, `llama-3.3-70b-versatile`,
  `qwen/qwen3.6-27b`; GPT-OSS 20B/120B support tools but not parallel calls.
  Strongest candidate on paper.
- **OpenRouter free (`:free`) models** — genuinely no-card at $0 balance, but
  thin (20 RPM / 50 req/day before any spend, 1,000/day after $10+ purchased)
  and the free-model lineup churns significantly month to month (models get
  delisted with no notice).
- **Together AI** — disqualified: no-card signup credit was retired, now
  requires a minimum $5 credit purchase to use at all.
- **Fireworks AI** — disqualified as a sustained free tier: $1 one-time trial
  credit, capped at 10 RPM without a card (unusable at that rate for a public
  app), full throughput requires adding a card.
- Tool-calling reliability for small/8B-class models on Groq specifically
  (the closest proxy to the local baseline) has some GitHub-reported friction
  (litellm #5195, autogen #3217) but nothing recent or matching this app's
  exact 3-tool usage pattern — flagged as needing the same local eval harness
  used for the Ollama baseline before trusting either provider in production.

Full report: `docs/research/2026-08-01-prod-llm-provider-options.md`

## Answer

Facts gathered, no-card no-billable-surprise requirement is the hard filter:

- **Cerebras is disqualified** — its own docs (checked 2026-08-01) now
  require a verified payment method at signup, contradicting the July 22
  research this project was originally planned around. Its model lineup
  also shrank to 3 models since then.
- **Groq is the strongest surviving candidate** — no card-requirement
  language anywhere in its docs, full OpenAI-compatible tool calling
  (including parallel calls) on `llama-3.1-8b-instant` and larger models,
  30 RPM/500K TPD free tier.
- **OpenRouter free models** are a genuinely no-card fallback but thin (50
  req/day) and the free lineup churns monthly — viable as a backup, not a
  primary.
- **Together AI and Fireworks AI are disqualified** — both now require a
  card or minimum purchase to be usable.
- Tool-calling reliability at 8B scale on a *hosted* provider (vs. local
  Ollama, where the eval harness measured 0% hallucination rate) is
  genuinely unverified — nothing found matches this app's exact 3-tool
  usage pattern closely enough to trust without re-running the local eval
  harness against the hosted candidate once one is wired up.

This resolves the fact-finding; it does not choose a provider. See ticket
08 for that decision, now that Groq and OpenRouter are the two live
options.
