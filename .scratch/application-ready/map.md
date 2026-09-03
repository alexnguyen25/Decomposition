# Application-ready

## Destination

Decomposition is ready to put in front of co-op recruiters: `improve-f1` is
committed with clean history, the BEATs classifier lives in `src/` (Alex's
own tutored implementation), a live demo (the evolved reference-app, chat
agent included) is deployed, the README sells the project in 90 seconds with
measured metrics, resume bullets are drafted and true, and three interview
stories are rehearsed. Target: **~September 1, 2026** (Northeastern co-op
recruiting for Spring 2027 opens early-to-mid September).

## Notes

- **Role boundary, two different modes on this one effort.** `src/` (the ML
  core: BEATs head, stem-domain fine-tune) is Alex's — tutored, Socratic,
  never written for him (see `claude-role-researcher-tutor-only` memory).
  The web app / deployment layer (reference-app going to production: Vercel,
  Modal, prod LLM wiring, frontend polish) is fair game for the agent to
  write — Alex drives the decisions and is deliberately using this as a
  chance to learn AI-assisted full-stack workflow.
- **Never commit or push without explicit permission.** This whole effort is
  planning until Alex says otherwise, even where a ticket's resolution
  produces something ready to commit.
- Domain: audio ML product — Demucs stem separation + BEATs-based instrument
  classifier + grounded LLM chat agent — built for Northeastern co-op
  applications (Spring 2027 cycle).
- Skills to consult: `grilling` + `domain-modeling` for the `src/` tutoring
  sessions; `prototype` for anything about how the frontend should look or
  behave; `research` for any remaining fact-finding (model/provider
  landscapes move fast — re-check rather than trust memory).
- **In-flight work** from the live `/goal` session (chat agent, grounding
  eval harness, chat UI, frontend art-direction prototypes, waveform-peak
  precompute, web-quality audit) is running in `reference-app/` outside this
  map's ticket numbering. It feeds Decisions-so-far as pieces land; anything
  it leaves undecided should still get a ticket here so the map stays a
  complete picture of what's left.
- Out-of-scope features already have feasibility research banked in
  `docs/research/2026-07-31-lyrics-chords-feasibility.md` — do not resurrect
  mid-effort; they start a fresh effort after applications are out.

## Decisions so far

- [Prod LLM provider research](issues/03-prod-llm-provider-research.md) —
  Cerebras/Together/Fireworks disqualified (all now require a card); Groq
  and OpenRouter free models are the live candidates. Provider choice
  itself deferred to [ticket 08](issues/08-choose-prod-llm-provider.md).

## Not yet specified

- Exact resume bullet phrasing and which 3 interview stories to rehearse —
  can't sharpen until the README ticket lands and final numbers (src/ F1,
  hallucination rate, live URL) are known.
- Whether the HF Space needs any changes once the full web app is live, or
  is fully superseded — depends on how ticket 04 resolves.
- Anything the frontend redesign (in-flight, not yet on this map) turns up
  that needs a real decision — folds in once that work reports back.

## Out of scope

- Lyrics transcription tool (Whisper on the Demucs vocals stem) — past the
  destination. Feasibility researched:
  `docs/research/2026-07-31-lyrics-chords-feasibility.md`. Revisit as a
  fresh effort after applications are out.
- Music theory tools (chord/structure detection, e.g. allin1/BTC) — same,
  past the destination, same research doc covers feasibility.
- Genre/origin classifier head + AcoustID lookup — same, past the
  destination.
- Audio-LLM leaderboard (GPT-4o-audio, Qwen-Audio comparison) and the
  Gemini prompt-technique ablation — previously deferred ideas, still past
  the destination.
