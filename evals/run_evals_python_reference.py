"""Eval harness for the chat agent: measured grounding, not vibes.

Runs every question in questions.jsonl against the real agent (real LLM,
real tools, real track data) and grades the FINAL user-visible replies:

- hallucination rate  — % of replies containing >=1 claim that fails
                        agent.check_grounding (instruments, BPM, key,
                        duration). This is the README headline number.
- trap resistance     — % of leading questions about absent instruments
                        ("that guitar solo!") that the agent does NOT play
                        along with.
- factual accuracy    — % of answerable questions whose reply contains the
                        expected fact.
- refusal correctness — % of out-of-scope questions (artist, lyrics, album)
                        answered with an explicit "the analysis doesn't
                        cover that" style response. Heuristic: refusal
                        language present. Documented limitation: a polite
                        hallucination with the word "not" in it would pass
                        this sub-metric — but it would still be caught by
                        the grounding check if it asserts checkable facts.

Model comes from the usual env config, so comparing models is:
  LLM_MODEL=llama3.2:3b python evals/run_evals.py
  LLM_MODEL=qwen3      python evals/run_evals.py

Writes evals/results_<model>.json next to this file.
"""

import contextlib
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import agent
import settings

EVALS_DIR = Path(__file__).parent

REFUSAL_MARKERS = ("not", "n't", "no ", "unable", "only", "beyond",
                   "unavailable", "unknown")


def grade(q: dict, reply: str, trace: list, grounded: bool) -> tuple[bool, str]:
    """Category-specific pass/fail for one reply (lowercased matching)."""
    text = reply.lower()
    cat = q["category"]
    if cat == "trap":
        # playing along = asserting the absent instrument; the grounding
        # checker already treats non-negated mentions as violations.
        return (grounded, "asserted the absent instrument" if not grounded
                else "ok")
    if cat == "out_of_scope":
        refused = any(m in text for m in REFUSAL_MARKERS)
        return ((grounded and refused),
                "ok" if refused else "answered instead of refusing")
    if cat == "temporal":
        used = any(t["tool"] == q["expect_tool"] for t in trace)
        return ((grounded and used),
                "ok" if used else f"never called {q['expect_tool']}")
    # factual
    hit = any(e in text for e in q["expect_any"])
    return ((grounded and hit),
            "ok" if hit else f"missing expected fact {q['expect_any']}")


def main():
    manifest = {ex["id"]: ex["result"] for ex in
                json.loads((settings.EXAMPLES_DIR / "manifest.json")
                           .read_text())}
    questions = [json.loads(line) for line in
                 (EVALS_DIR / "questions.jsonl").read_text().splitlines()
                 if line.strip()]

    rows, t_all = [], time.time()
    for q in questions:
        result = manifest[q["track"]]
        t0 = time.time()
        out = agent.chat(result, [{"role": "user", "content": q["q"]}])
        latency = time.time() - t0
        # re-check the final reply directly: hallucination rate must reflect
        # what the user actually reads, including the agent's refusal path.
        ok, violations = agent.check_grounding(result, out["reply"], [])
        # tool outputs aren't persisted here; allow anything the trace's
        # tools could have legitimately returned by re-running them.
        if not ok:
            replayed = []
            for t in out["trace"]:
                impl = agent.TOOL_IMPLS.get(t["tool"])
                if impl:
                    with contextlib.suppress(TypeError, ValueError):
                        replayed.append(impl(result, **t["args"]))
            ok, violations = agent.check_grounding(result, out["reply"],
                                                   replayed)
        passed, why = grade(q, out["reply"], out["trace"], ok)
        rows.append({**q, "reply": out["reply"], "latency_s": round(latency, 1),
                     "tools_used": [t["tool"] for t in out["trace"]],
                     "grounded": ok, "violations": violations,
                     "passed": passed, "why": why})
        flag = "✓" if passed else "✗"
        print(f"{flag} [{q['category']:12s}] {q['id']:22s} "
              f"{latency:5.1f}s  {why}")

    # ── summary ──────────────────────────────────────────────────────────
    n = len(rows)
    halluc = [r for r in rows if not r["grounded"]]
    by_cat = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r["passed"])
    summary = {
        "model": settings.LLM_MODEL,
        "n_questions": n,
        "hallucination_rate": round(len(halluc) / n, 3),
        "per_category_pass": {c: f"{sum(v)}/{len(v)}"
                              for c, v in sorted(by_cat.items())},
        "mean_latency_s": round(sum(r["latency_s"] for r in rows) / n, 1),
        "mean_tool_calls": round(sum(len(r["tools_used"]) for r in rows) / n, 1),
        "total_runtime_s": round(time.time() - t_all, 1),
    }
    print("\n" + json.dumps(summary, indent=2))
    if halluc:
        print("\nungrounded replies:")
        for r in halluc:
            print(f"  {r['id']}: {r['violations']} :: {r['reply'][:120]}")

    safe_model = re.sub(r"[^\w.-]", "_", settings.LLM_MODEL)
    out_path = EVALS_DIR / f"results_{safe_model}.json"
    out_path.write_text(json.dumps({"summary": summary, "rows": rows},
                                   indent=1))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
