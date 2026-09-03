"""Eval harness for the chat agent as deployed: measured grounding, not vibes.

Every question in questions.jsonl is sent to a running site's POST
/api/chat/{track}, and the FINAL user-visible reply is graded. That matters:
it measures the TypeScript agent that visitors actually talk to, over the
provider actually configured, rather than a Python function that no longer
ships.

Metrics
    hallucination rate   % of replies containing >=1 claim that fails the
                         grounding contract. The README headline number.
    trap resistance      % of leading questions about absent instruments
                         ("that guitar solo!") the agent does NOT play along with.
    factual accuracy     % of answerable questions whose reply has the expected fact.
    refusal correctness  % of out-of-scope questions (artist, lyrics, album)
                         answered with an explicit "not covered" response.
    validator agreement  how often the shipped TS validator and the independent
                         Python one reach the same verdict. Disagreement means
                         the port has drifted and one of them is wrong.

Usage
    python evals/run_evals_http.py --base-url http://localhost:3002
"""

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from grounding_reference import check_grounding

EVALS_DIR = Path(__file__).parent
MANIFEST = (Path(__file__).resolve().parents[1] / "reference-app" / "frontend"
            / "public" / "examples" / "manifest.json")

REFUSAL_MARKERS = ("not", "n't", "no ", "unable", "only", "beyond",
                   "unavailable", "unknown")


def ask(base_url: str, track: str, question: str, timeout: int) -> dict:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/chat/{track}",
        data=json.dumps({"messages": [{"role": "user", "content": question}]}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def grade(question: dict, reply: str, trace: list, grounded: bool) -> tuple[bool, str]:
    """Category-specific pass/fail for one reply (lowercased matching)."""
    text = reply.lower()
    category = question["category"]
    if category == "trap":
        return (grounded,
                "ok" if grounded else "asserted the absent instrument")
    if category == "out_of_scope":
        refused = any(marker in text for marker in REFUSAL_MARKERS)
        return ((grounded and refused),
                "ok" if refused else "answered instead of refusing")
    if category == "temporal":
        used = any(t["tool"] == question["expect_tool"] for t in trace)
        return ((grounded and used),
                "ok" if used else f"never called {question['expect_tool']}")
    hit = any(fact in text for fact in question["expect_any"])
    return ((grounded and hit),
            "ok" if hit else f"missing expected fact {question['expect_any']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:3002")
    parser.add_argument("--label", default="local",
                        help="tag for the output file, e.g. the model name")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    manifest = {ex["id"]: ex["result"] for ex in json.loads(MANIFEST.read_text())}
    questions = [json.loads(line) for line
                 in (EVALS_DIR / "questions.jsonl").read_text().splitlines()
                 if line.strip()]

    rows = []
    disagreements = 0
    started = time.time()

    for question in questions:
        result = manifest[question["track"]]
        t0 = time.time()
        try:
            response = ask(args.base_url, question["track"], question["q"], args.timeout)
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            print(f"✗ [{question['category']:12s}] {question['id']:22s} "
                  f"TRANSPORT ERROR {error}")
            rows.append({**question, "error": str(error), "passed": False})
            continue
        latency = time.time() - t0

        reply = response.get("reply", "")
        trace = response.get("trace", [])
        shipped_verdict = bool(response.get("grounded"))

        # Independent re-check. Tool outputs are not returned by the API, so
        # replay is impossible here — instead allow anything the analysis or a
        # timeline entry legitimately contains, which is what the tools expose.
        allowed_from_timeline = [{
            "instruments": [
                {"name": name, "confidence": probability}
                for entry in (result.get("timeline") or {}).get("instruments", [])
                for name, probability in entry["top"].items()
            ]
        }]
        independent_ok, violations = check_grounding(result, reply, allowed_from_timeline)
        # A refusal legitimately reads as grounded=False from the agent (the
        # model failed the contract) but clean to this checker (the visible text
        # asserts nothing). That is agreement, not drift, so skip refusals.
        is_refusal = reply.startswith("I couldn't give a reliable answer")
        if not is_refusal and independent_ok != shipped_verdict:
            disagreements += 1
            print(f"    ! validator drift on {question['id']}: "
                  f"shipped={shipped_verdict} independent={independent_ok}")

        passed, why = grade(question, reply, trace, independent_ok)
        rows.append({**question, "reply": reply, "latency_s": round(latency, 1),
                     "tools_used": [t["tool"] for t in trace],
                     "grounded_shipped": shipped_verdict,
                     "grounded_independent": independent_ok,
                     "violations": violations, "passed": passed, "why": why})
        print(f"{'✓' if passed else '✗'} [{question['category']:12s}] "
              f"{question['id']:22s} {latency:5.1f}s  {why}")

    graded = [r for r in rows if "reply" in r]
    if not graded:
        sys.exit("no questions completed — is the site running?")

    ungrounded = [r for r in graded if not r["grounded_independent"]]
    by_category: dict[str, list[bool]] = {}
    for row in graded:
        by_category.setdefault(row["category"], []).append(row["passed"])

    summary = {
        "label": args.label,
        "base_url": args.base_url,
        "n_questions": len(graded),
        "hallucination_rate": round(len(ungrounded) / len(graded), 3),
        "per_category_pass": {c: f"{sum(v)}/{len(v)}"
                              for c, v in sorted(by_category.items())},
        "validator_disagreements": disagreements,
        "mean_latency_s": round(sum(r["latency_s"] for r in graded) / len(graded), 1),
        "mean_tool_calls": round(sum(len(r["tools_used"]) for r in graded) / len(graded), 1),
        "total_runtime_s": round(time.time() - started, 1),
    }
    print("\n" + json.dumps(summary, indent=2))
    if ungrounded:
        print("\nungrounded replies:")
        for row in ungrounded:
            print(f"  {row['id']}: {row['violations']} :: {row['reply'][:120]}")

    safe_label = re.sub(r"[^\w.-]", "_", args.label)
    out_path = EVALS_DIR / f"results_{safe_label}.json"
    out_path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
