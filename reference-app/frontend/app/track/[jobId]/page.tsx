"use client";

import { use, useEffect, useState } from "react";
import Link from "next/link";
import Console from "@/components/Console";
import type { Result } from "@/components/Console";

type Job = {
  status: "queued" | "running" | "done" | "error";
  progress: number;
  stage: string;
  result?: Result;
  error?: string;
};

const STAGES = [
  "Loading audio",
  "Separating stems (Demucs) — the slow part",
  "Encoding stems",
  "Detecting instruments",
  "Estimating BPM and key",
  "Writing description",
];

/** Job page. Two data paths:
 *  - real jobs: poll GET /api/jobs/{id} every 2 s until done/error
 *  - examples (id starts with "ex_"): read the precomputed result from a
 *    static JSON file — no backend involved, so this path always works
 *  Polling is the simplest robust pattern for long jobs (see NOTES.md). */
export default function TrackPage({
  params,
}: {
  params: Promise<{ jobId: string }>;
}) {
  const { jobId } = use(params);
  const isExample = jobId.startsWith("ex_");
  const [job, setJob] = useState<Job | null>(null);

  useEffect(() => {
    let stop = false;
    async function loadExample() {
      const res = await fetch("/examples/manifest.json");
      const all = await res.json();
      const ex = all.find((e: { id: string }) => e.id === jobId);
      setJob(
        ex
          ? { status: "done", progress: 1, stage: "Done", result: ex.result }
          : { status: "error", progress: 0, stage: "", error: "Unknown example" },
      );
    }
    async function poll() {
      while (!stop) {
        const res = await fetch(`/api/jobs/${jobId}`);
        if (res.status === 404) {
          setJob({ status: "error", progress: 0, stage: "",
                   error: "Job not found — results expire ~30 min after completion." });
          return;
        }
        const data: Job = await res.json();
        setJob(data);
        if (data.status === "done" || data.status === "error") return;
        await new Promise((r) => setTimeout(r, 2000));
      }
    }
    if (isExample) loadExample();
    else poll();
    return () => {
      stop = true;
    };
  }, [jobId, isExample]);

  return (
    <main className="mx-auto max-w-5xl px-6 pb-24">
      <nav className="pt-8 text-xs text-[var(--muted)]">
        <Link href="/" className="hover:text-[var(--amber)] transition-colors">
          ← decomposition_
        </Link>
      </nav>

      {!job && <p className="mt-16 text-sm text-[var(--muted)]">connecting…</p>}

      {job && (job.status === "queued" || job.status === "running") && (
        <div className="mt-16 max-w-md rise">
          <div className="flex items-end gap-1 h-10">
            {[0, 1, 2, 3, 4, 5, 6].map((i) => (
              <span
                key={i}
                className="eqbar inline-block w-2 bg-[var(--amber)]"
                style={{ height: "100%", animationDelay: `${i * 0.1}s` }}
              />
            ))}
          </div>
          <h2 className="mt-6 text-2xl" style={{ fontFamily: "var(--font-display)" }}>
            analyzing…
          </h2>
          <div className="mt-6 space-y-2">
            {STAGES.map((s) => {
              const active = job.stage === s;
              const done =
                STAGES.indexOf(s) < STAGES.findIndex((x) => x === job.stage);
              return (
                <div key={s} className="flex items-center gap-3 text-xs">
                  <span
                    className={`h-1.5 w-1.5 rounded-full ${
                      active
                        ? "bg-[var(--amber)] led"
                        : done
                          ? "bg-[var(--muted)]"
                          : "bg-[var(--line)]"
                    }`}
                  />
                  <span className={active ? "text-[var(--text)]" : "text-[var(--muted)]"}>
                    {s}
                  </span>
                </div>
              );
            })}
          </div>
          <div className="mt-6 h-1 w-full bg-[var(--line)]">
            <div
              className="h-1 bg-[var(--amber)] transition-all duration-700"
              style={{ width: `${Math.round(job.progress * 100)}%` }}
            />
          </div>
          <p className="mt-2 text-[10px] text-[var(--muted)]">
            {Math.round(job.progress * 100)}% — separation dominates the wait;
            everything after it is fast.
          </p>
        </div>
      )}

      {job?.status === "error" && (
        <div className="mt-16 max-w-md border border-[#e85d4a] bg-[var(--panel)] p-6 text-sm">
          <div className="text-[#e85d4a]">▲ {job.error}</div>
          <Link href="/" className="mt-4 inline-block text-xs text-[var(--muted)] hover:text-[var(--amber)]">
            ← try another song
          </Link>
        </div>
      )}

      {job?.status === "done" && job.result && (
        <Console result={job.result} jobId={jobId} />
      )}
    </main>
  );
}
