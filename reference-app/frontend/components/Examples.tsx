"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

type Example = {
  id: string;
  title: string;
  duration_s: number;
  instruments: string[];
  attribution: string;
};

/** Example shelf. The manifest and the stem audio are static assets in
 *  public/examples, so this works with no backend running at all — which is
 *  the whole point: the deployed demo must never show a visitor a spinner. */
export default function Examples() {
  const [examples, setExamples] = useState<Example[] | null>(null);

  useEffect(() => {
    fetch("/examples/manifest.json")
      .then((r) => (r.ok ? r.json() : []))
      .then(setExamples)
      .catch(() => setExamples([]));
  }, []);

  if (examples === null)
    return <p className="mt-6 text-xs text-[var(--muted)]">loading…</p>;
  if (examples.length === 0)
    return (
      <p className="mt-6 text-xs text-[var(--muted)]">
        no examples bundled — see README
      </p>
    );

  return (
    <div className="mt-6 grid gap-4 md:grid-cols-3">
      {examples.map((ex, i) => (
        <Link
          key={ex.id}
          href={`/track/${ex.id}`}
          className="group border border-[var(--line)] bg-[var(--panel)] p-5 transition-all duration-300 hover:border-[var(--amber-dim)] hover:-translate-y-0.5"
        >
          <div className="flex items-center justify-between text-[10px] text-[var(--muted)]">
            <span>TRK {String(i + 1).padStart(2, "0")}</span>
            <span>{Math.round(ex.duration_s)}s</span>
          </div>
          <div
            className="mt-3 text-base leading-snug group-hover:text-[var(--amber)] transition-colors"
            style={{ fontFamily: "var(--font-display)" }}
          >
            {ex.title}
          </div>
          <div className="mt-3 flex flex-wrap gap-1.5">
            {ex.instruments.slice(0, 4).map((name) => (
              <span
                key={name}
                className="border border-[var(--line)] px-1.5 py-0.5 text-[10px] text-[var(--muted)]"
              >
                {name.replace("_", " ")}
              </span>
            ))}
          </div>
        </Link>
      ))}
    </div>
  );
}
