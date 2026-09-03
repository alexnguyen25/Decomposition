"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import WaveSurfer from "wavesurfer.js";
import Chat from "@/components/Chat";

export type Result = {
  duration_s: number;
  bpm: number;
  key: string | null;
  presence: Record<string, boolean>;
  instruments: { name: string; confidence: number }[];
  stems: Record<string, string>;
  description: {
    blurb: string;
    genre?: string | null;
    moods?: string[];
    energy?: string | null;
    source?: string;
  };
};

const STEM_ORDER = ["vocals", "drums", "bass", "other"] as const;
const STEM_COLOR: Record<string, string> = {
  vocals: "var(--c-vocals)",
  drums: "var(--c-drums)",
  bass: "var(--c-bass)",
  other: "var(--c-other)",
};

/** The results console: four channel strips (one per stem) with waveforms,
 *  a single master transport, mute/solo, instrument meters and the writeup.
 *
 *  Sync strategy: four independent WaveSurfer instances driven by one
 *  controller — play/pause/seek fan out to all. For a 4-stem demo this stays
 *  in sync well; sample-locked playback would use the Web Audio API with one
 *  AudioContext clock (noted in NOTES.md as the production upgrade). */
export default function Console({
  result,
  jobId,
}: {
  result: Result;
  jobId: string;
}) {
  const containers = useRef<Record<string, HTMLDivElement | null>>({});
  const surfers = useRef<Record<string, WaveSurfer>>({});
  const [ready, setReady] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [muted, setMuted] = useState<Set<string>>(new Set());
  const [solo, setSolo] = useState<string | null>(null);
  const [time, setTime] = useState(0);
  const [showJson, setShowJson] = useState(false);

  // create one WaveSurfer per stem, destroy on unmount
  useEffect(() => {
    STEM_ORDER.forEach((stem) => {
      const el = containers.current[stem];
      if (!el || surfers.current[stem]) return;
      const ws = WaveSurfer.create({
        container: el,
        url: result.stems[stem],
        height: 56,
        waveColor: "#3a3a44",
        progressColor: STEM_COLOR[stem].startsWith("var")
          ? getComputedStyle(document.documentElement).getPropertyValue(
              STEM_COLOR[stem].slice(4, -1),
            )
          : STEM_COLOR[stem],
        cursorColor: "#e8e4da",
        barWidth: 2,
        barGap: 1,
        interact: true,
      });
      ws.on("ready", () => setReady((n) => n + 1));
      // seeking any waveform seeks all of them
      ws.on("interaction", (newTime: number) => {
        Object.values(surfers.current).forEach((other) => {
          if (other !== ws) other.setTime(newTime);
        });
      });
      surfers.current[stem] = ws;
    });
    const first = () => surfers.current["vocals"];
    const tick = setInterval(() => {
      const ws = first();
      if (ws) setTime(ws.getCurrentTime());
    }, 250);
    return () => {
      clearInterval(tick);
      // Destroy AND clear the ref: React StrictMode runs mount→cleanup→mount
      // in dev, and stale destroyed instances in the ref would block
      // re-creation on the second mount (bug found via headless probe).
      Object.values(surfers.current).forEach((ws) => ws.destroy());
      surfers.current = {};
      setReady(0);
      setPlaying(false);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // mute/solo -> volumes (solo wins over mute)
  useEffect(() => {
    STEM_ORDER.forEach((stem) => {
      const ws = surfers.current[stem];
      if (!ws) return;
      const audible = solo ? stem === solo : !muted.has(stem);
      ws.setVolume(audible ? 1 : 0);
    });
  }, [muted, solo, ready]);

  const toggle = useCallback(() => {
    const all = Object.values(surfers.current);
    if (playing) all.forEach((ws) => ws.pause());
    else all.forEach((ws) => ws.play());
    setPlaying(!playing);
  }, [playing]);

  const fmt = (s: number) =>
    `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

  const loading = ready < STEM_ORDER.length;

  return (
    <div className="mt-10">
      {/* ── header: verdict chips ─────────────────────────────────────── */}
      <div className="rise flex flex-wrap items-center gap-3">
        <span className="border border-[var(--amber)] px-3 py-1.5 text-sm text-[var(--amber)]">
          {result.bpm} BPM
        </span>
        {result.key && (
          <span className="border border-[var(--line)] px-3 py-1.5 text-sm">
            {result.key}
          </span>
        )}
        <span className="border border-[var(--line)] px-3 py-1.5 text-sm text-[var(--muted)]">
          {fmt(result.duration_s)}
        </span>
        {result.description?.genre && (
          <span className="border border-[var(--line)] px-3 py-1.5 text-sm text-[var(--muted)]">
            {result.description.genre}
          </span>
        )}
      </div>

      {/* ── the writeup ───────────────────────────────────────────────── */}
      <blockquote
        className="rise mt-8 max-w-2xl border-l-2 border-[var(--amber)] pl-5 text-lg leading-relaxed"
        style={{ fontFamily: "var(--font-display)", animationDelay: "0.1s" }}
      >
        {result.description?.blurb}
        {result.description?.source && (
          <footer className="mt-2 text-[10px] tracking-wide text-[var(--muted)]" style={{ fontFamily: "var(--font-mono)" }}>
            — written by {result.description.source}, grounded in the analysis
          </footer>
        )}
      </blockquote>

      {/* ── transport + channel strips ────────────────────────────────── */}
      <div
        className="rise mt-10 border border-[var(--line)] bg-[var(--panel)]"
        style={{ animationDelay: "0.2s" }}
      >
        <div className="flex items-center gap-4 border-b border-[var(--line)] px-4 py-3">
          <button
            onClick={toggle}
            disabled={loading}
            className="grid h-10 w-10 shrink-0 place-items-center border border-[var(--amber)] text-[var(--amber)] transition-all hover:bg-[var(--amber)] hover:text-black disabled:opacity-40"
            aria-label={playing ? "Pause" : "Play"}
          >
            {playing ? "❚❚" : "▶"}
          </button>
          <span className="text-xs text-[var(--muted)]">
            {loading
              ? `decoding waveforms… ${ready}/4`
              : `${fmt(time)} / ${fmt(result.duration_s)}`}
          </span>
          <span className="ml-auto text-[10px] uppercase tracking-[0.25em] text-[var(--muted)]">
            master transport · click a waveform to seek
          </span>
        </div>

        {STEM_ORDER.map((stem) => {
          const audible = solo ? stem === solo : !muted.has(stem);
          return (
            <div
              key={stem}
              className="flex items-center gap-3 border-b border-[var(--line)] px-4 py-3 last:border-b-0"
            >
              <div className="w-20 shrink-0">
                <div
                  className="text-xs uppercase tracking-wider"
                  style={{ color: STEM_COLOR[stem] }}
                >
                  {stem}
                </div>
                {"presence" in result &&
                  stem !== "other" &&
                  result.presence[stem] === false && (
                    <div className="text-[9px] text-[var(--muted)]">absent</div>
                  )}
              </div>
              <div className="flex shrink-0 gap-1">
                <button
                  onClick={() =>
                    setMuted((m) => {
                      const next = new Set(m);
                      if (next.has(stem)) next.delete(stem);
                      else next.add(stem);
                      return next;
                    })
                  }
                  className={`h-7 w-7 shrink-0 border text-[10px] transition-colors ${
                    muted.has(stem) && !solo
                      ? "border-[#e85d4a] text-[#e85d4a]"
                      : "border-[var(--line)] text-[var(--muted)] hover:text-[var(--text)]"
                  }`}
                  title="mute"
                  aria-label={`${muted.has(stem) && !solo ? "unmute" : "mute"} ${stem}`}
                  aria-pressed={muted.has(stem) && !solo}
                >
                  M
                </button>
                <button
                  onClick={() => setSolo(solo === stem ? null : stem)}
                  className={`h-7 w-7 shrink-0 border text-[10px] transition-colors ${
                    solo === stem
                      ? "border-[var(--amber)] text-[var(--amber)]"
                      : "border-[var(--line)] text-[var(--muted)] hover:text-[var(--text)]"
                  }`}
                  title="solo"
                  aria-label={`${solo === stem ? "unsolo" : "solo"} ${stem}`}
                  aria-pressed={solo === stem}
                >
                  S
                </button>
              </div>
              <div
                className={`min-w-0 flex-1 transition-opacity duration-300 ${
                  audible ? "opacity-100" : "opacity-30"
                }`}
                ref={(el) => {
                  containers.current[stem] = el;
                }}
              />
              <a
                href={result.stems[stem]}
                download={`${stem}.mp3`}
                aria-label={`download ${stem} stem as mp3`}
                className="grid h-6 w-6 shrink-0 place-items-center text-[10px] text-[var(--muted)] hover:text-[var(--amber)] transition-colors"
              >
                ↓
              </a>
            </div>
          );
        })}
      </div>

      {/* ── instrument meters ─────────────────────────────────────────── */}
      <div className="rise mt-10" style={{ animationDelay: "0.3s" }}>
        <h2 className="border-b border-[var(--line)] pb-2 text-sm uppercase tracking-[0.25em] text-[var(--muted)]">
          instruments detected in the &ldquo;other&rdquo; stem
        </h2>
        {result.instruments.length === 0 ? (
          <p className="mt-4 text-xs text-[var(--muted)]">
            nothing above the confidence threshold — vocals/drums/bass are
            reported as stems above.
          </p>
        ) : (
          <div className="mt-5 space-y-3">
            {result.instruments.map((inst) => (
              <div key={inst.name} className="flex items-center gap-4">
                <span className="w-40 shrink-0 text-sm">
                  {inst.name.replace("_", " ")}
                </span>
                <div className="h-3 flex-1 bg-[var(--panel-2)]">
                  <div
                    className="h-3 transition-all duration-1000"
                    style={{
                      width: `${inst.confidence * 100}%`,
                      background:
                        "linear-gradient(90deg, var(--amber-dim), var(--amber))",
                    }}
                  />
                </div>
                <span className="w-12 shrink-0 text-right text-xs text-[var(--muted)]">
                  {(inst.confidence * 100).toFixed(0)}%
                </span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* ── grounded chat ─────────────────────────────────────────────── */}
      <Chat jobId={jobId} />

      {/* ── raw JSON ──────────────────────────────────────────────────── */}
      <div className="rise mt-10" style={{ animationDelay: "0.4s" }}>
        <button
          onClick={() => setShowJson(!showJson)}
          className="text-xs text-[var(--muted)] transition-colors hover:text-[var(--amber)]"
        >
          {showJson ? "▾ hide" : "▸ show"} raw analysis JSON
        </button>
        {showJson && (
          <pre className="mt-3 overflow-x-auto border border-[var(--line)] bg-[var(--panel)] p-4 text-[11px] leading-relaxed text-[var(--muted)]">
            {JSON.stringify(result, null, 2)}
          </pre>
        )}
      </div>
    </div>
  );
}
