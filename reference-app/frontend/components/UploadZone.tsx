"use client";

import { useCallback, useRef, useState } from "react";
import { useRouter } from "next/navigation";

/** Drag-drop / click-to-browse upload. POSTs to /api/analyze and routes to
 *  the job page. All server-side limits (size, duration, rate) surface here
 *  as inline error text — the backend is the source of truth for rules.
 *
 *  Analysis needs Demucs and a 345 MB model: minutes of CPU and gigabytes of
 *  RAM, which no free serverless host will run. So the deployed site ships
 *  without it and says so plainly, rather than offering a dropzone that fails.
 *  Set NEXT_PUBLIC_UPLOAD_ENABLED=1 with the backend running to turn it on. */
const uploadEnabled = process.env.NEXT_PUBLIC_UPLOAD_ENABLED === "1";

export default function UploadZone() {
  if (!uploadEnabled) return <RunItLocally />;
  return <UploadDropzone />;
}

/** Honest stand-in for the dropzone when no analysis backend is reachable. */
function RunItLocally() {
  return (
    <div className="border border-[var(--line)] bg-[var(--panel)] px-8 py-10">
      <div className="text-lg" style={{ fontFamily: "var(--font-display)" }}>
        analyse your own track
      </div>
      <p className="mt-3 max-w-xl text-xs leading-relaxed text-[var(--muted)]">
        Separation runs Demucs plus a 345&nbsp;MB model — minutes of CPU and
        several GB of RAM per song, which no free serverless host provides. The
        examples below are precomputed and load instantly; to analyse your own
        audio, run the pipeline locally:
      </p>
      <pre className="mt-4 overflow-x-auto border border-[var(--line)] bg-[var(--panel-2)] p-4 text-[11px] leading-relaxed text-[var(--muted)]">
{`git clone https://github.com/alexnguyen25/Decomposition
cd Decomposition
pip install -r requirements.txt
python scripts/fetch_models.py
python -m src.main path/to/song.mp3`}
      </pre>
    </div>
  );
}

function UploadDropzone() {
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);

  const submit = useCallback(
    async (file: File) => {
      setBusy(true);
      setError(null);
      const form = new FormData();
      form.append("file", file);
      try {
        const res = await fetch("/api/analyze", { method: "POST", body: form });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail ?? "Upload failed");
        router.push(`/track/${data.job_id}`);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Upload failed");
        setBusy(false);
      }
    },
    [router],
  );

  return (
    <div>
      <div
        onClick={() => !busy && inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          const f = e.dataTransfer.files?.[0];
          if (f && !busy) submit(f);
        }}
        className={`group relative cursor-pointer border transition-all duration-300 ${
          dragging
            ? "border-[var(--amber)] bg-[var(--panel-2)]"
            : "border-[var(--line)] bg-[var(--panel)] hover:border-[var(--amber-dim)]"
        }`}
      >
        {/* corner ticks, like a tape-deck window */}
        {["top-0 left-0", "top-0 right-0", "bottom-0 left-0", "bottom-0 right-0"].map(
          (pos) => (
            <span
              key={pos}
              className={`absolute ${pos} h-3 w-3 border-[var(--amber)] ${
                pos.includes("top") ? "border-t" : "border-b"
              } ${pos.includes("left") ? "border-l" : "border-r"}`}
            />
          ),
        )}
        <div className="px-8 py-14 text-center">
          {busy ? (
            <div className="flex items-end justify-center gap-1 h-8">
              {[0, 1, 2, 3, 4].map((i) => (
                <span
                  key={i}
                  className="eqbar inline-block w-1.5 bg-[var(--amber)]"
                  style={{ height: "100%", animationDelay: `${i * 0.12}s` }}
                />
              ))}
            </div>
          ) : (
            <>
              <div className="text-lg" style={{ fontFamily: "var(--font-display)" }}>
                drop a song here
              </div>
              <div className="mt-2 text-xs text-[var(--muted)]">
                mp3 / wav / ogg / flac / m4a — max 15 MB, 10 s–6 min · CPU
                processing takes a minute or two
              </div>
            </>
          )}
        </div>
        <input
          ref={inputRef}
          type="file"
          accept="audio/*"
          className="hidden"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) submit(f);
          }}
        />
      </div>
      {error && (
        <p className="mt-3 text-xs text-[#e85d4a]" role="alert">
          ▲ {error}
        </p>
      )}
    </div>
  );
}
