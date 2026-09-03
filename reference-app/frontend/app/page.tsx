import UploadZone from "@/components/UploadZone";
import Examples from "@/components/Examples";

// Landing page: hero, upload slot, example shelf. Server component shell;
// interactivity lives in the client components below.
export default function Home() {
  return (
    <main className="mx-auto max-w-5xl px-6 pb-24">
      <header className="pt-16 pb-2 rise">
        <div className="flex items-center gap-2 text-xs tracking-[0.3em] uppercase text-[var(--muted)]">
          <span className="inline-block h-2 w-2 rounded-full bg-[var(--amber)] led" />
          audio intelligence console
        </div>
        <h1
          className="mt-4 text-6xl md:text-8xl font-semibold leading-[0.95] tracking-tight"
          style={{ fontFamily: "var(--font-display)" }}
        >
          decomposition<span className="text-[var(--amber)] blink">_</span>
        </h1>
        <p className="mt-5 max-w-xl text-sm leading-relaxed text-[var(--muted)]">
          Hear a song pulled apart: four isolated stems, the instruments
          inside it, BPM &amp; key, and a written breakdown you can question.
          Separation by Demucs; instrument detection by a BEATs-based
          classifier trained for this project (macro-F1&nbsp;0.80 on
          OpenMIC-2018).
        </p>
      </header>

      <section className="mt-12 rise" style={{ animationDelay: "0.12s" }}>
        <UploadZone />
      </section>

      <section className="mt-16 rise" style={{ animationDelay: "0.24s" }}>
        <div className="flex items-baseline justify-between border-b border-[var(--line)] pb-2">
          <h2 className="text-sm uppercase tracking-[0.25em] text-[var(--muted)]">
            or audition an example
          </h2>
          <span className="text-[10px] text-[var(--muted)]">
            precomputed · instant
          </span>
        </div>
        <Examples />
      </section>

      <footer className="mt-24 border-t border-[var(--line)] pt-6 text-[11px] leading-relaxed text-[var(--muted)]">
        Research project — classifier trained on OpenMIC-2018; the example
        tracks are CC-licensed Jamendo recordings, analysed ahead of time and
        served as static files. Nothing you do here uploads audio anywhere.
      </footer>
    </main>
  );
}
