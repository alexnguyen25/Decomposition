"use client";

// PROTOTYPE — throwaway. Variant B: "SPECTRUM"
// Dark cinematic visualizer. The waveform IS the interface — mirrored,
// glowing, spectral. Giant condensed display type over full-bleed darkness,
// radial dials instead of bars. Motion: slow, smooth, luminous (no bounce).

import { useState } from "react";
import { CHAT, peaks, STEMS, TRACK, type Stem } from "./data";

const HUE: Record<Stem, string> = {
  vocals: "#FF4D6D",
  drums: "#FFB020",
  bass: "#4D7CFF",
  other: "#2CE8A0",
};

export const name = "Spectrum";

function Dial({ v, c }: { v: number; c: string }) {
  const R = 15;
  const C = 2 * Math.PI * R;
  return (
    <svg viewBox="0 0 36 36" className="sp-dial" aria-hidden>
      <circle cx="18" cy="18" r={R} className="sp-dial-bg" />
      <circle
        cx="18"
        cy="18"
        r={R}
        stroke={c}
        strokeDasharray={`${v * C} ${C}`}
        className="sp-dial-fg"
      />
    </svg>
  );
}

export default function VariantB() {
  const [playing, setPlaying] = useState(true);
  const [muted, setMuted] = useState<Set<Stem>>(new Set());

  return (
    <div className="v-spec">
      <style>{specCSS}</style>

      {/* ── full-bleed hero ──────────────────────────────────── */}
      <header className="sp-hero">
        <div className="sp-aurora" />
        <div className="sp-hero-wave" aria-hidden>
          {STEMS.map((s) => (
            <div key={s} className="sp-hw-layer" style={{ ["--h" as string]: HUE[s] }}>
              {peaks(s, 120).map((p, i) => (
                <i key={i} style={{ height: `${p * 100}%` }} />
              ))}
            </div>
          ))}
        </div>
        <div className="sp-scrim" aria-hidden />

        <div className="sp-hero-txt">
          <div className="sp-kicker">
            <span className="sp-live" /> decomposition engine
          </div>
          <h1 className="sp-title">
            EVERY LAYER<span>.</span>
            <br />
            LAID BARE<span>.</span>
          </h1>
          <p className="sp-sub">
            Demucs pulls the song into four stems. A BEATs-based classifier
            names what&rsquo;s playing inside. Then you can just ask it things.
          </p>
          <div className="sp-cta">
            <button className="sp-btn">Drop a track</button>
            <span className="sp-cta-s">or hear a demo — instant, precomputed</span>
          </div>
        </div>

        <div className="sp-metrics">
          {[
            ["0.80", "MACRO-F1"],
            ["4", "STEMS"],
            ["20", "CLASSES"],
            ["0%", "HALLUCINATION"],
          ].map(([n, l], i) => (
            <div key={l} style={{ animationDelay: `${400 + i * 70}ms` }}>
              <b>{n}</b>
              <span>{l}</span>
            </div>
          ))}
        </div>
      </header>

      {/* ── console ──────────────────────────────────────────── */}
      <section className="sp-console">
        <div className="sp-bar">
          <button
            className="sp-play"
            onClick={() => setPlaying((p) => !p)}
            aria-label={playing ? "Pause" : "Play"}
          >
            {playing ? "❚❚" : "▶"}
          </button>
          <div className="sp-time">1:24 / 4:17</div>
          <div className="sp-scrub">
            <div className="sp-scrub-f" />
          </div>
          <div className="sp-chips">
            <span>{TRACK.bpm} BPM</span>
            <span>{TRACK.key}</span>
          </div>
        </div>

        {STEMS.map((s, i) => {
          const off = muted.has(s);
          return (
            <button
              key={s}
              className={`sp-track${off ? " sp-off" : ""}`}
              style={{ ["--h" as string]: HUE[s], animationDelay: `${i * 70}ms` }}
              onClick={() =>
                setMuted((m) => {
                  const n = new Set(m);
                  n.has(s) ? n.delete(s) : n.add(s);
                  return n;
                })
              }
              aria-pressed={off}
            >
              <span className="sp-track-n">{s}</span>
              <span className="sp-track-w">
                {peaks(s, 100).map((p, j) => (
                  <i key={j} style={{ height: `${p * 100}%` }} />
                ))}
              </span>
              <span className="sp-track-s">{off ? "muted" : "live"}</span>
            </button>
          );
        })}
      </section>

      {/* ── readout ──────────────────────────────────────────── */}
      <section className="sp-readout">
        <div>
          <h2 className="sp-h2">Detected</h2>
          <div className="sp-inst">
            {TRACK.instruments.map((it, i) => (
              <div key={it.name} style={{ animationDelay: `${i * 55}ms` }}>
                <Dial v={it.confidence} c={HUE[STEMS[i % 4]]} />
                <span className="sp-inst-n">{it.name}</span>
                <span className="sp-inst-v">
                  {(it.confidence * 100).toFixed(0)}
                  <em>%</em>
                </span>
              </div>
            ))}
          </div>
        </div>

        <div>
          <h2 className="sp-h2">Readout</h2>
          <p className="sp-blurb">{TRACK.blurb}</p>

          <div className="sp-chat">
            {CHAT.map((m, i) =>
              m.role === "user" ? (
                <div key={i} className="sp-q">
                  {m.text}
                </div>
              ) : (
                <div key={i} className="sp-a">
                  {m.text}
                  <span className="sp-cite">{m.tools?.join(" · ")}</span>
                </div>
              ),
            )}
            <div className="sp-ask">
              <input placeholder="ask the engine…" readOnly />
              <kbd>↵</kbd>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}

const specCSS = `
.v-spec{
  --bg:#06060A; --fg:#EDEDF2; --dim:#7B7B8C; --edge:#1A1A24;
  background:var(--bg); color:var(--fg); min-height:100vh;
  font-family:var(--f-spec-mono),ui-monospace,monospace; overflow-x:hidden;
}
.v-spec *{box-sizing:border-box}
.v-spec ::selection{background:#2CE8A0;color:#06060A}

/* hero */
/* cap the hero so a very tall window doesn't leave a dead field of nothing */
.sp-hero{position:relative;min-height:min(88vh,880px);display:flex;flex-direction:column;
  justify-content:center;padding:clamp(24px,5vw,80px);overflow:hidden}
/* No filter:blur() here — radial gradients are already soft, and a 28px blur
   on a ~700px animated element promotes a huge composited layer that fails
   to repaint on scroll (caught in the browser preview: hero went white). */
.sp-aurora{position:absolute;inset:-30% -10% auto;height:80%;pointer-events:none;
  background:
    radial-gradient(50% 60% at 20% 40%,rgba(255,77,109,.22),transparent 70%),
    radial-gradient(45% 55% at 62% 30%,rgba(77,124,255,.22),transparent 70%),
    radial-gradient(40% 50% at 85% 55%,rgba(44,232,160,.18),transparent 70%);
  animation:sp-drift 22s ease-in-out infinite alternate;will-change:transform}
@keyframes sp-drift{to{transform:translate3d(-3%,2%,0) scale(1.08)}}

/* Four layers as a RIDGE, not a pile: each stem gets its own band height so
   all four colours stay legible instead of compositing into brown soup.
   Back-to-front = other (tallest, dimmest) → vocals (shortest, brightest). */
.sp-hero-wave{position:absolute;inset:auto 0 0;height:38%;display:grid;pointer-events:none;
  mask-image:linear-gradient(to top,#000 40%,transparent 100%)}
.sp-hw-layer{grid-area:1/1;display:flex;align-items:flex-end;gap:2px;
  padding:0 clamp(24px,5vw,80px);align-self:end}
/* glow lives on the LAYER, not on all 120 bars — one filter instead of 120 */
.sp-hw-layer i{flex:1;background:var(--h);border-radius:2px 2px 0 0;
  animation:sp-rise 900ms cubic-bezier(.23,1,.32,1) backwards}
.sp-hw-layer:nth-child(1){filter:drop-shadow(0 0 9px var(--h))}
.sp-hw-layer:nth-child(1){height:34%}   /* vocals  */
.sp-hw-layer:nth-child(2){height:56%}   /* drums   */
.sp-hw-layer:nth-child(3){height:78%}   /* bass    */
.sp-hw-layer:nth-child(4){height:100%}  /* other   */
.sp-hw-layer:nth-child(1) i{opacity:.85}
.sp-hw-layer:nth-child(2) i{opacity:.5}
.sp-hw-layer:nth-child(3) i{opacity:.3}
.sp-hw-layer:nth-child(4) i{opacity:.18}
@keyframes sp-rise{from{transform:scaleY(.05);opacity:0}}
/* scrim so the display type never fights the ridge behind it */
.sp-scrim{position:absolute;inset:0;pointer-events:none;
  background:linear-gradient(100deg,rgba(6,6,10,.94) 0%,rgba(6,6,10,.82) 42%,rgba(6,6,10,.15) 78%,transparent 100%)}

.sp-hero-txt{position:relative;max-width:1200px;margin:0 auto;width:100%}
.sp-kicker{display:inline-flex;align-items:center;gap:9px;font-size:10.5px;letter-spacing:.3em;
  text-transform:uppercase;color:var(--dim)}
.sp-live{width:6px;height:6px;border-radius:50%;background:#2CE8A0;
  box-shadow:0 0 10px #2CE8A0;animation:sp-blink 1.8s ease-in-out infinite}
@keyframes sp-blink{50%{opacity:.25}}
.sp-title{font-family:var(--f-spec-display),Impact,sans-serif;font-weight:400;
  font-size:clamp(52px,11.5vw,168px);line-height:.84;letter-spacing:-.015em;margin:18px 0 0;
  text-transform:uppercase;
  background:linear-gradient(96deg,#fff 18%,#B9B9CC 52%,#6E6E85 100%);
  -webkit-background-clip:text;background-clip:text;color:transparent}
.sp-title span{color:#2CE8A0;-webkit-text-fill-color:#2CE8A0}
.sp-sub{max-width:48ch;margin-top:22px;font-size:13.5px;line-height:1.75;color:var(--dim)}
.sp-cta{display:flex;align-items:center;gap:18px;margin-top:34px;flex-wrap:wrap}
.sp-btn{font-family:inherit;font-size:12.5px;letter-spacing:.14em;text-transform:uppercase;
  padding:14px 30px;border-radius:2px;border:1px solid rgba(255,255,255,.22);cursor:pointer;
  color:#06060A;background:#EDEDF2;
  transition:transform 150ms cubic-bezier(.23,1,.32,1),box-shadow 220ms ease}
.sp-btn:hover{box-shadow:0 0 34px rgba(237,237,242,.34)}
.sp-btn:active{transform:scale(.97)}
.sp-cta-s{font-size:11px;color:var(--dim)}
.sp-metrics{position:relative;display:flex;gap:clamp(20px,5vw,66px);margin-top:clamp(34px,6vh,64px);
  max-width:1200px;margin-inline:auto;width:100%;flex-wrap:wrap}
.sp-metrics div{opacity:0;animation:sp-fade 600ms cubic-bezier(.23,1,.32,1) forwards}
@keyframes sp-fade{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}
.sp-metrics b{display:block;font-family:var(--f-spec-display),sans-serif;font-size:34px;
  letter-spacing:.01em;line-height:1}
.sp-metrics span{font-size:9.5px;letter-spacing:.22em;color:var(--dim)}

/* console */
.sp-console{max-width:1200px;margin:clamp(40px,8vh,96px) auto 0;padding:0 clamp(24px,5vw,80px)}
.sp-bar{display:flex;align-items:center;gap:16px;padding-bottom:20px;border-bottom:1px solid var(--edge)}
.sp-play{width:44px;height:44px;flex:none;border-radius:50%;border:1px solid rgba(255,255,255,.2);
  background:rgba(255,255,255,.05);color:var(--fg);font-size:13px;cursor:pointer;
  transition:transform 150ms cubic-bezier(.23,1,.32,1),background 200ms ease}
.sp-play:hover{background:rgba(255,255,255,.12)}
.sp-play:active{transform:scale(.95)}
.sp-time{font-size:12px;color:var(--dim);font-variant-numeric:tabular-nums}
.sp-scrub{flex:1;height:2px;background:var(--edge);position:relative;min-width:60px}
.sp-scrub-f{position:absolute;inset:0 68% 0 0;background:linear-gradient(90deg,#4D7CFF,#2CE8A0)}
.sp-scrub-f::after{content:'';position:absolute;right:-4px;top:-3px;width:8px;height:8px;
  border-radius:50%;background:#2CE8A0;box-shadow:0 0 12px #2CE8A0}
.sp-chips{display:flex;gap:8px}
.sp-chips span{font-size:10.5px;letter-spacing:.1em;padding:5px 11px;border:1px solid var(--edge);
  color:var(--dim)}

.sp-track{display:flex;align-items:center;gap:18px;width:100%;padding:16px 0;background:none;
  border:0;border-bottom:1px solid var(--edge);cursor:pointer;text-align:left;color:inherit;
  font:inherit;opacity:0;animation:sp-fade 520ms cubic-bezier(.23,1,.32,1) forwards;
  transition:opacity 220ms ease}
.sp-track:hover .sp-track-w i{opacity:.95}
.sp-off{opacity:.28}
.sp-track-n{width:78px;flex:none;font-size:11px;letter-spacing:.18em;text-transform:uppercase;
  color:var(--h)}
.sp-track-w{flex:1;display:flex;align-items:center;gap:2px;height:46px;min-width:0}
.sp-track-w i{flex:1;background:var(--h);border-radius:1px;opacity:.68;
  filter:drop-shadow(0 0 5px var(--h));transition:opacity 200ms ease}
.sp-track-s{width:52px;text-align:right;font-size:9.5px;letter-spacing:.16em;
  text-transform:uppercase;color:var(--dim)}

/* readout */
.sp-readout{max-width:1200px;margin:clamp(40px,7vh,84px) auto 130px;
  padding:0 clamp(24px,5vw,80px);display:grid;grid-template-columns:1fr 1fr;gap:clamp(28px,5vw,72px)}
@media(max-width:880px){.sp-readout{grid-template-columns:1fr}}
.sp-h2{font-family:var(--f-spec-display),sans-serif;font-size:13px;letter-spacing:.3em;
  text-transform:uppercase;color:var(--dim);margin:0 0 22px;font-weight:400}
.sp-inst{display:grid;gap:3px}
.sp-inst>div{display:flex;align-items:center;gap:14px;padding:9px 0;opacity:0;
  animation:sp-fade 480ms cubic-bezier(.23,1,.32,1) forwards}
.sp-dial{width:34px;height:34px;flex:none;transform:rotate(-90deg)}
.sp-dial-bg{fill:none;stroke:var(--edge);stroke-width:2.5}
.sp-dial-fg{fill:none;stroke-width:2.5;stroke-linecap:round;
  transition:stroke-dasharray 700ms cubic-bezier(.23,1,.32,1)}
.sp-inst-n{flex:1;font-size:13.5px}
.sp-inst-v{font-size:15px;font-variant-numeric:tabular-nums}
.sp-inst-v em{font-size:9.5px;color:var(--dim);font-style:normal;margin-left:2px}
.sp-blurb{font-size:14.5px;line-height:1.8;color:#C9C9D6;margin:0 0 30px;max-width:52ch}
.sp-chat{border-left:1px solid var(--edge);padding-left:20px;display:grid;gap:16px}
.sp-q{font-size:13px;color:var(--fg)}
.sp-q::before{content:'? ';color:#2CE8A0}
.sp-a{font-size:13px;line-height:1.7;color:#C9C9D6}
.sp-cite{display:block;font-size:9.5px;letter-spacing:.14em;text-transform:uppercase;
  color:var(--dim);margin-top:9px}
.sp-ask{display:flex;align-items:center;gap:10px;border:1px solid var(--edge);padding:11px 14px;
  margin-top:6px}
.sp-ask input{flex:1;background:none;border:0;color:var(--fg);font:inherit;font-size:12.5px;outline:none}
.sp-ask kbd{font-size:10px;color:var(--dim);border:1px solid var(--edge);padding:2px 6px;border-radius:3px}
@media(prefers-reduced-motion:reduce){
  .v-spec *{animation-duration:.01ms!important;animation-iteration-count:1!important;transition-duration:.01ms!important}
}
`;
