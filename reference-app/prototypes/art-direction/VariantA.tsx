"use client";

// PROTOTYPE — throwaway. Variant A: "TAPE LAB"
// Warm analog maximalism. Light cream paper, chunky physical controls, hard
// offset shadows, tape-label stem colors. The delight is TACTILITY — things
// look pressable, tilt, and thunk. Motion: springy, slightly bouncy.

import { useState } from "react";
import { CHAT, peaks, STEMS, TRACK, type Stem } from "./data";

const LABEL: Record<Stem, { c: string; n: string }> = {
  vocals: { c: "#E9A23B", n: "A" },
  drums: { c: "#D64545", n: "B" },
  bass: { c: "#5B4FCF", n: "C" },
  other: { c: "#2E8B7A", n: "D" },
};

export const name = "Tape Lab";

export default function VariantA() {
  const [playing, setPlaying] = useState(false);
  const [solo, setSolo] = useState<Stem | null>(null);

  return (
    <div className="v-tape">
      <style>{tapeCSS}</style>

      {/* ── hero ─────────────────────────────────────────────── */}
      <header className="tp-hero">
        <div className="tp-hero-l">
          <div className="tp-badge">
            <span className="tp-dot" /> reel-to-reel audio analysis
          </div>
          <h1 className="tp-title">
            Take a song
            <br />
            <em>apart.</em>
          </h1>
          <p className="tp-sub">
            Four isolated stems, every instrument inside, tempo &amp; key — and
            a machine that will actually answer questions about it.
          </p>
          <div className="tp-stats">
            <div>
              <b>0.80</b>
              <span>macro-F1</span>
            </div>
            <div>
              <b>20</b>
              <span>instruments</span>
            </div>
            <div>
              <b>0%</b>
              <span>hallucination</span>
            </div>
          </div>
        </div>

        <div className="tp-drop">
          <div className="tp-reel tp-reel-a" />
          <div className="tp-reel tp-reel-b" />
          <div className="tp-drop-in">
            <div className="tp-drop-icon">↓</div>
            <div className="tp-drop-t">Drop a song on the deck</div>
            <div className="tp-drop-s">mp3 · wav · flac — up to 15 MB</div>
            <button className="tp-btn tp-btn-lg">Load a track</button>
          </div>
        </div>
      </header>

      {/* ── transport ────────────────────────────────────────── */}
      <section className="tp-deck">
        <div className="tp-deck-head">
          <button
            className="tp-play"
            onClick={() => setPlaying((p) => !p)}
            aria-label={playing ? "Pause" : "Play"}
          >
            {playing ? "❚❚" : "▶"}
          </button>
          <div className="tp-now">
            <div className="tp-now-t">{TRACK.title}</div>
            <div className="tp-now-s">
              {TRACK.bpm} BPM · {TRACK.key} · 4:17
            </div>
          </div>
          <div className="tp-vu">
            {Array.from({ length: 7 }).map((_, i) => (
              <span
                key={i}
                className={playing ? "tp-vu-b tp-vu-live" : "tp-vu-b"}
                style={{ animationDelay: `${i * 90}ms` }}
              />
            ))}
          </div>
        </div>

        {/* stems as tape strips */}
        <div className="tp-strips">
          {STEMS.map((s, i) => {
            const dim = solo !== null && solo !== s;
            return (
              <div
                key={s}
                className={`tp-strip${dim ? " tp-dim" : ""}`}
                style={{ ["--tape" as string]: LABEL[s].c, animationDelay: `${i * 60}ms` }}
              >
                <div className="tp-strip-lbl">
                  <span className="tp-strip-n">{LABEL[s].n}</span>
                  {s}
                </div>
                <button
                  className={`tp-solo${solo === s ? " tp-on" : ""}`}
                  onClick={() => setSolo(solo === s ? null : s)}
                >
                  solo
                </button>
                <div className="tp-wave">
                  {peaks(s, 72).map((p, j) => (
                    <i key={j} style={{ height: `${p * 100}%` }} />
                  ))}
                </div>
                <span className="tp-knurl" />
              </div>
            );
          })}
        </div>
      </section>

      {/* ── findings ─────────────────────────────────────────── */}
      <section className="tp-grid">
        <div className="tp-card">
          <h2 className="tp-h2">What&rsquo;s inside</h2>
          <ul className="tp-inst">
            {TRACK.instruments.map((it, i) => (
              <li key={it.name} style={{ animationDelay: `${i * 45}ms` }}>
                <span className="tp-inst-n">{it.name}</span>
                <span className="tp-meter">
                  {Array.from({ length: 12 }).map((_, k) => (
                    <b
                      key={k}
                      data-lit={k < Math.round(it.confidence * 12)}
                    />
                  ))}
                </span>
                <span className="tp-inst-p">
                  {Math.round(it.confidence * 100)}
                </span>
              </li>
            ))}
          </ul>
        </div>

        <div className="tp-card tp-card-note">
          <h2 className="tp-h2">The write-up</h2>
          <p className="tp-blurb">{TRACK.blurb}</p>
          <div className="tp-sig">— written by the model, grounded in the analysis</div>

          <h2 className="tp-h2 tp-h2-sp">Ask the deck</h2>
          <div className="tp-chat">
            {CHAT.map((m, i) =>
              m.role === "user" ? (
                <div key={i} className="tp-q">
                  {m.text}
                </div>
              ) : (
                <div key={i} className="tp-a">
                  {m.text}
                  <span className="tp-cite">↳ {m.tools?.join(" · ")}</span>
                </div>
              ),
            )}
          </div>
          <div className="tp-ask">
            <input placeholder="ask about this track…" readOnly />
            <button className="tp-btn">ask</button>
          </div>
        </div>
      </section>
    </div>
  );
}

const tapeCSS = `
.v-tape{
  --paper:#F2EBDD; --ink:#17130F; --ink-2:#6B6154; --hot:#E24E1B;
  --line:#17130F;
  background:var(--paper); color:var(--ink);
  font-family:var(--f-tape-mono),ui-monospace,monospace;
  min-height:100vh; padding:0 clamp(16px,4vw,56px) 120px;
  background-image:
    radial-gradient(circle at 12% 8%, rgba(226,78,27,.09), transparent 42%),
    radial-gradient(circle at 88% 0%, rgba(91,79,207,.08), transparent 38%);
}
.v-tape *{box-sizing:border-box}
.v-tape ::selection{background:var(--hot);color:var(--paper)}

/* hero */
.tp-hero{display:grid;grid-template-columns:1.05fr .95fr;gap:clamp(24px,4vw,64px);
  align-items:center;padding:clamp(40px,7vh,88px) 0 clamp(32px,5vh,64px);max-width:1240px;margin:0 auto}
@media(max-width:900px){.tp-hero{grid-template-columns:1fr}}
.tp-badge{display:inline-flex;align-items:center;gap:8px;font-size:11px;letter-spacing:.16em;
  text-transform:uppercase;color:var(--ink-2);border:2px solid var(--line);border-radius:99px;
  padding:6px 14px;background:#fff}
.tp-dot{width:7px;height:7px;border-radius:50%;background:var(--hot);
  box-shadow:0 0 0 0 rgba(226,78,27,.6);animation:tp-pulse 2.2s ease-out infinite}
@keyframes tp-pulse{0%{box-shadow:0 0 0 0 rgba(226,78,27,.55)}70%{box-shadow:0 0 0 12px rgba(226,78,27,0)}100%{box-shadow:0 0 0 0 rgba(226,78,27,0)}}
.tp-title{font-family:var(--f-tape-display),Georgia,serif;
  font-size:clamp(48px,8.5vw,104px);line-height:.92;letter-spacing:-.03em;margin:22px 0 0;
  font-variation-settings:'SOFT' 40,'WONK' 1,'opsz' 100;font-weight:600}
.tp-title em{font-style:italic;color:var(--hot)}
.tp-sub{max-width:44ch;margin-top:20px;font-size:14.5px;line-height:1.65;color:var(--ink-2)}
.tp-stats{display:flex;gap:34px;margin-top:32px}
.tp-stats b{display:block;font-family:var(--f-tape-display),serif;font-size:30px;
  font-variation-settings:'WONK' 1;line-height:1}
.tp-stats span{font-size:10.5px;letter-spacing:.12em;text-transform:uppercase;color:var(--ink-2)}

/* drop deck */
.tp-drop{position:relative;background:#fff;border:2.5px solid var(--line);border-radius:10px;
  box-shadow:8px 8px 0 var(--ink);padding:44px 30px;text-align:center;overflow:hidden}
.tp-reel{position:absolute;top:-46px;width:132px;height:132px;border-radius:50%;
  border:2.5px solid var(--line);background:
    repeating-conic-gradient(var(--paper) 0 12deg,#fff 12deg 24deg);opacity:.5;
  animation:tp-spin 9s linear infinite}
.tp-reel-a{left:-42px}.tp-reel-b{right:-42px;animation-direction:reverse}
@keyframes tp-spin{to{transform:rotate(360deg)}}
.tp-drop-in{position:relative}
.tp-drop-icon{font-size:26px;animation:tp-bob 2.4s ease-in-out infinite}
@keyframes tp-bob{50%{transform:translateY(7px)}}
.tp-drop-t{font-family:var(--f-tape-display),serif;font-size:24px;margin-top:8px;
  font-variation-settings:'WONK' 1}
.tp-drop-s{font-size:11.5px;color:var(--ink-2);margin-top:6px;letter-spacing:.04em}
.tp-btn{font-family:inherit;font-size:12.5px;letter-spacing:.06em;background:var(--hot);
  color:#fff;border:2.5px solid var(--line);border-radius:7px;padding:10px 20px;cursor:pointer;
  box-shadow:3px 3px 0 var(--ink);transition:transform 140ms cubic-bezier(.23,1,.32,1),box-shadow 140ms cubic-bezier(.23,1,.32,1)}
.tp-btn:hover{transform:translate(-1px,-1px);box-shadow:5px 5px 0 var(--ink)}
.tp-btn:active{transform:translate(3px,3px);box-shadow:0 0 0 var(--ink)}
.tp-btn-lg{margin-top:22px;padding:13px 30px;font-size:14px}

/* deck */
.tp-deck{max-width:1240px;margin:0 auto;background:#fff;border:2.5px solid var(--line);
  border-radius:12px;box-shadow:8px 8px 0 var(--ink);overflow:hidden}
.tp-deck-head{display:flex;align-items:center;gap:18px;padding:18px 22px;
  border-bottom:2.5px solid var(--line);background:linear-gradient(#fff,#F7F2E8)}
.tp-play{width:56px;height:56px;flex:none;border-radius:50%;border:2.5px solid var(--line);
  background:var(--hot);color:#fff;font-size:17px;cursor:pointer;box-shadow:3px 3px 0 var(--ink);
  transition:transform 140ms cubic-bezier(.23,1,.32,1),box-shadow 140ms cubic-bezier(.23,1,.32,1)}
.tp-play:active{transform:translate(3px,3px) scale(.97);box-shadow:0 0 0}
.tp-now-t{font-family:var(--f-tape-display),serif;font-size:19px;font-variation-settings:'WONK' 1}
.tp-now-s{font-size:11.5px;color:var(--ink-2);margin-top:3px;letter-spacing:.05em}
.tp-vu{display:flex;gap:4px;align-items:flex-end;margin-left:auto;height:30px}
.tp-vu-b{width:6px;height:30%;background:var(--ink);border-radius:1px;transform-origin:bottom}
.tp-vu-live{animation:tp-vu 780ms ease-in-out infinite alternate;background:var(--hot)}
@keyframes tp-vu{to{transform:scaleY(3.2)}}

/* tape strips */
.tp-strips{padding:6px 0}
.tp-strip{display:flex;align-items:center;gap:14px;padding:13px 22px;position:relative;
  border-bottom:1.5px dashed #D9CFBC;opacity:0;animation:tp-in 460ms cubic-bezier(.23,1,.32,1) forwards;
  transition:background 160ms ease,opacity 200ms ease}
.tp-strip:last-child{border-bottom:0}
.tp-strip:hover{background:#FCF9F3}
.tp-dim{opacity:.32}
@keyframes tp-in{from{opacity:0;transform:translateX(-10px)}to{opacity:1;transform:none}}
.tp-strip-lbl{width:104px;flex:none;display:flex;align-items:center;gap:9px;font-size:12.5px;
  letter-spacing:.1em;text-transform:uppercase}
.tp-strip-n{width:22px;height:22px;display:grid;place-items:center;border-radius:4px;
  background:var(--tape);color:#fff;font-size:10.5px;border:2px solid var(--line)}
.tp-solo{font-family:inherit;font-size:10px;letter-spacing:.08em;padding:4px 9px;border-radius:5px;
  border:2px solid var(--line);background:#fff;cursor:pointer;
  transition:transform 130ms cubic-bezier(.23,1,.32,1),background 130ms ease}
.tp-solo:active{transform:scale(.95)}
.tp-on{background:var(--tape);color:#fff}
.tp-wave{flex:1;display:flex;align-items:center;gap:1.5px;height:44px;min-width:0}
.tp-wave i{flex:1;background:var(--tape);border-radius:1px;opacity:.85;
  transition:height 200ms cubic-bezier(.23,1,.32,1)}
.tp-knurl{width:16px;height:34px;flex:none;border:2px solid var(--line);border-radius:3px;
  background:repeating-linear-gradient(90deg,var(--paper) 0 2px,#fff 2px 4px)}

/* findings */
.tp-grid{max-width:1240px;margin:34px auto 0;display:grid;grid-template-columns:1fr 1.15fr;gap:26px}
@media(max-width:900px){.tp-grid{grid-template-columns:1fr}}
.tp-card{background:#fff;border:2.5px solid var(--line);border-radius:12px;
  box-shadow:8px 8px 0 var(--ink);padding:26px 26px 30px}
.tp-card-note{background:#FFFDF7}
.tp-h2{font-family:var(--f-tape-display),serif;font-size:22px;font-variation-settings:'WONK' 1;
  margin:0 0 18px;letter-spacing:-.01em}
.tp-h2-sp{margin-top:30px}
.tp-inst{list-style:none;margin:0;padding:0;display:grid;gap:11px}
.tp-inst li{display:flex;align-items:center;gap:12px;font-size:13px;opacity:0;
  animation:tp-in 420ms cubic-bezier(.23,1,.32,1) forwards}
.tp-inst-n{width:96px;flex:none}
.tp-meter{flex:1;display:flex;gap:2.5px}
.tp-meter b{flex:1;height:15px;border-radius:2px;background:#EFE7D8;border:1.5px solid #DCD1BE}
.tp-meter b[data-lit="true"]{background:var(--hot);border-color:var(--ink)}
.tp-inst-p{width:26px;text-align:right;font-size:11.5px;color:var(--ink-2)}
.tp-blurb{font-family:var(--f-tape-display),serif;font-size:17.5px;line-height:1.55;margin:0;
  font-variation-settings:'opsz' 20}
.tp-sig{font-size:10.5px;color:var(--ink-2);margin-top:12px;letter-spacing:.04em}
.tp-chat{display:grid;gap:12px}
.tp-q{font-size:13px}.tp-q::before{content:'▸ ';color:var(--hot)}
.tp-a{font-size:13px;line-height:1.6;border-left:3px solid var(--tape,#2E8B7A);padding-left:12px}
.tp-cite{display:block;font-size:10px;color:var(--ink-2);margin-top:6px}
.tp-ask{display:flex;gap:9px;margin-top:16px}
.tp-ask input{flex:1;font-family:inherit;font-size:12.5px;padding:11px 13px;border-radius:7px;
  border:2.5px solid var(--line);background:var(--paper)}
@media(prefers-reduced-motion:reduce){
  .v-tape *{animation-duration:.01ms!important;animation-iteration-count:1!important;transition-duration:.01ms!important}
}
`;
