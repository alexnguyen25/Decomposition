# Delightful audio UI — research for the Decomposition frontend overhaul

**Date:** 2026-08-02
**Scope:** What to steal, and what to avoid, in redesigning Decomposition's frontend so it is
genuinely enjoyable to use while still reading as credible ML engineering.
**Method:** Web research plus direct measurement of live sites (computed styles, CSS custom
properties, loaded font files, keyframe names) and direct queries against the Baseline
(`api.webstatus.dev`) and MDN BCD datasets. Where a popular blog contradicted primary data,
primary data wins and the conflict is flagged. Anything unverified is marked as such.

**Current state being redesigned** (`reference-app/frontend/`): a dark "studio console" theme —
`--bg: #0b0b0e`, `--amber: #f5a623`, per-stem channel colors (`vocals #f5a623`, `drums #e85d4a`,
`bass #9d7bff`, `other #3ecfb2`), Bricolage Grotesque display + mono for data, a phosphor grid and
grain overlay, three keyframes (`rise`, `eq`, `blink`). Four independent WaveSurfer instances
driven by one controller; mute/solo implemented as `setVolume(0/1)`; the playhead clock is a
`setInterval(…, 250)`.

---

## (a) Executive summary — the 8 principles that matter

**1. The delight has to be in the substance, not bolted onto it.**
NN/g's distinction is the right frame: *surface delight* (animations, microcopy, imagery) is local
and cheap and wears off; *deep delight* comes from the product anticipating what you need. For this
app the deepest available delight is not a new color palette — it is **showing the model's
time-resolved output**. The classifier already runs on 5-second windows and aggregates with
top-3-mean; today the UI throws the time axis away and prints four percentages. Putting it back
(see the "activity ribbon", §c) is simultaneously the most fun feature and the strongest ML signal
available.

**2. Animate the separation. It is the product's one hero moment and the UI currently skips it.**
Open on the *mixed* waveform, then split it into four lanes. `djay Pro`'s Neural Mix does exactly
this — one waveform that progressively fans out into 2 → 3 → 4 stem layers on demand — and it is
the best interaction model found in this entire survey. It also literally animates the word
"decomposition."

**3. Precompute the reactive data; do not chase a live AnalyserNode for everything.**
The backend already computes framed RMS per stem for presence detection. Ship that envelope to the
client and index it by playhead time. You get jitter-free meters, no cross-origin taint failures,
no AudioContext plumbing for visuals, motion that is *honest* (it is literally the data the model
saw), and — the delightful part — **the visualization works while paused**, so you can see where
the vocals are before you press play.

**4. The transport foundation is a design problem, not just an engineering one.**
The playhead currently updates at 4 fps (`setInterval` at 250 ms). Nothing you do with color or
type will overcome that; it is the single most damaging detail in the current build. And four
`<audio>`-backed WaveSurfer instances is the wrong sync primitive — the Web Audio spec has a
documented gap for `MediaElementAudioSourceNode` timing, and Safari is the worst offender. Split
the layers: **visual = canvas from precomputed peaks; transport = `AudioBufferSourceNode`s sharing
one `t0` on one `AudioContext`**, which is sample-accurate by construction because there is only
one clock.

**5. Scale the interface like a faceplate, not a document.**
Teenage Engineering's entire "this feels like hardware" effect is one trick, verified directly on
their site: `--client-width: 100vw`, a fixed design width of 980, and *every* token expressed as
`calc(fraction * var(--client-width))` — type, line-height, gaps, radii, header height. One
breakpoint at 768px where mobile is exactly 2×. The page grows as a unit instead of reflowing.

**6. Fun and credible is not a tradeoff — it is resolved by choosing *what* you make playful.**
Playful chrome over serious substance reads as a toy. Playful treatment *of the substance* reads as
mastery. Ciechanowski's interactive articles, Georgia Tech's CNN/Transformer Explainers, and
Ableton's Learning Music all work the same way: the technical content itself is the thing you get
to touch. Don't add confetti; make the classifier's confidence-over-time draggable.

**7. Budget the motion. Pick five moments; make everything else instant.**
Emil Kowalski's frequency rule: 100+ times/day → never animate; occasional → standard animation;
rare/first-time → you may add delight. In this app the rare moments are *results reveal*, *first
solo*, *first chat answer*, *share*. Everything a user does repeatedly (mute, solo, seek, scrub)
should be instant with ≤160 ms feedback. Under 300 ms, `ease-out`, `transform`/`opacity` only,
interruptible.

**8. Continuous motion in an audio app is the accessibility risk, and it is easy to miss.**
A level meter bouncing at 60 fps is unsolicited perpetual motion — WCAG 2.2.2 (Pause, Stop, Hide)
territory, and the most motion-heavy thing on the page. It doesn't feel like an "animation," so it
gets forgotten. The current `.eqbar` is worse than that: it's a *fake* visualizer that animates
identically regardless of audio, inside a product whose entire premise is that it listens
accurately. Kill it or make it real.

---

## (b) Reference gallery

### B1 — Stem separation and multitrack tools

#### stem.fm (formerly stemplayer.com) — the Kanye/Kano Stem Player, now a real web app
`https://www.stem.fm` (`stemplayer.com` 301-redirects here) · device background:
`https://en.wikipedia.org/wiki/Stem_Player`

The device: beige 7 cm puck, **four touch-sensitive LED sliders cut as crisscrossed grooves**, six
hardware buttons, no screen. Reviewers reach for the word *braille* — you find the controls by
touch. The delight is weight + circularity + grooves your fingers fall into, plus instant response.

The web app is the surprise, and it is measurably good:
- Pure `#000`; custom typefaces **`STEM`, `STEM Mono`, `DeepSea`** (Inter/RobotoMono fallbacks).
- `--app-secondary-rgb: 196 186 168` → **`#C4BAA8`** — *the device's beige promoted to the web
  app's accent color.* The hardware colors the software.
- Loading `/` **immediately mints a session** and redirects to `/session/{id}`. No landing page, no
  login wall, shareable URL from second zero.
- Ambient background = two stacked `.background-gradients__layer` divs (opacity 1 and 0), each a
  stack of `radial-gradient(80% 60% at 0% 30%, …)` sampled from artwork, crossfading on track
  change. ~20 lines of CSS for enormous perceived polish.
- Glass: `backdrop-filter: blur(16px) saturate(1.5)`; sidebar `blur(60px) saturate(1.8)`.
- Easings in production: `cubic-bezier(0.22, 1, 0.36, 1)` (easeOutQuint), `cubic-bezier(0.16, 1,
  0.3, 1)` (easeOutExpo), `cubic-bezier(0, 0, 0.2, 1)`. Durations cluster at
  **0.1 / 0.15 / 0.16 / 0.18 / 0.2 / 0.3 / 0.45s**. Touch targets 44×44. Pills at `9999px`.

**The keyframe names are the real intelligence** — they tell you exactly what they decided was
worth animating: `pulse-stem-container`, `click-stem-container`, `stems-dot-refresh-pulse`,
`stems-artwork-refresh-pulse`, `track-title-pulse`, `track-title-transition-glow`,
`create-session-sheen`, `animateOneLoader` … `animateFiveLoader`.

**Steal:** the stem container has *both* an idle pulse (alive) and a distinct click animation;
re-separation is *felt* via artwork/dot pulses rather than reported; five numbered loader keyframes
= a staggered per-stem loading state, which is exactly the shape of a Demucs job.

**Caveat:** the device drew real backlash ($200, no wireless transfer, misfiring haptics), and the
brand carries unavoidable baggage. Steal the interface ideas, not the association.

#### djay Pro / Neural Mix (Algoriddim, powered by AudioShake) — the best waveform model found
`https://help.algoriddim.com/user-manual/djay-pro-mac/neural-mix/tracks`

A **progressively splittable waveform**: 2-stem (drums / harmonics) → 3-stem (+ vocals) → 4-stem
(drums / bass / harmonics / vocals), real-time and on-device, opened from the waveform's own corner
rather than a separate mode. Multiple Apple Design Awards.

**Steal:** this exact model. Do not open on four disconnected tracks. Open on the mix and let the
user *cause* the split. It matches how people think about a song and it makes separation feel like
an action rather than a fait accompli.

#### Moises.ai — the metadata is what turns a separator into a tool
`https://moises.ai` · design writeup:
`https://moises.ai/newsroom/awards-recognition/apple-design-awards-nomination/` (2025-06-04)

Apple Design Award **finalist 2025** (Innovation), 2024 iPad App of the Year. Marketing site is
`#000`, typeface **Articulat CF at weight 400 only**, headings 48px / `letter-spacing: -0.96px`,
fully-round pill CTAs. Up to 7 stems.

The named delight is the **iPad chord view**: in Moises' own words, when chords are on during
playback *"the interface adapts instantly. The chords take over the screen in large, readable
letters, perfectly synced with the music."* No menu, no chrome — the UI reconfigures to the task.

Design chief Jardson Almeida's framing is the thesis for this whole project: *"turning complex
AI-powered source separation into something as intuitive as moving a volume slider."*

**Steal:** (1) a **mode that takes over the whole screen** instead of adding another panel;
(2) ship key/BPM/sections/chords *alongside* stems — you already have key and BPM, and section
detection is the obvious next rung; (3) loop a detected section.

**Caveat:** the in-app mixer (fader layout, per-stem colors) is behind auth and was **not
verified**. Don't cite specifics about it.

#### LALAL.AI — one good idea, two hostile ones
`https://www.lalal.ai` · Roobert throughout, CTA yellow `#F8D231` on `#1B191C`, `border-radius: 6px`.
(Note: `body` declares a dark background but a wrapper overrides it — the rendered page is light.)

**The good idea:** *"Choose what to extract, then upload your file."* Specifying the job before
upload means the progress bar means something specific. Also: a 30-second stem preview before you
spend credits.

**The two to avoid:** unused minutes don't roll over (loudest complaint by far), and
**one instrument per pass** — multi-instrument extraction means repeated processing, burning both
time and credits. That is an actively hostile interaction loop.

#### stemplayer-js — a real open-source architecture reference
`https://github.com/stemplayer-js/stemplayer-js`

Four custom elements (`stemplayer-js`, `-controls`, `-stem`, `-workspace`). The important
architectural notes, quoted: waveforms **must be pre-generated** because "we don't download the
entire audio file, we cannot analyse the audio"; playback uses one shared `AudioController` over
HLS chunks; and *"UI progress is driven by the player UI loop; this avoids coupling high-frequency
render work to engine internals."* That last sentence is the fix for the current 250 ms interval.

#### Stemdeck — the mixer-interaction details worth copying verbatim
`https://github.com/stemdeckapp/stemdeck`

Vanilla JS + Web Audio, canvas min/max sample rendering, live per-stem VU meters (post-gain RMS,
peak hold, slow falloff), gold playhead overlay, draggable loop region on the ruler.
**Fader behavior:** 1:1 drag, double-click resets to 0 dB, `Shift`+wheel coarse / plain wheel fine.
**Solo semantics:** additive (multiple solos stay audible), with a separate **Monitor** action that
isolates one stem and clears the others.

#### Ableton Live 12.3 (2025-11-25) — the restraint lesson
`https://cdm.link/ableton-live-12-3-guide/`

Native stem separation into drums / vocals / bass / other, and the separated stems land as
**ordinary audio clips**. No special mode, no new UI to learn. When separation is table stakes, the
differentiator is what you do *with* the stems.

#### Suno — non-numeric sliders
`https://suno.com/blog/songeditor` (2025-06-03)

Song Editor exposes three creative sliders governing how *weird / structured / reference-driven*
generations get. Named in plain adjectives, not 0–100. That is the right pattern for any control
whose units a user cannot reason about.
(**Udio:** no 2025–2026 stem view could be verified. Do not cite one.)

---

### B2 — Brands whose web design feels like hardware

#### Teenage Engineering — the single most transferable technical finding in this report
`https://teenage.engineering` · `https://teenage.engineering/products/ep-133` ·
`https://teenage.engineering/products/field-system`

Measured directly, because the widely-cited write-ups are wrong (see caveat below).

**Typography.** `--te-20: "te-20"` and `--te-40: "te-40"` resolve to **`UniversTE20T` /
`UniversTE40L`** — bespoke **Univers** cuts, *not monospace*. Exactly two weights:
`--fw-thin: 100`, `--fw-light: 300`. All UI text lowercase. Plus bespoke display faces *per
product* (`franxurter`, `TechnoType`, `swingus`, `riddim`) — each product gets its own voice.

**Color.**
```
--te-black: #0F0E12   --te-white: #F5F5F5   --te-grey-100: #E5E5E5  --te-grey-1000: #272727
--te-blue:  #0071BB   --te-green: #006837   --te-orange:   #F05A24
--te-red:   #B81D13   --te-yellow:#FAB413
```

**The trick that makes it feel like an object.** Every dimension is a fraction of viewport width:
```css
:root { --client-width: 100vw; }              /* base design width: 980 */
--fs-20:  calc(.0183673 * var(--client-width));  /* 18/980 */
--lh-20:  calc(.0204082 * var(--client-width));  /* 20/980 */
--space-sm: calc(.0204082 * var(--client-width));
--tile-border-radius: calc(.0255102 * var(--client-width));
--header-height: calc(.0816327 * var(--client-width));
```
At 980px this resolves to font sizes `9 / 13 / 18 / 23 / 27 / 36px` and line-heights of exactly
`10 / 15 / 20 / 30 / 30 / 40px` — **the token names *are* their pixel values**. Spacing:
xs 5, sm 10, md 15, lg 22.5, xl 45. **One breakpoint at 768px, where mobile is exactly 2× desktop.**
The page doesn't reflow like a document; it scales like a vector faceplate.

**Per-product theming from the physical colorway** (verified): EP–133 `--theme-body-bc: #F9FAF9`
with `#ABB5BA` accent (its grey-blue chassis); TP–7 `#E5E5E5` (aluminium); OP–XY and OB–4 `#000`.

**And:** the EP–133 page has six product videos with `muted: false, autoplay: false` — click to
play, **with sound**. The product page lets you hear the instrument. Never autoplays.

> **Correction to a circulating source.** `blakecrosley.com/guides/design/teenage-engineering`
> claims TE is monospace-only and uses `#ff6600`. Both are wrong against the live site (Univers-
> derived faces; `#F05A24`). That article is a stylistic reconstruction, not documentation.

#### Endel — discipline as personality
`https://endel.io`

Pure `#000`. **One typeface (`apercuPro`) at one weight (400)** across the entire marketing site.
Two accents in the whole system: `--color-orange: #FF7A00`, `--color-green: #82D133`. Rigid 12-col
grid (`--max-width: 1280px`, `--gutter: 64px`, `--gap: 24px`). Slightly *positive* heading tracking
(`letter-spacing: 0.35px`), which reads as calm. **Only three CSS keyframes exist in the entire
site** — all the generative motion is prerendered video.

**Steal:** the budget discipline. One typeface, one weight, two accents, zero decorative CSS
animation — then spend the entire motion allowance on one hero visual that represents live system
state. Endel's generative visual is a *status readout for an invisible system*, which is exactly
what a mix visualization is.

---

### B3 — Story, reveal, and shareability

#### Spotify Wrapped — read 2024 as the cautionary tale and 2025 as the recipe
2025 design writeup: `https://spotifynews.substack.com/p/designing-2025-wrapped-turning-a`
(2025-12-04) · campaign: `https://newsroom.spotify.com/2025-12-03/wrapped-marketing-campaign/` ·
analysis: `https://uxplaybook.org/articles/spotify-wrapped-ux-design-lessons` (2025-12-08) ·
2024 backlash: `https://www.forbes.com/sites/danidiplacido/2024/12/05/spotify-wrapped-2024-backlash-controversy-and-memes/`

**2024 failed, and the reason is directly relevant to this app.** It was called "AI-generated
slop." Three causes: invented genre names (*"Pink Pilates Princess Strut Pop"*), an AI podcast that
delivered empty observations, and the removal of beloved accurate features (Sound Town, genre
highlights). The lesson is not "AI bad." It is that **generated whimsy reads as cheap the moment
the underlying data feels less accurate than before.** For an instrument classifier: a wrong
instrument name costs far more trust than a missing one.

**2025 course-corrected into physical media.** Design lead Rasmus Wangelin: the reference is
1980s–90s audio culture — *"people were making mixtapes, doodling on cassette inserts, and
designing club flyers."* Creative director Jeremy Wirth on motion: *"Every element feels like it's
forming in real time, with shapes moving, textures overlapping, and type dancing like sound
waves."* Black-and-white foundation with selective accent, condensed type on a grid, scratchy
textures, gritty stop-motion feel.

**Four transferable mechanics:** emotional framing over raw metrics; **shareability as a primary
design driver, not an afterthought**; progressive card-by-card disclosure; no badges or streaks.

Spotify's core system is also worth noting for one move: **extract color from the artwork and
recolor the page from it** so the UI recedes and the content glows.

---

### B4 — Playful web that is still respected

#### Ableton Learning Music / Learning Synths — the gold standard for this project's genre
`https://learningmusic.ableton.com` · `https://learningsynths.ableton.com`

A full playable synth plus lessons, in the browser, free, no signup, works on any device (2019,
later localized to seven languages). It is the best available model for "explain audio by letting
people touch it immediately," and it is more relevant to a stem-separation explainer than most
commercial tools. *(Both are heavily client-rendered and resisted automated fetching; the
description here is from documentation and reputation, not measurement.)*

#### Chrome Music Lab and the Google Creative Lab experiments
`https://musiclab.chromeexperiments.com`

Zero-friction is the whole design: no account, open an experiment and play. Bold flat color, large
targets, one idea per experiment (Song Maker, Spectrogram, Kandinsky, Rhythm, Oscillators).
**Steal the entry cost, not the aesthetic** — the "audition an example, instantly, no upload" path
this app already has is the same insight and deserves to be much more prominent.

#### Blob Opera — the single best "ML model as toy" precedent
`https://artsandculture.google.com/experiment/blob-opera/AAHWrq360NcGbw` · David Li × Google Arts &
Culture *(originally released December 2020; the Arts & Culture page carries a later date — treat
the exact launch date as unverified)*

A neural net trained on ~16 hours of four professional opera singers (tenor Christian Joel, bass
Frederick Tong, mezzo-soprano Joanna Gamble, soprano Olivia Doutney). You drag four blobs **up/down
for pitch, forward/back for vowel**, and the model generates the harmony. You are hearing the
model's interpretation, not the original voices — and the page says so.

**Why it's the right precedent for this project:** it is a *research model*, presented as a toy, with
its provenance stated plainly, and it is one of the most-shared ML demos ever made. The delight
comes from the model being directly manipulable, and the credibility comes from naming the training
data. That is precisely the combination Decomposition needs.

**Steal:** two-axis direct manipulation as the entire control surface (no sliders, no menus), and
the honest one-line provenance note sitting right next to the fun.

#### Poolsuite (formerly Poolside.fm) — the committed bit
`https://poolsuite.net` (`poolside.fm` 301-redirects here)

A retro-OS desktop metaphor — windowed "apps," boot screen, period-accurate chrome — wrapped around
an internet radio station. Its lesson is commitment: the conceit is applied to *everything*
(cursor, scrollbars, boot sequence, error dialogs), which is why it reads as an artifact rather than
a theme. A half-applied metaphor reads as a gimmick; a fully-applied one reads as a world.
*(This site is heavily client-rendered and resisted automated inspection — description is from
reputation and the boot assets, not measurement.)*

#### Every Noise at Once — the anti-design that worked, and its cautionary ending
`https://everynoise.com`

A wall of unstyled colored text that was delightful because the *data* was delightful and
navigable. Worth knowing the ending: Glenn McDonald was laid off from Spotify on **2023-12-04**,
losing the internal data access the site depended on; most of it is now a static snapshot on his
own server, and by January 2025 he noted Spotify appeared to be quietly removing genres from the
API. **Lesson:** a delightful interface built entirely on someone else's data feed has a
half-life. This app owns its pipeline, which is an underrated advantage worth showing off.

#### Bruno Simon — the ceiling, and the reason not to aim at it
`https://bruno-simon.com`

A drivable 3D world as a portfolio: WASD/arrow steering, SHIFT boost, SPACE jump, gamepad support,
an achievements system, a racing circuit with leaderboards, and a 30-character "whisper" system for
visitors. Three.js with WebGPU/WebGL via TSL, Rapier physics, Howler.js audio, MIT-licensed with the
Blender files included.

**The honest read for this project:** this works because Bruno Simon *is* a creative WebGL developer
— the site is a direct demonstration of the exact skill he's hired for. Decomposition is hired for
ML engineering, so a driving game would be a category error. What *is* stealable is the underlying
move: **make the interface a demonstration of the thing you actually do.** For Bruno that's WebGL;
here that's audio ML, which is why the activity ribbon and the draggable analysis window are the
right kind of showing-off and a 3D scene is not. Also worth copying: the site's open-source-with-
source-files posture is itself a credibility signal.

#### Tim Holman — the case for one deliberate joke
`https://tholman.com` · "Fun.css", CSSconf EU 2015 · `https://youtube.com/watch?v=5HP6k43T0yM`

Elevator.js, console.frog, cursor-effects, The Useless Web. The talk's argument is that a single
piece of unnecessary craft signals confidence. The operative word is *single*. One easter egg in a
technically serious tool reads as personality; five read as unserious.

---

### B5 — Technically credible *and* delightful (the exact target for this project)

#### Bartosz Ciechanowski, "Sound"
`https://ciechanow.ski/sound/` (2022-10-18)

The most directly relevant reference in this entire report, and it's a coincidence worth
exploiting: it's an interactive explainer about **waveform decomposition** — sine sums, Fourier
series, Fourier transforms, and how instruments produce sound — with custom-coded, no-framework
interactive figures. Every concept has a thing you can drag.

**Steal:** the pedagogical stance. Decomposition's differentiator versus Moises/LALAL is that it
can *explain itself*. An expandable "how this works" figure next to the instrument list — one
draggable 5-second window over the waveform showing the classifier's per-window scores — is
Ciechanowski's move applied to your own model.

#### CNN Explainer and Transformer Explainer (Georgia Tech, Polo Chau's lab)
`https://poloclub.github.io/cnn-explainer/` · `https://poloclub.github.io/transformer-explainer/`

Both went viral; CNN Explainer was invited to SIGGRAPH, Transformer Explainer won IEEE VIS Best
Poster. Transformer Explainer runs **a live GPT-2 in the browser** so you type your own input and
watch the internals respond.

**Why this matters for recruiters specifically:** these are the canonical artifacts of "ML person
who can build interfaces." Being visibly in that lineage is worth more than any amount of visual
polish. The move to copy is *live model output made manipulable*, not static screenshots of it.

---

### B6 — Craft and motion sources (the primary literature)

- **Emil Kowalski** · "Great animations" `https://emilkowal.ski/ui/great-animations` (≈June 2024) ·
  "7 Practical Animation Tips" · "You Don't Need Animations" · "Building a Drawer Component"
  (Vaul internals) · standards file `github.com/emilkowalski/skills`. Concrete rules in §c5.
  *(Note: there is no article at `emilkowal.ski/ui/speed` — the perceived-speed material lives
  inside "Great animations" and "You Don't Need Animations".)* Two details worth surfacing here:
  Vaul uses **500 ms** for its sheet — the one place he goes long, matching iOS sheets; and Vaul hit
  real jank at ~20+ list items from driving child transforms through a **parent CSS variable**,
  which forced a style recalc on every child. He now writes `transform: translateY()` directly on
  the moving element. Also his tooltip rule: the *first* tooltip gets a delay + animation; once the
  user is "in" tooltip mode, subsequent ones open with no delay and `transition-duration: 0ms`.
- **Rauno Freiberg — "Invisible details of interaction design"** ·
  `https://rauno.me/craft/interaction-design` (undated; between March 2022 and July 2023).
  Now a 23-chapter course at `https://devouringdetails.com/`. Twelve categories; the ones that
  apply here: **metaphors** (gestures are learned once and compound); **kinetics** (a thrown
  element retains both momentum *and angle*); **thresholds scale with consequence** — lightweight
  overlays trigger partway through a swipe, destructive actions only on gesture *end* regardless of
  distance; **responsive gestures** — pinch scale delta applies *immediately*, before any animation
  threshold, and iOS Settings never blocks a second back-swipe on an in-flight animation;
  **spatial consistency** (apps launch from their icon's actual location); **frequency and novelty**
  — macOS context menus appear with *no* motion, and Cmd-Tab released immediately switches without
  ever drawing the switcher; **content visibility under touch** — the caret loupe renders *above*
  the finger, and a slider keeps dragging when your finger wanders off it; **fidgetability**;
  **Fitts's Law** (screen corners are infinite targets).
- **Josh Comeau** · "An Interactive Guide to CSS Transitions"
  `https://www.joshwcomeau.com/animation/css-transitions/` (2021-02-09, upd. 2026-05-05) — his
  example durations are **250 ms** standard hover, **125 ms enter / 450 ms exit** for asymmetric
  hover, **400 ms** dropdown opacity; "action-driven animation" means enter and exit get different
  durations *and* easings. · "A Friendly Introduction to Spring Physics" (2020-09-21, upd.
  2025-11-03) — sandbox config `{ mass: 1.75, tension: 200, friction: 12 }`; react-spring's default
  is `{ mass: 1, tension: 170, friction: 26 }`; springs are for *motion*, not color or opacity. ·
  **"Springs and Bounces in Native CSS"** `https://www.joshwcomeau.com/animation/linear-timing-function/`
  (2025-10-28, upd. 2026-05-05) — `linear()` spring approximation, e.g.
  `linear(0, 1.25, 1, 0.9, 1.04, 0.99, 1.005, 0.996, 1.001, 0.999, 1)`; the advanced syntax with
  time percentages cuts point count roughly **50 → 25**; honest caveat that a `linear()` spring
  **cannot absorb existing velocity** on interruption. · **"A Million Little Secrets"**
  `https://www.joshwcomeau.com/blog/whimsical-animations/` (2025-02-24) — the most relevant essay
  in this list, see §B7.
- **Harrison, Amento, Kuznetsov & Bell — "Rethinking the Progress Bar" (UIST '07)** ·
  `https://chrisharrison.net/projects/progressbars/ProgBarHarrison.pdf`. Ribbing that moves
  **backwards, decelerating** measurably shortens perceived duration: a 5.0 s solid bar felt
  equivalent to a 5.61 s ribbed one (**12.2%**), and a follow-up measured an **11%** reduction in
  perceived duration. Directly applicable to a 60–300 s Demucs job.
- **Smart Interface Design Patterns — loading/progress thresholds** ·
  `https://smart-interface-design-patterns.com/articles/designing-better-loading-progress-ux/`.
  <1 s: no indicator. 1–3 s: skeleton or spinner. 3–10 s: determinate bar. 10 s+: bar **plus**
  percentage **plus** status text. Users only perceive speed changes of **≥20%**. Progress feels
  faster when it **starts fast and slows at the end**. Avoid "queue jumping" (later tasks finishing
  first) — it destroys confidence that anything succeeded.
- **NN/g — "Three Pillars of User Delight" / "A Theory of User Delight"** ·
  `https://www.nngroup.com/articles/pillars-user-delight/`. Surface vs. deep delight; embellishments
  can only ever produce the former.
- **NN/g — "The Role of Animation and Motion in UX"** (Page Laubheimer, 2020-01-12) ·
  `https://www.nngroup.com/articles/animation-purpose-ux/`. Animation has exactly four legitimate
  jobs: **feedback, state-change communication, spatial navigation, signifiers.** Everything else is
  gratuitous. Motion should be "unobtrusive, brief, and subtle."

### B7 — Josh Comeau, "A Million Little Secrets" — the whimsy essay that is really about audio

`https://www.joshwcomeau.com/blog/whimsical-animations/` (2025-02-24, upd. 2026-05-05)

This is the most directly transferable piece of writing found, because his sound-design decision is
the exact decision this app faces:

- **"A charming, delightful effect becomes mundane and annoying surprisingly quickly."** Novelty is
  the active ingredient; reusable formulas kill it.
- He adds sound effects for **discrete** actions (button press, slider tick) and deliberately leaves
  the *continuously running* fireworks **silent** — "perpetual audio becomes irritating," and he
  would rather omit it than add a mute button people never find. **The visual analogue is exact:
  an always-on visualizer is a fundamentally different and riskier design object than one that
  appears on playback and stills on pause.**
- Multi-sample trick: record **5+ variations** of each sound and pick randomly so repetition doesn't
  read as robotic.
- **"Boop" animations** — instead of a sustained `:hover` transform, apply a transform that
  **unsets itself after a short interval** whether or not the pointer is still there. Bursts, not
  lingering states. (This is the right model for a "stem just got soloed" flourish.)
- Particle randomization in polar coordinates: random angle **200–240°**, random distance
  **30–60px**, converted with `calc(cos(var(--angle)) * var(--distance))`.
- Sprite-sheet consolidation of 22 shapes took **~2 MB → under 200 KB** while keeping P3 color.
- **Caveat: the article never mentions `prefers-reduced-motion`.** Don't inherit that omission.

### B8 — The published motion specs worth copying numbers from

**Material 3 Expressive spring tokens.** The m3.material.io motion pages are JS-rendered and
unfetchable; these come from the implementation of record — `ExpressiveMotionTokens.kt` and
`StandardMotionTokens.kt` on `androidx-main`, fetched 2026-08-02. Phone-class values (M3 states
tokens differ per form factor).

| Token | Expressive damping / stiffness | Standard damping / stiffness |
|---|---|---|
| Spatial fast | **0.6 / 800** | 0.9 / 1400 |
| Spatial default | **0.8 / 380** | 0.9 / 700 |
| Spatial slow | **0.8 / 200** | 0.9 / 300 |
| Effects fast | 1.0 / 3800 | 1.0 / 3800 |
| Effects default | 1.0 / 1600 | 1.0 / 1600 |
| Effects slow | 1.0 / 800 | 1.0 / 800 |

The structural rule is the takeaway: **Spatial** tokens (position, size, shape) are underdamped and
allowed to overshoot; **Effects** tokens (color, opacity) are pinned at damping **1.0** — critically
damped, zero bounce, always. Expressive differs from Standard *only* in the spatial springs.
Legacy M3 duration tokens: Short 50/100/150/200 ms · Medium 250/300/350/400 · Long 450/500/550/600 ·
Emphasized easing `cubic-bezier(0.2, 0, 0, 1)`, Emphasized decelerate `(0.05, 0.7, 0.1, 1)`.

**Motion (formerly Framer Motion), 2025–2026.** Package renamed `framer-motion` → **`motion`**,
import path `motion/react`; API unchanged. `animateView` (a View Transitions wrapper) graduated into
the main library **2026-06-23**. Spring defaults: `stiffness: 100, damping: 10, mass: 1`.
Duration-based parameterization defaults to `bounce: 0.25`. **Use `visualDuration`, not `duration`**
— it specifies when the spring *visually appears* to land, ignoring the long settling tail, which is
what you actually mean for UI. Tween default is 0.3 s. There is a footgun: calling `spring()`
directly takes duration in **milliseconds** for historical reasons. There is **no official Motion
doc recommending per-component spring values** — the button/sheet/drag preset tables in circulation
are community convention.

**Apple.** The HIG Motion page is JS-rendered and unfetchable; the numbers live in
**WWDC23 session 10078, "Design considerations for vision and motion"**
(`https://developer.apple.com/videos/play/wwdc2023/10078/`). The single hardest number Apple
publishes, and it is directly relevant here:

> **Avoid oscillations, in particular those with frequencies around 0.2 Hz — one oscillation per
> five seconds.** If you can't avoid oscillating motion, keep the amplitude low and make the content
> semitransparent.

**A slow ambient "breathing" glow at roughly one cycle per five seconds is the worst possible rate
for nausea.** Beat-rate pulsing (1–3 Hz) is far safer than a slow swell. Other Apple guidance: keep
moving objects small and distant rather than large and close; use low-luminance-contrast textures
when something large must move; prefer **lazy-follow** (slowly chase a destination) over hard-locking
to input; under Reduce Motion, **replace sliding transitions with crossfades** and provide an
oscillation-free alternative. SwiftUI's `.spring` default is `response: 0.55, dampingFraction: 0.825`,
with `.smooth` / `.snappy` / `.bouncy` added in iOS 17.

**RAIL response thresholds** (`https://web.dev/articles/rail`) — the arithmetic behind the 300 ms
ceiling: **<100 ms** feels instantaneous · **100–300 ms** slight perceptible delay · **300 ms–1 s**
"things are happening," user stays on task · **>1 s** user loses focus. 16 ms per frame for 60 fps.
An animation *is* latency from the user's point of view. The rule that follows: **animate waiting,
never animate responding.**

---

## (c) Recommendations for Decomposition's four surfaces

### c0 — Foundations to fix first (nothing else lands without these)

**F1. Replace the transport.** One `AudioContext`; decode each stem to an `AudioBuffer`; play by
creating `AudioBufferSourceNode`s and calling `start(t0, offset)` with an **identical `t0`** for all
four. Sample-accurate by construction — one clock, no drift. Seek = discard nodes, create new ones
from the same (immutable, shared) buffers.

```js
const t0 = ctx.currentTime + 0.1;               // lookahead past the 128-sample render quantum
buffers.forEach((buf, i) => { const s = ctx.createBufferSource();
  s.buffer = buf; s.connect(gains[i]); s.start(t0, offsetSec); });
startedAt = t0 - offsetSec;
const playhead = ctx.currentTime - startedAt - ctx.outputLatency;  // subtract output latency
```
`ctx.outputLatency` matters: on Bluetooth the visual leads the audio by 10–40 ms without it.

**Memory budget:** a 4-minute stereo stem as float32 at 44.1 kHz is ~84 MB; four stems ≈ **340 MB
resident**. Fine for a 3–4 minute song on desktop, dangerous on low-end mobile. Mitigate by capping
upload length (already done at 6 min — consider 4), decoding mono where only one channel is
needed, and calling `ctx.close()` on unmount.

**F2. Precompute peaks server-side.** The pipeline already runs Python per job; emit an 8-bit
`.dat` per stem (BBC `audiowaveform`, `-b 8`) in the same job. ~50 KB of peaks versus ~3 MB of
audio, and the waveform paints *before* the audio finishes downloading. wavesurfer's own docs
confirm MediaElement + precomputed peaks is "almost the same as simply playing audio" in CPU terms.
Target **1,000–2,000 peak points per channel** (their recommendation).

**F3. Precompute the RMS envelope and the per-window classifier scores.** `stem_presence.py`
already frames RMS. Emit it at ~20 Hz per stem (a 4-minute track = 4,800 floats ≈ 10 KB as
`Uint8`). Emit the per-5s-window instrument scores too, instead of discarding them after top-3-mean
aggregation. These two arrays unlock the two best features in this document (activity ribbon,
honest meters) for essentially zero additional compute.

**F4. Playhead on rAF, as a transform, never a canvas redraw.**
```js
function frame() { const t = ctx.currentTime - startedAt - ctx.outputLatency;
  head.style.transform = `translate3d(${t * pxPerSec}px,0,0)`;   // one 1px div per lane
  requestAnimationFrame(frame); }
```
Draw the static waveform to canvas **once**; only the overlay moves. This alone is the difference
between "dead" and "alive."

Why this matters more than it sounds: the current 250 ms interval is *the same order of magnitude as
the browser's own `timeupdate` rate*, which is unspecified and measured in the wild at Safari 250 ms,
Edge 268 ms, Firefox 270 ms, **Chrome 390 ms** (Holzmann, 2018). Anything driven off either clock
steps visibly. Two proven architectures:

- **rAF + interpolation** — bind to `play`/`pause`, cache duration and container width once, and
  interpolate position every frame. Put the bar at 100% width inside an `overflow: hidden` wrapper
  and slide it, so nothing ever touches layout.
- **Audio clock as master** (Fender Engineering, 2020) — `AudioContext.currentTime` is the master
  because it is hardware-derived and does not drift the way `Date.now()`/`setTimeout` do; a **Web
  Worker `setInterval`** produces reliably-timed ticks that queue events (a worker's interval isn't
  offset by main-thread jank); rAF on the main thread drains the queue right before paint. They
  report a consistent 60+ fps profile. This is the right architecture once playback runs on
  `AudioBufferSourceNode`s.

The cleanest *API* shape comes from peaks.js (PR #206), which deleted its tweened-playhead machinery
entirely and replaced it with a rAF loop that emits **synthetic high-rate `timeupdate` events**
(`_fireFakeTimeUpdate()`). Every consumer — playhead, meters, chat highlight, activity ribbon —
subscribes to one high-rate time signal instead of each animating itself. Do that.

Also: `will-change: transform` on the playhead. It avoids the CPU→GPU handoff artifact and, per
Comeau, improves sub-pixel rendering quality. Animating `width` is the single most common cause of a
chunky playhead — it snaps to integer pixels and forces layout every frame.

**Scrolling model.** Three options, and the middle one is a trap: (a) *static waveform, moving
playhead* — cheap, calm, correct default; (b) *fixed centered playhead, waveform scrolls underneath*
(Logic's "Catch + Scroll in Play", Ableton's song follow) — smoother and keeps the point of interest
at a stable screen position, but it puts **the entire waveform in continuous large-area motion**,
which is exactly the vestibular trigger Val Head and Apple both flag; (c) page-flip — cheap but
loses continuity. Use (a), and only switch to (b) when zoomed past the viewport.

**F5. Ramp gains; never hard-set volume.** `gain.gain.setTargetAtTime(v, ctx.currentTime, 0.008)`
or a 15–25 ms `linearRampToValueAtTime`. A hard `setVolume(0)` produces a click, which in an audio
product is an audible bug.

---

### c1 — Landing / upload

**Current:** hero + dashed-ish bordered drop zone with corner ticks + example shelf. It's tidy but
it doesn't demonstrate anything, and the strongest asset (instant precomputed examples) is below
the fold in a section headed "or audition an example."

**Recommendations:**

1. **Lead with a live artifact, not a description.** Put one precomputed example *in the hero* as a
   playable 4-lane micro-console (30 seconds, ~200px tall). The product's claim is visual and
   audible; making people read two sentences and then upload a file before seeing anything is the
   biggest conversion and delight loss on the page. Chrome Music Lab's entire thesis is zero
   friction to first play.
2. **Mint a session immediately** (stem.fm). `/` → `/s/{id}`. Shareable from the first frame, and
   the upload lands into an already-existing space rather than causing a navigation.
3. **Spec the job before upload** (LALAL.AI's one good idea), but keep it to one honest control:
   a "4 stems · instruments · BPM & key · written breakdown" checklist that is all-on by default and
   whose only real function is to tell the user what the next 90 seconds buys them.
4. **Drop zone physicality.** On `dragover`: scale the target to `1.01`, raise the corner ticks
   outward by 3px, and add a `color-mix(in oklch, var(--accent) 12%, transparent)` wash — 160 ms
   `ease-out`. On drop, the ticks snap inward to bracket the filename. Keep the tape-deck corner
   ticks; they're the best existing idea in the current design.
5. **The example shelf becomes the demo, not the fallback.** Reframe the heading from "or audition
   an example" (apologetic) to something that asserts they're the fast path.
6. **Micro-honesty as a credibility signal.** Show the real constraint inline with real numbers:
   "CPU inference, single worker — ~1 song at a time, ~90 s for 3 minutes of audio." Naming your own
   capacity plan reads as engineering maturity, not as an excuse.

---

### c2 — Processing / waiting (the most under-exploited surface in the app)

This is a 60–300 second wait. Per the loading thresholds, 10 s+ requires **bar + percentage +
status text** — the current UI has all three, which is why it's competent. What it lacks is a
reason to watch.

1. **Reveal results as they land, don't gate on completion.** The pipeline is
   `Loading → Demucs → Encoding → Classifier → BPM/key → LLM`. BPM and key are cheap and could run
   *first*, off the mix, before Demucs. Then the wait has content: the key and BPM chips fill in at
   ~5 s, the mixed waveform paints at ~8 s, then stems land **one at a time** with a stagger as
   Demucs finishes each — vocals, drums, bass, other. This is stem.fm's `animateOneLoader` …
   `animateFiveLoader` pattern, and it converts dead time into a show. Streaming partial results
   measurably cuts perceived wait (reported 55–70% in AI-UI testing) even at identical total time.
2. **Use the decelerating-ribbing trick.** Harrison et al. measured an 11–12% reduction in perceived
   duration from a progress bar with ribbing animating **backwards and decelerating**. Fifteen lines
   of CSS on a bar you already have. Also: progress should move fast early and slow late — never
   linear.
3. **Animate the bar with `transform: scaleX()`, not `width`.** The current
   `transition-all duration-700` on `width` triggers layout every tick.
4. **Make the stage list a real state machine.** Each stage gets `pending → active → done` with an
   elapsed time that freezes on completion (`Separating stems — 71.2 s`). Showing per-stage timings
   is simultaneously the most useful and most credible thing on the screen; it is what an engineer
   would want and what a recruiter reads as instrumentation.
5. **Kill the fake EQ bars.** They animate on a fixed 0.9 s sine regardless of audio. In an audio
   *analysis* product, a lying visualizer is a self-inflicted credibility wound. Replace with
   something honest: a slow scan line sweeping the mixed waveform in time with the stage that's
   actually running, or the real mixed-waveform envelope drawing itself left-to-right.
6. **Teach during the wait, at low cost.** One rotating line of genuinely interesting technical
   copy, tied to the current stage: *"Demucs is a U-Net over the spectrogram + waveform — it's the
   slow part because it processes the whole track four times, once per source."* This is
   Ciechanowski's stance in miniature, and it costs nothing.
7. **Do not block the tab.** For 3+ minute waits, offer "notify me" via the Notifications API or
   just make the tab title tick (`⟳ 42% · decomposition`). Guidance is explicit that for operations
   over 2–3 minutes, holding someone on a dedicated waiting screen is counterproductive.

---

### c3 — Results console

**The single highest-value addition: the activity ribbon.**

Under each stem lane (or as a stacked band beneath the transport), render a thin horizontal ribbon
per detected instrument, shaded by that instrument's confidence in each 5-second window. This:
- answers the app's own flagship chat question ("when do the vocals come in?") *visually*;
- turns four static percentages into a map of the song;
- is the strongest possible ML-credibility artifact, because it exposes the model's time-resolved
  output rather than a summary statistic;
- costs almost nothing — the per-window scores already exist inside the aggregation step.

Apple validated this exact pattern at WWDC 2026: their Music Understanding framework exposes
instrument `activity` as `TimedValue` floats over time and calls it *"a great source to drive
audio-reactive animations."* Render it as a per-window opacity ramp on a single canvas row, not as
individual DOM nodes.

**Other console recommendations:**

1. **Open on the mix; fan out on demand** (Neural Mix). The results page's first frame is one
   waveform at ~96px. A prominent control (or the automatic reveal animation) splits it into four
   56px lanes with a **50 ms stagger, 420 ms duration, `cubic-bezier(0.22, 1, 0.36, 1)`**. This is
   the hero moment; it is the only place in the app that deserves a >300 ms animation.
2. **Hold-to-solo.** Press-and-hold a stem's `S` for momentary solo that releases on pointer-up;
   click for latching solo. This is a hardware-console behavior, it is instantly discoverable once
   found, and it is the most *tactile* single interaction you can add. (Rauno: lightweight actions
   trigger mid-gesture; destructive ones wait for release.)
3. **Additive solo + a separate Monitor** (Stemdeck). Multiple solos stay audible; `Monitor` isolates
   one and clears the rest. Mute supersedes solo, per console convention.
4. **Real faders, not just M/S buttons.** 1:1 drag, double-click resets to unity, wheel = fine,
   `Shift`+wheel = coarse (Stemdeck). Extend the hit tolerance well beyond the fader's visual bounds
   and use pointer capture so the drag survives leaving the element (Rauno's slider-tolerance note).
5. **Real per-stem meters, driven by the precomputed envelope** indexed by playhead time — with peak
   hold and slow falloff. Because the data is precomputed, the meter can also render a *ghost* of
   the whole track, so you can see the shape of the vocal performance while paused. No live
   AnalyserNode required, no CORS-taint failure mode, no jitter.

   **Tuning constants**, borrowed from `audioMotion-analyzer` (a mature production visualizer) —
   these are the most directly reusable numbers found in this research:
   `gravity: 3.8` (peaks fall to zero in **~750 ms** on a 1080px canvas) · `peakHoldTime: 500ms` ·
   `peakFadeTime: 750ms`. Note it ships `smoothing: 0.5`, *below* the AnalyserNode spec default of
   0.8, because it pairs low smoothing with a large FFT and its own peak-decay system.

   If you do add a live spectrum view, four moves separate "reads as music" from "reads as noise",
   and only the first is obvious:
   (i) **log or perceptual frequency axis** — FFT bins are linear, hearing is logarithmic, so a raw
   linear plot "looks like a downhill" because nearly every bin sits in the inaudible high end;
   audioMotion defaults to `'log'` over 20 Hz–22 kHz and also offers **bark** and **mel** scales
   (and you already compute mel spectrograms server-side, which is a nice consistency);
   (ii) **log amplitude** — set `minDecibels`/`maxDecibels` to your actual material's dynamic range
   instead of leaving the defaults, or the display either flatlines or clips;
   (iii) **perceptual weighting** (A/B/C/D or ITU-R 468) applied to the *visualization only* — this
   is the "why do the bass bars always dominate" fix;
   (iv) **asymmetric attack/decay** — `smoothingTimeConstant` smooths rises and falls *equally*,
   which is musically wrong. Transients want fast attack, slow release; that is what the
   gravity/peak-hold system above provides.
   Recommended `fftSize` for music visualization is **4096 or 8192** (audioMotion defaults to 8192);
   drop to **128** on mobile and cap the bar count. Allocate the `Uint8Array` **once** and reuse it;
   drive with rAF, never `setInterval`. Bin→Hz is `sampleRate / fftSize` (≈21.5 Hz per bin at
   44.1 kHz / 2048).

   And the honest note: amplitude alone never syncs convincingly to music — **beat/onset detection
   does**, and you already have the BPM. Drive discrete visual events off the beat grid and
   continuous ones off the envelope.
6. **Beat-snapped seeking.** You have the BPM. `[` / `]` jump one bar; holding `Shift` while
   scrubbing snaps the playhead to the beat grid. This is a genuine fun-and-credible feature: it is
   only possible because the analysis worked, and it feels great.
7. **Keyboard completeness — the cheapest credibility signal that exists.**
   `Space` play/pause · `1–4` solo · `Shift+1–4` mute · `←/→` ±5 s · `[ ]` bar jumps ·
   `0` return to start · `?` shortcut sheet. Per Emil's frequency rule and Rauno's note on
   high-frequency interactions, **none of these should animate.**
8. **Demote muted stems properly.** Current: `opacity: 0.3` on a full-size lane. Better: desaturate
   the waveform to grey *and* drop the lane height to ~70% with a 200 ms transition, so the mix's
   visual weight actually reflects what you're hearing.
9. **Confidence display: keep numbers, add calibration.** Keep the percentage (it's honest), but pair
   it with plain-language bands and never hide low-confidence results. Below threshold, say so
   explicitly rather than omitting — the current empty state already does this well and should be
   kept. Never invent a label (the Wrapped 2024 lesson).
10. **Keep the raw JSON toggle and the model provenance.** Add a small badge: model, checkpoint,
    macro-F1 0.8045 on OpenMIC-2018, aggregation = top-3-mean, threshold per class. This is the
    single line most likely to matter to an ML reader, and it belongs on the results page, not just
    the README.
11. **Chromatic identity from the analysis** (optional, high-payoff). Derive the page's ambient glow
    hue from the detected key using a circle-of-fifths mapping in OKLCH (fixed L and C, hue steps of
    30°). Implement as two stacked `radial-gradient` layers crossfading (stem.fm's technique) with
    the hue in an `@property`-registered `<angle>`/`<color>` so it interpolates. **Label it
    honestly** — "hue mapped from key, after Scriabin's colour wheel" — which converts a decorative
    flourish into a small piece of music history. Keep the four stem colors *fixed*; they are the
    taxonomy, not decoration.

---

### c4 — Chat panel

The terminal styling and the `· consulted stem activity: vocals` trace line are the best-designed
parts of the current app. They are a working implementation of the 2026 grounding-UI consensus:
*citations are the user-visible proof of grounding.* Elevate rather than replace.

1. **Make timestamps live.** When the agent says "vocals enter around 0:23," render `0:23` as a chip
   that seeks the transport and pulses the vocals lane. This is the "deep-link to source passage"
   citation pattern applied to time instead of documents, and it collapses verification from a
   minute to two seconds. It is also, straightforwardly, the most delightful single interaction
   available in this app.
2. **Make the trace hoverable, then dismissible.** Hover a trace chip → a small popover showing the
   actual tool arguments and returned values. Use the **Popover API** (Baseline Newly available
   2025-01-27) so you get the top layer, light dismiss, and focus management free.
3. **Highlight the ribbon while answering.** While the agent consults `get_stem_activity(vocals)`,
   briefly outline the vocals ribbon. You are showing the model *pointing at its evidence*.
4. **Stream the reply.** Streaming cuts perceived wait 55–70% at identical total time. The current
   `▮ checking the analysis…` is fine but a token stream is strictly better.
5. **Starters should change based on the analysis.** If the classifier found no guitar, don't offer
   "when does the guitar come in?" Generate starters from the actual result. Cheap, and it makes
   the panel feel like it knows what it's looking at.
6. **`field-sizing: content`** on the input for an auto-growing composer (Baseline Newly available
   **2026-06-16** — six weeks old at time of writing, so pair with `min/max-width` and let it
   degrade to a fixed field).

---

### c5 — Cross-cutting design system

**Motion tokens** (curves measured off stem.fm, cross-checked against Emil Kowalski's published
standards — his own out-quint is `cubic-bezier(0.23, 1, 0.32, 1)`, effectively the same curve):
```css
:root {
  --ease-out:     cubic-bezier(0.22, 1, 0.36, 1);    /* out-quint — default */
  --ease-out-exp: cubic-bezier(0.16, 1, 0.3, 1);     /* bigger reveals */
  --ease-in-out:  cubic-bezier(0.77, 0, 0.175, 1);   /* on-screen movement */
  --ease-drawer:  cubic-bezier(0.32, 0.72, 0, 1);    /* iOS sheet curve (Vaul) */
  --dur-press: 120ms;  --dur-fast: 160ms;  --dur: 220ms;
  --dur-panel: 320ms;  --dur-hero: 420ms;
  --travel: 8px;  --stagger: 50ms;
}
@media (prefers-reduced-motion: reduce) {
  :root { --dur-press:1ms; --dur-fast:1ms; --dur:1ms; --dur-panel:1ms; --dur-hero:1ms; --travel:0px; }
}
```
Duration bands to keep honest, from Emil's standards: button press **100–160 ms** · tooltips
**125–200 ms** · dropdowns **150–250 ms** · modals and drawers **200–500 ms**. Adopt M3's structural
split too: **spatial** properties (position, size, shape) may overshoot slightly; **effects**
(color, opacity) are always critically damped with zero bounce.
Driving *every* duration and translate distance through tokens means one media query neutralizes
the whole system — and critically, animations still **run and complete**, so `animationend` and
React transition callbacks still fire. A blanket `animation: none !important` breaks those and
leaves UI stuck.

**Rules to apply:**
- **Never `ease-in` on UI.** `ease-out` at 200 ms *feels* faster than `ease-in` at 200 ms.
- **UI motion stays under 300 ms** — with exactly one exception here, the 420 ms fan-out.
- **Never `scale(0)`** — enter from `scale(0.97)` + `opacity: 0`.
- **Press feedback:** `transform: scale(0.97)` on `:active`, 120–160 ms `ease-out`. Applies to the
  transport button, M/S, and the example cards.
- **Only animate `transform` and `opacity`.** Current violations: the progress bar (`width`), the
  instrument confidence bars (`width`, and `duration-1000` is far too slow — use `scaleX` at 320 ms
  with a `transform-origin: left`).
- **Stagger 30–80 ms.** Current `rise` delays are 100/120/200/240/300/350/400 ms — roughly 3–5×
  too slow, which is why the page feels like it's assembling rather than arriving. Compress to
  50 ms steps, cap total at ~250 ms.
- **Prefer transitions over keyframes for anything retargetable** — transitions interrupt and
  retarget from the current value; keyframes restart from zero. Use `@starting-style` (Baseline
  since 2024-08-06) for entry animations without a mount flag.
- **Springs only where there's a gesture.** Faders and any drag; `{ type: "spring", visualDuration:
  0.4, bounce: 0.2 }` in the Apple-style parameterization (use `visualDuration`, not `duration` —
  it targets when the spring *appears* to land rather than when it mathematically settles), or
  `mass: 1, tension: 170, friction: 26` (react-spring default). Keep bounce ≤ 0.3. Use Motion's
  **`inertia`** type for fling/scrub release. Everything else uses the easing tokens. Note the
  tradeoff if you go the CSS `linear()` route instead: a `linear()` spring **cannot absorb existing
  velocity** on interruption, which is exactly what you want a fader drag to do.
- **Gate hover motion:** `@media (hover: hover) and (pointer: fine)` — touch fires false hovers.
- **Use "boop", not sustained hover.** For flourishes (a stem lane acknowledging a solo), apply a
  transform that **unsets itself after a short interval** whether or not the pointer stayed. Bursts
  read as alive; lingering states read as sticky.
- **Threshold scales with consequence** (Rauno). A volume nudge applies live; a scrub commits on
  release; anything destructive waits for gesture end regardless of distance travelled. And apply
  drag deltas *immediately*, before any animation — a fader that animates toward your finger feels
  broken.

**Color.** Move the palette to **OKLCH** (Baseline *Widely available* since 2025-11-09) and derive
states with `color-mix(in oklch, …)` (also Widely available) instead of hand-picking tints. The
existing stem hues are good and worth keeping — just re-express them so a lightness ramp is
perceptually even. Register the accent as an `@property` `<color>` so it can actually interpolate.

**Typography.** Two weights maximum, following TE. The current Bricolage Grotesque display + mono
pairing is a real point of view — keep it. But the 9–10px `--muted` metadata is too small to be
comfortable regardless of contrast (roughly 5.3:1 by my calculation of `#8d8a80` on `#131318`,
which passes AA — the problem is size, not contrast). Floor it at 11–12px. Consider `te-20`-style
proportional scaling (§B2) as the layout system so the console scales as an object.

**Shareable artifact.** Generate a 1200×630 OG card per result with Satori / Next.js
`ImageResponse`: four stem waveforms stacked in their channel colors, key + BPM + top instruments,
the track title. Wrapped's lesson is that shareability is a primary design driver, not an
afterthought — and for a portfolio project, a good share card is what actually travels.
(Satori supports inline flexbox only — no Tailwind classes — and is CPU-intensive, so cache it.)

---

## (d) Pitfalls

**Audio-specific**
1. **Fake visualizers.** The current `.eqbar` animates identically whether the audio is a drum solo
   or silence. In a product whose premise is accurate listening, this is the worst possible lie to
   tell. Either drive it from real data or remove it.
2. **Never autoplay.** Chrome creates an `AudioContext` in the `suspended` state without a user
   gesture; you must `resume()` from a gesture, and iOS is strictest. TE's product pages get this
   exactly right: click-to-play *with sound*, never automatic.
3. **Do not sync stems with multiple `<audio>` elements.** The spec gap for
   `MediaElementAudioSourceNode` timing is real and documented; reported behavior ranges from tight
   in Chromium to audible flanging in Firefox to multi-millisecond slip in Safari. *(This ordering
   comes from W3C spec-issue discussion, not a formal conformance report — treat the ranking as
   directional, the gap as real.)*
4. **`createMediaElementSource(el)` can be called only once per element** — a second call throws
   `InvalidStateError`, and React StrictMode's dev double-mount *will* trigger it. The same class of
   bug already bit this codebase once with WaveSurfer instances.
5. **Cross-origin taint silently zeroes the analyser.** Without CORS headers plus
   `crossOrigin="anonymous"`, you get a visualizer that runs and displays nothing. #1 cause of "my
   visualizer is dead."
6. **Don't add UI sound effects.** In a music app the content is the sound; interface chirps compete
   with it. Restraint here is not timidity, it's correctness.
7. **340 MB of decoded PCM for four stems** — cap duration, decode mono where possible,
   `ctx.close()` on unmount.

**Motion and accessibility**
8. **Continuous meter motion is unsolicited motion.** WCAG **2.2.2 Pause, Stop, Hide (Level A)**
   covers anything that moves, blinks, scrolls or auto-updates, starts **automatically**, lasts
   **more than 5 seconds**, and runs **in parallel with other content**. A continuously-running
   visualizer sitting alongside a stem list is squarely in scope. The exemption is content where the
   motion is *essential to the activity* — a visualizer that *is* the activity can argue
   essentiality; ambient motion behind other UI cannot. Safest read: give it an off switch. Under
   `prefers-reduced-motion: reduce`, freeze it or slow it heavily (`smoothingTimeConstant` → 0.95,
   throttle to ~10 fps). A linear scrolling playhead is fine — it conveys information and is slow.
9. **Run the flash-rate arithmetic before shipping a beat-synced strobe.** WCAG **2.3.1 Three
   Flashes (Level A)**: nothing may flash more than **3 times in any 1-second period** unless below
   the general and red flash thresholds, where a flash is a pair of opposing relative-luminance
   changes of **≥25%** of max screen luminance. Exemption if the flashing area is **≤25% of any
   10-degree field**, operationalized as a **341 × 256 px block**. Do the math for music: a
   four-on-the-floor kick at 128 BPM is **2.13 flashes/sec — safe**; a 16th-note hi-hat at 128 BPM
   is **8.5/sec — a violation** if the flash is large and high-contrast. Keep beat-reactive flashing
   either small, low-contrast, or locked to the downbeat.
10. **Do not build a slow ambient pulse at ~0.2 Hz.** Apple names one oscillation per five seconds
    as the specific frequency to avoid (WWDC23 session 10078). If you want an ambient "breathing"
    glow, run it at beat rate (1–3 Hz), or keep amplitude low and the element semitransparent.
11. **The `opacity: 0` trap.** If a reveal animation starts at `opacity: 0` and reduced motion sets
    `animation: none`, the content is invisible. Make motion the enhancement over a complete,
    correct, static layout — never the mechanism that makes content appear. Relatedly: a blanket
    `animation: none !important` breaks `animationend`/`transitionend` handlers and leaves UI stuck.
12. **Reduced motion means reduced, not zero.** Keep opacity crossfades (under ~200 ms), color
    changes, focus rings, and sub-10px shifts; drop parallax, large slide-ins, zoom transitions,
    auto-playing carousels, and rotation. Val Head's three risk factors in order: **relative size of
    the movement**, **mismatched direction/speed between layers** (parallax is "almost universally
    listed as a trigger"), and **distance covered**. Directional slides are the highest-risk
    pattern; morphs and crossfades are comparatively safe.
13. **The doom flicker.** If hovering an element moves it, it can move out from under the cursor →
    unhover → snap back → rehover. The fix is structural: **the hover target must not move; a child
    moves.** Same class of bug: hover states that change `border`, `padding` or `height` shift layout
    and cause flicker — express them as `transform` / `box-shadow` / `outline`, or pre-reserve the
    space. And use transitions, not `@keyframes`, for hover — users interrupt hover constantly.
14. **Stem color must not be the only channel.** `#f5a623` / `#e85d4a` / `#9d7bff` / `#3ecfb2` are a
    good set, but pair color with a persistent label and stable vertical position so the mapping
    survives color-vision deficiency and greyscale printing.
15. **Don't over-stagger.** 100–400 ms delays read as sluggish assembly.
16. **When everything moves, nothing stands out.** Motion is a scarce attention resource; spending
    it uniformly bankrupts hierarchy. NN/g's test: does this animation provide *feedback*,
    communicate a *state change*, aid *spatial navigation*, or act as a *signifier*? If not, cut it.

**Platform**
17. **View transitions freeze `<canvas>`.** A view transition captures *static snapshots* — an
    animating waveform or spectrum becomes a still image for the entire transition. Use view
    transitions for landing → results (nothing is playing yet); never for a transition *during*
    playback, where 400 ms of frozen waveform reads as a stutter.
18. **Skip cross-document view transitions.** Chrome 126 / Safari 18.2, **Firefox: absent** (not
    flagged, not preview — simply not implemented). There is also a silent **4-second timeout**
    measured from navigation start, including TTFB, that aborts without a console warning.
19. **Don't polyfill scroll-driven animations.** Chrome 115, Safari 26, **Firefox still Nightly-only
    behind `layout.css.scroll-driven-animations.enabled`** as of Firefox 153 (2026-07-21), despite
    being an Interop 2026 focus. The `flackr/scroll-timeline` polyfill runs everything on the main
    thread, i.e. it serves Firefox users a *worse-performing* version of an effect Chrome users get
    free. Use `@supports (animation-timeline: view())` and let Firefox see static content. And
    never put comprehension-critical information in a scroll animation.
20. **React `<ViewTransition>` gotchas.** It is Canary/experimental only, though Next.js App Router
    ships React canary internally (`experimental: { viewTransition: true }`). Only Transitions,
    `<Suspense>` reveals, and `useDeferredValue` activate it — plain `setState` does nothing. `name`
    must be globally unique or React throws. **Without `default="none"`, every ViewTransition on the
    page animates on every transition.** And React does *not* handle reduced motion for you.
21. **Scroll-state container queries and `interpolate-size`/`calc-size()` are Chrome-only** and not
    in Interop 2026. Decorative enhancement at most.
22. **Web haptics on iOS is dead.** The `<input type=checkbox switch>` label-click trick worked from
    iOS 17.4 through 26.4; **Apple patched it in iOS 26.5.** Do not build tactility on it.
23. **`wavesurfer-multitrack` (last publish 2024-07-16) and `waveform-playlist` (2022-02-26) are
    unmaintained.** Don't adopt either.

**Positioning**
24. **Generated whimsy reads as slop the moment accuracy slips** (Wrapped 2024). The LLM blurb is
    the highest-risk surface in this app: keep it short, keep it grounded, keep the validator, and
    keep the "written by X, grounded in the analysis" attribution line — that line is doing real
    work.
25. **One easter egg, not five.** A single unnecessary piece of craft signals confidence; a
    scattering of them signals an unserious tool. For a portfolio piece read in under 90 seconds by
    a recruiter, ambiguity about seriousness is expensive.
26. **Don't let playfulness delay first value.** Playful portfolio sites measurably beat static
    grids on stickiness but lose on recruiter speed. Keep the fastest possible path to a working
    demo — which is why the hero should *be* a demo.

---

## (e) Sources

**Products and design systems (measured or documented)**
- stem.fm — https://www.stem.fm · Stem Player background — https://en.wikipedia.org/wiki/Stem_Player
- Moises — https://moises.ai · ADA writeup — https://moises.ai/newsroom/awards-recognition/apple-design-awards-nomination/
- LALAL.AI — https://www.lalal.ai
- djay Neural Mix — https://help.algoriddim.com/user-manual/djay-pro-mac/neural-mix/tracks · https://www.audioshake.ai/case-studies/algoriddim
- Ableton Live 12.3 — https://cdm.link/ableton-live-12-3-guide/
- Suno Song Editor — https://suno.com/blog/songeditor · stems — https://suno.com/release-notes/advanced-stems
- stemplayer-js — https://github.com/stemplayer-js/stemplayer-js
- Stemdeck — https://github.com/stemdeckapp/stemdeck
- Teenage Engineering — https://teenage.engineering · https://teenage.engineering/products/ep-133 · https://teenage.engineering/products/field-system
- Endel — https://endel.io
- Spotify Wrapped 2025 — https://spotifynews.substack.com/p/designing-2025-wrapped-turning-a · https://newsroom.spotify.com/2025-12-03/wrapped-marketing-campaign/ · analysis https://uxplaybook.org/articles/spotify-wrapped-ux-design-lessons · 2024 backlash https://www.forbes.com/sites/danidiplacido/2024/12/05/spotify-wrapped-2024-backlash-controversy-and-memes/ · Daylist https://www.behance.net/gallery/180097711/Spotify-Daylist
- Ableton Learning Music / Learning Synths — https://learningmusic.ableton.com · https://learningsynths.ableton.com
- Chrome Music Lab — https://musiclab.chromeexperiments.com
- Blob Opera (David Li × Google Arts & Culture) — https://artsandculture.google.com/experiment/blob-opera/AAHWrq360NcGbw
- Poolsuite (ex-Poolside.fm) — https://poolsuite.net
- Every Noise at Once — https://everynoise.com · https://en.wikipedia.org/wiki/Every_Noise_at_Once
- Bruno Simon — https://bruno-simon.com · Three.js Journey https://threejs-journey.com
- Tim Holman — https://tholman.com · "Fun.css", CSSconf EU 2015 — https://www.youtube.com/watch?v=5HP6k43T0yM

**Craft, motion, and delight literature**
- Emil Kowalski — "Great animations" https://emilkowal.ski/ui/great-animations · "7 Practical Animation Tips" https://emilkowal.ski/ui/7-practical-animation-tips · "You Don't Need Animations" https://emilkowal.ski/ui/you-dont-need-animations · "Good vs Great Animations" https://emilkowal.ski/ui/good-vs-great-animations · "Building a Drawer Component" https://emilkowal.ski/ui/building-a-drawer-component · standards https://github.com/emilkowalski/skills/blob/main/skills/review-animations/STANDARDS.md · course https://animations.dev/
- Rauno Freiberg — "Invisible details of interaction design" https://rauno.me/craft/interaction-design · republished https://every.to/p/invisible-details-of-interaction-design · course https://devouringdetails.com/
- Josh Comeau — CSS transitions https://www.joshwcomeau.com/animation/css-transitions/ · spring physics https://www.joshwcomeau.com/animation/a-friendly-introduction-to-spring-physics/ · `linear()` springs https://www.joshwcomeau.com/animation/linear-timing-function/ · "A Million Little Secrets" https://www.joshwcomeau.com/blog/whimsical-animations/
- Motion (ex-Framer Motion) — https://motion.dev/docs/react-transitions · https://motion.dev/docs/spring · https://motion.dev/changelog
- Material 3 Expressive spring tokens (source of record) — https://raw.githubusercontent.com/androidx/androidx/androidx-main/compose/material3/material3/src/commonMain/kotlin/androidx/compose/material3/tokens/ExpressiveMotionTokens.kt · Standard https://…/StandardMotionTokens.kt · legacy durations https://cs.android.com/androidx/platform/frameworks/support/+/androidx-main:compose/material3/material3/src/commonMain/kotlin/androidx/compose/material3/tokens/MotionTokens.kt
- Apple — WWDC23 session 10078, "Design considerations for vision and motion" https://developer.apple.com/videos/play/wwdc2023/10078/ · HIG Motion https://developer.apple.com/design/human-interface-guidelines/motion
- RAIL response thresholds — https://web.dev/articles/rail
- Harrison, Amento, Kuznetsov & Bell, "Rethinking the Progress Bar", UIST '07 — https://chrisharrison.net/projects/progressbars/ProgBarHarrison.pdf · follow-up https://www.chrisharrison.net/projects/progressbars2/ProgressBarsHarrison.pdf
- Smart Interface Design Patterns, loading & progress UX — https://smart-interface-design-patterns.com/articles/designing-better-loading-progress-ux/
- NN/g, "Three Pillars of User Delight" — https://www.nngroup.com/articles/pillars-user-delight/ · "A Theory of User Delight" — https://www.nngroup.com/articles/theory-user-delight/ · "The Role of Animation and Motion in UX" — https://www.nngroup.com/articles/animation-purpose-ux/
- Val Head, "Designing Safer Web Animation For Motion Sensitivity", A List Apart — https://alistapart.com/article/designing-safer-web-animation-for-motion-sensitivity/
- Sophie Paxton, "Stop Gratuitous UI Animation" — https://medium.com/@sophie_paxtonUX/stop-gratuitous-ui-animation-9ece9aa9eb97

**Playhead, scrubbing, and visualizer engineering**
- Ralph Holzmann, "Creating a jank-free media progress bar" (measured `timeupdate` rates) — https://medium.com/@ralphholzmann/creating-a-jank-free-media-progress-bar-3f31db3d1c43
- Fender Engineering, "Near-Realtime Animations with Synchronized Audio in JavaScript" — https://medium.com/fender-engineering/near-realtime-animations-with-synchronized-audio-in-javascript-6d845afcf1c5
- peaks.js PR #206 (rAF playhead + synthetic `timeupdate`) — https://github.com/bbc/peaks.js/pull/206/files
- audioMotion-analyzer (gravity / peak-hold / weighting / log-bark-mel scales) — https://github.com/hvianna/audioMotion-analyzer
- Boris Smus, *Web Audio API*, ch. 5 — https://webaudioapi.com/book/Web_Audio_API_Boris_Smus_html/ch05.html
- addpipe, `getByteFrequencyData` guide (2024-12-11) — https://blog.addpipe.com/understanding-audio-frequency-analysis-in-javascript-a-guide-to-using-analysernode-and-getbytefrequencydata/
- Codrops, audio-reactive particles in three.js — https://tympanus.net/codrops/2023/12/19/creating-audio-reactive-visuals-with-dynamic-particles-in-three-js/
- gl-spectrogram / Spectro (WebGL mel spectrogram) — https://github.com/dy/gl-spectrogram · https://calebj0seph.github.io/spectro/
- Logic Pro "Catch" + "Scroll in Play" — https://support.apple.com/guide/logicpro/control-windows-using-catch-modes-lgcp5cbf1727/mac

**Credible-and-delightful explainers**
- Bartosz Ciechanowski, "Sound" — https://ciechanow.ski/sound/
- CNN Explainer — https://poloclub.github.io/cnn-explainer/ · paper https://poloclub.github.io/papers/20-vis-cnnexplainer.pdf
- Transformer Explainer — https://poloclub.github.io/transformer-explainer/ · https://arxiv.org/html/2408.04619v1

**Platform data (queried directly, 2026-08-02)**
- Baseline API — https://api.webstatus.dev/v1/features · MDN BCD — https://bcd.developer.mozilla.org/bcd/api/v0/current/ · Firefox releases — https://product-details.mozilla.org/1.0/firefox_history_major_releases.json
- MDN: View Transition API — https://developer.mozilla.org/en-US/docs/Web/API/View_Transition_API · AnalyserNode — https://developer.mozilla.org/en-US/docs/Web/API/AnalyserNode · scroll-driven animations — https://developer.mozilla.org/en-US/docs/Web/CSS/Guides/Scroll-driven_animations · Firefox experimental features — https://developer.mozilla.org/en-US/docs/Mozilla/Firefox/Experimental_features
- Next.js view transitions — https://nextjs.org/docs/app/guides/view-transitions · React `<ViewTransition>` — https://react.dev/reference/react/ViewTransition
- web.dev, Interop 2026 — https://web.dev/blog/interop-2026 · "A Tale of Two Clocks" — https://web.dev/articles/audio-scheduling
- CSS-Tricks, cross-document view transition gotchas — https://css-tricks.com/cross-document-view-transitions-part-1/
- wavesurfer.js docs — https://wavesurfer.xyz/docs/ · performance — https://wavesurfer.xyz/docs/performance
- BBC audiowaveform — https://github.com/bbc/audiowaveform · peaks.js — https://github.com/bbc/peaks.js · waveform-data.js — https://github.com/bbc/waveform-data.js
- Chrome autoplay policy — https://developer.chrome.com/blog/autoplay
- CSS-Tricks, reduced motion — https://css-tricks.com/nuking-motion-with-prefers-reduced-motion/ · WCAG 2.2.2 — https://hidde.blog/meeting-2-22-pause-stop-hide-with-prefers-reduced-motion/
- Apple Music Understanding (WWDC 2026) — https://developer.apple.com/videos/play/wwdc2026/253/ · summary https://blakecrosley.com/blog/music-understanding-framework-ios-27
- iOS web haptics (and the 26.5 patch) — https://github.com/tijnjh/ios-haptics
- OKLCH — https://evilmartians.com/chronicles/oklch-in-css-why-quit-rgb-hsl
- Satori / Next.js `ImageResponse` — https://og-image.org/docs/dynamic-og
- Grounding/citation UI patterns — https://www.shapeof.ai/patterns/citations · https://aiuxplayground.com/teardowns/perplexity/citations/
- Scriabin's colour wheel — https://en.wikipedia.org/wiki/Chromesthesia

**Library maintenance status (npm, checked 2026-08-02)**

| Library | Latest | Published | Read |
|---|---|---|---|
| `wavesurfer.js` | 7.12.11 | 2026-07-17 | actively maintained |
| `@wavesurfer/react` | 1.0.12 | 2025-12-04 | maintained |
| `peaks.js` | 4.0.0 | 2025-08-30 | maintained |
| `waveform-data` | 4.5.2 | 2025-06-07 | maintained |
| `audiomotion-analyzer` | 4.5.4 | 2026-01-09 | maintained |
| `wavesurfer-multitrack` | 0.4.12 | 2024-07-16 | **stale — avoid** |
| `waveform-playlist` | 4.3.3 | 2022-02-26 | **abandoned** |

---

## Appendix — Baseline status table (verified 2026-08-02)

Definitions: **Newly** = shipped in all core browsers as of the low date. **Widely** = 30 months
after that. **Limited** = at least one core engine missing.

| Feature | Baseline | Since | Chrome | Firefox | Safari |
|---|---|---|---|---|---|
| View transitions (same-doc) | **Newly** | 2025-10-14 | 111 | 144 | 18 |
| `view-transition-class` | **Newly** | 2025-10-14 | 125 | 144 | 18.2 |
| View transition **types** | cross-engine | 2026-01-13 | 125 | 147 | 18.2 |
| Cross-document `@view-transition` | **Limited** | — | 126 | none | 18.2 |
| Scroll-driven animations | **Limited** | — | 115 | Nightly flag | 26 |
| `animation-trigger` | **Limited** | — | 146 | — | — |
| `@property` | **Newly** | 2024-07-09 | 85 | 128 | 16.4 |
| Container size queries + `cqi` | **Widely** | 2025-08-14 | 105 | 110 | 16 |
| Container **style** queries | **Newly** | 2026-05-19 | 111 | 151 | 18 |
| `container-type: scroll-state` | **Limited** | — | 133 | — | — |
| Anchor positioning (group) | **Limited** | — | 125–129 | 147 | 26 |
| `@starting-style` | **Newly** | 2024-08-06 | 117 | 129 | 17.5 |
| `transition-behavior: allow-discrete` | **Newly** | 2024-08-06 | 117 | 129 | 17.4 |
| Popover API | **Newly** | 2025-01-27 | 114 | 125 | 17 (iOS 18.3) |
| `text-wrap: balance` | **Newly** | 2024-05-13 | 114 | 121 | 17.5 |
| `text-wrap: pretty` | **Limited** | — | 130 | — | 26 |
| `color-mix()` | **Widely** | 2025-11-09 | 111 | 113 | 16.2 |
| Oklab / OKLCH | **Widely** | 2025-11-09 | 111 | 113 | 15.4 |
| `light-dark()` | **Newly** | 2024-05-13 | 123 | 120 | 17.5 |
| `field-sizing` | **Newly** | 2026-06-16 | 123 | 152 | 26.2 |
| `interpolate-size` / `calc-size()` | **Limited** | — | 129 | — | — |
| `linear()` easing | **Widely** | 2026-06-11 | 113 | 112 | 17.2 |
| `animation-composition` | **Widely** | 2026-01-04 | 112 | 115 | 16 |
| `:has()` | **Widely** | 2026-06-19 | 105 | 121 | 15.4 |
| OffscreenCanvas | **Widely** | 2025-09-27 | 69 | 105 | 16.4 |
| WebGPU | **Limited** | — | 144 | 141 partial | 26 |
| Web Audio / AudioWorklet | **Widely** | 2023-10-26 | 66 | 76 | 14.1 |
| `prefers-reduced-motion` | **Widely** | 2022-07-15 | 74 | 63 | 10.1 |

**Verdicts for this app.** *Build on:* `@property`, container size queries + `cqi`, `:has()`,
OKLCH + `color-mix()`, `linear()`, `@starting-style` + `allow-discrete`, Popover API, one
`AudioContext` + `AudioBufferSourceNode`s, one `AnalyserNode` per stem at `fftSize: 512` polled
once per rAF, server-precomputed 8-bit `.dat` peaks. *Enhancement only:* same-document view
transitions for landing → results (never mid-playback), scroll reveals behind `@supports`, style
queries for density, `text-wrap: pretty`, `interpolate-size`. *Skip:* cross-document view
transitions, the scroll-timeline polyfill, scroll-state container queries, WebGPU.

**Known gaps in this research.**
- Moises' in-app mixer (fader layout, per-stem colors) is behind auth and was not inspected. Udio's
  stem view could not be verified to exist. Suno's Song Editor *visual* design was not verified.
- Learning Synths, Learning Music, Chrome Music Lab, Poolside.fm and Bruno Simon's site are heavily
  client-rendered and resisted automated inspection — descriptions here are from documentation and
  reputation, not measurement.
- The per-browser ranking of `MediaElementAudioSourceNode` sync slip comes from W3C spec-issue
  discussion rather than a conformance report; the spec gap itself is documented.
- Apple's HIG Motion page and Material 3's motion pages are both JS-rendered and unfetchable. Apple
  numbers come from WWDC23 session 10078; M3 spring tokens come from the androidx Compose source,
  which is the implementation of record but is phone-class only and may not match the docs' framing.
- The `minDecibels: -100` / `maxDecibels: -30` AnalyserNode defaults are widely reported but the
  W3C spec tables did not render for verification.
- No authoritative benchmarks were found for **OffscreenCanvas in audio waveform rendering
  specifically**. It's an obvious fit; there is no measurement behind that intuition.
- Two commonly-cited articles do not exist under those titles: "Emil Kowalski — Speed" and
  "Josh Comeau — The Magical World of Particles." The nearest real sources are listed above.
- One active correction: `blakecrosley.com/guides/design/teenage-engineering` claims TE uses
  monospace exclusively and `#ff6600`. The live site uses Univers-derived `te-20`/`te-40` and
  `#F05A24`. Several 2026 SEO posts also claim Firefox 132+ supports scroll-driven animations and
  Safari 18+; BCD says Firefox is Nightly-only and Safari is 26. Override both.
