# Deploying

The site is one Vercel project. Everything the public demo needs — the example
analyses and their stem audio — is a static file, so there is no server to keep
warm and no cold start. The only runtime dependency is an LLM endpoint for chat.

Total cost: **$0**, no card anywhere.

---

## 1. Get a Groq API key

Groq's free tier needs no credit card: 30 requests/minute, 14,400/day,
500,000 tokens/day on `llama-3.1-8b-instant`, with tool calling.

1. Sign up at [console.groq.com](https://console.groq.com).
2. **API Keys → Create API Key.** Copy it.

> Rate limits are enforced per **organization**, not per key. If you reuse this
> account for another project, both draw from the same 30 req/min — generate a
> separate key per project so you can revoke one without breaking the other,
> but know the quota is shared.

## 2. Import the repo into Vercel

1. [vercel.com/new](https://vercel.com/new) → import `Decomposition`.
2. Set **Root Directory** to `reference-app/frontend`. This matters — the repo
   root is a Python project and Vercel will not find the Next.js app otherwise.
3. Framework preset should auto-detect as **Next.js**. Leave the build and
   output settings alone.

## 3. Set environment variables

**Settings → Environment Variables**, for Production and Preview:

| Name | Value |
|---|---|
| `LLM_API_KEY` | your Groq key |
| `LLM_BASE_URL` | `https://api.groq.com/openai/v1` |
| `LLM_MODEL` | `llama-3.1-8b-instant` |

Do **not** set `BACKEND_ORIGIN` or `NEXT_PUBLIC_UPLOAD_ENABLED` in production.
Leaving them unset is what makes the site serve examples statically and show
the honest "run it locally" panel instead of a dropzone that cannot work.

## 4. Deploy and verify

Push to `main` and Vercel builds automatically. Then check the three things
that actually break:

```bash
curl -s -o /dev/null -w "%{http_code}\n" https://YOUR-APP.vercel.app/examples/manifest.json
```

```bash
curl -s -X POST https://YOUR-APP.vercel.app/api/chat/ex_867662 -H "Content-Type: application/json" -d '{"messages":[{"role":"user","content":"what instruments are in this track?"}]}'
```

The chat response must come back with `"grounded": true` and a non-empty
`trace`. An empty trace means the model answered without calling a tool, which
is the one failure mode worth watching after a provider change.

Then run the full grounding suite against production:

```bash
python evals/run_evals_http.py --base-url https://YOUR-APP.vercel.app --label groq-llama-3.1-8b
```

**Put the number this produces in the README, not the local one.** The
committed result (0/30 hallucinations, 2.7 s mean latency) was measured against
local `llama3.2:3b` via Ollama. Groq runs a different model and may score
differently — report what it actually does.

## 5. Add the URL

Put the live URL in the README hero, the GitHub repo **About** field, your
resume, and LinkedIn.

---

## Switching LLM provider

The agent talks to any OpenAI-compatible endpoint with tool calling, so
changing provider is two environment variables and a redeploy.

| | `LLM_BASE_URL` | `LLM_MODEL` |
|---|---|---|
| Groq | `https://api.groq.com/openai/v1` | `llama-3.1-8b-instant` |
| OpenRouter | `https://openrouter.ai/api/v1` | any `:free` model with tool support |
| Ollama (local) | `http://localhost:11434/v1` | `llama3.2:3b` |

OpenRouter's free tier is genuinely no-card but allows only **50 requests/day**
at a zero balance, and its free model lineup changes monthly — usable as a
manual fallback, too thin to be the primary.

If the LLM is unreachable the app degrades to an honest message and the rest of
the analysis still renders.

## Why not Hugging Face Spaces

The original plan was one all-in-one Docker Space. **Docker Spaces now require
HF PRO ($9/month)** — free personal accounts get static Spaces and up to two
ZeroGPU Gradio Spaces only. Vercel Hobby gives a real Next.js site for free with
no card, so the site moved there and analysis stayed local.

## Running the full pipeline locally

Uploads need Demucs and the 345 MB backbone — minutes of CPU and several GB of
RAM per song, which no free serverless tier provides.

```bash
cd reference-app/backend && ../../.venv/bin/uvicorn app:app --port 8000
```

```bash
cd reference-app/frontend && BACKEND_ORIGIN=http://localhost:8000 NEXT_PUBLIC_UPLOAD_ENABLED=1 npm run dev
```
