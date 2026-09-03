---
title: Decomposition
emoji: 🎚️
colorFrom: yellow
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

> **⚠️ OBSOLETE — not the deployment path.** Docker Spaces now require HF PRO
> ($9/month), so this cannot be deployed on a free account. Kept as a record of
> the approach. The live site deploys to Vercel — see
> [`DEPLOY.md`](../../DEPLOY.md).


# Deploying this Space (free, no card, public URL)

1. Create a free HF account → New Space → **Docker** SDK → CPU Basic (free).
2. From `reference-app/`, assemble the Space repo:
   - copy `backend/`, `frontend/`, `examples/`, and this `space/Dockerfile`
     (as `Dockerfile` at the repo root) into the Space
   - create `models/` containing: `beats/BEATs_iter3_plus_AS2M.pt` (352 MB),
     `ckpt_E10_stem_recalib.pt`, `ckpt_E10_beats_head.pt`
     (from `data/f1_research/`) — use `git lfs track "*.pt"` first
3. (Optional LLM) In Space settings → Variables & secrets, set
   `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY` to a free hosted open model
   (e.g. Cerebras/Groq — no card attached, so no bill possible). Without
   them the app still works via the template description fallback.
4. `git push` → live at `https://huggingface.co/spaces/<you>/decomposition`.

Notes: free CPU = 1–5 min per song (the UI shows staged progress); the Space
sleeps after ~48 h idle and cold-starts in ~1 min. No card anywhere in this
setup → hard stops, never a bill.
