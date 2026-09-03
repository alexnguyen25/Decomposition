# HF Space (Docker SDK) — runs the FastAPI backend + Next.js frontend in one
# container. Space exposes port 7860; Next.js listens there and proxies
# /api/* to uvicorn on 8000 inside the container (see next.config.ts).
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg curl ca-certificates && \
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# backend deps first (layer caching: deps change less often than code)
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# app code + model assets (copy checkpoints into space/models/ before build:
#   BEATs_iter3_plus_AS2M.pt, ckpt_E10_stem_recalib.pt, ckpt_E10_beats_head.pt)
COPY backend backend
COPY models models
COPY examples examples
COPY frontend frontend

# build the frontend (API proxied to localhost inside the container)
RUN cd frontend && npm ci && BACKEND_ORIGIN=http://localhost:8000 npm run build

ENV MODEL_DIR=/app/models \
    EXAMPLES_DIR=/app/examples \
    WORK_DIR=/tmp/jobs \
    HOME=/tmp

EXPOSE 7860
# uvicorn (internal) + next (public on 7860)
CMD sh -c "cd backend && uvicorn app:app --host 0.0.0.0 --port 8000 & \
           cd frontend && npx next start -p 7860 -H 0.0.0.0"
