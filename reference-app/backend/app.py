"""FastAPI server: upload -> async job -> poll -> results + stem audio.

Run locally:  cd reference-app/backend && ../../.venv/bin/uvicorn app:app --port 8000

Architecture notes (the parts worth learning):
- LONG JOBS: HTTP requests must return fast, but Demucs takes minutes. So
  POST /api/analyze only *starts* a job (returns job_id) and a single worker
  thread processes jobs one at a time. The client polls GET /api/jobs/{id}.
  This is the smallest possible version of the queue pattern every ML product
  uses (the "grown-up" versions swap the dict for Redis and the thread for a
  worker fleet — the shape is identical).
- ONE worker on purpose: Demucs needs ~3 GB; two concurrent jobs would blow
  the RAM budget of a free host. Serializing is the honest capacity.
- HARDENING (public internet = adversarial): size cap before reading the
  body, magic-byte sniffing (extensions lie), duration cap after decode,
  per-IP cooldown, global daily cap, TTL cleanup of outputs.
"""

import queue
import shutil
import threading
import time
import uuid
from collections import defaultdict
from pathlib import Path

import librosa
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

import settings
from pipeline import analyze

app = FastAPI(title="Decomposition API")

# CORS: locally the Next.js dev server proxies /api (no CORS needed), but a
# deployed frontend on another origin (Cloudflare Pages) needs this.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

# --- in-memory job store -----------------------------------------------------
JOBS: dict[str, dict] = {}
JOB_QUEUE: "queue.Queue[str]" = queue.Queue()
LAST_JOB_BY_IP: dict[str, float] = defaultdict(float)
DAILY = {"day": time.strftime("%Y-%m-%d"), "count": 0}

MAGIC = {b"ID3": "mp3", b"\xff\xfb": "mp3", b"\xff\xf3": "mp3",
         b"\xff\xf2": "mp3", b"RIFF": "wav", b"OggS": "ogg",
         b"fLaC": "flac"}


def _sniff(head: bytes) -> str | None:
    for magic, kind in MAGIC.items():
        if head.startswith(magic):
            return kind
    if head[4:8] == b"ftyp":                       # m4a/mp4 container
        return "m4a"
    return None


def _worker():
    """Single background thread: pull job ids, run the pipeline."""
    while True:
        job_id = JOB_QUEUE.get()
        job = JOBS[job_id]
        job.update(status="running", progress=0.01, stage="Starting")
        try:
            def cb(frac, stage):
                job["progress"], job["stage"] = round(frac, 3), stage

            out_dir = settings.WORK_DIR / job_id
            result = analyze(Path(job["upload_path"]), out_dir, cb)
            result["stems"] = {k: f"/api/files/{job_id}/{v}"
                               for k, v in result["stems"].items()}
            job.update(status="done", result=result, done_at=time.time())
        except Exception as e:                      # noqa: BLE001
            job.update(status="error", error=str(e)[:300])
        finally:
            Path(job["upload_path"]).unlink(missing_ok=True)  # delete upload


def _janitor():
    """Delete result files past their TTL — uploads must stay ephemeral."""
    while True:
        time.sleep(60)
        now = time.time()
        for job_id, job in list(JOBS.items()):
            if job.get("done_at") and now - job["done_at"] > settings.RESULT_TTL_S:
                shutil.rmtree(settings.WORK_DIR / job_id, ignore_errors=True)
                JOBS.pop(job_id, None)


threading.Thread(target=_worker, daemon=True).start()
threading.Thread(target=_janitor, daemon=True).start()


@app.post("/api/analyze")
async def create_job(request: Request, file: UploadFile):
    ip = request.client.host if request.client else "unknown"

    # -- abuse guards (cheap checks first) ------------------------------------
    if time.strftime("%Y-%m-%d") != DAILY["day"]:
        DAILY.update(day=time.strftime("%Y-%m-%d"), count=0)
    if DAILY["count"] >= settings.GLOBAL_DAILY_CAP:
        raise HTTPException(429, "Daily capacity reached — try tomorrow.")
    since = time.time() - LAST_JOB_BY_IP[ip]
    if since < settings.PER_IP_COOLDOWN_S:
        raise HTTPException(429, f"One song per {settings.PER_IP_COOLDOWN_S}s "
                                 f"— retry in {int(settings.PER_IP_COOLDOWN_S - since)}s.")
    if any(j["status"] in ("queued", "running") and j.get("ip") == ip
           for j in JOBS.values()):
        raise HTTPException(429, "You already have a job running.")

    # -- upload validation -----------------------------------------------------
    head = await file.read(12)
    kind = _sniff(head)
    if kind is None:
        raise HTTPException(400, "Not a recognized audio file "
                                 "(mp3/wav/ogg/flac/m4a).")
    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    body = head + await file.read(max_bytes + 1)
    if len(body) > max_bytes:
        raise HTTPException(413, f"File too large (max {settings.MAX_UPLOAD_MB} MB).")

    job_id = uuid.uuid4().hex[:12]
    settings.WORK_DIR.mkdir(parents=True, exist_ok=True)
    upload_path = settings.WORK_DIR / f"{job_id}_upload.{kind}"
    upload_path.write_bytes(body)

    # duration check needs a decode; do it now so bad files fail fast.
    try:
        duration = librosa.get_duration(path=str(upload_path))
    except Exception as e:                          # noqa: BLE001
        upload_path.unlink(missing_ok=True)
        raise HTTPException(400, "Could not decode audio.") from e
    if not settings.MIN_DURATION_S <= duration <= settings.MAX_DURATION_S:
        upload_path.unlink(missing_ok=True)
        raise HTTPException(400, f"Duration must be {settings.MIN_DURATION_S}"
                                 f"–{settings.MAX_DURATION_S}s "
                                 f"(got {duration:.0f}s).")

    LAST_JOB_BY_IP[ip] = time.time()
    DAILY["count"] += 1
    JOBS[job_id] = {"status": "queued", "progress": 0.0, "stage": "Queued",
                    "ip": ip, "upload_path": str(upload_path),
                    "created_at": time.time()}
    JOB_QUEUE.put(job_id)
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job (results expire after "
                                 f"{settings.RESULT_TTL_S // 60} min).")
    public = {k: job.get(k) for k in
              ("status", "progress", "stage", "result", "error")}
    return JSONResponse(public)


@app.get("/api/files/{job_id}/{name}")
def job_file(job_id: str, name: str):
    # basic path-traversal guard: only serve flat names we created.
    if "/" in name or ".." in name:
        raise HTTPException(400, "Bad filename.")
    path = settings.WORK_DIR / job_id / name
    if not path.exists():
        path = settings.EXAMPLES_DIR / job_id / name   # examples share route
        if not path.exists():
            raise HTTPException(404, "File not found or expired.")
    return FileResponse(path)


@app.get("/api/examples")
def examples():
    """Precomputed demo songs: instant results, zero compute, zero LLM calls."""
    manifest = settings.EXAMPLES_DIR / "manifest.json"
    if not manifest.exists():
        return []
    import json
    return json.loads(manifest.read_text())


# --- chat agent ---------------------------------------------------------------

def _find_result(job_id: str) -> dict | None:
    """A track's analysis, wherever it lives: finished job or bundled example."""
    job = JOBS.get(job_id)
    if job and job.get("status") == "done":
        return job["result"]
    if job_id.startswith("ex_"):
        import json
        manifest = settings.EXAMPLES_DIR / "manifest.json"
        if manifest.exists():
            for ex in json.loads(manifest.read_text()):
                if ex["id"] == job_id:
                    return ex["result"]
    return None


class ChatBody(BaseModel):
    messages: list[dict]


@app.post("/api/chat/{job_id}")
def chat_with_track(job_id: str, body: ChatBody):
    """Grounded Q&A about one analyzed track (see agent.py for the contract).

    Stateless on purpose: the client sends the whole (capped) message history
    each time, so the server needs no chat session store — the same property
    that lets the job store be a dict lets this be nothing at all.

    Sync `def`, not `async def`: agent.chat blocks for seconds on the LLM,
    and a blocking call inside the event loop would freeze every other
    request (polling, file serving) for its whole duration. FastAPI runs
    sync endpoints in a threadpool — the honest way to wait.
    """
    result = _find_result(job_id)
    if result is None:
        raise HTTPException(404, "Unknown or expired track.")
    messages = body.messages
    if not messages:
        raise HTTPException(400, "messages must be a non-empty list.")
    if len(messages) > 24:
        messages = messages[-24:]              # cap context, keep the tail
    if sum(len(str(m.get("content", ""))) for m in messages) > 24_000:
        raise HTTPException(400, "Conversation too long — refresh to reset.")
    from agent import chat as agent_chat
    return agent_chat(result, messages)


@app.get("/api/health")
def health():
    return {"ok": True, "queue": JOB_QUEUE.qsize(),
            "daily_used": DAILY["count"], "daily_cap": settings.GLOBAL_DAILY_CAP}
