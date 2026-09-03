"""Central configuration — everything tunable lives here, driven by env vars.

WHY a settings module: the same code must run in three places (your laptop,
an HF Space, any future host) with different paths/limits. Env-var config is
the 12-factor-app pattern: the code never changes, only the environment.
"""

import os
from pathlib import Path

# --- paths -----------------------------------------------------------------
BACKEND_DIR = Path(__file__).parent
REPO_ROOT = BACKEND_DIR.parent.parent

# Model artifacts live in data/f1_research by default (they're big; we don't
# copy them). For deployment you copy them next to the app and set MODEL_DIR.
MODEL_DIR = Path(os.environ.get("MODEL_DIR", REPO_ROOT / "data" / "f1_research"))

# Where uploads and generated stems live. Ephemeral by design (see TTL below).
WORK_DIR = Path(os.environ.get("WORK_DIR", BACKEND_DIR / "jobs_data"))

# Precomputed example results (safe to serve without compute).
EXAMPLES_DIR = Path(os.environ.get(
    "EXAMPLES_DIR", BACKEND_DIR.parent / "frontend" / "public" / "examples"))

# --- LLM (swappable, OpenAI-compatible) -------------------------------------
# Local dev default: Ollama. For prod set e.g.
#   LLM_BASE_URL=https://api.cerebras.ai/v1  LLM_MODEL=llama3.1-8b  LLM_API_KEY=csk-...
# or Groq: LLM_BASE_URL=https://api.groq.com/openai/v1  LLM_MODEL=llama-3.1-8b-instant
# The app works without any LLM (falls back to a template description).
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "llama3.2:3b")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")  # empty is fine for Ollama
LLM_TIMEOUT_S = int(os.environ.get("LLM_TIMEOUT_S", "60"))

# Chat agent: max LLM round-trips per message (each tool batch costs one).
# Small models occasionally loop on tools; this is the circuit breaker.
CHAT_MAX_ROUNDS = int(os.environ.get("CHAT_MAX_ROUNDS", "6"))

# --- classifier -------------------------------------------------------------
# "stem_recalib" = head fine-tuned on Demucs-processed clips (better real-song
# recall, measured); "e10" = the plain champion head (best on OpenMIC test).
CLASSIFIER_HEAD = os.environ.get("CLASSIFIER_HEAD", "stem_recalib")
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.5"))

# --- upload validation / abuse guards ---------------------------------------
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "15"))
MAX_DURATION_S = int(os.environ.get("MAX_DURATION_S", "360"))   # 6 minutes
MIN_DURATION_S = int(os.environ.get("MIN_DURATION_S", "10"))
PER_IP_COOLDOWN_S = int(os.environ.get("PER_IP_COOLDOWN_S", "120"))
GLOBAL_DAILY_CAP = int(os.environ.get("GLOBAL_DAILY_CAP", "200"))
RESULT_TTL_S = int(os.environ.get("RESULT_TTL_S", "1800"))      # 30 min

# Demucs stems the classifier must NOT report (they're already separate stems)
DEAD_WEIGHT = {"bass", "cymbals", "drums", "voice"}
