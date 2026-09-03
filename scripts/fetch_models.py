"""Download the model weights this project needs but does not keep in git.

Only the 345 MB BEATs backbone is third-party and too big for the repo, so it
lives on the Hugging Face Hub. The trained heads are small and are committed to
git — they are the part of the model that was actually trained here — but they
are mirrored on the Hub too so this script has a single source.

Usage:
    python scripts/fetch_models.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import HEAD_FILES, HF_MODEL_REPO, MODEL_DIR

FILES = ["beats/BEATs_iter3_plus_AS2M.pt", *HEAD_FILES.values()]


def main() -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("huggingface_hub is missing — run: pip install -r requirements.txt")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = MODEL_DIR / name
        if target.exists():
            print(f"  have  {name}")
            continue
        print(f"  fetch {name} ...", flush=True)
        # token=False: the repo is public, so never send credentials and never
        # fail for a user who has none.
        downloaded = hf_hub_download(HF_MODEL_REPO, name, token=False)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(downloaded).read_bytes())
        print(f"  done  {name} ({target.stat().st_size / 1e6:.0f} MB)")

    print(f"\nAll weights present in {MODEL_DIR.resolve()}")


if __name__ == "__main__":
    main()
