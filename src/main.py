"""Run one audio file through the full pipeline and emit a structured result."""

import json
import sys
from pathlib import Path

import torch

from src.classification.classifier import classify, load_model
from src.classification.model import Model
from src.config import CHECKPOINT_PATH
from src.feature_extraction.feature_extraction import extract_features
from src.preprocessing.processing import process_audio
from src.preprocessing.validation import validAudio
from src.separation.separation import separate
from src.separation.stem_presence import check_stem_presence

# where Demucs writes stems: data/separated/<model>/<track>/
SEPARATED_DIR = Path("data/separated")

# vocals/drums/bass get a silence check; "other" goes to the classifier instead
PRESENCE_STEMS = ("vocals", "drums", "bass")


def setup() -> tuple[Model, torch.device]:
    """Pick device and load the trained classifier once per process."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = load_model(CHECKPOINT_PATH, device)
    return model, device


def analyze(file_path: Path, model: Model, device: torch.device) -> dict:
    """One path in, one JSON-ready dict out — the full pipeline for one song."""
    validAudio(file_path)
    processed_path = Path(process_audio(file_path))

    stems = separate(processed_path, SEPARATED_DIR)
    features = extract_features(stems)
    presence = check_stem_presence({name: stems[name] for name in PRESENCE_STEMS})
    instruments = classify(stems["other"], model, device)

    return {
        "bpm": features["bpm"],
        "key": features["key"],
        "stems": {name: str(path) for name, path in stems.items()},
        "presence": presence,
        "instruments": instruments,
    }


def main(file_path: Path) -> dict:
    """Set up the model once, then run one file through the pipeline."""
    model, device = setup()
    return analyze(file_path, model, device)


if __name__ == "__main__":
    result = main(Path(sys.argv[1]))
    print(json.dumps(result, indent=2))
