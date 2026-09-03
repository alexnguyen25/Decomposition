"""Run one audio file through the full pipeline and emit a structured result."""

import json
import sys
from pathlib import Path

from src.classification.classifier import Classifier, load_classifier
from src.feature_extraction.feature_extraction import extract_features
from src.preprocessing.processing import process_audio
from src.preprocessing.validation import validAudio
from src.separation.separation import separate
from src.separation.stem_presence import check_stem_presence

# where Demucs writes stems: data/separated/<model>/<track>/
SEPARATED_DIR = Path("data/separated")

# vocals/drums/bass get a silence check; "other" goes to the classifier instead
PRESENCE_STEMS = ("vocals", "drums", "bass")


def setup() -> Classifier:
    """Load the classifier once per process — weights are expensive, songs are not."""
    classifier = load_classifier()
    print(f"Using backend {classifier.backend} on {classifier.device}")
    return classifier


def analyze(file_path: Path, classifier: Classifier) -> dict:
    """One path in, one JSON-ready dict out — the full pipeline for one song.

    Exceptions from ``validAudio`` are deliberately not caught: how a bad file
    should surface differs per caller (a CLI wants stderr and a non-zero exit,
    a web handler wants a 400), so that choice belongs to the caller.
    """
    validAudio(file_path)
    processed_path = Path(process_audio(file_path))

    stems = separate(processed_path, SEPARATED_DIR)
    features = extract_features(stems)
    presence = check_stem_presence({name: stems[name] for name in PRESENCE_STEMS})
    instruments, timeline = classifier.predict(stems["other"])

    return {
        "bpm": features["bpm"],
        "key": features["key"],
        "stems": {name: str(path) for name, path in stems.items()},
        "presence": presence,
        "instruments": instruments,
        "timeline": {"chunk_s": 10, "instruments": timeline},
    }


def main(file_path: Path) -> dict:
    """Set up the model once, then run one file through the pipeline."""
    return analyze(file_path, setup())


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m src.main <audio-file>")
    print(json.dumps(main(Path(sys.argv[1])), indent=2))
