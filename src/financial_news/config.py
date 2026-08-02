from pathlib import Path

# Sentiment
LABEL_TO_ID = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}

ID_TO_LABEL = {
    value: key
    for key, value in LABEL_TO_ID.items()
}

NUM_CLASSES = len(LABEL_TO_ID)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "phrasebank"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
