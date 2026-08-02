from pathlib import Path

import kagglehub
from kagglehub import KaggleDatasetAdapter


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "phrasebank"
RAW_DIR.mkdir(parents=True, exist_ok=True)

DATASET_NAME = "ankurzing/sentiment-analysis-for-financial-news"

def download_phrasebank() -> None:
    dataset = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        DATASET_NAME,
        "all-data.csv",
        pandas_kwargs={
            "header": None,
            "names": ["sentiment", "text"],
            "encoding": "utf-8",
        },
    )

    print(dataset.head())
    print(dataset.shape)
    print(dataset["sentiment"].value_counts())

    dataset.to_csv(
        RAW_DIR / "all-data.csv",
        index=False,
        encoding="utf-8"
    )


if __name__ == "__main__":
    download_phrasebank()