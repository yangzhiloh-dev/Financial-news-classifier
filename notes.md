# Financial News Sentiment Classifier — Learning Notes

## 1. Project goal

This project has two main parts:

1. Train NLP models to classify financial news as **negative**, **neutral**, or **positive**.
2. Later, test whether the model's dated predictions could have been turned into market signals in an educational backtest.

The current milestone stops at the **majority-class baseline**. The purpose of this stage is to build a reliable data pipeline and a simple reference model before implementing TF-IDF or neural networks.

### Learning order

Do not begin with a GRU, FinBERT, Gmail integration, or live trading. First understand:

- text labels and numeric class IDs;
- raw versus processed data;
- cleaning, validation, duplicates, and data leakage;
- training, validation, and test splits;
- baseline models and evaluation metrics;
- later: token IDs, tensors, embeddings, `Dataset`, `DataLoader`, `CrossEntropyLoss`, backpropagation, and the training loop.

---

## 2. Current project structure

Status labels:

- **Done** — implemented in the current milestone.
- **Next** — the majority baseline to implement next.
- **Later** — placeholder for a later stage of the project.
- **Generated** — created by Python tooling and not maintained manually.

```text
Financial-News-Classifier/
├── .gitignore                         # Files Git should ignore
├── README.md                          # Final project overview and instructions [Later]
├── notes.md                           # These reusable implementation notes [Done]
├── pyproject.toml                     # Python package/build configuration [Done]
├── requirements.txt                   # Third-party dependencies [Done]
│
├── app/
│   └── dashboard.py                   # Streamlit dashboard [Later]
│
├── artifacts/                         # Outputs created by experiments [Later]
│   ├── checkpoints/                   # Saved PyTorch model weights
│   ├── figures/                       # Charts and confusion matrices
│   ├── metrics/                       # Evaluation results
│   └── vectorizers/                   # Fitted TF-IDF/vectorizer objects
│
├── data/
│   ├── raw/
│   │   ├── phrasebank/
│   │   │   └── all-data.csv           # Immutable downloaded PhraseBank data [Done]
│   │   └── market_news/               # Dated market-linked news [Later]
│   └── processed/
│       ├── sentiment_data.csv         # Cleaned data before deduplication [Done]
│       ├── sentiment_data_no_duplicates.csv  # Final deduplicated data [Done]
│       ├── sentiment_data_train.csv   # 70% training split [Done]
│       ├── sentiment_data_val.csv     # 15% validation split [Done]
│       └── sentiment_data_test.csv    # 15% test split [Done]
│
├── docs/
│   ├── data_card.md                   # Dataset origin, limits, and intended use [Later]
│   ├── labeling_guide.md              # Definitions and examples of labels [Later]
│   └── methodology.md                 # Full NLP/backtest methodology [Later]
│
├── learning/
│   ├── 01_tensors.py                  # Tensor exercises [Later]
│   ├── 02_autograd.py                 # Gradient/autograd exercises [Later]
│   ├── 03_linear_regression.py        # Small training-loop exercise [Later]
│   └── 04_embedding_exercise.py       # Embedding exercise [Later]
│
├── scripts/
│   ├── download_phrasebank.py         # Download and save the raw dataset [Done]
│   ├── prepare_sentiment_data.py      # Clean, validate, deduplicate, and split [Done]
│   ├── train_baseline.py              # Run/evaluate the majority baseline [Next]
│   ├── train_neural_model.py          # Train neural models [Later]
│   ├── evaluate_model.py              # Detailed model evaluation [Later]
│   └── run_backtest.py                # Educational market backtest [Later]
│
├── src/
│   └── financial_news/
│       ├── __init__.py                # Marks the folder as an importable package
│       ├── config.py                  # Shared label IDs and constants [Done]
│       ├── seed.py                    # Reproducible random seeds [Later]
│       ├── inference.py               # Prediction probabilities [Later]
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   ├── load_phrasebank.py     # Load and validate the raw CSV [Done]
│       │   ├── validation.py          # Resolve duplicate/conflicting texts [Done]
│       │   ├── splitting.py           # Stratified train/val/test split [Done]
│       │   ├── cleaning.py            # Reusable text cleaning [Later]
│       │   └── load_market_news.py    # Load dated market news [Later]
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   ├── baselines.py           # Majority baseline class/function [Next]
│       │   ├── embedding_mlp.py        # Embedding-average MLP [Later]
│       │   └── gru_classifier.py       # GRU model [Later]
│       │
│       ├── text/
│       │   ├── __init__.py
│       │   ├── tokenizer.py            # Convert text into tokens [Later]
│       │   ├── vocabulary.py           # Map tokens to integer IDs [Later]
│       │   └── padding.py              # Padding and attention masks [Later]
│       │
│       ├── training/
│       │   ├── __init__.py
│       │   ├── dataset.py              # PyTorch Dataset/DataLoader support [Later]
│       │   ├── loops.py                # Training and validation loops [Later]
│       │   └── checkpointing.py        # Save/load best model [Later]
│       │
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── metrics.py              # Accuracy, macro-F1, confusion matrix [Later]
│       │   ├── error_analysis.py       # Inspect incorrect predictions [Later]
│       │   └── calibration.py          # Inspect probability quality [Later]
│       │
│       └── finance/
│           ├── __init__.py
│           ├── prices.py               # Download SPY/ticker prices [Later]
│           ├── alignment.py            # Align news with trading sessions [Later]
│           ├── signals.py              # Convert sentiment into signals [Later]
│           └── backtest.py             # Backtest with transaction costs [Later]
│
└── tests/                              # Automated tests [Later]
```

Folders named `*.egg-info` are generated when the project is installed in editable mode. They are not source code and should not be edited. They are ignored by Git and can be regenerated with `pip install -e .`.

### Why separate `scripts/` and `src/`?

- `scripts/` contains entry points that run complete jobs.
- `src/financial_news/` contains small reusable functions and classes.
- A script should coordinate reusable code instead of containing every detail itself.

For example, `prepare_sentiment_data.py` runs the preparation job, while `load_phrasebank.py`, `validation.py`, and `splitting.py` each handle one responsibility.

---

## 3. Step 1 — Create the project and virtual environment

### Create and activate the environment in PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

If PowerShell blocks activation, the policy may need to be changed for the current terminal session. The environment can also be used directly through `.venv\Scripts\python.exe`.

### Select the interpreter in VS Code

1. Open the command palette.
2. Choose **Python: Select Interpreter**.
3. Select the interpreter inside `.venv`.

The selected interpreter determines which Python installation and packages VS Code uses. If an import such as `train_test_split` is underlined, first confirm that VS Code is using `.venv`, then confirm that `scikit-learn` is installed in that environment.

```powershell
python -c "from sklearn.model_selection import train_test_split; print('Import works')"
```

### Required libraries at this stage

- `pandas`: tables, CSV files, cleaning, grouping, and splitting outputs.
- `numpy`: numerical work used throughout the project.
- `scikit-learn`: data splitting, baselines, TF-IDF, metrics, and Logistic Regression.
- `kagglehub[pandas-datasets]`: downloading the PhraseBank CSV from Kaggle.
- `pytest`: automated tests later.

PyTorch, Streamlit, yfinance, plotting libraries, and joblib are already listed for future stages.

### What `pyproject.toml` does

`pyproject.toml` describes the local Python package and tells Python that packages live inside `src/`.

Important settings:

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "financial-news-classifier"
version = "0.1.0"
requires-python = ">=3.11"

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]
```

Running `pip install -e .` creates an **editable install**. This makes imports such as the following work without copying the package:

```python
from financial_news.config import LABEL_TO_ID
```

Changes inside `src/financial_news/` are immediately available to the environment.

### Key things to remember

- Do not install project packages globally when a virtual environment can be used.
- `requirements.txt` lists third-party dependencies.
- `pyproject.toml` describes this project as an installable package.
- Commit configuration files, but do not commit `.venv`, caches, or `*.egg-info`.

---

## 4. Step 2 — Define the three sentiment labels

The shared mapping is stored in `src/financial_news/config.py`:

```python
LABEL_TO_ID = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}

ID_TO_LABEL = {class_id: label for label, class_id in LABEL_TO_ID.items()}
NUM_CLASSES = 3
```

### Why numeric labels are required

Models and loss functions work with numbers, not label strings. Later, PyTorch's `CrossEntropyLoss` will expect one integer class index for each example:

- `0` means negative;
- `1` means neutral;
- `2` means positive.

The IDs do not describe intensity or distance. Positive is not “twice” neutral. They are only class identifiers.

### Why define the mapping once?

If each script creates its own mapping, one script could accidentally use a different order. A shared configuration ensures that preprocessing, training, evaluation, and inference interpret every class consistently.

### Key things to remember

- Never silently change label IDs after training a model.
- Keep both directions: text-to-ID for training and ID-to-text for readable predictions.
- Validate labels before mapping them.
- `CrossEntropyLoss` later needs raw model scores shaped `[batch_size, 3]` and targets shaped `[batch_size]` containing these IDs.

---

## 5. Step 3 — Download and inspect Financial PhraseBank

### Dataset used

Kaggle dataset identifier:

```python
DATASET_NAME = "ankurzing/sentiment-analysis-for-financial-news"
```

The download script requests `all-data.csv` through KaggleHub's pandas adapter:

```python
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
```

It then saves the result here:

```text
data/raw/phrasebank/all-data.csv
```

### What the arguments mean

- `KaggleDatasetAdapter.PANDAS`: load the dataset as a pandas `DataFrame`.
- `DATASET_NAME`: identify the Kaggle dataset.
- `"all-data.csv"`: choose the file inside that dataset.
- `header=None`: the downloaded source is treated as having no header row.
- `names=["sentiment", "text"]`: assign meaningful column names while loading.
- `encoding="utf-8"`: decode text consistently.

After saving it locally, the project CSV contains the `sentiment` and `text` headers. Therefore, the later loader should use normal `pd.read_csv(...)` behavior and should not add the names again.

### Why keep a download script?

A reproducible project should be able to rebuild its data from a known source. `download_phrasebank.py` records:

- where the data came from;
- which file was selected;
- how it was decoded;
- what column names were assigned;
- where the raw file belongs.

The script also prints the first rows, shape, and sentiment counts so the download can be inspected immediately.

### Raw-data rule

Treat files in `data/raw/` as immutable source material:

- do not clean them in place;
- do not manually delete rows from them;
- write transformations to `data/processed/`;
- keep the source and download method documented.

This preserves **data provenance**: the ability to explain where processed data came from and recreate it.

### Run the download

```powershell
python scripts/download_phrasebank.py
```

Then check:

- the destination file exists;
- its columns are `sentiment` and `text`;
- it has 4,846 rows before duplicate handling;
- sentiment values are only negative, neutral, and positive;
- class counts are visibly imbalanced.

### Why `load_phrasebank.py` is still useful

The downloader and loader have different responsibilities:

- `download_phrasebank.py` obtains the data and writes the raw file.
- `load_phrasebank.py` reads the existing raw file and verifies its schema whenever another script needs it.

This means model development does not redownload data every time. The loader also gives all future scripts one consistent way to read PhraseBank.

---

## 6. Step 4 — Create the processed sentiment dataset

Run:

```powershell
python scripts/prepare_sentiment_data.py
```

The preparation pipeline performs this sequence:

```text
raw CSV
  → load and validate required columns
  → preserve original text/label values
  → check missing values
  → normalize labels and text
  → validate known sentiments
  → map strings to numeric labels
  → create stable hashes and IDs
  → save cleaned pre-deduplication data
  → remove duplicate/conflicting text
  → split into train/validation/test data
  → verify split integrity
  → save all outputs
```

### Columns in the processed data

| Column | Meaning | Why it is useful |
|---|---|---|
| `news_id` | Stable readable identifier derived from the text hash | Easier to refer to one example in logs or error analysis |
| `text` | Normalized text used by the models | Provides consistent model input |
| `sentiment` | Normalized string label | Human-readable class name |
| `label` | Numeric class ID (`0`, `1`, or `2`) | Used by ML models and loss functions |
| `source` | Dataset origin, such as `FinancialPhraseBank` | Supports provenance when datasets are combined later |
| `text_hash` | SHA-256 fingerprint of normalized text | Finds duplicates and checks leakage across splits |
| `raw_text` | Original text before normalization | Preserves an auditable copy |
| `raw_sentiment` | Original label before normalization | Preserves an auditable copy |

### Text normalization

The current preparation performs light normalization:

- convert values to strings after missing-value validation;
- remove whitespace from the beginning and end;
- collapse repeated whitespace inside text;
- lowercase and trim the sentiment label.

Example:

```text
"  Operating   profit rose.  "
```

becomes:

```text
"Operating profit rose."
```

Keep text cleaning conservative. Punctuation, percentages, currency symbols, minus signs, and numbers may carry financial meaning. Aggressive cleaning can destroy useful information.

### Validate before mapping labels

Mapping is normally performed with:

```python
df["label"] = df["sentiment"].map(LABEL_TO_ID)
```

If an unexpected value such as `"positve"` is present, `.map(...)` returns `NaN` for that row. Allowing the pipeline to continue would create an invalid target and cause confusing errors later.

The safe order is:

```python
unknown_sentiments = sorted(set(df["sentiment"]) - set(LABEL_TO_ID))

if unknown_sentiments:
    raise ValueError(f"Unknown sentiment values: {unknown_sentiments}")

df["label"] = df["sentiment"].map(LABEL_TO_ID)

if df["label"].isna().any():
    raise ValueError("At least one sentiment could not be mapped to a label")
```

This is called **fail-fast validation**: stop close to the source of the problem and report the invalid values clearly.

### Stable `text_hash`

```python
import hashlib

def make_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
```

A hash converts the normalized text into a deterministic fingerprint. Identical normalized text produces the same hash. A small text change produces a different hash.

Uses in this project:

- detect repeated articles/sentences;
- prevent the same text from appearing in multiple splits;
- create reproducible IDs;
- trace examples during error analysis.

A hash is not encryption and is not a sentiment feature. It is metadata used for data integrity.

### Stable `news_id`

The readable identifier can be made from part of the hash:

```python
df["news_id"] = "fpb_" + df["text_hash"].str[:12]
```

This is preferable to using only a row number because row numbers change when rows are reordered or removed. A hash-derived ID remains associated with the same normalized text.

### `source`

```python
df["source"] = "FinancialPhraseBank"
```

This may look repetitive now, but it becomes important when PhraseBank is combined with dated news from another provider. It lets the project group, filter, audit, and report performance by data source.

---

## 7. Step 5 — Remove duplicates safely

Duplicate handling must occur **before splitting**. Otherwise, the same sentence could appear in training and validation/test data, making performance look better than it really is.

There are two distinct situations:

1. **Same text, same label** — keep one copy.
2. **Same text, conflicting labels** — remove the entire text group because there is no unambiguous target.

Simply calling `drop_duplicates("text_hash")` immediately is unsafe for conflicting groups. It keeps whichever row happens to appear first and silently discards the other label. Reordering the CSV could therefore change the retained target.

### Conflict detection

```python
conflict_mask = (
    df.groupby("text_hash")["label"]
    .transform("nunique")
    > 1
)
```

Line-by-line meaning:

1. `groupby("text_hash")` groups rows with identical normalized text.
2. `["label"]` selects the target labels in each group.
3. `nunique()` counts distinct labels in each group.
4. `transform(...)` returns that count aligned to every original row.
5. `> 1` produces `True` for every row belonging to a conflicting group.

Then:

```python
df = df[~conflict_mask]
```

- `~` means logical NOT.
- Conflicting rows are `True` in `conflict_mask`.
- `~conflict_mask` keeps only the non-conflicting rows.
- Therefore, **all rows** in every conflicting group are removed.

After conflicts are gone:

```python
df = df.drop_duplicates(subset=["text_hash"]).copy()
```

Every remaining repeated text has only one label, so it is now safe to keep one copy.

### Current dataset result

- Rows before duplicate handling: **4,846**.
- Rows after safe duplicate handling: **4,836**.
- Duplicate text groups found: **8**.
- Six duplicate groups had matching labels.
- Two duplicate groups had conflicting neutral/positive labels and were removed completely.

### Key things to remember

- Deduplication is not only about reducing file size; it prevents data leakage.
- Use normalized text or its hash to detect semantic row identity.
- Do not arbitrarily keep one target when duplicate texts disagree.
- Log how many rows and conflict groups were removed.
- Keep the original raw file unchanged.

---

## 8. Step 6 — Split the sentiment data

The deduplicated data is split into:

- **Training set (70%)**: fit model parameters and learn preprocessing objects.
- **Validation set (15%)**: compare choices and tune the model.
- **Test set (15%)**: one final, unbiased evaluation after choices are complete.

### Why three datasets?

Evaluating on training data measures memory as well as generalization. Repeatedly checking the test set also leaks information into development decisions. The validation set allows experimentation while the test set remains untouched.

### Two-stage split

Scikit-learn directly creates two outputs, so a 70/15/15 split is made in two stages:

```python
train_df, temporary_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["label"],
)

val_df, test_df = train_test_split(
    temporary_df,
    test_size=0.50,
    random_state=42,
    stratify=temporary_df["label"],
)
```

The first call reserves 30% outside training. The second divides that 30% equally, producing 15% validation and 15% test.

### What `stratify` does

```python
stratify=df["label"]
```

This asks scikit-learn to preserve approximately the same negative/neutral/positive proportions in each split. It matters because PhraseBank is imbalanced, with neutral as the dominant class.

### What `random_state=42` does

The split is random but reproducible. Running the pipeline again with the same data, code, and seed produces the same row assignment. The number 42 has no special statistical meaning; consistency is what matters.

### Current split sizes and class balance

| Split | Rows | Negative | Neutral | Positive |
|---|---:|---:|---:|---:|
| Train | 3,385 | 423 (12.50%) | 2,009 (59.35%) | 953 (28.15%) |
| Validation | 725 | 90 (12.41%) | 431 (59.45%) | 204 (28.14%) |
| Test | 726 | 91 (12.53%) | 431 (59.37%) | 204 (28.10%) |

The proportions are close across all three sets, showing that stratification worked.

### Integrity checks after splitting

At minimum, verify:

```python
assert len(train_df) + len(val_df) + len(test_df) == len(deduplicated_df)

assert set(train_df["text_hash"]).isdisjoint(val_df["text_hash"])
assert set(train_df["text_hash"]).isdisjoint(test_df["text_hash"])
assert set(val_df["text_hash"]).isdisjoint(test_df["text_hash"])
```

Also verify:

- no split is empty;
- every label belongs to `{0, 1, 2}`;
- required columns exist;
- there are no missing target labels;
- class counts are printed for inspection.

### Critical data-leakage rule

Anything that learns from text must be fitted on the training set only. This includes:

- the future TF-IDF vectorizer;
- vocabulary/token counts;
- normalization statistics that depend on the dataset;
- feature selection;
- class-weight calculations.

Validation and test text may be **transformed** by a fitted object, but must not help fit it.

---

## 9. Step 7 — Build the majority-class baseline

**Status: next implementation step.**

A baseline answers: “Is the real model better than an extremely simple strategy?” Because neutral is about 59% of PhraseBank, accuracy alone can look reasonable even when a model never identifies positive or negative news.

### Majority-baseline rule

1. Examine only the training labels.
2. Find the most frequent training class.
3. Predict that one class for every example.

For the current training data:

```text
majority class = neutral
majority class ID = 1
```

Do not use validation or test labels to choose the majority class. Even this simple decision must be learned from training data only.

### File responsibility: `src/financial_news/models/baselines.py`

Create a small reusable class:

```python
import numpy as np


class MajorityClassBaseline:
    def __init__(self) -> None:
        self.majority_class_: int | None = None

    def fit(self, labels) -> "MajorityClassBaseline":
        values, counts = np.unique(labels, return_counts=True)
        self.majority_class_ = int(values[np.argmax(counts)])
        return self

    def predict(self, number_of_examples: int) -> np.ndarray:
        if self.majority_class_ is None:
            raise RuntimeError("The baseline must be fitted before prediction")

        return np.full(
            shape=number_of_examples,
            fill_value=self.majority_class_,
            dtype=np.int64,
        )
```

Concepts in this class:

- `fit(...)` learns one value from training data: the most common class.
- The trailing underscore in `majority_class_` follows the scikit-learn convention for an attribute learned during fitting.
- `predict(...)` repeats that class once for each requested example.
- The fitted-state check prevents accidental prediction before `fit(...)`.
- `np.int64` gives predictions an integer type suitable for class labels.

An alternative is scikit-learn's `DummyClassifier(strategy="most_frequent")`. Implementing the tiny class yourself first makes the baseline logic explicit; comparing it with `DummyClassifier` is a useful test.

### File responsibility: `scripts/train_baseline.py`

The script should:

1. Load `sentiment_data_train.csv` and `sentiment_data_val.csv`.
2. Validate that both contain a `label` column.
3. Fit the baseline using `train_df["label"]` only.
4. Predict one label for every validation row.
5. calculate accuracy, macro-F1, and a confusion matrix.
6. Print the learned class as both its ID and readable name.
7. Print the metrics clearly.

Suggested implementation outline:

```python
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from financial_news.config import ID_TO_LABEL
from financial_news.models.baselines import MajorityClassBaseline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def main() -> None:
    train_df = pd.read_csv(PROCESSED_DIR / "sentiment_data_train.csv")
    val_df = pd.read_csv(PROCESSED_DIR / "sentiment_data_val.csv")

    required_columns = {"label"}
    for split_name, split_df in {"train": train_df, "validation": val_df}.items():
        missing = required_columns - set(split_df.columns)
        if missing:
            raise ValueError(f"{split_name} is missing columns: {sorted(missing)}")

    model = MajorityClassBaseline().fit(train_df["label"].to_numpy())
    predictions = model.predict(len(val_df))
    targets = val_df["label"].to_numpy()

    print(
        "Majority class:",
        model.majority_class_,
        ID_TO_LABEL[model.majority_class_],
    )
    print("Accuracy:", accuracy_score(targets, predictions))
    print("Macro-F1:", f1_score(targets, predictions, average="macro"))
    print("Confusion matrix:\n", confusion_matrix(targets, predictions, labels=[0, 1, 2]))
    print(
        classification_report(
            targets,
            predictions,
            labels=[0, 1, 2],
            target_names=[ID_TO_LABEL[i] for i in [0, 1, 2]],
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()
```

Run it with:

```powershell
python scripts/train_baseline.py
```

### Metrics to understand

#### Accuracy

```text
correct predictions / all predictions
```

The majority baseline should obtain about **59.45% validation accuracy**, because 431 of 725 validation rows are neutral.

#### Per-class precision, recall, and F1

- **Precision**: of the examples predicted as a class, how many were correct?
- **Recall**: of the real examples in a class, how many were found?
- **F1**: harmonic mean of precision and recall.

The majority baseline has zero recall for negative and positive because it never predicts those classes.

#### Macro-F1

Macro-F1 calculates F1 separately for negative, neutral, and positive, then gives all three classes equal weight:

```text
macro-F1 = (F1_negative + F1_neutral + F1_positive) / 3
```

The expected validation macro-F1 is only about **0.249**, even though accuracy is about 59%. This exposes the model's failure on minority classes.

#### Confusion matrix

With label order `[0, 1, 2]`, rows are true classes and columns are predicted classes. Since every prediction is neutral, all counts appear in the neutral prediction column.

### Baseline acceptance checks

The majority-baseline step is complete when:

- the majority class is calculated from training data only;
- the learned class is neutral (`1`) for the current dataset;
- the number of predictions equals the number of validation rows;
- predictions contain only class `1`;
- validation accuracy is approximately `431 / 725 = 0.5945`;
- validation macro-F1 is approximately `0.249`;
- the confusion matrix and per-class report are printed;
- no model choice has been made using the test set.

The test split should remain untouched during development. It can be used later for the final comparison of selected models.

---

## 10. Commands from a clean checkout to the current milestone

```powershell
# 1. Create and activate an isolated environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies and the local package
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# 3. Download the immutable raw dataset
python scripts/download_phrasebank.py

# 4. Clean, validate, deduplicate, split, and save processed data
python scripts/prepare_sentiment_data.py

# 5. Run the majority baseline after implementing it
python scripts/train_baseline.py
```

If an import fails, diagnose the active interpreter before changing code:

```powershell
python -c "import sys; print(sys.executable)"
python -c "import pandas, sklearn; print('Core imports work')"
```

---

## 11. Reusable lessons from this milestone

### Reproducibility

- Record dependency names and project configuration.
- Use deterministic seeds for random operations.
- Keep raw data unchanged.
- Generate processed files through scripts rather than manual spreadsheet edits.
- Print row counts and class distributions at each important stage.

### Data integrity

- Validate required columns immediately after loading.
- Check missing values before converting columns to strings.
- Validate category values before mapping them to IDs.
- Detect contradictory duplicates before ordinary deduplication.
- Assert that data splits have no overlapping text hashes.

### Leakage prevention

- Deduplicate before splitting.
- Choose the majority class from training labels only.
- Fit all future vocabularies and vectorizers on training text only.
- Use validation data for development decisions.
- Reserve test data for final evaluation.

### Evaluation

- Never judge an imbalanced classifier from accuracy alone.
- Compare every learned model with a simple baseline.
- Report macro-F1 and per-class metrics.
- Inspect the confusion matrix to see which classes are ignored or confused.

### Maintainability

- Keep reusable logic in `src/financial_news/`.
- Keep executable workflows in `scripts/`.
- Give each module one clear responsibility.
- Use `pathlib.Path` instead of depending on the terminal's current folder.
- Protect executable script code with `if __name__ == "__main__":`.

---

## 12. Milestone checklist

- [x] Create the VS Code project and virtual environment.
- [x] Configure dependencies and editable package installation.
- [x] Define the three sentiment labels.
- [x] Download and inspect Financial PhraseBank.
- [x] Save raw data in `data/raw/phrasebank/all-data.csv`.
- [x] Create the processed sentiment CSV.
- [x] Normalize and validate the text and labels.
- [x] Add `news_id`, `source`, and `text_hash` metadata.
- [x] Remove same-label duplicates safely.
- [x] Remove contradictory duplicate groups.
- [x] Create stratified 70/15/15 splits.
- [x] Verify sizes, class balance, labels, and hash separation.
- [ ] Implement and run the majority-class baseline.

After the final unchecked item is complete, the next project step is **TF-IDF Logistic Regression**. Do not begin neural-network tokenization until that classical baseline has been evaluated and understood.
