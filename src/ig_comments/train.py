from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

LABELS = ["pozytywny", "neutralny", "negatywny"]


def train(data_path: Path, output_path: Path) -> None:
    frame = pd.read_csv(data_path)
    required = {"text", "label"}
    if not required.issubset(frame.columns):
        raise ValueError("CSV musi zawierać kolumny: text,label")
    frame = frame.dropna(subset=["text", "label"])
    frame["text"] = frame["text"].astype(str).str.strip()
    frame = frame[frame["text"] != ""]
    unknown = sorted(set(frame["label"]) - set(LABELS))
    if unknown:
        raise ValueError(f"Nieobsługiwane etykiety: {unknown}; dozwolone: {LABELS}")
    counts = frame["label"].value_counts()
    if len(counts) < 2 or counts.min() < 2:
        raise ValueError("Każda z co najmniej dwóch klas musi mieć minimum 2 przykłady")

    x_train, x_test, y_train, y_test = train_test_split(
        frame["text"], frame["label"], test_size=0.25, random_state=42, stratify=frame["label"]
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=50000,
                                  strip_accents=None, sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    report = classification_report(y_test, predictions, labels=LABELS, output_dict=True,
                                   zero_division=0)
    metrics = {
        "n_rows": int(len(frame)),
        "class_counts": {str(k): int(v) for k, v in counts.items()},
        "macro_f1": report["macro avg"]["f1-score"],
        "per_class": {label: report[label] for label in LABELS},
        "confusion_matrix_labels_order": LABELS,
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=LABELS).tolist(),
        "note": "Wyniki na małym holdoucie; nie traktować jako dowodu gotowości produkcyjnej.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    output_path.with_suffix(".metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/model.joblib"))
    args = parser.parse_args()
    train(args.data, args.output)


if __name__ == "__main__":
    main()
