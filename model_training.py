"""Train, compare and save the final used-car price model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_cleaning import clean_data
from src.data_preprocessing import split_features_target
from src.feature_engineering import add_features
from src.model_comparison import compare_models
from src.model_evaluation import prediction_examples


def train_project(data_path: str, output_dir: str = "."):
    root = Path(output_dir)
    models_dir = root / "models"
    reports_dir = root / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(data_path)
    cleaned = clean_data(raw)
    featured = add_features(cleaned)
    X, y = split_features_target(featured)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results, fitted = compare_models(X_train, X_test, y_train, y_test)
    best_name = str(results.iloc[0]["model"])
    best_model = fitted[best_name]
    best_predictions = np.maximum(best_model.predict(X_test), 0)

    joblib.dump(best_model, models_dir / "car_price_model.joblib", compress=3)
    results.to_csv(reports_dir / "model_comparison.csv", index=False)
    prediction_examples(y_test, best_predictions, count=20).to_csv(
        reports_dir / "prediction_examples.csv", index=False
    )

    metadata = {
        "best_model": best_name,
        "rows_raw": int(len(raw)),
        "rows_after_cleaning": int(len(cleaned)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "test_size": 0.2,
        "random_state": 42,
        "target_transformation": "log1p",
        "metrics": {
            key: float(results.iloc[0][key]) for key in ["MAE", "MSE", "RMSE", "R2"]
        },
    }
    (reports_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("\nModel comparison:\n", results.to_string(index=False))
    print(f"\nSaved best model ({best_name}) to {models_dir / 'car_price_model.joblib'}")
    return results, best_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the car-price regression project.")
    parser.add_argument("--data", default="data/cars.csv")
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()
    train_project(args.data, args.output_dir)


if __name__ == "__main__":
    main()

