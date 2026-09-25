"""Regression metrics and command-line evaluation for a saved model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.data_cleaning import clean_data
from src.data_preprocessing import split_features_target
from src.feature_engineering import add_features


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def prediction_examples(y_true, y_pred, count: int = 15) -> pd.DataFrame:
    examples = pd.DataFrame(
        {
            "actual_price_usd": np.asarray(y_true),
            "predicted_price_usd": np.maximum(np.asarray(y_pred), 0),
        }
    )
    examples["absolute_error_usd"] = (
        examples["actual_price_usd"] - examples["predicted_price_usd"]
    ).abs()
    return examples.head(count).round(2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the saved car-price model.")
    parser.add_argument("--data", default="data/cars.csv")
    parser.add_argument("--model", default="models/car_price_model.joblib")
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()

    data = add_features(clean_data(pd.read_csv(args.data)))
    X, y = split_features_target(data)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = joblib.load(args.model)
    predictions = np.maximum(model.predict(X_test), 0)
    metrics = regression_metrics(y_test, predictions)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "final_model_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    prediction_examples(y_test, predictions).to_csv(
        output_dir / "prediction_examples.csv", index=False
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

