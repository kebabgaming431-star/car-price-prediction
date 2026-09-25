"""Predict a used-car price with the saved end-to-end pipeline."""

from __future__ import annotations

import argparse

import joblib
import pandas as pd

from src.data_cleaning import clean_data
from src.data_preprocessing import split_features_target
from src.feature_engineering import add_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate the market price of one car.")
    parser.add_argument("--model-path", default="models/car_price_model.joblib")
    parser.add_argument("--make", default="volkswagen")
    parser.add_argument("--car-model", default="golf")
    parser.add_argument("--year", type=int, default=2014)
    parser.add_argument("--condition", default="with mileage")
    parser.add_argument("--mileage", type=float, default=180000)
    parser.add_argument("--fuel-type", default="diesel")
    parser.add_argument("--volume", type=float, default=1600)
    parser.add_argument("--color", default="black")
    parser.add_argument("--transmission", default="mechanics")
    parser.add_argument("--drive-unit", default="front-wheel drive")
    parser.add_argument("--segment", default="C")
    args = parser.parse_args()

    row = pd.DataFrame(
        [
            {
                "make": args.make,
                "model": args.car_model,
                "priceUSD": 1.0,
                "year": args.year,
                "condition": args.condition,
                "mileage(kilometers)": args.mileage,
                "fuel_type": args.fuel_type,
                "volume(cm3)": args.volume,
                "color": args.color,
                "transmission": args.transmission,
                "drive_unit": args.drive_unit,
                "segment": args.segment,
            }
        ]
    )
    prepared = add_features(clean_data(row))
    X, _ = split_features_target(prepared)
    model = joblib.load(args.model_path)
    prediction = max(float(model.predict(X)[0]), 0)
    print(f"Estimated price: ${prediction:,.2f}")


if __name__ == "__main__":
    main()

