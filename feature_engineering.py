"""Feature engineering for used-car price prediction."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REFERENCE_YEAR = 2020


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create interpretable predictors without using the target."""

    df = data.copy()
    df["car_age"] = (REFERENCE_YEAR - df["year"]).clip(lower=1)
    df["mileage_per_year"] = df["mileage_km"] / df["car_age"]
    df["log_mileage"] = np.log1p(df["mileage_km"].clip(lower=0))
    df["engine_volume_liters"] = df["engine_volume_cm3"] / 1_000
    df["make_model"] = (
        df["make"].fillna("unknown").astype(str)
        + "__"
        + df["model"].fillna("unknown").astype(str)
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Create model features.")
    parser.add_argument("--input", default="data/cars_clean.csv")
    parser.add_argument("--output", default="data/cars_features.csv")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    featured = add_features(pd.read_csv(args.input))
    featured.to_csv(output_path, index=False)
    print(f"Saved {len(featured):,} rows with engineered features to {output_path}")


if __name__ == "__main__":
    main()

