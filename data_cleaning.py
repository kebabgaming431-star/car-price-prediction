"""Data cleaning utilities for the used-car dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


COLUMN_RENAMES = {
    "mileage(kilometers)": "mileage_km",
    "volume(cm3)": "engine_volume_cm3",
}
CATEGORICAL_COLUMNS = [
    "make",
    "model",
    "condition",
    "fuel_type",
    "color",
    "transmission",
    "drive_unit",
    "segment",
]


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy while leaving missing predictors for the imputer.

    Only impossible or very implausible records are removed. The target remains
    untouched except for requiring a positive value; the model handles its
    strong right skew with a logarithmic target transformation.
    """

    df = data.copy()
    df.columns = [str(col).strip() for col in df.columns]
    df = df.rename(columns=COLUMN_RENAMES)

    required = {
        "make",
        "model",
        "priceUSD",
        "year",
        "condition",
        "mileage_km",
        "fuel_type",
        "engine_volume_cm3",
        "color",
        "transmission",
        "drive_unit",
        "segment",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    for column in ["priceUSD", "year", "mileage_km", "engine_volume_cm3"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    for column in CATEGORICAL_COLUMNS:
        normalized = (
            df[column]
            .astype("string")
            .str.strip()
            .str.lower()
            .replace({"": pd.NA, "nan": pd.NA, "none": pd.NA})
        )
        # scikit-learn imputers handle np.nan reliably in object columns.
        df[column] = normalized.astype(object).where(normalized.notna(), np.nan)

    df = df.drop_duplicates().copy()
    valid = (
        df["priceUSD"].gt(0)
        & df["year"].between(1950, 2020)
        & df["mileage_km"].between(0, 2_000_000)
        & (df["engine_volume_cm3"].isna() | df["engine_volume_cm3"].between(500, 8_000))
    )
    df = df.loc[valid].reset_index(drop=True)

    df["year"] = df["year"].astype(int)
    df["priceUSD"] = df["priceUSD"].astype(float)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean the used-car dataset.")
    parser.add_argument("--input", default="data/cars.csv")
    parser.add_argument("--output", default="data/cars_clean.csv")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned = clean_data(pd.read_csv(args.input))
    cleaned.to_csv(output_path, index=False)
    print(f"Saved {len(cleaned):,} cleaned rows to {output_path}")


if __name__ == "__main__":
    main()
