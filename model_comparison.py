"""Comparable regression candidates trained on the same split."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from src.data_preprocessing import build_preprocessor
from src.model_evaluation import regression_metrics


def candidate_models(random_state: int = 42):
    return {
        "Ridge": Ridge(alpha=10.0, solver="lsqr"),
        "DecisionTree": DecisionTreeRegressor(
            max_depth=24, min_samples_leaf=3, random_state=random_state
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=25,
            max_depth=24,
            min_samples_leaf=2,
            max_features=0.8,
            n_jobs=-1,
            random_state=random_state,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=25,
            max_depth=None,
            min_samples_leaf=2,
            max_features=0.9,
            n_jobs=-1,
            random_state=random_state,
        ),
    }


def make_pipeline(regressor):
    pipeline = Pipeline(
        steps=[("preprocessor", build_preprocessor()), ("regressor", regressor)]
    )
    return TransformedTargetRegressor(
        regressor=pipeline,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )


def compare_models(X_train, X_test, y_train, y_test):
    rows = []
    fitted = {}
    for name, estimator in candidate_models().items():
        model = make_pipeline(estimator)
        model.fit(X_train, y_train)
        predictions = np.maximum(model.predict(X_test), 0)
        rows.append({"model": name, **regression_metrics(y_test, predictions)})
        fitted[name] = model
        print(f"Finished {name}: MAE=${rows[-1]['MAE']:,.2f}")
    results = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    return results, fitted
