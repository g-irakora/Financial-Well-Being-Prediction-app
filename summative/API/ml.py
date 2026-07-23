"""Shared logic for the financial well-being regression model.

Kept in one module so the notebook (Task 1), the /predict endpoint and the
/retrain endpoint (Task 2) all use exactly the same cleaning, feature
engineering and preprocessing. That way a model trained in the notebook behaves
the same way when it is served by the API.

Use case / mission
------------------
Predict a person's Financial Well-Being score (0-100, continuous) from a small
set of things they can actually influence - their financial skill, financial
knowledge, income, savings, employment, age, education and household size.
This supports the mission of moving young people toward financial freedom: the
score tells someone where they stand and the features show which levers move it.

Data: CFPB National Financial Well-Being Survey 2016 Public Use File (public,
from consumerfinance.gov / data.gov). Target column: FWBscore.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

TARGET = "FWBscore"

# Numeric features are standardised. FSscore/KHscore are continuous scores,
# household size is a small count.
NUMERIC = ["FSscore", "KHscore", "PPHHSIZE"]

# Ordinal features have a natural low-to-high order, so they are ordinal encoded.
ORDINAL = ["PPINCIMP", "SAVINGSRANGES", "agecat", "PPEDUC"]
ORDINAL_CATEGORIES = [
    [1, 2, 3, 4, 5, 6, 7, 8, 9],   # PPINCIMP  household income band (low -> high)
    [1, 2, 3, 4, 5, 6, 7],         # SAVINGSRANGES  savings band (low -> high)
    [1, 2, 3, 4, 5, 6, 7, 8],      # agecat  age band (young -> old)
    [1, 2, 3, 4, 5],               # PPEDUC  education (less -> more)
]

# Employment status has no natural order, so it is one hot encoded.
NOMINAL = ["EMPLOY"]

FEATURES = NUMERIC + ORDINAL + NOMINAL

# Valid coded values per feature, used for cleaning and to drive the API
# validation. Sentinel codes such as -1 (skip), 98 (don't know) and 99 (refused)
# are treated as missing and dropped.
VALID_VALUES = {
    "PPINCIMP": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    "SAVINGSRANGES": [1, 2, 3, 4, 5, 6, 7],
    "agecat": [1, 2, 3, 4, 5, 6, 7, 8],
    "PPEDUC": [1, 2, 3, 4, 5],
    "EMPLOY": [1, 2, 3, 4, 5, 6, 7, 8],
    "PPHHSIZE": [1, 2, 3, 4, 5],
}


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the survey sentinel/missing codes so only real answers remain.

    Safe to run on the raw PUF or on already-clean data; it only keeps rows whose
    values are valid, and needs just the model columns to be present.
    """
    df = df.copy()

    # target: valid FWBscore is 0-100; -1 and -4 are missing codes
    if TARGET in df.columns:
        df = df[(df[TARGET] >= 0) & (df[TARGET] <= 100)]

    # FSscore / KHscore: negative FSscore is a missing code (-1); KHscore is a
    # standardised knowledge score that is legitimately negative, so leave it.
    if "FSscore" in df.columns:
        df = df[df["FSscore"] >= 0]

    # categorical / ordinal features: keep only the valid coded values
    for col, allowed in VALID_VALUES.items():
        if col in df.columns:
            df = df[df[col].isin(allowed)]

    return df.reset_index(drop=True)


def build_preprocessor() -> ColumnTransformer:
    """Standardise numeric columns, ordinal-encode ordered columns (then scale
    them so every input is on the same footing for the linear models), and one
    hot encode the unordered employment column."""
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUMERIC),
        ("ord", Pipeline([
            ("enc", OrdinalEncoder(categories=ORDINAL_CATEGORIES,
                                   handle_unknown="use_encoded_value", unknown_value=-1)),
            ("scale", StandardScaler()),
        ]), ORDINAL),
        ("nom", OneHotEncoder(handle_unknown="ignore", sparse_output=False), NOMINAL),
    ])


class BatchGDRegressor(BaseEstimator, RegressorMixin):
    """Linear regression trained with full-batch gradient descent, written from
    scratch. Standardises the target internally so the learning rate behaves
    well, and records the training loss (MSE) history for the loss curve."""

    def __init__(self, lr: float = 0.2, n_epochs: int = 1000):
        self.lr = lr
        self.n_epochs = n_epochs

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        self._y_mean, self._y_std = y.mean(), y.std() + 1e-12
        y_s = (y - self._y_mean) / self._y_std
        n, m = X.shape
        self.w_ = np.zeros(m)
        self.b_ = 0.0
        self.loss_history_ = []
        for _ in range(self.n_epochs):
            pred = X @ self.w_ + self.b_
            err = pred - y_s
            self.loss_history_.append(float(np.mean(err ** 2)))
            self.w_ -= self.lr * (2 / n) * (X.T @ err)
            self.b_ -= self.lr * (2 / n) * err.sum()
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return (X @ self.w_ + self.b_) * self._y_std + self._y_mean


def candidate_models() -> dict[str, Pipeline]:
    """The four models compared for the task.

    1. SGD Linear Regression - stochastic gradient descent (scikit-learn).
    2. Batch GD Linear Regression - full-batch gradient descent, from scratch.
    3. Random Forest - ensemble of trees.
    4. Decision Tree - single tree.

    The two linear models are the focus; the tree models are the "other
    implementations" they are compared against. Hyperparameters are fixed here so
    retraining stays fast; the notebook does the heavier tuning.
    """
    return {
        "SGD Linear Regression": Pipeline([
            ("pre", build_preprocessor()),
            ("model", TransformedTargetRegressor(
                regressor=SGDRegressor(loss="squared_error", penalty="l2", alpha=1e-3,
                                       learning_rate="invscaling", eta0=0.01,
                                       max_iter=2000, tol=1e-4, random_state=42),
                transformer=StandardScaler())),
        ]),
        "Batch GD Linear Regression": Pipeline([
            ("pre", build_preprocessor()),
            ("model", BatchGDRegressor(lr=0.2, n_epochs=1000)),
        ]),
        "Random Forest": Pipeline([
            ("pre", build_preprocessor()),
            ("model", RandomForestRegressor(n_estimators=200, max_depth=12,
                                            min_samples_leaf=3, random_state=42, n_jobs=-1)),
        ]),
        "Decision Tree": Pipeline([
            ("pre", build_preprocessor()),
            ("model", DecisionTreeRegressor(max_depth=8, min_samples_leaf=10, random_state=42)),
        ]),
    }


def train_and_select(df: pd.DataFrame) -> tuple[Pipeline, str, dict]:
    """Clean the data, split it, train the four models, and return the best
    pipeline, its name, and every model's metrics. Best = lowest test RMSE, which
    is the loss metric used throughout the project (lower is better)."""
    data = clean(df)
    missing = [c for c in FEATURES + [TARGET] if c not in data.columns]
    if missing:
        raise ValueError(f"Training data missing required columns: {missing}")

    X, y = data[FEATURES], data[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    metrics: dict[str, dict] = {}
    fitted: dict[str, Pipeline] = {}
    for name, pipe in candidate_models().items():
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        metrics[name] = {
            "test_rmse": float(np.sqrt(mean_squared_error(y_te, pred))),
            "test_mae": float(mean_absolute_error(y_te, pred)),
            "test_r2": float(r2_score(y_te, pred)),
        }
        fitted[name] = pipe

    # Pick the lowest test RMSE, but prefer a linear model when it is within 1%
    # of the best: on this data the linear models tie the trees while generalising
    # better and being far lighter to serve. Same rule the notebook documents.
    raw_best = min(metrics, key=lambda k: metrics[k]["test_rmse"])
    linear = ["SGD Linear Regression", "Batch GD Linear Regression"]
    best_linear = min(linear, key=lambda k: metrics[k]["test_rmse"])
    if metrics[best_linear]["test_rmse"] <= metrics[raw_best]["test_rmse"] * 1.01:
        best_name = best_linear
    else:
        best_name = raw_best
    return fitted[best_name], best_name, metrics
