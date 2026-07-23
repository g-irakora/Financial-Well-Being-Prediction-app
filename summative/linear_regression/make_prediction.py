"""Standalone prediction script (Task 1 -> feeds Task 2).

Loads the best saved model and turns one raw profile into a predicted Financial
Well-Being score. The FastAPI /predict endpoint in Task 2 reuses this exact logic.

Run:  uv run --project .. python make_prediction.py
"""
import os
import sys

import joblib
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "API")))
import ml  # noqa: E402

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "API", "model", "best_model.pkl")


def predict_wellbeing(profile: dict) -> float:
    """profile: raw feature values keyed by ml.FEATURES. Returns FWBscore (0-100)."""
    model = joblib.load(MODEL_PATH)
    row = pd.DataFrame([profile])[ml.FEATURES]
    return float(model.predict(row)[0])


if __name__ == "__main__":
    example = {
        "FSscore": 55, "KHscore": 0.4, "PPHHSIZE": 3, "PPINCIMP": 7,
        "SAVINGSRANGES": 5, "agecat": 2, "PPEDUC": 4, "EMPLOY": 2,
    }
    print("Profile:", example)
    print("Predicted Financial Well-Being score: %.1f / 100" % predict_wellbeing(example))
