"""FastAPI service for the Financial Well-Being prediction model (Task 2).

Routes:
    GET  /         redirects to the Swagger UI at /docs
    GET  /health   status + which model is currently loaded
    POST /predict  predicts FWBscore (0-100) from a validated profile
    POST /retrain  appends optional new labelled data and retrains the model

Run locally:
    uv run --project .. uvicorn prediction:app --reload --port 8000
Then open http://localhost:8000/docs
"""
from __future__ import annotations

import io
import json
import os
from enum import IntEnum

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

import ml  # shared cleaning / feature engineering / training logic

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
META_PATH = os.path.join(MODEL_DIR, "model_metadata.json")
DATA_PATH = os.path.join(MODEL_DIR, "training_data.csv")

STATE: dict = {"model": None, "meta": {}}


def load_model() -> None:
    STATE["model"] = joblib.load(MODEL_PATH)
    with open(META_PATH) as f:
        STATE["meta"] = json.load(f)


load_model()


# ---- coded categorical inputs (IntEnum -> Swagger shows the allowed values) ----
class Income(IntEnum):
    lt20k = 1; k20_30 = 2; k30_40 = 3; k40_50 = 4; k50_60 = 5
    k60_75 = 6; k75_100 = 7; k100_150 = 8; k150k_plus = 9


class Savings(IntEnum):
    zero = 1; d1_99 = 2; d100_999 = 3; d1k_5k = 4
    d5k_20k = 5; d20k_75k = 6; d75k_plus = 7


class AgeBand(IntEnum):
    a18_24 = 1; a25_34 = 2; a35_44 = 3; a45_54 = 4
    a55_61 = 5; a62_69 = 6; a70_74 = 7; a75_plus = 8


class Education(IntEnum):
    less_than_hs = 1; high_school = 2; some_college = 3
    bachelors = 4; graduate = 5


class Employment(IntEnum):
    self_employed = 1; full_time = 2; part_time = 3; homemaker = 4
    student = 5; sick_disabled = 6; unemployed = 7; retired = 8


class ProfileRequest(BaseModel):
    """One person's profile. Every field is typed, and numeric fields carry a
    realistic range; the coded fields only accept their valid categories."""

    FSscore: float = Field(..., ge=0, le=100,
        description="Financial skill score, 0-100 (higher = more skilled).", examples=[55])
    KHscore: float = Field(..., ge=-4, le=4,
        description="Financial knowledge score, a standardised value ~ -3 to 3.", examples=[0.4])
    PPHHSIZE: int = Field(..., ge=1, le=12,
        description="Household size (number of people).", examples=[3])
    PPINCIMP: Income = Field(..., description="Household income band 1 (<$20k) to 9 ($150k+).")
    SAVINGSRANGES: Savings = Field(..., description="Savings band 1 ($0) to 7 ($75k+).")
    agecat: AgeBand = Field(..., description="Age band 1 (18-24) to 8 (75+).")
    PPEDUC: Education = Field(..., description="Education 1 (<HS) to 5 (graduate).")
    EMPLOY: Employment = Field(..., description="Employment status 1-8 (see labels).")

    model_config = {
        "json_schema_extra": {
            "example": {
                "FSscore": 55, "KHscore": 0.4, "PPHHSIZE": 3, "PPINCIMP": 7,
                "SAVINGSRANGES": 5, "agecat": 2, "PPEDUC": 4, "EMPLOY": 2,
            }
        }
    }


class PredictionResponse(BaseModel):
    predicted_wellbeing_score: float = Field(..., description="Predicted FWBscore, 0-100.")
    interpretation: str
    model_used: str
    scale: str = "0-100 (higher is better)"


class RetrainResponse(BaseModel):
    status: str
    best_model: str
    n_training_rows: int
    metrics: dict


app = FastAPI(
    title="Financial Well-Being Prediction API",
    description=("Predicts a person's Financial Well-Being score (0-100) from their financial "
                 "skill, knowledge, income, savings, employment, age, education and household "
                 "size. Built to help move young people toward financial freedom."),
    version="1.0.0",
)

# ---- CORS ----
# Set to specific values rather than a wildcard. Reasoning:
#  * Origins: only our own frontends (the deployed API host, a Flutter web dev
#    server, and localhost). A random website therefore cannot call this API from
#    a visitor's browser. Native mobile apps are NOT browsers, so CORS does not
#    apply to them and the Flutter Android/iOS app keeps working regardless.
#  * Methods: only GET, POST and OPTIONS - the verbs this API actually uses.
#  * Headers: only Content-Type - all a JSON API needs.
#  * Credentials: off, because the API is stateless (no cookies or sessions).
ALLOWED_ORIGINS = [
    "http://localhost",
    "http://localhost:3000",   # Flutter web dev server
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "https://financial-wellbeing-api.onrender.com",  # deployed host; update to your Render URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
    allow_credentials=False,
    max_age=600,
)


def _interpret(score: float) -> str:
    if score < 40:
        return "Low financial well-being - building savings and financial skill would help most."
    if score < 55:
        return "Below average - steady progress on savings and income would lift this."
    if score < 70:
        return "Moderate financial well-being - a reasonably secure footing."
    return "High financial well-being - strong financial security."


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": STATE["model"] is not None,
        "model_used": STATE["meta"].get("best_model"),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: ProfileRequest) -> PredictionResponse:
    """Predict the Financial Well-Being score (0-100) for one profile."""
    if STATE["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    row = pd.DataFrame([{
        "FSscore": payload.FSscore,
        "KHscore": payload.KHscore,
        "PPHHSIZE": payload.PPHHSIZE,
        "PPINCIMP": int(payload.PPINCIMP),
        "SAVINGSRANGES": int(payload.SAVINGSRANGES),
        "agecat": int(payload.agecat),
        "PPEDUC": int(payload.PPEDUC),
        "EMPLOY": int(payload.EMPLOY),
    }])[ml.FEATURES]
    try:
        pred = float(STATE["model"].predict(row)[0])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")
    pred = max(0.0, min(100.0, pred))  # clamp to the valid score range
    return PredictionResponse(
        predicted_wellbeing_score=round(pred, 1),
        interpretation=_interpret(pred),
        model_used=STATE["meta"].get("best_model", "unknown"),
    )


@app.post("/retrain", response_model=RetrainResponse)
async def retrain(file: UploadFile | None = File(default=None)) -> RetrainResponse:
    """Retrain the model, optionally on newly uploaded data.

    Upload a CSV of new labelled rows (same columns as the survey, including
    `FWBscore`). The new rows are cleaned, added to the stored training data, all
    four models are retrained, and the best one replaces the current model. Called
    without a file, it simply retrains on the data already stored - useful for
    streamed data that has been appended to the store by another process.
    """
    base = pd.read_csv(DATA_PATH)

    if file is not None:
        content = await file.read()
        try:
            new_df = pd.read_csv(io.BytesIO(content))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not parse uploaded CSV: {exc}")
        new_df = ml.clean(new_df)
        missing = [c for c in ml.FEATURES + [ml.TARGET] if c not in new_df.columns]
        if missing:
            raise HTTPException(status_code=422,
                                detail=f"Uploaded data missing required columns: {missing}")
        combined = pd.concat([base, new_df[ml.FEATURES + [ml.TARGET]]], ignore_index=True)
    else:
        combined = base

    if len(combined) < 50:
        raise HTTPException(status_code=422, detail="Not enough data to train, need at least 50 rows.")

    try:
        best_model, best_name, metrics = ml.train_and_select(combined)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {exc}")

    combined.to_csv(DATA_PATH, index=False)
    joblib.dump(best_model, MODEL_PATH)
    meta = STATE["meta"]
    meta["best_model"] = best_name
    meta["metrics"] = {k: {"Test RMSE": round(v["test_rmse"], 3),
                           "Test MAE": round(v["test_mae"], 3),
                           "Test R2": round(v["test_r2"], 3)} for k, v in metrics.items()}
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    load_model()

    return RetrainResponse(status="retrained", best_model=best_name,
                           n_training_rows=len(combined), metrics=metrics)
