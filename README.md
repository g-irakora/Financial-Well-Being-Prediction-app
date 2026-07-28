# Financial Well-Being Prediction

## Mission and problem
My mission is to move young people toward financial freedom and economic independence.
This project predicts a person's **Financial Well-Being score (0–100)** from a few things
they can influence — financial skill, knowledge, income, savings, employment, age, education
and household size — so they can see where they stand and which levers matter most. It is a
regression problem with a continuous target (deliberately not house-price prediction).

## Public API endpoint
> Deploy the `summative/API` folder to Render (steps below), then replace this URL with yours.

- Base URL: **https://financial-wellbeing-api.onrender.com**
- **Swagger UI (tests run here): https://financial-wellbeing-api.onrender.com/docs**
- Predict (POST): `https://financial-wellbeing-api.onrender.com/predict`
- Health: `https://financial-wellbeing-api.onrender.com/health`

The free tier sleeps when idle, so the first request after a while can take ~30–60s to wake.

Example request body:
```json
{
  "FSscore": 55, "KHscore": 0.4, "PPHHSIZE": 3, "PPINCIMP": 7,
  "SAVINGSRANGES": 5, "agecat": 2, "PPEDUC": 4, "EMPLOY": 2
}
```

## YouTube video demo
Demo (≤ 7 minutes): **https://youtu.be/7N-zqpvMVPk**

## Dataset
CFPB *National Financial Well-Being Survey 2016* Public Use File — a public U.S. government
dataset (consumerfinance.gov / data.gov), 6,394 adults, 217 columns. Target: `FWBscore`.
A copy is at `summative/linear_regression/data/financial_wellbeing.csv`. Source:
https://www.consumerfinance.gov/data-research/financial-well-being-survey-data/

## Model, features and results
Four regression models were compared using **RMSE** as the loss metric (lower is better).
The two linear models are the focus; the tree models are the comparison baselines.

| Model | Type | Test RMSE | Test MAE | Test R² |
|---|---|---|---|---|
| **SGD Linear Regression** (chosen) | linear, stochastic GD | **9.76** | 7.43 | 0.53 |
| Batch GD Linear Regression | linear, batch GD (from scratch) | 9.76 | 7.42 | 0.53 |
| Random Forest | ensemble | 9.71 | 7.44 | 0.54 |
| Decision Tree | tree | 10.20 | 7.82 | 0.49 |

The scores are close. Random Forest has the marginally lowest test RMSE but overfits (train
RMSE 6.79 vs test 9.71). The **linear model is chosen**: it ties the trees on test error while
generalising far better (train ≈ test), is interpretable, and is tiny to serve. It is saved as
`summative/API/model/best_model.pkl`.

8 features are used: numeric `FSscore`, `KHscore`, `PPHHSIZE` (standardised); ordinal `PPINCIMP`,
`SAVINGSRANGES`, `agecat`, `PPEDUC` (ordinal-encoded); nominal `EMPLOY` (one-hot). The ~200 raw
scale items were dropped to avoid leaking the target. All preprocessing lives inside the saved
pipeline, so the API only passes it raw feature values.

## CORS configuration
The API uses CORS middleware with specific values, not a wildcard:
- **Origins** — only our own frontends (the deployed host, the Flutter web dev server, localhost).
  A random website therefore cannot call the API from a visitor's browser. Native mobile apps are
  not browsers, so CORS does not apply to them and the Flutter app works regardless.
- **Methods** — only `GET`, `POST`, `OPTIONS` (the verbs the API uses).
- **Headers** — only `Content-Type` (all a JSON API needs).
- **Credentials** — off, because the API is stateless (no cookies or sessions).

## Data types and range validation
`ProfileRequest` (Pydantic) enforces types and ranges; anything invalid returns HTTP **422**:
`FSscore` float 0–100, `KHscore` float −4 to 4, `PPHHSIZE` int 1–12, and `PPINCIMP` (1–9),
`SAVINGSRANGES` (1–7), `agecat` (1–8), `PPEDUC` (1–5), `EMPLOY` (1–8) each accept only their
valid coded categories.

## Repository layout
```
linear_regression_model/
  render.yaml
  README.md
  summative/
    pyproject.toml
    uv.lock
    linear_regression/
      multivariate.ipynb        # Task 1 notebook (outputs saved inline)
      build_notebook.py         # regenerates+executes the notebook
      make_prediction.py        # standalone prediction script
      data/financial_wellbeing.csv
    API/
      prediction.py             # FastAPI app (predict, retrain, health)
      ml.py                     # shared cleaning / preprocessing / training
      requirements.txt
      model/                    # best_model.pkl, model_metadata.json, training_data.csv
    FlutterApp/                 # single-page mobile app
```

## How to run

### 1. Notebook (Task 1) — uses `uv`
```bash
cd summative
uv sync --extra notebook
cd linear_regression
uv run --project .. --extra notebook python build_notebook.py   # trains + writes the notebook
# or open it interactively:
uv run --project .. --extra notebook jupyter lab multivariate.ipynb
```

### 2. API locally (Task 2)
```bash
cd summative
uv sync
cd API
uv run --project .. uvicorn prediction:app --reload --port 8000
# open http://localhost:8000/docs
```
Retrain (optionally with new labelled rows in the same format, including `FWBscore`):
```bash
curl -X POST http://127.0.0.1:8000/retrain -F "file=@new_rows.csv"   # or no file to retrain on stored data
```

### 3. Deploy the API to Render
1. Push this repo to GitHub.
2. Render → **New → Blueprint** and point it at the repo (it reads `render.yaml`), or a Web Service:
   root dir `summative/API`, build `pip install -r requirements.txt`,
   start `uvicorn prediction:app --host 0.0.0.0 --port $PORT`.
3. Copy the public URL; update it in this README, in `summative/API/prediction.py`
   (`ALLOWED_ORIGINS`), and in the Flutter app (`kApiBaseUrl` in `lib/main.dart`).

### 4. Mobile app (Task 3)
```bash
cd summative/FlutterApp
flutter pub get
flutter run           # on a connected Android/iOS device or emulator
```
The app points at the deployed API via `kApiBaseUrl` in `lib/main.dart`. Fill the 8 fields,
tap **Predict**, and it shows the predicted score or a clear error for missing/out-of-range values.
```bash
flutter build apk --release   # optional: build an installable APK
```
