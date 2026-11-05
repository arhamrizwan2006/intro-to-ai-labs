# app/main.py
"""Titanic survival prediction service.

Endpoints
  GET  /health            liveness + which artefacts are loaded
  POST /predict           pickled encoders + SVC; unknown category -> 400
  POST /predict/pipeline  same request through the bundled Pipeline (Exercise 1)
  POST /predict/lenient   pipeline, unknown category -> code -1 and reported (Exercise 2)
  GET  /metrics           Prometheus counters and latency histogram (Exercise 3)

Artefacts are loaded once at start-up (lifespan), never per request.
"""

import pickle
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import (CONTENT_TYPE_LATEST, CollectorRegistry, Counter,
                               Histogram, generate_latest)

from .schemas import LenientResponse, PredictRequest, PredictResponse

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"
FEATURES = ["PClass", "Gender", "Sibling", "Embarked"]
ENCODER_FILES = {"PClass": "pclass_encoder.pkl", "Gender": "gender_encoder.pkl",
                 "Sibling": "sibling_encoder.pkl", "Embarked": "embarked_encoder.pkl"}
LABELS = {0: "NOT SURVIVED", 1: "SURVIVED"}


def _load_pickle(name):
    with open(ARTIFACTS / name, "rb") as handle:
        return pickle.load(handle)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = _load_pickle("svc_trained_model.pkl")
    app.state.encoders = {f: _load_pickle(file) for f, file in ENCODER_FILES.items()}
    app.state.pipeline = joblib.load(ARTIFACTS / "titanic_pipeline.joblib")
    yield


app = FastAPI(title="Titanic Survival API", version="1.0.0", lifespan=lifespan)

# Own registry so that re-importing the app (tests) never double-registers a metric.
registry = CollectorRegistry()
REQUESTS = Counter("api_requests_total", "Requests handled", ["path", "method", "status"],
                   registry=registry)
LATENCY = Histogram("api_request_seconds", "Request latency in seconds", ["path"],
                    registry=registry)


@app.middleware("http")
async def measure(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    path = request.scope.get("route").path if request.scope.get("route") else "unmatched"
    LATENCY.labels(path).observe(time.perf_counter() - start)
    REQUESTS.labels(path, request.method, str(response.status_code)).inc()
    return response


def _frame(body: PredictRequest) -> pd.DataFrame:
    return pd.DataFrame([body.model_dump()])[FEATURES]


def _reply(label: int, probability: float) -> dict:
    return {"predicted_label": label, "prediction": LABELS[label], "confidence": round(probability, 4)}


@app.get("/health")
def health(request: Request):
    return {"status": "ok", "model_loaded": hasattr(request.app.state, "model"),
            "encoders": sorted(request.app.state.encoders)}


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest, request: Request):
    state = request.app.state
    codes = []
    for feature in FEATURES:
        known = list(state.encoders[feature].classes_)
        value = getattr(body, feature)
        if value not in known:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown value '{value}' for {feature}. Allowed: {known}")
        codes.append(int(state.encoders[feature].transform([value])[0]))
    row = pd.DataFrame([codes], columns=FEATURES)
    proba = state.model.predict_proba(row)[0]
    label = int(state.model.predict(row)[0])
    return _reply(label, float(proba[list(state.model.classes_).index(label)]))


@app.post("/predict/pipeline", response_model=PredictResponse)
def predict_pipeline(body: PredictRequest, request: Request):
    pipe = request.app.state.pipeline
    row = _frame(body)
    for feature, cats in zip(FEATURES, pipe.named_steps["encode"].categories_):
        if getattr(body, feature) not in list(cats):
            raise HTTPException(
                status_code=400,
                detail=f"Unknown value '{getattr(body, feature)}' for {feature}. Allowed: {list(cats)}")
    proba = pipe.predict_proba(row)[0]
    label = int(pipe.predict(row)[0])
    return _reply(label, float(proba[list(pipe.classes_).index(label)]))


@app.post("/predict/lenient", response_model=LenientResponse)
def predict_lenient(body: PredictRequest, request: Request):
    pipe = request.app.state.pipeline
    unknown = [f for f, cats in zip(FEATURES, pipe.named_steps["encode"].categories_)
               if getattr(body, f) not in list(cats)]
    row = _frame(body)
    proba = pipe.predict_proba(row)[0]
    label = int(pipe.predict(row)[0])
    return {**_reply(label, float(proba[list(pipe.classes_).index(label)])),
            "unknown_fields": unknown}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)
