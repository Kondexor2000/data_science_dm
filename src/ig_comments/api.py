from __future__ import annotations

import os
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/model.joblib"))
app = FastAPI(title="Instagram Comment Classifier", version="0.1.0")
model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok" if model is not None else "model_missing"}


@app.post("/predict")
def predict(request: PredictionRequest) -> dict:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not available")
    probabilities = model.predict_proba([request.text])[0]
    classes = model.classes_.tolist()
    scores = {label: float(score) for label, score in zip(classes, probabilities)}
    return {"label": max(scores, key=scores.get), "probabilities": scores,
            "human_review_recommended": True}
