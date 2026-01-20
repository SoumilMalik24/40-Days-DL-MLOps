"""
Day 30 — Model Serving with FastAPI
Simple inference API for a trained ML model
"""

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load model at startup
model = joblib.load("models/model.joblib")

app = FastAPI(title="ML Model Inference API")

class PredictionRequest(BaseModel):
    features: list[float]

class PredictionResponse(BaseModel):
    prediction: float

@app.get("/")
def root():
    return {"message": "ML Model Inference API is running"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    features = np.array(request.features).reshape(1, -1)
    prediction = model.predict(features)[0]
    return {"prediction": float(prediction)}
