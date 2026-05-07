# src/deploymen/online/api.py

from .request import PassengerFeatures
from .response import PredictionResponse, HealthResponse

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, status

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Global state 
model_store: dict = {}


# ── Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = Path(os.environ.get("MODEL_PATH", "models/logistic_pipeline.pkl"))

    if not model_path.exists():
        raise RuntimeError(
            f"Model file not found: {model_path}. "
            "Run the training pipeline first or mount the models directory."
        )

    logger.info("Loading model from: %s", model_path)
    model_store["pipeline"] = joblib.load(model_path)
    model_store["model_file"] = str(model_path)
    logger.info("Model loaded successfully.")

    yield

    model_store.clear()
    
# ── App
app = FastAPI(
    title="Titanic Predictor — No Registry",
    description="Serves predictions from a local model file",
    version="1.0.0",
    lifespan=lifespan,
)

# ── Dependencies 
def get_pipeline():
    if "pipeline" not in model_store:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model_store["pipeline"]

# ── Endpoints 

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="healthy" if "pipeline" in model_store else "unhealthy",
        model_loaded="pipeline" in model_store,
        model_file=model_store.get("model_file", ""),
    )
    
@app.post("/api/v1/predict", response_model=PredictionResponse)
def predict(
    passenger: PassengerFeatures,
    pipeline=Depends(get_pipeline),
):
    input_df = pd.DataFrame([passenger.model_dump()])

    try:
        prediction = int(pipeline.predict(input_df)[0])
        probability = round(float(pipeline.predict_proba(input_df)[0][1]), 4)
    except Exception as e:
        logger.error("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return PredictionResponse(
        survived=prediction,
        probability=probability,
        model_file=model_store.get("model_file", ""),
    )