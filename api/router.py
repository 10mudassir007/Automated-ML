"""
Application entrypoint.

Run locally:
    uvicorn main:app --reload
"""

import os
import json
import pickle
import shutil
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.working import run

app = FastAPI(title="Automated ML Pipeline")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_FOLDER = Path("models")
UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)

class SampleInput(BaseModel):
    target: str  # model name
    features: dict

# --- ML Pipeline Endpoints ---

@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    """Upload a CSV file, run the ML pipeline on it, and return results."""
    if not file.filename.endswith(".csv"):
        return JSONResponse(
            content={"error": "Only CSV files are allowed"}, status_code=400
        )

    file_path = UPLOAD_FOLDER / file.filename
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = run(str(file_path))
    except Exception:
        try:
            result = run(str(file_path))
        except Exception as e:
            return JSONResponse(content={"error": str(e)}, status_code=500)

    df = pd.read_csv(file_path)
    target_column = df.columns[-1]

    feature_order_lines = [
        line
        for line in result.split("\n")
        if line.startswith("[") and line.endswith("]")
    ]
    if feature_order_lines:
        line = feature_order_lines[-1]
        start = line.find("[")
        end = line.rfind("]")
        if start != -1 and end != -1:
            feature_order_str = line[start : end + 1]
            feature_order = json.loads(feature_order_str.replace("'", '"'))
        else:
            feature_order = df.columns[:-1].tolist()
    else:
        feature_order = df.columns[:-1].tolist()

    feature_order_path = MODELS_FOLDER / f"{target_column}_features.json"
    MODELS_FOLDER.mkdir(exist_ok=True)
    with open(feature_order_path, "w") as f:
        json.dump(feature_order, f)

    return {
        "filename": file.filename,
        "model_name": target_column,
        "results": result.split("\n"),
        "feature_order": feature_order,
    }


@app.post("/predict")
async def predict(sample: SampleInput):
    """Predict for a single sample provided as JSON."""
    target = sample.target

    model_path = MODELS_FOLDER / f"{target}_model.pkl"
    scaler_path = MODELS_FOLDER / f"{target}_scaler.pkl"
    feature_order_path = MODELS_FOLDER / f"{target}_features.json"

    if (
        not model_path.exists()
        or not scaler_path.exists()
        or not feature_order_path.exists()
    ):
        raise HTTPException(
            status_code=400,
            detail=f"Model, scaler, or feature order not found for '{target}'",
        )

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    with open(feature_order_path, "r") as f:
        feature_order = json.load(f)

    df = pd.DataFrame([sample.features])

    missing = [f for f in feature_order if f not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing features: {missing}")

    X = df[feature_order]
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)

    return {"target": target, "prediction": pred.tolist()[0]}

# --- Frontend Serving Routes ---

# Assumes your dist folder sits in the root directory relative to main.py
# --- Frontend Serving Routes ---

# .parent gets the 'api' folder, second .parent gets the 'automated-ml' root folder
dist_path = Path(__file__).resolve().parent.parent / "dist"

# Debug verification line to print in your terminal window on boot
print(f"\n--> FRONTEND PATH CHECK: {dist_path} | EXISTS: {dist_path.exists()}\n")

if dist_path.exists():
    # 1. Mount static assets folder so index.html can load JS and CSS files
    app.mount("/assets", StaticFiles(directory=str(dist_path / "assets")), name="assets")

    # 2. Explicitly handle the root URL path
    @app.get("/")
    def serve_root():
        return FileResponse(str(dist_path / "index.html"))

    # 3. Catch-all route to serve the index.html for SPA router fallback support
    @app.get("/{catchall:path}")
    def serve_frontend(catchall: str):
        # Allow backend 404 errors for API requests to pass through cleanly
        if catchall.startswith("api"):
            return JSONResponse(content={"error": "Not Found"}, status_code=404)
        return FileResponse(str(dist_path / "index.html"))
else:
    print(f"\n[WARNING] Frontend dist folder not found at: {dist_path}\n")