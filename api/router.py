from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from src.working import run
import shutil
from pathlib import Path
import pandas as pd
import json
import pickle
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Automated ML Pipeline")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:3000"] for stricter control
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_FOLDER = Path("models")
# Temporary upload folder
UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)


class SampleInput(BaseModel):
    target: str  # model name
    features: dict


@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    """Upload a CSV file, run the ML pipeline on it, and return results."""
    if not file.filename.endswith(".csv"):
        return JSONResponse(
            content={"error": "Only CSV files are allowed"}, status_code=400
        )

    # Save the uploaded file to disk
    file_path = UPLOAD_FOLDER / file.filename
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run your existing pipeline
    try:
        result = run(str(file_path))
    except Exception:
        # one retry (kept from original behavior)
        try:
            result = run(str(file_path))
        except Exception as e:
            return JSONResponse(content={"error": str(e)}, status_code=500)

    df = pd.read_csv(file_path)
    target_column = df.columns[-1]

    # Try to extract feature order from printed output
    feature_order_lines = [
        line
        for line in result.split("\n")
        if line.startswith("[") and line.endswith("]")
    ]
    if feature_order_lines:
        line = feature_order_lines[-1]
        # Find first [ and last ] to extract the list
        start = line.find("[")
        end = line.rfind("]")
        if start != -1 and end != -1:
            feature_order_str = line[start : end + 1]
            feature_order = json.loads(feature_order_str.replace("'", '"'))
        else:
            feature_order = df.columns[:-1].tolist()
    else:
        feature_order = df.columns[:-1].tolist()

    # Save feature order for prediction
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

    # Load model and scaler
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

    # Convert input dict to DataFrame
    df = pd.DataFrame([sample.features])

    # Check missing features
    missing = [f for f in feature_order if f not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing features: {missing}")

    # Reorder columns according to training
    X = df[feature_order]

    # Scale and predict
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)

    return {"target": target, "prediction": pred.tolist()[0]}
