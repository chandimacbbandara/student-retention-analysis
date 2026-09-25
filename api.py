import os
import glob
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List

# Patch for older sklearn versions (solves _RemainderColsList loading issue)
import sklearn.compose._column_transformer
class _RemainderColsList(list):
    pass
sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList

app = FastAPI(title="Student Retention API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

models_cache = {}
model_features = {}
label_encoder = None

def load_models():
    global label_encoder
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    
    try:
        le_path = os.path.join(models_dir, "target_label_encoder.joblib")
        if os.path.exists(le_path):
            label_encoder = joblib.load(le_path)
    except Exception as e:
        print(f"Error loading label encoder: {e}")

    joblib_files = [
        os.path.join(models_dir, "final_elastic_net_model.joblib"),
        os.path.join(models_dir, "final_lasso_model.joblib"),
        os.path.join(models_dir, "final_xgboost_model.joblib")
    ]
    for f in joblib_files:
        filename = os.path.basename(f)
        if filename == "target_label_encoder.joblib":
            continue
            
        try:
            model_name = filename.replace(".joblib", "")
            artifacts = joblib.load(f)
            
            features = []
            pipeline = None
            
            if isinstance(artifacts, dict):
                pipeline = artifacts.get("pipeline")
                if "feature_names_in" in artifacts:
                    features = list(artifacts["feature_names_in"])
                elif hasattr(pipeline, "feature_names_in_"):
                    features = list(pipeline.feature_names_in_)
            else:
                pipeline = artifacts
                if hasattr(pipeline, "feature_names_in_"):
                    features = list(pipeline.feature_names_in_)
                    
            if pipeline is not None and features:
                models_cache[model_name] = pipeline
                model_features[model_name] = features
                print(f"Loaded {model_name} with {len(features)} features.")
        except Exception as e:
            print(f"Failed to load {filename}: {e}")

@app.on_event("startup")
async def startup_event():
    load_models()

@app.get("/models")
async def get_models():
    """Returns a list of available models and their required features"""
    return {
        "models": [
            {
                "id": model_id,
                "name": model_id.replace("_", " ").title().replace("Xgboost", "XGBoost"),
                "features": features
            }
            for model_id, features in model_features.items()
        ]
    }

class PredictRequest(BaseModel):
    model_id: str
    features: Dict[str, float]

@app.post("/predict")
async def predict(request: PredictRequest):
    if request.model_id not in models_cache:
        raise HTTPException(status_code=404, detail="Model not found")
        
    if label_encoder is None:
        raise HTTPException(status_code=500, detail="Label encoder not loaded")
        
    pipeline = models_cache[request.model_id]
    required_features = model_features[request.model_id]
    
    # Construct the input row based on required features, defaulting missing ones to 0
    row = {f: request.features.get(f, 0.0) for f in required_features}
    
    # Create DataFrame enforcing column order
    df = pd.DataFrame([row], columns=required_features)
    
    try:
        proba = pipeline.predict_proba(df)[0]
        pred_int = int(np.argmax(proba))
        pred_label = label_encoder.inverse_transform([pred_int])[0]
        
        probabilities = {
            str(cls): float(p) for cls, p in zip(label_encoder.classes_, proba)
        }
        
        return {
            "prediction": pred_label,
            "probabilities": probabilities
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Mount static files and handle SPA routing
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Check if dist directory exists (for local development vs docker)
dist_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "dist")

if os.path.exists(dist_dir):
    # Mount the 'assets' directory (and other static dirs) directly if needed,
    # but the easiest way is to mount the whole dist folder under a specific path,
    # or handle the root manually.
    
    # We will mount the static assets
    app.mount("/assets", StaticFiles(directory=os.path.join(dist_dir, "assets")), name="assets")
    
    # Also mount the public forecast image if it's in the dist root
    @app.get("/forecast.png")
    async def serve_forecast():
        return FileResponse(os.path.join(dist_dir, "forecast.png"))
        
    # Catch-all route to serve the SPA index.html
    @app.get("/{catchall:path}")
    async def serve_spa(catchall: str):
        index_file = os.path.join(dist_dir, "index.html")
        if os.path.exists(os.path.join(dist_dir, catchall)) and os.path.isfile(os.path.join(dist_dir, catchall)):
            return FileResponse(os.path.join(dist_dir, catchall))
        return FileResponse(index_file)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
