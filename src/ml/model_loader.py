import os
import joblib
from typing import Dict, Any

ml_assets: Dict[str, Any] = {}


def get_asset_path(filename: str) -> str:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(base_dir, "outputs", "models", filename)


def load_ml_assets() -> Dict[str, Any]:
    global ml_assets

    pipeline_path = get_asset_path("preprocessing_pipeline.joblib")
    if not os.path.exists(pipeline_path):
        pipeline_path = get_asset_path("pipeline.joblib")

    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(f"Preprocessing pipeline joblib not found at: {pipeline_path}")

    model_path = get_asset_path("isolation_forest.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Isolation Forest joblib not found at: {model_path}")

    pipeline = joblib.load(pipeline_path)
    model_dict = joblib.load(model_path)

    ml_assets["pipeline"] = pipeline
    ml_assets["model"] = model_dict["model"]
    ml_assets["selected_features"] = model_dict.get("selected_features", [])
    ml_assets["high_impact_cols"] = model_dict.get("high_impact_cols", [])

    print("Loaded ML preprocessing pipeline and Isolation Forest model into memory.")
    return ml_assets


def get_ml_assets() -> Dict[str, Any]:
    if not ml_assets:
        return load_ml_assets()
    return ml_assets
