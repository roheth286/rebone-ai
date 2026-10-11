import pandas as pd
import numpy as np
from typing import Dict, Any
from src.ml.model_loader import get_ml_assets

SCORE_MIN = -0.1500
SCORE_MAX = 0.1500
THRESHOLD = 0.8420


def run_inference(patient_dict: Dict[str, Any]) -> Dict[str, Any]:
    assets = get_ml_assets()
    pipeline = assets["pipeline"]
    model = assets["model"]
    selected_features = assets["selected_features"]
    high_impact_cols = assets["high_impact_cols"]

    from src.preprocessing.pipeline import get_pipeline_feature_names

    # 1. Convert input to DataFrame and map L1_4T key to L1.4T
    input_data = patient_dict.copy()
    if "L1_4T" in input_data:
        input_data["L1.4T"] = input_data.pop("L1_4T")

    # Canonical 15 clinical features used during pipeline fitting and model training
    clinical_features_15 = [
        "Gender", "Age", "BMI", "L1.4T", "FNT", "TLT",
        "Calsium", "Calcitriol", "Bisphosphonate", "Calcitonin",
        "VT", "VD", "OP", "Smoking", "Drinking"
    ]

    # Ensure required raw columns exist with appropriate defaults
    raw_record = {}
    for feat in clinical_features_15:
        raw_record[feat] = input_data.get(feat, 0.0)

    df_raw = pd.DataFrame([raw_record])

    # Convert Gender string or integer to standard format expected by pipeline
    if "Gender" in df_raw.columns:
        df_raw["Gender"] = df_raw["Gender"].apply(
            lambda val: 1 if str(val).strip().lower() in ["female", "f", "1", "2"] and str(val).strip().lower() in ["female", "f", "2"] else (0 if str(val).strip().lower() in ["male", "m", "1", "0"] else val)
        )

    # 2. Preprocess using fitted pipeline
    if not hasattr(pipeline, "transform"):
        raise RuntimeError("Loaded preprocessing pipeline does not support 'transform'.")

    transformed_arr = pipeline.transform(df_raw)
    if hasattr(transformed_arr, "toarray"):
        transformed_arr = transformed_arr.toarray()

    # Determine column names in the exact order output by ColumnTransformer
    processed_cols = get_pipeline_feature_names(pipeline, df_raw)
    df_preprocessed = pd.DataFrame(transformed_arr, columns=processed_cols)

    # 3. Select 12 Isolation Forest features
    df_sel = pd.DataFrame()
    for col in selected_features:
        if col in df_preprocessed.columns:
            df_sel[col] = df_preprocessed[col]
        else:
            raise KeyError(f"Expected model feature '{col}' missing from preprocessed data columns: {list(df_preprocessed.columns)}")

    # 4. Weighted Feature Duplication
    df_weighted = df_sel.copy()
    for col in high_impact_cols:
        if col in df_sel.columns:
            df_weighted[f"{col}_dup1"] = df_sel[col]
            df_weighted[f"{col}_dup2"] = df_sel[col]

    # 5. Raw Anomaly Score calculation
    raw_decision = model.decision_function(df_weighted)[0]
    raw_score = -float(raw_decision)

    # 6. Score Normalization & Clipping to [0, 1]
    normalized_score = (raw_score - SCORE_MIN) / (SCORE_MAX - SCORE_MIN)
    normalized_score = float(np.clip(normalized_score, 0.0, 1.0))
    normalized_score = round(normalized_score, 4)

    # 7. Threshold Classification
    if normalized_score >= THRESHOLD:
        classification = "High Risk"
        decision = "Anomaly Detected"
    else:
        classification = "Low Risk"
        decision = "Normal"

    return {
        "risk_score": normalized_score,
        "classification": classification,
        "threshold_used": THRESHOLD,
        "decision": decision,
        "model_version": "isolation_forest_v1.0"
    }
