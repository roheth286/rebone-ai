import os
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

def train_isolation_forest_model(train_path, model_output_path, random_state=42):
    # 1. Load data
    train_df = pd.read_csv(train_path)
    X_train = train_df.drop(columns=["Fracture"])
    y_train = train_df["Fracture"]
    
    print(f"Loaded training data for Isolation Forest: {X_train.shape}")
    
    # 2. Select clinical features (Idea 2)
    selected_features = [
        "Gender", "Age", "BMI", "VT", "VD", "Calcitriol", "Calcitonin", 
        "OP", "Calsium", "L1.4T", "FNT", "TLT"
    ]
    
    # Check if all selected features exist in training data
    missing_feats = [f for f in selected_features if f not in X_train.columns]
    if missing_feats:
        print(f"Warning: Selected features {missing_feats} not found. Using available columns.")
        selected_features = [f for f in selected_features if f in X_train.columns]
        
    X_train_sel = X_train[selected_features].copy()
    
    # 3. Filter training set to include ONLY healthy patients (Semi-Supervised)
    healthy_mask = (y_train == 0)
    X_train_healthy = X_train_sel[healthy_mask].copy()
    
    # Apply Method 2: Clean the healthy training set by removing borderline cases (OP=1 or T-score < -2.0)
    # This establishes a "super-healthy" baseline profile
    if "OP" in X_train_healthy.columns:
        # Exclude diagnosed osteoporosis
        X_train_healthy = X_train_healthy[X_train_healthy["OP"] <= 0]
    if "TLT" in X_train_healthy.columns:
        # Exclude severe osteopenia/osteoporosis in Total Hip T-score
        X_train_healthy = X_train_healthy[X_train_healthy["TLT"] > -2.0]
    if "FNT" in X_train_healthy.columns:
        # Exclude severe osteopenia/osteoporosis in Femoral Neck T-score
        X_train_healthy = X_train_healthy[X_train_healthy["FNT"] > -2.0]
        
    print(f"Filtered training set to {X_train_healthy.shape[0]} 'super-healthy' patients (discarded fracture cases and osteoporotic baseline patients).")
    
    # 4. Method 4: Weighted Feature Duplication (duplicate high-impact features)
    high_impact_cols = [col for col in ["VT", "VD", "Calcitriol", "Calcitonin", "OP"] if col in X_train_healthy.columns]
    X_train_weighted = X_train_healthy.copy()
    for col in high_impact_cols:
        X_train_weighted[col + "_dup1"] = X_train_healthy[col]
        X_train_weighted[col + "_dup2"] = X_train_healthy[col]
        
    print(f"Applied weighted feature duplication. Shape of training matrix: {X_train_weighted.shape}")
    
    # 5. Initialize and train Isolation Forest
    model = IsolationForest(contamination=0.02, random_state=random_state)
    model.fit(X_train_weighted)
    
    # 6. Save model and metadata (selected features and duplication details) to disk
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    model_data = {
        'model': model,
        'selected_features': selected_features,
        'high_impact_cols': high_impact_cols
    }
    joblib.dump(model_data, model_output_path)
    print(f"Isolation Forest model and metadata saved successfully to: {model_output_path}")
    
    return model_data
