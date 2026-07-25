import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, fbeta_score, precision_recall_curve

def cross_validate_isolation_forest(dataset_path, metrics_output_path, num_folds=5, random_state=42):
    # 1. Load the full cleaned dataset
    df = pd.read_csv(dataset_path)
    target = 'Fracture'
    
    X = df.drop(columns=[target])
    y = df[target]
    
    # 2. Select clinical features
    selected_features = [
        "Gender", "Age", "BMI", "VT", "VD", "Calcitriol", "Calcitonin", 
        "OP", "Calsium", "L1.4T", "FNT", "TLT"
    ]
    
    # Check if all selected features exist
    missing_feats = [f for f in selected_features if f not in X.columns]
    if missing_feats:
        selected_features = [f for f in selected_features if f in X.columns]
        
    X_sel = X[selected_features].copy()
    
    # Preprocess categorical columns if any
    for col in X_sel.columns:
        if not pd.api.types.is_numeric_dtype(X_sel[col]):
            X_sel[col] = X_sel[col].astype(str).astype('category').cat.codes
            
    # 3. Initialize Stratified K-Fold
    cv = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_state)
    
    cv_results = []
    
    print(f"Starting Stratified {num_folds}-Fold Cross-Validation on: {dataset_path}")
    print(f"Selected Features ({len(selected_features)}): {selected_features}")
    
    # 4. Run CV loop
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_sel, y), 1):
        X_train, X_val = X_sel.iloc[train_idx].copy(), X_sel.iloc[val_idx].copy()
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Impute missing values
        imputer = SimpleImputer(strategy='median')
        X_train_imp = imputer.fit_transform(X_train)
        X_val_imp = imputer.transform(X_val)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imp)
        X_val_scaled = scaler.transform(X_val_imp)
        
        # Separate healthy class for training
        healthy_mask = (y_train == 0)
        
        # Filter training fold to remove borderline osteoporotic cases (OP=1 or T-score < -2.0)
        orig_OP = X_train.iloc[healthy_mask.values]["OP"] if "OP" in X_train.columns else None
        orig_TLT = X_train.iloc[healthy_mask.values]["TLT"] if "TLT" in X_train.columns else None
        orig_FNT = X_train.iloc[healthy_mask.values]["FNT"] if "FNT" in X_train.columns else None
        
        clean_mask = np.ones(healthy_mask.sum(), dtype=bool)
        if orig_OP is not None:
            clean_mask &= (orig_OP <= 0).values
        if orig_TLT is not None:
            clean_mask &= (orig_TLT > -2.0).values
        if orig_FNT is not None:
            clean_mask &= (orig_FNT > -2.0).values
            
        X_train_healthy = X_train_scaled[healthy_mask][clean_mask]
        
        # Weighted Feature Duplication (Method 4)
        high_impact_cols = [col for col in ["VT", "VD", "Calcitriol", "Calcitonin", "OP"] if col in X_train.columns]
        
        # Duplicate in training
        X_train_weighted = pd.DataFrame(X_train_healthy, columns=selected_features)
        for col in high_impact_cols:
            X_train_weighted[col + "_dup1"] = X_train_weighted[col]
            X_train_weighted[col + "_dup2"] = X_train_weighted[col]
            
        # Duplicate in validation
        X_val_weighted = pd.DataFrame(X_val_scaled, columns=selected_features)
        for col in high_impact_cols:
            X_val_weighted[col + "_dup1"] = X_val_weighted[col]
            X_val_weighted[col + "_dup2"] = X_val_weighted[col]
            
        # Train model
        model = IsolationForest(contamination=0.02, random_state=random_state)
        model.fit(X_train_weighted)
        
        # Predict anomaly scores
        val_scores = -model.decision_function(X_val_weighted)
        
        # Normalize scores to [0, 1] range
        s_min, s_max = val_scores.min(), val_scores.max()
        if s_max > s_min:
            val_probs = (val_scores - s_min) / (s_max - s_min)
        else:
            val_probs = val_scores
            
        # Calculate ROC-AUC
        auc_score = roc_auc_score(y_val, val_probs)
        
        # Optimize threshold for F1-score
        precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)
        f1_scores = np.zeros_like(thresholds, dtype=float)
        for idx, (p, r) in enumerate(zip(precisions[:-1], recalls[:-1])):
            if (p + r) > 0:
                f1_scores[idx] = (2 * p * r) / (p + r)
                
        if len(thresholds) > 0:
            opt_idx = np.argmax(f1_scores)
            opt_thresh = thresholds[opt_idx]
            best_prec = precisions[opt_idx]
            best_rec = recalls[opt_idx]
            best_f1 = f1_scores[opt_idx]
            best_f2 = (5 * best_prec * best_rec) / (4 * best_prec + best_rec + 1e-9)
        else:
            opt_thresh = 0.5
            best_prec = precision_score(y_val, val_probs >= 0.5, zero_division=0)
            best_rec = recall_score(y_val, val_probs >= 0.5)
            best_f1 = f1_score(y_val, val_probs >= 0.5, zero_division=0)
            best_f2 = fbeta_score(y_val, val_probs >= 0.5, beta=2, zero_division=0)
            
        cv_results.append({
            'fold': fold,
            'auc': auc_score,
            'precision': best_prec,
            'recall': best_rec,
            'f1': best_f1,
            'f2': best_f2
        })
        print(f"  Fold {fold}: AUC = {auc_score:.4f} | Precision = {best_prec:.4f} | Recall = {best_rec:.4f} | F1 = {best_f1:.4f}")

    # 5. Compute average results
    results_df = pd.DataFrame(cv_results)
    mean_auc = results_df['auc'].mean()
    mean_prec = results_df['precision'].mean()
    mean_rec = results_df['recall'].mean()
    mean_f1 = results_df['f1'].mean()
    mean_f2 = results_df['f2'].mean()
    
    print("\n=== 5-Fold Cross-Validation Averages ===")
    print(f"ROC-AUC:   {mean_auc:.4f} (+/- {results_df['auc'].std():.4f})")
    print(f"Precision: {mean_prec:.4f} (+/- {results_df['precision'].std():.4f})")
    print(f"Recall:    {mean_rec:.4f} (+/- {results_df['recall'].std():.4f})")
    print(f"F1-Score:  {mean_f1:.4f} (+/- {results_df['f1'].std():.4f})")
    print(f"F2-Score:  {mean_f2:.4f} (+/- {results_df['f2'].std():.4f})")
    
    # 6. Save CV metrics to JSON
    os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
    output_data = {
        "model_type": "IsolationForest_CV",
        "num_folds": num_folds,
        "metrics": {
            "roc_auc": float(mean_auc),
            "precision": float(mean_prec),
            "recall": float(mean_rec),
            "f1_score": float(mean_f1),
            "f2_score": float(mean_f2)
        },
        "fold_details": cv_results
    }
    
    with open(metrics_output_path, "w") as f:
        json.dump(output_data, f, indent=4)
    print(f"Cross-Validation metrics saved to: {metrics_output_path}")
    
    return output_data
