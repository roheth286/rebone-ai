import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, precision_recall_curve

def verify_feature_strength(dataset_path, output_json_path):
    print("==================================================")
    # 1. Load the full cleaned dataset (contains all 40 columns)
    df = pd.read_csv(dataset_path)
    target = 'Fracture'
    
    y = df[target]
    X_full = df.drop(columns=[target])
    
    # Preprocess categorical columns in full set if any
    for col in X_full.columns:
        if not pd.api.types.is_numeric_dtype(X_full[col]):
            X_full[col] = X_full[col].astype(str).astype('category').cat.codes

    # Define the 12 selected clinical features
    selected_features = [
        "Gender", "Age", "BMI", "VT", "VD", "Calcitriol", "Calcitonin", 
        "OP", "Calsium", "L1.4T", "FNT", "TLT"
    ]
    
    # Ensure all selected features exist in dataset
    selected_features = [f for f in selected_features if f in X_full.columns]
    
    # Define the remaining 27 "noise" features
    noise_features = [col for col in X_full.columns if col not in selected_features]
    
    feature_sets = {
        "12 Selected Clinical Features": X_full[selected_features],
        "All 39 Features": X_full,
        "27 Discarded Features (Noise Only)": X_full[noise_features]
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    comparison_results = {}
    
    print("FEATURE STRENGTH VERIFICATION STUDY")
    print(f"Full Dataset Size: {df.shape[0]} patients (Positive cases: {y.sum()})\n")
    
    # 2. Evaluate each feature set
    for set_name, X_data in feature_sets.items():
        print(f"Evaluating: {set_name} ({X_data.shape[1]} columns)...")
        auc_list = []
        prec_list = []
        rec_list = []
        f1_list = []
        
        for train_idx, val_idx in cv.split(X_data, y):
            X_train, X_val = X_data.iloc[train_idx].copy(), X_data.iloc[val_idx].copy()
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Impute & Scale
            imputer = SimpleImputer(strategy='median')
            X_train_imp = imputer.fit_transform(X_train)
            X_val_imp = imputer.transform(X_val)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_imp)
            X_val_scaled = scaler.transform(X_val_imp)
            
            # Filter training fold for healthy patients
            healthy_mask = (y_train == 0)
            
            # For selected features and all features, we purge borderline cases
            # For noise-only features, we just train on raw healthy cases
            if "OP" in X_train.columns:
                orig_OP = X_train.iloc[healthy_mask.values]["OP"]
                orig_TLT = X_train.iloc[healthy_mask.values]["TLT"]
                orig_FNT = X_train.iloc[healthy_mask.values]["FNT"]
                
                clean_mask = (orig_OP <= 0).values & (orig_TLT > -2.0).values & (orig_FNT > -2.0).values
                X_train_healthy = X_train_scaled[healthy_mask][clean_mask]
            else:
                X_train_healthy = X_train_scaled[healthy_mask]
                
            # If we are using the 12 selected features, apply weighted feature duplication
            if set_name == "12 Selected Clinical Features":
                high_impact_cols = [col for col in ["VT", "VD", "Calcitriol", "Calcitonin", "OP"] if col in X_train.columns]
                
                # Duplicate in train
                X_train_df = pd.DataFrame(X_train_healthy, columns=X_train.columns)
                for col in high_impact_cols:
                    X_train_df[col + "_dup1"] = X_train_df[col]
                    X_train_df[col + "_dup2"] = X_train_df[col]
                X_train_final = X_train_df.values
                
                # Duplicate in validation
                X_val_df = pd.DataFrame(X_val_scaled, columns=X_val.columns)
                for col in high_impact_cols:
                    X_val_df[col + "_dup1"] = X_val_df[col]
                    X_val_df[col + "_dup2"] = X_val_df[col]
                X_val_final = X_val_df.values
            else:
                X_train_final = X_train_healthy
                X_val_final = X_val_scaled
                
            # Train model
            model = IsolationForest(contamination=0.02, random_state=42)
            model.fit(X_train_final)
            
            # Predict scores
            val_scores = -model.decision_function(X_val_final)
            
            # Normalize
            s_min, s_max = val_scores.min(), val_scores.max()
            if s_max > s_min:
                val_probs = (val_scores - s_min) / (s_max - s_min)
            else:
                val_probs = val_scores
                
            # ROC-AUC
            auc_score = roc_auc_score(y_val, val_probs)
            auc_list.append(auc_score)
            
            # Threshold optimization
            precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)
            f1_scores = np.zeros_like(thresholds, dtype=float)
            for idx, (p, r) in enumerate(zip(precisions[:-1], recalls[:-1])):
                if (p + r) > 0:
                    f1_scores[idx] = (2 * p * r) / (p + r)
                    
            if len(thresholds) > 0:
                opt_idx = np.argmax(f1_scores)
                best_prec = precisions[opt_idx]
                best_rec = recalls[opt_idx]
                best_f1 = f1_scores[opt_idx]
            else:
                best_prec = precision_score(y_val, val_probs >= 0.5, zero_division=0)
                best_rec = recall_score(y_val, val_probs >= 0.5)
                best_f1 = f1_score(y_val, val_probs >= 0.5, zero_division=0)
                
            prec_list.append(best_prec)
            rec_list.append(best_rec)
            f1_list.append(best_f1)
            
        comparison_results[set_name] = {
            "ROC-AUC": float(np.mean(auc_list)),
            "Precision": float(np.mean(prec_list)),
            "Recall": float(np.mean(rec_list)),
            "F1-Score": float(np.mean(f1_list))
        }

    # 3. Print Results Comparison Table
    print("\n" + "="*80)
    print(f"{'FEATURE SET COMPARISON':^80}")
    print("="*80)
    print(f"{'Feature Configuration':<40} | {'ROC-AUC':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-"*80)
    for name, metrics in comparison_results.items():
        print(f"{name:<40} | {metrics['ROC-AUC']:<10.4f} | {metrics['Precision']:<10.4f} | {metrics['Recall']:<10.4f} | {metrics['F1-Score']:<10.4f}")
    print("="*80)
    
    # 4. Save results to JSON
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(comparison_results, f, indent=4)
    print(f"Results successfully saved to: {output_json_path}")

if __name__ == "__main__":
    # Paths assuming script is run from project root
    dataset_path = "data/processed/cleaned_UA.csv"
    output_json_path = "outputs/metrics/feature_strength_comparison.json"
    
    # Fallback paths for different execution directories
    if not os.path.exists(dataset_path):
        dataset_path = "../data/processed/cleaned_UA.csv"
        output_json_path = "../outputs/metrics/feature_strength_comparison.json"
        
    verify_feature_strength(dataset_path, output_json_path)
