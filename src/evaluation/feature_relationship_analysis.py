import os
import pandas as pd
import numpy as np

def run_feature_analysis(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: Raw data file not found at {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    target = 'Fracture'
    
    # 1. Separate features and target
    X = df.drop(columns=[target])
    y = df[target]
    
    # 2. Preprocess columns to handle correlation computation
    X_encoded = X.copy()
    for col in X_encoded.columns:
        if not pd.api.types.is_numeric_dtype(X_encoded[col]):
            X_encoded[col] = X_encoded[col].astype(str).astype('category').cat.codes
            
    # Impute missing values with medians
    X_encoded = X_encoded.fillna(X_encoded.median(numeric_only=True))
    
    # 3. Compute Pearson correlation with the Fracture target
    correlations = []
    for col in X_encoded.columns:
        corr_val = X_encoded[col].corr(y, method='pearson')
        correlations.append({
            'Feature': col,
            'Abs_Correlation': abs(corr_val),
            'Correlation': corr_val
        })
        
    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values(by='Abs_Correlation', ascending=False).reset_index(drop=True)
    
    # 12 Chosen features for the Isolation Forest model
    chosen_features = [
        "Gender", "Age", "BMI", "VT", "VD", "Calcitriol", "Calcitonin", 
        "OP", "Calsium", "L1.4T", "FNT", "TLT"
    ]
    
    # 4. Print results
    print("==================================================")
    print("FEATURE RELATIONSHIP ANALYSIS (ALL 39 FEATURES)")
    print("==================================================")
    print("This analysis ranks all 39 features by their absolute correlation with the target (Fracture).")
    print("Features selected for the final Isolation Forest model are highlighted with [CHOSEN].\n")
    
    print(f"{'Rank':4s} | {'Feature Name':16s} | {'Abs Corr':8s} | {'Direction':9s} | {'Status':8s}")
    print("-" * 60)
    
    for i, row in enumerate(corr_df.itertuples(), 1):
        status = "[CHOSEN]" if row.Feature in chosen_features else "Noise"
        direction = "Positive" if row.Correlation >= 0 else "Negative"
        print(f"{i:4d} | {row.Feature:16s} | {row.Abs_Correlation:.6f} | {direction:9s} | {status}")
        
    print("\n==================================================")
    print("SUMMARY")
    print("==================================================")
    print(f"Total features: {len(corr_df)}")
    print(f"Selected features: {len(chosen_features)}")
    
    # Check average correlation of chosen vs noise
    chosen_corr = corr_df[corr_df['Feature'].isin(chosen_features)]['Abs_Correlation'].mean()
    noise_corr = corr_df[~corr_df['Feature'].isin(chosen_features)]['Abs_Correlation'].mean()
    
    print(f"Average Absolute Correlation of [CHOSEN] features: {chosen_corr:.6f}")
    print(f"Average Absolute Correlation of Noise features:     {noise_corr:.6f}")
    print(f"The chosen features have {chosen_corr/noise_corr:.1f}x stronger relationships on average than the noise features.")

if __name__ == "__main__":
    csv_path = r"C:\Users\rohet\OneDrive\Documents\CS_WORK\rebone-ai\data\raw\UA.csv"
    run_feature_analysis(csv_path)
