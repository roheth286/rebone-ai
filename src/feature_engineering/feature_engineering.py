import os
import pandas as pd
import numpy as np

def engineer_features(df):
    """
    Applies clinical feature engineering rules on the raw columns:
    1. BMI Categories (Underweight, Normal, Overweight, Obese)
    2. Post-menopausal Risk (Female and Age >= 50)
    3. Senior Status Flag (Age >= 65)
    4. Lifestyle Severity Score (Smoking + Drinking)
    """
    df_feat = df.copy()
    
    # --- Idea 1: BMI Categories (Binary Indicators) ---
    df_feat["BMI_Underweight"] = (df_feat["BMI"] < 18.5).astype(int)
    df_feat["BMI_Normal"] = ((df_feat["BMI"] >= 18.5) & (df_feat["BMI"] < 25.0)).astype(int)
    df_feat["BMI_Overweight"] = ((df_feat["BMI"] >= 25.0) & (df_feat["BMI"] < 30.0)).astype(int)
    df_feat["BMI_Obese"] = (df_feat["BMI"] >= 30.0).astype(int)
    
    # --- Idea 2: Post-menopausal Risk ---
    # Assuming Gender raw values: 1 is Male, 2 is Female (mapping happens in pipeline later)
    df_feat["Postmenopausal_Risk"] = ((df_feat["Gender"] == 2) & (df_feat["Age"] >= 50.0)).astype(int)
    
    # --- Idea 4: Senior Status Flag ---
    df_feat["Is_Senior"] = (df_feat["Age"] >= 65.0).astype(int)
    
    # --- Idea 5: Lifestyle Severity Score (Smoking + Drinking) ---
    df_feat["Lifestyle_Score"] = (df_feat["Smoking"] + df_feat["Drinking"]).astype(int)
    
    print("Feature engineering complete!")
    print(f"Original shape: {df.shape} -> Engineered shape: {df_feat.shape}")
    print(f"New columns added: ['BMI_Underweight', 'BMI_Normal', 'BMI_Overweight', 'BMI_Obese', 'Postmenopausal_Risk', 'Is_Senior', 'Lifestyle_Score']")
    
    return df_feat

def engineer_dataset_from_file(input_path, output_path):
    """
    Utility function to load a CSV file, engineer features, and save it to a new path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.read_csv(input_path)
    df_engineered = engineer_features(df)
    df_engineered.to_csv(output_path, index=False)
    print(f"Saved engineered dataset to: {output_path}")
    return output_path
