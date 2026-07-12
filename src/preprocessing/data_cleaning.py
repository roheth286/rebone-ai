import os
import pandas as pd

def clean_data(raw_data_path, output_dir):
    """
    Loads raw data, drops rows with missing values, selects the 8 recommended features,
    and saves both the full cleaned dataset and the selected features dataset.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load raw dataset (support both CSV and Excel)
    if raw_data_path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(raw_data_path)
    else:
        df = pd.read_csv(raw_data_path)
    print(f"Loaded raw dataset from {raw_data_path}: {df.shape}")

    
    # 2. Impute missing values using column medians (preserving all patient rows)
    df_clean = df.fillna(df.median(numeric_only=True)).copy()
    print(f"Imputed missing values with column medians. Cleaned shape: {df_clean.shape}")

    
    # Save cleaned full dataset
    cleaned_path = os.path.join(output_dir, "cleaned_UA.csv")
    df_clean.to_csv(cleaned_path, index=False)
    print(f"Saved full cleaned dataset to: {cleaned_path}")
    
    # 3. Select the 15 clinical features + target column
    selected_cols = [
        "Gender", "Age", "BMI", "L1.4T", "FNT", "TLT", 
        "Calsium", "Calcitriol", "Bisphosphonate", "Calcitonin", 
        "VT", "VD", "OP", "Smoking", "Drinking", "Fracture"
    ]
    df_selected = df_clean[selected_cols].copy()
    
    clean_dataset_path = os.path.join(output_dir, "cleaned_dataset.csv")
    df_selected.to_csv(clean_dataset_path, index=False)
    print(f"Cleaned clinical dataset saved to {clean_dataset_path}: {df_selected.shape}")
    
    return clean_dataset_path


