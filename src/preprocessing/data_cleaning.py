import os
import pandas as pd

def clean_data(raw_data_path, output_dir):
    """
    Loads raw data, drops rows with missing values, selects the 8 recommended features,
    and saves both the full cleaned dataset and the selected features dataset.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load raw dataset
    df = pd.read_csv(raw_data_path)
    print(f"Loaded raw dataset from {raw_data_path}: {df.shape}")
    
    # 2. Drop missing rows
    df_clean = df.dropna().copy()
    print(f"Dropped rows with missing values. Cleaned shape: {df_clean.shape}")
    
    # Save cleaned full dataset
    cleaned_path = os.path.join(output_dir, "cleaned_UA.csv")
    df_clean.to_csv(cleaned_path, index=False)
    print(f"Saved full cleaned dataset to: {cleaned_path}")
    
    # 3. Save all 40 columns as the cleaned dataset
    clean_dataset_path = os.path.join(output_dir, "cleaned_dataset.csv")
    df_clean.to_csv(clean_dataset_path, index=False)
    print(f"Cleaned dataset saved to {clean_dataset_path}: {df_clean.shape}")
    
    return clean_dataset_path

