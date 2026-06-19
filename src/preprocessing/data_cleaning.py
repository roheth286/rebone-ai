import os
import pandas as pd
from sklearn.model_selection import train_test_split

def clean_and_split_data(raw_data_path, output_dir, test_size=0.1, random_state=42):
    """
    Loads raw data, drops rows with missing values, selects recommended features,
    splits into train/test sets, and saves them.
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
    
    # 3. Select recommended features
    selected_cols = ["Gender", "Age", "Height", "Weight", "BMI", "Smoking", "Drinking", "Fracture"]
    df_rec = df_clean[selected_cols].copy()
    
    rec_path = os.path.join(output_dir, "reccomended_features_dataset.csv")
    df_rec.to_csv(rec_path, index=False)
    print(f"Selected recommended features. Saved to {rec_path}: {df_rec.shape}")
    
    # 4. Train/Test Split
    X = df_rec.drop(columns=["Fracture"])
    y = df_rec["Fracture"]
    
    # Stratify split on target to maintain class proportions
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    # Save train and test sets
    train_path = os.path.join(output_dir, "train_reccomended_features.csv")
    test_path = os.path.join(output_dir, "test_reccomended_features.csv")
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Saved split train dataset to {train_path}: {train_df.shape}")
    print(f"Saved split test dataset to {test_path}: {test_df.shape}")
    
    return train_path, test_path
