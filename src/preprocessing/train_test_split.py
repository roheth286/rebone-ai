import os
import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset(dataset_path, output_dir, target_col="Fracture", test_size=0.1, random_state=42):
    """
    Loads a dataset, performs a stratified train-test split, and saves the 
    resulting train and test datasets.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load dataset
    df = pd.read_csv(dataset_path)
    print(f"Loaded dataset from {dataset_path} for splitting: {df.shape}")
    
    # 2. Perform train/test split
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Stratify split on the target to maintain the 0/1 class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    # Recombine features and targets
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    # Extract base filename to name outputs dynamically
    base_name = os.path.basename(dataset_path).replace(".csv", "")
    train_path = os.path.join(output_dir, f"train_{base_name}.csv")
    test_path = os.path.join(output_dir, f"test_{base_name}.csv")
    
    # 3. Save splits to disk
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Saved stratified train split to {train_path}: {train_df.shape}")
    print(f"Saved stratified test split to {test_path}: {test_df.shape}")
    
    return train_path, test_path
