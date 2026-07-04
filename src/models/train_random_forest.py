import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def train_random_forest_model(train_path, model_output_path, random_state=42):
    # 1. Load data
    train_df = pd.read_csv(train_path)
    X_train = train_df.drop(columns=["Fracture"])
    y_train = train_df["Fracture"]
    
    print(f"Loaded training data for Random Forest: {X_train.shape}")
    print(f"Target distribution:\n{y_train.value_counts()}")
    
    # 2. Initialize Random Forest Classifier
    # Note: No class balancing used, standard weights (None).
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=random_state
    )

    
    # 3. Fit the model
    print("Training Random Forest model...")
    model.fit(X_train, y_train)
    
    # 4. Save model to disk
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(model, model_output_path)
    print(f"Random Forest model saved successfully to: {model_output_path}")
    
    return model
