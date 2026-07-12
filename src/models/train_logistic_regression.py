import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

def train_logistic_model(train_path, model_output_path, random_state=42):
    # 1. Load data
    train_df = pd.read_csv(train_path)
    X_train = train_df.drop(columns=["Fracture"])
    y_train = train_df["Fracture"]
    
    print(f"Loaded training data: {X_train.shape}")
    print(f"Target distribution:\n{y_train.value_counts()}")
    
    # 2. Apply SMOTE to balance the dataset
    print("Applying SMOTE to balance dataset...")
    smote = SMOTE(sampling_strategy=0.25, random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"Resampled training data: {X_train_res.shape}")
    print(f"Resampled target distribution:\n{y_train_res.value_counts()}")
    
    # 3. Initialize Logistic Regression
    # class_weight='balanced' weights penalties inversely proportional to class frequencies
    model = LogisticRegression(
        C=0.1,  # Added stronger L2 regularization to prevent overfitting on 46 features
        random_state=random_state, 
        max_iter=1000
    )

    
    # 4. Fit the model
    print("Training Logistic Regression model...")
    model.fit(X_train_res, y_train_res)
    
    # 4. Save model to disk
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(model, model_output_path)
    print(f"Model saved successfully to: {model_output_path}")
    
    return model
