import os
import joblib
import pandas as pd
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

def train_xgboost_model(train_path, model_output_path, random_state=42):
    # 1. Load data
    train_df = pd.read_csv(train_path)
    X_train = train_df.drop(columns=["Fracture"])
    y_train = train_df["Fracture"]
    
    print(f"Loaded training data for XGBoost: {X_train.shape}")
    print(f"Target distribution:\n{y_train.value_counts()}")
    
    # 2. Apply SMOTE to balance the dataset
    print("Applying SMOTE to balance dataset...")
    smote = SMOTE(sampling_strategy=0.25, random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"Resampled training data: {X_train_res.shape}")
    print(f"Resampled target distribution:\n{y_train_res.value_counts()}")
    
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.metrics import fbeta_score, make_scorer
    
    # 3. Set up Grid Search parameters
    base_model = XGBClassifier(
        random_state=random_state,
        eval_metric="logloss"
    )
    
    # Custom F2-scorer (beta=2 values recall twice as highly as precision)
    f2_scorer = make_scorer(fbeta_score, beta=2, zero_division=0)
    
    param_grid = {
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "n_estimators": [50, 100, 150],
        "reg_alpha": [0, 0.1, 1.0],
        "reg_lambda": [0.5, 1.0, 2.0]
    }
    
    print("Initializing Grid Search over XGBoost hyperparameters (optimizing for F2-Score)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring=f2_scorer,
        cv=cv,
        n_jobs=-1,
        verbose=1
    )
    
    # 4. Fit Grid Search on resampled data
    print("Fitting Grid Search on resampled training data...")
    grid_search.fit(X_train_res, y_train_res)
    
    model = grid_search.best_estimator_
    print(f"Grid Search Complete! Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation F2-Score: {grid_search.best_score_:.4f}")

    
    # 5. Save model to disk
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(model, model_output_path)
    print(f"XGBoost model saved successfully to: {model_output_path}")
    
    return model
