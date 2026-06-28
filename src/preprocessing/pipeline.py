import os
import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

# --- Custom Transformers ---

class IQROutlierCapper(BaseEstimator, TransformerMixin):

    def __init__(self, columns=None, factor=1.5):
        self.columns = columns
        self.factor = factor

    def fit(self, X, y=None):
        self.bounds_ = {}

        X_df = pd.DataFrame(X)

        cols = self.columns if self.columns is not None else X_df.columns

        for col in cols:
            q1 = X_df[col].quantile(0.25)
            q3 = X_df[col].quantile(0.75)

            iqr = q3 - q1

            lower = q1 - self.factor * iqr
            upper = q3 + self.factor * iqr

            self.bounds_[col] = (lower, upper)

        return self
        
    def transform(self, X):
        X_capped = pd.DataFrame(X).copy()
        for col, (lower, upper) in self.bounds_.items():
            if col in X_capped.columns:
                X_capped[col] = X_capped[col].clip(lower=lower, upper=upper)
        return X_capped


class GenderEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, column="Gender"):
        self.column = column

    def fit(self, X, y=None):
        self.is_fitted_ = True   
        return self

    def transform(self, X):
        X_encoded = pd.DataFrame(X).copy()
        if self.column in X_encoded.columns:
            X_encoded[self.column] = X_encoded[self.column].map({1: 0, 2: 1})
        return X_encoded

# --- Pipeline Builder ---

def build_preprocessing_pipeline(num_cols=None, gender_col="Gender"):
    if num_cols is None:
        num_cols = [
            "Age", "Height", "Weight", "BMI",
            "L1-4", "L1.4T", "FN", "FNT", "TL", "TLT",
            "ALT", "AST", "BUN", "CREA", "URIC", "FBG", "HDL-C", "LDL-C", "Ca", "P", "Mg"
        ]


    
    # 1. Custom transformer pipeline (Outlier Capping & Gender Encoding)
    custom_preprocessor = Pipeline(steps=[
        ("outlier_capper", IQROutlierCapper(columns=num_cols)),
        ("gender_encoder", GenderEncoder(column=gender_col))
    ])
    
    # 2. Scaler applied only to numeric columns (leaving encoded Gender, Smoking, Drinking alone)
    numeric_scaler = ColumnTransformer(
        transformers=[
            ("scaler", StandardScaler(), num_cols)
        ],
        remainder="passthrough" # Keep all other columns (Gender, Smoking, Drinking) untouched
    )
    
    # Combine into unified pipeline
    full_pipeline = Pipeline(steps=[
        ("custom_steps", custom_preprocessor),
        ("scaling_step", numeric_scaler)
    ])
    
    return full_pipeline

# --- Utilities ---

def save_pipeline(pipeline, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(pipeline, filepath)
    print(f"Fitted pipeline saved successfully to {filepath}")

def load_pipeline(filepath):
    return joblib.load(filepath)

def get_pipeline_feature_names(pipeline, X_input):
    """
    Returns the list of column names in the order they are output by the pipeline.
    """
    # 1. Identify scaled columns from the ColumnTransformer
    ct = pipeline.named_steps["scaling_step"]
    num_cols = ct.transformers_[0][2]
    
    # 2. Get the columns entering the ColumnTransformer (after custom preprocessor steps)
    X_custom = pipeline.named_steps["custom_steps"].transform(X_input)
    
    # 3. The remaining columns are those not scaled
    passthrough_cols = [col for col in X_custom.columns if col not in num_cols]
    
    # ColumnTransformer outputs scaled columns first, then passthrough columns
    return list(num_cols) + passthrough_cols

