import argparse
import os
import sys
import pandas as pd

from src.preprocessing.data_cleaning import clean_data
from src.preprocessing.train_test_split import split_dataset
from src.preprocessing.pipeline import (
    build_preprocessing_pipeline,
    save_pipeline,
    get_pipeline_feature_names,
)
from src.models.train_isolation_forest import train_isolation_forest_model


def retrain_pipeline(raw_data_path: str, output_models_dir: str = "outputs/models", processed_dir: str = "data/processed"):
    """
    End-to-end retraining pipeline:
    1. Ingests a new raw dataset (CSV or Excel).
    2. Cleans and imputes missing values using medians.
    3. Performs stratified train/test split.
    4. Builds and fits the preprocessing pipeline (StandardScaler + outlier capping).
    5. Saves the fitted preprocessing pipeline to outputs/models/preprocessing_pipeline.joblib.
    6. Trains the Isolation Forest model on the clean baseline cohort.
    7. Saves the trained model to outputs/models/isolation_forest.joblib.
    """
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Raw dataset not found at: {raw_data_path}")

    print("=" * 60)
    print(f"Starting Rebone AI Retraining Pipeline on: {raw_data_path}")
    print("=" * 60)

    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(output_models_dir, exist_ok=True)

    # Step 1: Clean data and select 15 clinical features + Fracture target
    print("\n[Step 1/5] Cleaning and imputing raw data...")
    cleaned_dataset_path = clean_data(raw_data_path, processed_dir)

    # Step 2: Stratified train/test split
    print("\n[Step 2/5] Creating stratified train/test split...")
    train_path, test_path = split_dataset(
        dataset_path=cleaned_dataset_path,
        output_dir=processed_dir,
        target_col="Fracture",
        test_size=0.3,
        random_state=42,
    )

    # Step 3: Build & fit preprocessing pipeline
    print("\n[Step 3/5] Fitting preprocessing pipeline on training split...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=["Fracture"])
    y_train = train_df["Fracture"]

    X_test = test_df.drop(columns=["Fracture"])
    y_test = test_df["Fracture"]

    pipeline = build_preprocessing_pipeline(gender_col="Gender")
    X_train_processed = pipeline.fit_transform(X_train)
    X_test_processed = pipeline.transform(X_test)

    processed_columns = get_pipeline_feature_names(pipeline, X_train)

    train_df_processed = pd.concat([
        pd.DataFrame(X_train_processed, columns=processed_columns),
        y_train.reset_index(drop=True),
    ], axis=1)

    test_df_processed = pd.concat([
        pd.DataFrame(X_test_processed, columns=processed_columns),
        y_test.reset_index(drop=True),
    ], axis=1)

    train_scaled_path = os.path.join(processed_dir, "train_features_dataset_scaled.csv")
    test_scaled_path = os.path.join(processed_dir, "test_features_dataset_scaled.csv")

    train_df_processed.to_csv(train_scaled_path, index=False)
    test_df_processed.to_csv(test_scaled_path, index=False)

    # Step 4: Save fitted preprocessing pipeline
    print("\n[Step 4/5] Saving fitted preprocessing pipeline...")
    pipeline_output_path = os.path.join(output_models_dir, "preprocessing_pipeline.joblib")
    save_pipeline(pipeline, pipeline_output_path)

    # Step 5: Train & save Isolation Forest model
    print("\n[Step 5/5] Training and saving Isolation Forest model...")
    model_output_path = os.path.join(output_models_dir, "isolation_forest.joblib")
    train_isolation_forest_model(train_scaled_path, model_output_path, random_state=42)

    print("\n" + "=" * 60)
    print("Retraining completed successfully!")
    print(f"Updated Preprocessing Pipeline: {pipeline_output_path}")
    print(f"Updated Isolation Forest Model: {model_output_path}")
    print("The backend will automatically serve these newly trained artifacts.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebone AI Model Retraining Pipeline")
    parser.add_argument(
        "--data",
        type=str,
        default="data/raw/UA.csv",
        help="Path to the new raw dataset file (CSV or Excel)",
    )
    parser.add_argument(
        "--output-models",
        type=str,
        default="outputs/models",
        help="Directory to save the trained model artifacts",
    )
    parser.add_argument(
        "--output-processed",
        type=str,
        default="data/processed",
        help="Directory to save processed datasets",
    )

    args = parser.parse_args()
    retrain_pipeline(
        raw_data_path=args.data,
        output_models_dir=args.output_models,
        processed_dir=args.output_processed,
    )
