import os
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, roc_curve, confusion_matrix
)

def evaluate_model(model_path, test_path, metrics_output_path, plots_dir):

    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
    
    # 1. Load model and test dataset
    model = joblib.load(model_path)
    test_df = pd.read_csv(test_path)
    
    X_test = test_df.drop(columns=["Fracture"])
    y_test = test_df["Fracture"]
    
    # 2. Get predictions and probabilities
    y_probs = model.predict_proba(X_test)[:, 1] # Probability of positive class (Fracture=1)
    
    # 3. Find the optimal decision threshold (maximizes F1-score)
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
    f1_scores = np.zeros_like(thresholds, dtype=float)
    for idx, (p, r) in enumerate(zip(precisions[:-1], recalls[:-1])):
        if (p + r) > 0:
            f1_scores[idx] = (2 * p * r) / (p + r)
            
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]
    
    # Apply optimal threshold to get final binary predictions
    y_preds = (y_probs >= optimal_threshold).astype(int)
    
    # 4. Calculate final metrics
    precision = precision_score(y_test, y_preds, zero_division=0)
    recall = recall_score(y_test, y_preds)
    f1 = f1_score(y_test, y_preds, zero_division=0)
    accuracy = accuracy_score(y_test, y_preds)
    roc_auc = roc_auc_score(y_test, y_probs)
    pr_auc = auc(recalls, precisions)
    
    # 5. Extract hyperparameters from the model object
    model_params = model.get_params()
    selected_hyperparameters = {
        "class_weight": model_params.get("class_weight"),
        "C": float(model_params.get("C")) if model_params.get("C") is not None else None,
        "penalty": model_params.get("penalty"),
        "solver": model_params.get("solver"),
        "max_iter": int(model_params.get("max_iter")) if model_params.get("max_iter") is not None else None
    }
    
    # Print metrics to console
    print("\n=== Model Evaluation Results ===")
    print(f"Optimal Threshold: {optimal_threshold:.4f} (max F1={optimal_f1:.4f})")
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Precision:         {precision:.4f}")
    print(f"Recall (Sensitivity): {recall:.4f}")
    print(f"F1-Score:          {f1:.4f}")
    print(f"ROC-AUC:           {roc_auc:.4f}")
    print(f"PR-AUC:            {pr_auc:.4f}")
    
    # 6. Save metrics and hyperparameters together to JSON
    output_data = {
        "model_type": type(model).__name__,
        "hyperparameters": selected_hyperparameters,
        "metrics": {
            "optimal_threshold": float(optimal_threshold),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc)
        }
    }
    
    with open(metrics_output_path, "w") as f:
        json.dump(output_data, f, indent=4)
    print(f"Metrics and hyperparameters saved to: {metrics_output_path}")
    
    # 7. Generate and save Plots
    # Plot A: Confusion Matrix
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix (Threshold={optimal_threshold:.2f})")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.savefig(os.path.join(plots_dir, "logistic_regression_confusion_matrix.png"), dpi=200, bbox_inches="tight")
    plt.close()
    
    # Plot B: ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, "logistic_regression_roc_curve.png"), dpi=200, bbox_inches="tight")
    plt.close()
    
    # Plot C: Precision-Recall Curve
    plt.figure(figsize=(7, 6))
    plt.plot(recalls, precisions, color="blue", lw=2, label=f"PR Curve (AUC = {pr_auc:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, "logistic_regression_pr_curve.png"), dpi=200, bbox_inches="tight")
    plt.close()
    
    print(f"Saved evaluation plots to: {plots_dir}")
