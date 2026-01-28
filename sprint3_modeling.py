from src.data_loader import load_data
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
from src.preprocessing import (
    create_target,
    remove_leakage,
    handle_missing_values
)
from src.modeling import (
    train_logistic_regression,
    train_random_forest,
    train_xgboost,
    evaluate_model,
    save_model
)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib
import os

# -----------------------------
# Paths
# -----------------------------
DATA_PATH = "data/ecommerce_returns_synthetic_data.csv"
MODEL_PATH = "models/return_risk_model.pkl"
ENCODER_PATH = "models/label_encoders.pkl"

os.makedirs("models", exist_ok=True)

# -----------------------------
# FINAL TRAINING FEATURES
# -----------------------------
FEATURE_COLUMNS = [
    "Product_Category",
    "Product_Price",
    "Order_Quantity",
    "Discount_Applied",
    "Payment_Method",
    "Shipping_Method",
    "User_Age",
    "User_Gender",
    "User_Location"
]


def main():
    # -----------------------------
    # Load data
    # -----------------------------
    df = load_data(DATA_PATH)

    # -----------------------------
    # Preprocessing
    # -----------------------------
    df = create_target(df)
    df = remove_leakage(df)
    df = handle_missing_values(df)

    # -----------------------------
    # Feature selection
    # -----------------------------
    X = df[FEATURE_COLUMNS]
    y = df["returned"]

    # -----------------------------
    # Encode categorical features
    # -----------------------------
    label_encoders = {}
    for col in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    # -----------------------------
    # Train–test split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # -----------------------------
    # Logistic Regression
    # -----------------------------
    print("\n--- Logistic Regression ---")
    lr_model = train_logistic_regression(X_train, y_train)
    lr_metrics = evaluate_model(lr_model, X_test, y_test)

    # -----------------------------
    # Random Forest
    # -----------------------------
    print("\n--- Random Forest ---")
    rf_model = train_random_forest(X_train, y_train)
    rf_metrics = evaluate_model(rf_model, X_test, y_test)

    # -----------------------------
    # XGBoost
    # -----------------------------
    print("\n--- XGBoost ---")
    xgb_model = train_xgboost(X_train, y_train)
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test)

    # -----------------------------
    # Model Comparison
    # -----------------------------
    print("\nModel Comparison:")
    print("Logistic Regression:", lr_metrics)
    print("Random Forest:", rf_metrics)
    print("XGBoost:", xgb_metrics)

    # -----------------------------
    # Final Model Selection
    # -----------------------------
    best_model = xgb_model
    print("\n✅ Selected model: XGBoost")
        # -----------------------------
    # Feature Importance (XGBoost)
    # -----------------------------
    print("\n--- Feature Importance (XGBoost) ---")

    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": best_model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    print(importance_df)

    # Plot feature importance
    plt.figure(figsize=(8, 5))
    plt.barh(
        importance_df["Feature"],
        importance_df["Importance"]
    )
    plt.gca().invert_yaxis()
    plt.xlabel("Importance Score")
    plt.title("Feature Importance – Return Risk Prediction")
    plt.tight_layout()
    plt.show()

        # -----------------------------
    # Probability Calibration
    # -----------------------------
    print("\n--- Calibrating probabilities (Isotonic) ---")
    calibrated_model = CalibratedClassifierCV(
        estimator=best_model,
        method="isotonic",
        cv=5
    )
    calibrated_model.fit(X_train, y_train)

    # -----------------------------
    # Threshold-based Evaluation
    # -----------------------------
    y_prob = calibrated_model.predict_proba(X_test)[:, 1]
    thresholds = [0.3, 0.5, 0.7]

    print("\n--- Threshold Analysis ---")
    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)

        precision = precision_score(y_test, y_pred_t)
        recall = recall_score(y_test, y_pred_t)
        f1 = f1_score(y_test, y_pred_t)

        print(f"\nThreshold = {t}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall:    {recall:.3f}")
        print(f"F1-score:  {f1:.3f}")

        # -----------------------------
    # Threshold Curves (Precision / Recall / F1)
    # -----------------------------
    thresholds = np.linspace(0.01, 0.99, 50)

    precisions = []
    recalls = []
    f1_scores = []

    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)

        precisions.append(precision_score(y_test, y_pred_t))
        recalls.append(recall_score(y_test, y_pred_t))
        f1_scores.append(f1_score(y_test, y_pred_t))

    # Plot curves
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, precisions, label="Precision")
    plt.plot(thresholds, recalls, label="Recall")
    plt.plot(thresholds, f1_scores, label="F1-score")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Threshold vs Precision / Recall / F1")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # Confusion Matrix (Chosen Threshold)
    # -----------------------------
    chosen_threshold = 0.5
    y_pred = (y_prob >= chosen_threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Not Returned", "Returned"]
    )
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix – Threshold {chosen_threshold}")
    plt.show()

    # -----------------------------
    # Save Model & Encoders
    # -----------------------------
    save_model(calibrated_model, MODEL_PATH)
    joblib.dump(label_encoders, ENCODER_PATH)

    print("\n✅ Calibrated model saved")
    print(f"📁 Model path: {MODEL_PATH}")
    print(f"📁 Encoders path: {ENCODER_PATH}")


if __name__ == "__main__":
    main()
