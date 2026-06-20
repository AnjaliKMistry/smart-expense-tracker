"""
Train an XGBoost model to predict expense amount from categorical features.

Usage:
    python train_model.py

Requires:
    final_dataset_updated.csv (run generate_dataset.py first)

Outputs:
    model.pkl, encoders.pkl, metrics.pkl
"""

import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

CSV_FILE = "final_dataset_updated.csv"
MODEL_FILE = "model.pkl"
ENCODERS_FILE = "encoders.pkl"
METRICS_FILE = "metrics.pkl"

FEATURE_COLS = [
    "category",
    "location",
    "paymentmethod",
    "merchanttype",
    "timeofday",
    "month",
    "day_of_week",
]
TARGET_COL = "amount"

# Columns encoded as strings with LabelEncoder
CATEGORICAL_COLS = [
    "category",
    "location",
    "paymentmethod",
    "merchanttype",
    "timeofday",
    "day_of_week",
]


def load_and_prepare_data():
    """Load CSV, parse dates, and derive month/day_of_week if missing."""
    df = pd.read_csv(CSV_FILE)
    df.columns = [c.strip().lower().replace(" ", "") for c in df.columns]

    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df.dropna(subset=[TARGET_COL])

    # Parse date and fill time-based features
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if "month" not in df.columns:
            df["month"] = df["date"].dt.month
        if "day_of_week" not in df.columns:
            df["day_of_week"] = df["date"].dt.strftime("%A")

    df["month"] = pd.to_numeric(df["month"], errors="coerce").fillna(1).astype(int)
    df["day_of_week"] = df["day_of_week"].fillna("Monday").astype(str)

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)

    return df


def main():
    df = load_and_prepare_data()
    encoders = {}

    df_encoded = df.copy()
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        encoders[col] = le

    X = df_encoded[FEATURE_COLS].values
    y = df_encoded[TARGET_COL].values

    # Proper 80/20 train/test split — evaluate ONLY on held-out test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)

    mae = float(mean_absolute_error(y_test, y_pred_test))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
    r2 = float(r2_score(y_test, y_pred_test))

    metrics = {"mae": mae, "rmse": rmse, "r2": r2}

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    with open(ENCODERS_FILE, "wb") as f:
        pickle.dump(encoders, f)
    with open(METRICS_FILE, "wb") as f:
        pickle.dump(metrics, f)

    print("Model trained successfully.")
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Features: {', '.join(FEATURE_COLS)}")
    print("\nModel Evaluation Metrics (test set only)")
    print(f"   MAE  (Mean Absolute Error)          : {mae:.2f}")
    print(f"   RMSE (Root Mean Squared Error)      : {rmse:.2f}")
    print(f"   R2   (Coefficient of Determination) : {r2:.4f}")
    print(f"\nSaved: {MODEL_FILE}, {ENCODERS_FILE}, {METRICS_FILE}")


if __name__ == "__main__":
    main()
