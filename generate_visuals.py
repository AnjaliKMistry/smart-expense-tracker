"""
Generate ML visualization charts for the insights page and portfolio.

Usage:
    python generate_visuals.py

Requires:
    final_dataset_updated.csv, model.pkl, encoders.pkl

Outputs:
    static/charts/feature_importance.png
    static/charts/actual_vs_predicted.png
    static/charts/residuals.png
"""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

CSV_FILE = "final_dataset_updated.csv"
MODEL_FILE = "model.pkl"
ENCODERS_FILE = "encoders.pkl"
OUTPUT_DIR = os.path.join("static", "charts")

FEATURE_COLS = [
    "category",
    "location",
    "paymentmethod",
    "merchanttype",
    "timeofday",
    "month",
    "day_of_week",
]
CATEGORICAL_COLS = [
    "category",
    "location",
    "paymentmethod",
    "merchanttype",
    "timeofday",
    "day_of_week",
]

# Consistent color theme
PRIMARY = "#4f46e5"
ACCENT = "#06b6d4"
TEXT = "#1e293b"
GRID = "#e2e8f0"


def load_test_data():
    """Recreate the same train/test split used in train_model.py."""
    df = pd.read_csv(CSV_FILE)
    df.columns = [c.strip().lower().replace(" ", "") for c in df.columns]

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["amount"])

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if "month" not in df.columns:
            df["month"] = df["date"].dt.month
        if "day_of_week" not in df.columns:
            df["day_of_week"] = df["date"].dt.strftime("%A")

    df["month"] = pd.to_numeric(df["month"], errors="coerce").fillna(1).astype(int)
    df["day_of_week"] = df["day_of_week"].fillna("Monday").astype(str)

    with open(ENCODERS_FILE, "rb") as f:
        encoders = pickle.load(f)

    for col in CATEGORICAL_COLS:
        df[col] = df[col].fillna("Unknown").astype(str)
        le = encoders[col]
        # Handle unseen labels gracefully
        df[col] = df[col].apply(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )

    X = df[FEATURE_COLS].values
    y = df["amount"].values

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_test, y_test


def setup_style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
            "font.size": 11,
        }
    )


def plot_feature_importance(model, output_path):
    importances = model.feature_importances_
    indices = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        [FEATURE_COLS[i] for i in indices],
        importances[indices],
        color=PRIMARY,
        edgecolor="white",
    )
    ax.set_xlabel("Importance Score")
    ax.set_title("XGBoost Feature Importance", fontweight="bold", pad=12)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_actual_vs_predicted(y_test, y_pred, output_path):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test, y_pred, alpha=0.5, color=PRIMARY, s=25, edgecolors="white")
    max_val = max(y_test.max(), y_pred.max()) * 1.05
    ax.plot([0, max_val], [0, max_val], "--", color=ACCENT, linewidth=2, label="Perfect prediction")
    ax.set_xlabel("Actual Amount (₹)")
    ax.set_ylabel("Predicted Amount (₹)")
    ax.set_title("Actual vs Predicted Expenses (Test Set)", fontweight="bold", pad=12)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_residuals(y_test, y_pred, output_path):
    residuals = y_pred - y_test
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_pred, residuals, alpha=0.5, color=ACCENT, s=25, edgecolors="white")
    ax.axhline(0, color=PRIMARY, linestyle="--", linewidth=1.5)
    ax.set_xlabel("Predicted Amount (₹)")
    ax.set_ylabel("Residual (Predicted - Actual) (₹)")
    ax.set_title("Residual Plot (Test Set)", fontweight="bold", pad=12)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    setup_style()

    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)

    X_test, y_test = load_test_data()
    y_pred = model.predict(X_test)

    plot_feature_importance(
        model, os.path.join(OUTPUT_DIR, "feature_importance.png")
    )
    plot_actual_vs_predicted(
        y_test, y_pred, os.path.join(OUTPUT_DIR, "actual_vs_predicted.png")
    )
    plot_residuals(y_test, y_pred, os.path.join(OUTPUT_DIR, "residuals.png"))

    print("Charts saved to static/charts/:")
    print("  - feature_importance.png")
    print("  - actual_vs_predicted.png")
    print("  - residuals.png")


if __name__ == "__main__":
    main()
