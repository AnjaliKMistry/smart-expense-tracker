from flask import Flask, render_template, request, redirect, url_for, session, flash
import sqlite3
import pandas as pd
import pickle
import numpy as np
import os
from datetime import datetime
from collections import Counter, defaultdict

from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret-change-me")

DB_FILE = os.getenv("DATABASE_URL", "expense_tracker.db").replace("sqlite:///", "")
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
CATEGORICAL_COLS = [
    "category",
    "location",
    "paymentmethod",
    "merchanttype",
    "timeofday",
    "day_of_week",
]


def get_db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def get_dropdown_values():
    if not os.path.exists(CSV_FILE):
        return {
            k: []
            for k in [
                "category",
                "paymentmethod",
                "location",
                "accounttype",
                "deviceused",
                "merchanttype",
                "timeofday",
            ]
        }
    df = pd.read_csv(CSV_FILE)
    df.columns = [c.strip().lower().replace(" ", "") for c in df.columns]

    def uniq(col):
        if col in df.columns:
            return sorted([str(v) for v in df[col].dropna().unique().tolist()])
        return []

    return {
        "category": uniq("category"),
        "paymentmethod": uniq("paymentmethod"),
        "location": uniq("location"),
        "accounttype": uniq("accounttype") or ["Savings", "Current"],
        "deviceused": uniq("deviceused") or ["Mobile", "Desktop", "Tablet"],
        "merchanttype": uniq("merchanttype"),
        "timeofday": uniq("timeofday"),
    }


def load_dataset_stats():
    """Load dataset-wide averages for prediction explanations."""
    if not os.path.exists(CSV_FILE):
        return None
    df = pd.read_csv(CSV_FILE)
    df.columns = [c.strip().lower().replace(" ", "") for c in df.columns]
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["amount"])
    return df


def build_prediction_explanation(form_data, predicted_amount):
    """Compare inputs against dataset averages to explain the prediction."""
    df = load_dataset_stats()
    if df is None or df.empty:
        return "Prediction based on trained model patterns."

    overall_avg = float(df["amount"].mean())
    reasons = []

    for col, label in [
        ("category", "category"),
        ("location", "location"),
        ("merchanttype", "merchant type"),
        ("paymentmethod", "payment method"),
        ("timeofday", "time of day"),
    ]:
        value = form_data.get(col, "")
        if not value or col not in df.columns:
            continue
        subset = df[df[col].astype(str) == str(value)]
        if len(subset) < 5:
            continue
        feature_avg = float(subset["amount"].mean())
        if feature_avg > overall_avg * 1.08:
            reasons.append(f"{value} {label} typically costs more (avg ₹{feature_avg:.0f})")
        elif feature_avg < overall_avg * 0.92:
            reasons.append(f"{value} {label} typically costs less (avg ₹{feature_avg:.0f})")

    if predicted_amount > overall_avg * 1.1:
        intro = "This is higher than the dataset average"
        if reasons:
            intro += " because " + ", and ".join(reasons[:3]) + "."
        else:
            intro += f" (₹{overall_avg:.0f} avg) due to the combined effect of your selected features."
    elif predicted_amount < overall_avg * 0.9:
        intro = "This is lower than the dataset average"
        if reasons:
            intro += " because " + ", and ".join(reasons[:3]) + "."
        else:
            intro += f" (₹{overall_avg:.0f} avg) — your choices point to modest spending."
    else:
        intro = "This is close to the typical expense amount"
        if reasons:
            intro += ". " + " ".join(r.capitalize() + "." for r in reasons[:2])
        else:
            intro += f" (dataset avg ₹{overall_avg:.0f})."

    return intro


def encode_features(form_data):
    """Encode form inputs for model prediction."""
    with open(ENCODERS_FILE, "rb") as f:
        encoders = pickle.load(f)

    now = datetime.now()
    month = now.month
    day_of_week = now.strftime("%A")

    raw_values = {
        "category": form_data.get("category", ""),
        "location": form_data.get("location", ""),
        "paymentmethod": form_data.get("paymentmethod", ""),
        "merchanttype": form_data.get("merchanttype", ""),
        "timeofday": form_data.get("timeofday", ""),
        "month": month,
        "day_of_week": day_of_week,
    }

    features = []
    for col in FEATURE_COLS:
        if col == "month":
            features.append(int(raw_values["month"]))
        elif col in encoders:
            val = str(raw_values[col])
            le = encoders[col]
            try:
                features.append(int(le.transform([val])[0]))
            except ValueError:
                features.append(0)
        else:
            features.append(0)

    return np.array([features]), raw_values


def compute_monthly_forecast(expenses, budget):
    """Forecast next month's spending using trend analysis."""
    month_totals = defaultdict(float)
    category_totals = defaultdict(float)

    for e in expenses:
        amount = float(e["amount"] or 0)
        category_totals[e["category"] or "Other"] += amount
        if e["date"]:
            month_key = str(e["date"])[:7]
            month_totals[month_key] += amount

    sorted_months = sorted(month_totals.keys())
    monthly_values = [month_totals[m] for m in sorted_months]

    if len(monthly_values) >= 3:
        x = np.arange(len(monthly_values))
        recent = monthly_values[-6:]
        x_recent = np.arange(len(recent))
        slope, intercept = np.polyfit(x_recent, recent, 1)
        forecast = float(intercept + slope * len(recent))
        method = f"linear trend from your last {len(recent)} month(s) of spending"
    elif monthly_values:
        forecast = float(np.mean(monthly_values))
        method = "average of your available monthly spending"
    else:
        forecast = 0.0
        method = "no spending history yet"

    forecast = max(0, forecast)

    # Top categories driving the forecast (share of recent spending)
    total_cat = sum(category_totals.values()) or 1
    top_categories = sorted(
        category_totals.items(), key=lambda x: x[1], reverse=True
    )[:3]
    drivers = [
        {"name": cat, "share": round(val / total_cat * 100, 1), "total": val}
        for cat, val in top_categories
    ]

    if budget <= 0:
        status = "no_budget"
        status_label = "Set a budget to compare"
        diff = 0
    elif forecast <= budget * 0.85:
        status = "under"
        status_label = "Under budget"
        diff = budget - forecast
    elif forecast <= budget:
        status = "on_track"
        status_label = "On track"
        diff = budget - forecast
    else:
        status = "over"
        status_label = f"Likely to exceed budget by ₹{forecast - budget:.2f}"
        diff = forecast - budget

    return {
        "forecast": round(forecast, 2),
        "budget": budget,
        "status": status,
        "status_label": status_label,
        "diff": round(abs(diff), 2),
        "method": method,
        "monthly_labels": sorted_months,
        "monthly_values": [round(v, 2) for v in monthly_values],
        "drivers": drivers,
    }


# ─── AUTH ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    if "user" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        if not username or not password:
            flash("All fields required.", "error")
            return render_template("register.html")
        try:
            conn = get_db()
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO users (username, password) VALUES (?, ?)",
                (username, password),
            )
            conn.commit()
            conn.close()
            flash("Registered successfully. Please login.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Username already exists.", "error")
        except Exception as e:
            flash(f"Error: {e}", "error")
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        conn = get_db()
        cur = conn.cursor()

        cur.execute("SELECT * FROM users WHERE username=?", (username,))
        existing = cur.fetchone()

        if not existing:
            conn.close()
            flash("User not found. Please register first.", "error")
            return render_template("login.html")

        cur.execute(
            "SELECT * FROM users WHERE username=? AND password=?",
            (username, password),
        )
        user = cur.fetchone()
        conn.close()

        if user:
            session["user"] = username
            flash(f"Welcome back, {username}!", "success")
            return redirect(url_for("dashboard"))
        flash("Incorrect password.", "error")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("Logged out successfully.", "success")
    return redirect(url_for("login"))


# ─── DASHBOARD ────────────────────────────────────────────────────────────────
@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))
    user = session["user"]
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM expenses WHERE username=? ORDER BY date DESC", (user,)
    )
    expenses = cur.fetchall()

    total = sum(float(e["amount"] or 0) for e in expenses)
    avg = total / len(expenses) if expenses else 0
    highest = max((float(e["amount"] or 0) for e in expenses), default=0)

    cats = Counter(e["category"] for e in expenses if e["category"])
    top_cat = cats.most_common(1)[0][0] if cats else "N/A"

    cat_totals = {}
    for e in expenses:
        c = e["category"] or "Other"
        cat_totals[c] = cat_totals.get(c, 0) + float(e["amount"] or 0)

    cur.execute("SELECT budget FROM users WHERE username=?", (user,))
    user_row = cur.fetchone()
    budget = float(user_row["budget"] or 0) if user_row else 0
    remaining = budget - total

    conn.close()
    cat_labels = list(cat_totals.keys())
    cat_values = [float(v) for v in cat_totals.values()]

    return render_template(
        "dashboard.html",
        expenses=expenses,
        total=total,
        avg=avg,
        highest=highest,
        top_cat=top_cat,
        cat_labels=cat_labels,
        cat_values=cat_values,
        user=user,
        budget=budget,
        total_spent=total,
        remaining=remaining,
    )


# ─── ADD EXPENSE ──────────────────────────────────────────────────────────────
@app.route("/add_expense", methods=["GET", "POST"])
def add_expense():
    if "user" not in session:
        return redirect(url_for("login"))
    dropdowns = get_dropdown_values()
    if request.method == "POST":
        f = request.form
        user = session["user"]
        try:
            date_val = f.get("date") or datetime.now().strftime("%Y-%m-%d")
            conn = get_db()
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO expenses
                (username, date, description, amount, category, paymentmethod, location,
                 accounttype, deviceused, timeofday, merchanttype)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    user,
                    date_val,
                    f.get("description"),
                    f.get("amount"),
                    f.get("category"),
                    f.get("paymentmethod"),
                    f.get("location"),
                    f.get("accounttype"),
                    f.get("deviceused"),
                    f.get("timeofday"),
                    f.get("merchanttype"),
                ),
            )
            conn.commit()
            conn.close()
            flash("Expense added successfully.", "success")
            return redirect(url_for("dashboard"))
        except Exception as e:
            flash(f"Error adding expense: {e}", "error")
    return render_template(
        "add_expense.html",
        dropdowns=dropdowns,
        today=datetime.now().strftime("%Y-%m-%d"),
    )


# ─── EDIT EXPENSE ─────────────────────────────────────────────────────────────
@app.route("/edit_expense/<int:expense_id>", methods=["GET", "POST"])
def edit_expense(expense_id):
    if "user" not in session:
        return redirect(url_for("login"))
    dropdowns = get_dropdown_values()
    conn = get_db()
    cur = conn.cursor()
    if request.method == "POST":
        f = request.form
        try:
            cur.execute(
                """
                UPDATE expenses SET date=?, description=?, amount=?, category=?,
                paymentmethod=?, location=?, accounttype=?,
                deviceused=?, merchanttype=?, timeofday=? WHERE id=? AND username=?
                """,
                (
                    f.get("date"),
                    f.get("description"),
                    f.get("amount"),
                    f.get("category"),
                    f.get("paymentmethod"),
                    f.get("location"),
                    f.get("accounttype"),
                    f.get("deviceused"),
                    f.get("merchanttype"),
                    f.get("timeofday"),
                    expense_id,
                    session["user"],
                ),
            )
            conn.commit()
            flash("Expense updated.", "success")
            return redirect(url_for("dashboard"))
        except Exception as e:
            flash(f"Error: {e}", "error")
    cur.execute(
        "SELECT * FROM expenses WHERE id=? AND username=?",
        (expense_id, session["user"]),
    )
    expense = cur.fetchone()
    conn.close()
    if not expense:
        flash("Expense not found.", "error")
        return redirect(url_for("dashboard"))
    return render_template("edit_expense.html", expense=expense, dropdowns=dropdowns)


# ─── DELETE EXPENSE ───────────────────────────────────────────────────────────
@app.route("/delete_expense/<int:expense_id>", methods=["POST"])
def delete_expense(expense_id):
    if "user" not in session:
        return redirect(url_for("login"))
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "DELETE FROM expenses WHERE id=? AND username=?",
        (expense_id, session["user"]),
    )
    conn.commit()
    conn.close()
    flash("Expense deleted.", "success")
    return redirect(url_for("dashboard"))


# ─── PREDICT ──────────────────────────────────────────────────────────────────
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if "user" not in session:
        return redirect(url_for("login"))

    dropdowns = get_dropdown_values()
    result = None

    if request.method == "POST":
        try:
            with open(MODEL_FILE, "rb") as f:
                model = pickle.load(f)
            with open(METRICS_FILE, "rb") as f:
                model_metrics = pickle.load(f)

            X, raw_values = encode_features(request.form)
            predicted = float(model.predict(X)[0])
            explanation = build_prediction_explanation(request.form, predicted)

            result = {
                "amount": round(predicted, 2),
                "mae": round(model_metrics["mae"], 2),
                "rmse": round(model_metrics["rmse"], 2),
                "r2": round(model_metrics["r2"], 2),
                "alert": predicted > 10000,
                "explanation": explanation,
                "inputs": raw_values,
            }
        except FileNotFoundError:
            flash("Model not found. Run: python train_model.py", "error")
        except Exception as e:
            flash(f"Prediction error: {e}", "error")

    return render_template(
        "predict.html", dropdowns=dropdowns, result=result, user=session["user"]
    )


# ─── FORECAST ─────────────────────────────────────────────────────────────────
@app.route("/forecast")
def forecast():
    if "user" not in session:
        return redirect(url_for("login"))

    user = session["user"]
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM expenses WHERE username=? ORDER BY date", (user,))
    expenses = cur.fetchall()
    cur.execute("SELECT budget FROM users WHERE username=?", (user,))
    user_row = cur.fetchone()
    budget = float(user_row["budget"] or 0) if user_row else 0
    conn.close()

    forecast_data = compute_monthly_forecast(expenses, budget)

    return render_template(
        "forecast.html",
        user=user,
        forecast=forecast_data,
    )


# ─── INSIGHTS ─────────────────────────────────────────────────────────────────
@app.route("/insights")
def insights():
    if "user" not in session:
        return redirect(url_for("login"))
    user = session["user"]
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM expenses WHERE username=?", (user,))
    expenses = cur.fetchall()
    conn.close()

    cat_totals = defaultdict(float)
    loc_totals = defaultdict(float)
    for e in expenses:
        cat_totals[e["category"] or "Other"] += float(e["amount"] or 0)
        loc_totals[e["location"] or "Unknown"] += float(e["amount"] or 0)

    top_cat = max(cat_totals, key=cat_totals.get) if cat_totals else "N/A"
    top_loc = max(loc_totals, key=loc_totals.get) if loc_totals else "N/A"
    highest = max(expenses, key=lambda e: float(e["amount"] or 0), default=None)

    suggestions = []
    if cat_totals.get(top_cat, 0) > 5000:
        suggestions.append(f"You're spending heavily on {top_cat}. Consider budgeting.")
    if len(expenses) > 50:
        suggestions.append("High transaction volume. Review recurring expenses.")
    suggestions.append("Track daily expenses to stay within budget.")

    chart_files = [
        "feature_importance.png",
        "actual_vs_predicted.png",
        "residuals.png",
    ]
    charts = [
        f for f in chart_files
        if os.path.exists(os.path.join("static", "charts", f))
    ]

    return render_template(
        "insights.html",
        top_cat=top_cat,
        top_loc=top_loc,
        highest=highest,
        cat_totals=dict(cat_totals),
        suggestions=suggestions,
        user=user,
        charts=charts,
    )


# ─── CHARTS ───────────────────────────────────────────────────────────────────
@app.route("/chart")
def chart():
    if "user" not in session:
        return redirect(url_for("login"))
    user = session["user"]
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM expenses WHERE username=? ORDER BY date", (user,))
    expenses = cur.fetchall()
    conn.close()

    cat_totals = defaultdict(float)
    month_totals = defaultdict(float)
    for e in expenses:
        cat_totals[e["category"] or "Other"] += float(e["amount"] or 0)
        m = str(e["date"])[:7] if e["date"] else "Unknown"
        month_totals[m] += float(e["amount"] or 0)

    return render_template(
        "chart.html",
        cat_labels=list(cat_totals.keys()),
        cat_values=[float(v) for v in cat_totals.values()],
        month_labels=list(month_totals.keys()),
        month_values=[float(v) for v in month_totals.values()],
        user=user,
    )


# ─── SET BUDGET ───────────────────────────────────────────────────────────────
@app.route("/set_budget", methods=["POST"])
def set_budget():
    if "user" not in session:
        return redirect(url_for("login"))
    budget = request.form.get("budget", 0)
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            "UPDATE users SET budget=? WHERE username=?",
            (float(budget), session["user"]),
        )
        conn.commit()
        conn.close()
        flash("Budget updated successfully.", "success")
    except Exception as e:
        flash(f"Error updating budget: {e}", "error")
    return redirect(url_for("dashboard"))


if __name__ == "__main__":
    app.run(debug=False)
