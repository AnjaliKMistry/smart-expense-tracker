"""
Generate a realistic synthetic expense dataset with learnable relationships
between features and amount.

Usage:
    python generate_dataset.py

Output:
    final_dataset_updated.csv  (~4000 rows, last 12 months)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ─── CONFIG ──────────────────────────────────────────────────────────────────
OUTPUT_FILE = "final_dataset_updated.csv"
NUM_ROWS = 4000
RANDOM_SEED = 42

CATEGORIES = [
    "Bills",
    "Entertainment",
    "Food",
    "Grocery",
    "Healthcare",
    "Shopping",
    "Travel",
]

# Base average amounts (₹) — Travel/Healthcare higher, Grocery/Food lower
CATEGORY_BASE = {
    "Bills": 1200,
    "Entertainment": 800,
    "Food": 350,
    "Grocery": 600,
    "Healthcare": 2500,
    "Shopping": 1500,
    "Travel": 4500,
}

LOCATIONS = ["Ahmedabad", "Bangalore", "Delhi", "Mumbai", "Pune"]

# Tier-1 cities cost ~20% more than tier-2 on average
LOCATION_MULTIPLIER = {
    "Mumbai": 1.22,
    "Bangalore": 1.20,
    "Delhi": 1.18,
    "Ahmedabad": 1.0,
    "Pune": 1.0,
}

PAYMENT_METHODS = ["Card", "Cash", "NetBanking", "UPI"]
# UPI is most common in India
PAYMENT_WEIGHTS = [0.12, 0.08, 0.15, 0.65]

TIME_OF_DAY = ["Morning", "Afternoon", "Evening", "Night"]
MERCHANT_TYPES = ["Offline", "Online", "Services", "Subscription"]

# Small multipliers for merchant type and time of day
MERCHANT_MULTIPLIER = {
    "Offline": 1.0,
    "Online": 0.95,
    "Services": 1.10,
    "Subscription": 0.85,
}

TIME_MULTIPLIER = {
    "Morning": 0.95,
    "Afternoon": 1.0,
    "Evening": 1.05,
    "Night": 1.08,
}

DESCRIPTIONS = {
    "Bills": ["Electricity bill", "Internet bill", "Mobile recharge", "Water bill"],
    "Entertainment": ["Movie tickets", "Streaming subscription", "Concert", "Gaming"],
    "Food": ["Restaurant lunch", "Cafe coffee", "Street food", "Food delivery"],
    "Grocery": ["Supermarket run", "Vegetables", "Daily essentials", "Snacks"],
    "Healthcare": ["Doctor visit", "Pharmacy", "Lab test", "Dental checkup"],
    "Shopping": ["Clothing", "Electronics", "Home decor", "Accessories"],
    "Travel": ["Flight ticket", "Hotel booking", "Train ticket", "Cab ride"],
}


def generate_dates(n_rows, end_date):
    """Random dates spread across the last 12 months."""
    start_date = end_date - timedelta(days=365)
    days_range = (end_date - start_date).days
    offsets = np.random.randint(0, days_range + 1, size=n_rows)
    dates = [start_date + timedelta(days=int(d)) for d in offsets]
    return dates


def compute_amount(category, location, merchanttype, timeofday, is_weekend):
    """Compute expense amount with built-in, learnable relationships."""
    base = CATEGORY_BASE[category]

    amount = base * LOCATION_MULTIPLIER[location]
    amount *= MERCHANT_MULTIPLIER[merchanttype]
    amount *= TIME_MULTIPLIER[timeofday]

    # Weekend spending bump for Entertainment and Food
    if is_weekend and category in ("Entertainment", "Food"):
        amount *= 1.15

    # Realistic Gaussian noise (~15% std dev, minimum ₹50)
    noise = np.random.normal(1.0, 0.15)
    amount = max(50, amount * noise)

    return round(amount, 2)


def main():
    np.random.seed(RANDOM_SEED)
    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    dates = generate_dates(NUM_ROWS, end_date)

    rows = []
    for dt in dates:
        category = np.random.choice(CATEGORIES)
        location = np.random.choice(LOCATIONS)
        paymentmethod = np.random.choice(PAYMENT_METHODS, p=PAYMENT_WEIGHTS)
        timeofday = np.random.choice(TIME_OF_DAY)
        merchanttype = np.random.choice(MERCHANT_TYPES)
        is_weekend = dt.weekday() >= 5  # Saturday=5, Sunday=6

        amount = compute_amount(category, location, merchanttype, timeofday, is_weekend)
        description = np.random.choice(DESCRIPTIONS[category])

        rows.append(
            {
                "date": dt.strftime("%Y-%m-%d"),
                "category": category,
                "description": description,
                "amount": amount,
                "merchanttype": merchanttype,
                "location": location,
                "timeofday": timeofday,
                "paymentmethod": paymentmethod,
                "month": dt.month,
                "day_of_week": dt.strftime("%A"),
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("date").reset_index(drop=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Generated {len(df)} rows -> {OUTPUT_FILE}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print("\nAverage amount by category:")
    print(df.groupby("category")["amount"].mean().round(2).sort_values(ascending=False))
    print("\nAverage amount by location:")
    print(df.groupby("location")["amount"].mean().round(2).sort_values(ascending=False))
    print("\nPayment method distribution:")
    print(df["paymentmethod"].value_counts(normalize=True).mul(100).round(1).astype(str) + "%")


if __name__ == "__main__":
    main()
