"""
Initialize the SQLite database for Smart Expense Tracker.

Usage:
    python init_db.py

Creates expense_tracker.db with users and expenses tables.
Optionally seeds demo data from final_dataset_updated.csv if the DB is empty.
"""

import os
import random
import sqlite3
from datetime import datetime, timedelta

import pandas as pd

DB_FILE = "expense_tracker.db"
CSV_FILE = "final_dataset_updated.csv"


def get_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def create_tables(conn):
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            budget REAL DEFAULT 0
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS expenses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            date TEXT,
            description TEXT,
            amount REAL,
            category TEXT,
            paymentmethod TEXT,
            location TEXT,
            accounttype TEXT,
            deviceused TEXT,
            timeofday TEXT,
            merchanttype TEXT
        )
        """
    )

    conn.commit()


def seed_from_csv(conn):
    """Insert demo users and expenses from CSV if table is empty."""
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM expenses")
    if cur.fetchone()[0] > 0:
        print("Database already has expenses — skipping seed.")
        return

    if not os.path.exists(CSV_FILE):
        print(f"CSV not found ({CSV_FILE}). Tables created, no seed data.")
        return

    print(f"Seeding from {CSV_FILE}...")
    df = pd.read_csv(CSV_FILE)
    df.columns = [c.strip().lower().replace(" ", "") for c in df.columns]

    demo_users = ["demo", "student", "professional"]
    for u in demo_users:
        cur.execute(
            "INSERT OR IGNORE INTO users (username, password, budget) VALUES (?, ?, ?)",
            (u, u, 15000),
        )

    inserted = 0
    for _, row in df.iterrows():
        username = random.choice(demo_users)

        raw_date = str(row.get("date", "")).strip()
        parsed = pd.to_datetime(raw_date, errors="coerce")
        if pd.isnull(parsed):
            date_val = (datetime.now() - timedelta(days=random.randint(1, 60))).strftime(
                "%Y-%m-%d"
            )
        else:
            date_val = parsed.strftime("%Y-%m-%d")

        def g(col, default=""):
            v = row.get(col, default)
            return str(v).strip() if pd.notna(v) else default

        cur.execute(
            """
            INSERT INTO expenses
            (username, date, description, amount, category, paymentmethod, location,
             accounttype, deviceused, timeofday, merchanttype)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                username,
                date_val,
                g("description", "Expense"),
                float(g("amount", "0") or 0),
                g("category"),
                g("paymentmethod"),
                g("location"),
                "Savings",
                "Mobile",
                g("timeofday"),
                g("merchanttype"),
            ),
        )
        inserted += 1

    conn.commit()
    print(f"Seeded {inserted} expense rows for demo users: {', '.join(demo_users)}")
    print("Login with username=password (e.g. demo / demo)")


def main():
    conn = get_connection()
    create_tables(conn)
    seed_from_csv(conn)
    conn.close()
    print(f"Database ready: {DB_FILE}")


if __name__ == "__main__":
    main()
