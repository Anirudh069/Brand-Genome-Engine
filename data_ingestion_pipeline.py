import os
import pandas as pd
import sqlite3

CSV_FILE = "raw_texts_watches.csv"
DB_FILE = "watches.db"
TABLE_NAME = "brand_texts"

REQUIRED_COLS = ["text_id", "brand_id", "brand_name", "text"]

def load_data_idempotent():
    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(f"CSV not found: {CSV_FILE}")

    df = pd.read_csv(CSV_FILE, encoding="utf-8", on_bad_lines="skip")

    # Fix column-name issues (e.g., "page_name ")
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Basic cleanup
    df["text_id"] = df["text_id"].astype(str).str.strip()
    df["brand_id"] = df["brand_id"].astype(str).str.strip()
    df["brand_name"] = df["brand_name"].astype(str).str.strip()
    df["text"] = df["text"].astype(str).str.strip()

    # Drop empty texts
    df = df[df["text"].str.len() > 0].copy()

    # Dedup within the batch (just in case)
    df = df.drop_duplicates(subset=["text_id"])

    with sqlite3.connect(DB_FILE) as conn:
        # Create table with PRIMARY KEY on text_id
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS "{TABLE_NAME}" (
                text_id TEXT PRIMARY KEY,
                brand_id TEXT,
                brand_name TEXT,
                segment TEXT,
                country TEXT,
                source_type TEXT,
                page_name TEXT,
                category TEXT,
                year_range TEXT,
                text TEXT,
                url TEXT,
                data_collected TEXT
            );
        """)

        # Idempotent UPSERT (SQLite >= 3.24 supports ON CONFLICT DO UPDATE)
        conn.executemany(
            f"""
            INSERT INTO "{TABLE_NAME}" (
                text_id, brand_id, brand_name, segment, country, source_type,
                page_name, category, year_range, text, url, data_collected
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(text_id) DO UPDATE SET
                brand_id=excluded.brand_id,
                brand_name=excluded.brand_name,
                segment=excluded.segment,
                country=excluded.country,
                source_type=excluded.source_type,
                page_name=excluded.page_name,
                category=excluded.category,
                year_range=excluded.year_range,
                text=excluded.text,
                url=excluded.url,
                data_collected=excluded.data_collected;
            """,
            df[[
                "text_id","brand_id","brand_name","segment","country","source_type",
                "page_name","category","year_range","text","url","data_collected"
            ]].astype(str).values.tolist()
        )

        conn.commit()
        count = conn.execute(f'SELECT COUNT(*) FROM "{TABLE_NAME}";').fetchone()[0]

    print(f"Ingestion complete. {count} rows in {TABLE_NAME} (idempotent).")

if __name__ == "__main__":
    load_data_idempotent()