import sqlite3
import os

DB_PATH = "data/brand_data.db"

def run_migrations():
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    try:
        # 1. Create analysis_history table
        print("Creating analysis_history table...")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS analysis_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT DEFAULT (datetime('now')),
                brand_id TEXT NOT NULL,
                score_before_json TEXT NOT NULL,
                score_after_json TEXT,
                text_length INTEGER,
                improved BOOLEAN
            )
        """)

        # 2. Add snippets_json to brand_profiles
        print("Adding snippets_json to brand_profiles...")
        # Check if column exists first
        cur.execute("PRAGMA table_info(brand_profiles)")
        columns = [col[1] for col in cur.fetchall()]
        if 'snippets_json' not in columns:
            cur.execute("ALTER TABLE brand_profiles ADD COLUMN snippets_json TEXT")
        
        conn.commit()
        print("Migrations complete.")
    except Exception as e:
        print(f"Error during migration: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    run_migrations()
