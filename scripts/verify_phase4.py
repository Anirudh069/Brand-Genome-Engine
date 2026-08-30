import sqlite3
import json
import requests
import os

DB_PATH = "data/brand_data.db"
API_URL = "http://localhost:8000/api"

def check_db():
    print("--- 1. Database Verification ---")
    if not os.path.exists(DB_PATH):
        print(f"ERROR: Database file {DB_PATH} not found.")
        return
        
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    # Check Schema
    print("\n[Schema: brand_profiles]")
    cur.execute("PRAGMA table_info(brand_profiles)")
    for col in cur.fetchall():
        print(f"  {col['name']}: {col['type']}")
        
    print("\n[Schema: analysis_history]")
    cur.execute("PRAGMA table_info(analysis_history)")
    for col in cur.fetchall():
        print(f"  {col['name']}: {col['type']}")
        
    # Check Persistence
    print("\n[Data Persistence Check]")
    cur.execute("SELECT brand_id, brand_name, snippets_json FROM brand_profiles WHERE brand_id = 'user_brand'")
    row = cur.fetchone()
    if row:
        print(f"  User Brand Profile Found: {row['brand_name']}")
        snippets = json.loads(row['snippets_json']) if row['snippets_json'] else []
        print(f"  Snippets Persisted: {len(snippets)} / 7")
    else:
        print("  User Brand Profile NOT FOUND in DB.")
        
    cur.execute("SELECT COUNT(*) FROM analysis_history")
    count = cur.fetchone()[0]
    print(f"  Total Analysis Logs in History: {count}")
    conn.close()

def check_api():
    print("\n--- 2. API Verification ---")
    endpoints = ["profile", "analytics"]
    for ep in endpoints:
        try:
            r = requests.get(f"{API_URL}/{ep}")
            print(f"\n[GET /{ep} Response (Sample)]")
            print(json.dumps(r.json(), indent=2)[:500] + "...")
        except Exception as e:
            print(f"  Failed to connect to {ep}: {e}")
            
    # Benchmark
    try:
        payload = {"my_brand": "user_brand", "competitor": "rolex", "metric": "Sentiment Distribution"}
        r = requests.post(f"{API_URL}/benchmark", json=payload)
        print("\n[POST /benchmark Response]")
        print(json.dumps(r.json(), indent=2))
    except Exception as e:
        print(f"  Benchmark API Failed: {e}")

def check_analytics_logic():
    print("\n--- 3. Analytics Logic Verification ---")
    cache_path = "data/processed/analytics_cache.json"
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            data = json.load(f)
            print(f"  Analytics Cache Exists: {len(data.get('tsne_points', []))} t-SNE points, {len(data.get('heatmap', {}).get('pillars', []))} pillars")
            
            # K-NN Verification
            user_point = next((p for p in data.get('tsne_points', []) if p['brand_id'] == 'user_brand'), None)
            if user_point:
                print(f"  User Brand t-SNE Position: ({user_point['x']}, {user_point['y']})")
    else:
        print("  ERROR: Analytics cache missing.")

if __name__ == "__main__":
    check_db()
    check_api()
    check_analytics_logic()
