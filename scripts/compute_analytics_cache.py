import pandas as pd
import numpy as np
import json
import os
from sklearn.manifold import TSNE
import sqlite3

FEATURES_PATH = "data/processed/features.parquet"
DB_PATH = "data/brand_data.db"
OUTPUT_PATH = "data/processed/analytics_cache.json"

PILLARS = ["craftsmanship", "heritage", "innovation", "luxury", "performance"]

def compute_analytics():
    if not os.path.exists(FEATURES_PATH):
        print(f"Features file not found: {FEATURES_PATH}")
        return

    print("Loading features...")
    df = pd.read_parquet(FEATURES_PATH)
    
    # 1. Compute Centroids and t-SNE
    print("Computing t-SNE centroids...")
    brands = df['brand_id'].unique()
    centroids = []
    brand_labels = []
    
    for brand_id in brands:
        brand_df = df[df['brand_id'] == brand_id]
        # Embeddings are stored as lists/arrays in parquet
        embeddings = np.stack(brand_df['embedding'].values)
        centroid = embeddings.mean(axis=0)
        centroids.append(centroid)
        brand_labels.append(brand_id)
    
    centroids = np.array(centroids)
    
    # Run t-SNE on centroids
    tsne = TSNE(n_components=2, perplexity=min(3, len(brands)-1), random_state=42, init='pca', learning_rate='auto')
    tsne_results = tsne.fit_transform(centroids)
    
    tsne_data = []
    for i, brand_id in enumerate(brand_labels):
        brand_name = df[df['brand_id'] == brand_id]['brand_name'].iloc[0]
        tsne_data.append({
            "brand_id": brand_id,
            "brand_name": brand_name,
            "x": float(tsne_results[i, 0]),
            "y": float(tsne_results[i, 1]),
            "embedding": centroids[i].tolist() # Store centroid embedding for K-NN projection
        })

    # 2. Compute Heatmap (Pillars vs Brands)
    print("Computing Heatmap matrix...")
    heatmap_matrix = []
    for brand_id in brands:
        brand_df = df[df['brand_id'] == brand_id]
        brand_name = brand_df['brand_name'].iloc[0]
        
        # We aggregate topic weights for our 5 pillars
        # topic_weights and top_topics are stored in the parquet
        pillar_weights = {p: [] for p in PILLARS}
        
        for _, row in brand_df.iterrows():
            topics = row['top_topics']
            weights = row['topic_weights']
            for t, w in zip(topics, weights):
                if t in pillar_weights:
                    pillar_weights[t].append(w)
        
        # Average weight per pillar
        avg_weights = [float(np.mean(pillar_weights[p])) if pillar_weights[p] else 0.0 for p in PILLARS]
        heatmap_matrix.append({
            "brand_id": brand_id,
            "brand_name": brand_name,
            "weights": avg_weights
        })

    # 3. Tone Histogram (Global Formality Distribution)
    print("Computing Tone Histogram...")
    formality_scores = df['formality'].values
    hist, bin_edges = np.histogram(formality_scores, bins=10, range=(0, 1))
    tone_histogram = {
        "counts": hist.tolist(),
        "bins": bin_edges.tolist()
    }

    # Save to Cache
    cache_data = {
        "tsne_points": tsne_data,
        "heatmap": {
            "pillars": PILLARS,
            "brands": heatmap_matrix
        },
        "tone_histogram": tone_histogram,
        "last_updated": pd.Timestamp.now().isoformat()
    }
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(cache_data, f, indent=2)
    
    print(f"Analytics cache saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    compute_analytics()
