"""
main_pipeline.py
================
AI-Powered E-commerce Trend & Product Intelligence System
---------------------------------------------------------
Entry point. Runs the full pipeline:

    1. Load data
    2. Clean & normalise
    3. Feature engineering (popularity, value scores)
    4. Cross-platform product matching (TF-IDF)
    5. Trend detection (rule-based + optional ML)
    6. Report generation (JSON + CSV)

Usage:
    python main_pipeline.py
    python main_pipeline.py --data data/products.csv --top 20
"""

import argparse
import os
import sys

import pandas as pd

# Add src to path when running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_cleaner       import clean_dataframe
from feature_engineering import build_features
from product_matcher    import match_products, price_comparison_summary
from trend_detector     import (
    detect_trending_rule_based,
    train_popularity_ranker,
    predict_popularity,
    category_trend_summary,
    platform_summary,
)
from report_generator   import generate_full_report, print_summary


# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_DATA_PATH  = "data/products.csv"
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_TOP_N      = 15
MATCH_THRESHOLD    = 0.72   # Cosine similarity threshold for product matching
USE_ML_RANKER      = True   # Train & use Linear Regression ranker


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(data_path: str, output_dir: str, top_n: int) -> dict:

    print("\n" + "="*55)
    print("  E-commerce AI Intelligence Pipeline  ")
    print("="*55)

    # ── Step 1: Load ──────────────────────────────────────────────────────────
    print("\n[1/6] Loading data...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    raw_df = pd.read_csv(data_path)
    print(f"      Loaded {len(raw_df)} rows from {data_path}")

    # ── Step 2: Clean ─────────────────────────────────────────────────────────
    print("\n[2/6] Cleaning & normalising...")
    clean_df = clean_dataframe(raw_df)
    print(f"      {len(clean_df)} products after cleaning")
    print(f"      Platforms: {sorted(clean_df['platform'].unique())}")
    print(f"      Categories: {sorted(clean_df['category'].unique())}")

    # ── Step 3: Feature engineering ───────────────────────────────────────────
    print("\n[3/6] Engineering features...")
    featured_df = build_features(clean_df)
    print(f"      popularity_score range: "
          f"{featured_df['popularity_score'].min():.3f} – "
          f"{featured_df['popularity_score'].max():.3f}")

    # Optional: ML-based ranker
    if USE_ML_RANKER:
        print("\n      Training ML ranker (Linear Regression)...")
        model, feat_cols = train_popularity_ranker(featured_df)
        featured_df = predict_popularity(model, feat_cols, featured_df)

    # ── Step 4: Product matching ──────────────────────────────────────────────
    print("\n[4/6] Matching products across platforms...")
    matches_df = match_products(featured_df, threshold=MATCH_THRESHOLD)
    print(f"      Found {len(matches_df)} cross-platform product matches")

    # ── Step 5: Trend detection ───────────────────────────────────────────────
    print("\n[5/6] Detecting trends...")
    trending_df      = detect_trending_rule_based(featured_df, top_n=top_n)
    cat_summary      = category_trend_summary(featured_df)
    plat_summary     = platform_summary(featured_df)
    print(f"      Top trending: {trending_df.iloc[0]['product_name'][:40]}...")

    # ── Step 6: Generate report ───────────────────────────────────────────────
    print("\n[6/6] Generating reports...")
    report = generate_full_report(
        df               = featured_df,
        matches_df       = matches_df,
        trending_df      = trending_df,
        category_summary = cat_summary,
        platform_summary = plat_summary,
        output_dir       = output_dir,
        top_n            = top_n,
    )

    # Print summary to console
    print_summary(report)
    return report


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="AI-Powered E-commerce Trend & Product Intelligence"
    )
    parser.add_argument("--data",   default=DEFAULT_DATA_PATH,
                        help="Path to input CSV file")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR,
                        help="Output directory for reports")
    parser.add_argument("--top",    type=int, default=DEFAULT_TOP_N,
                        help="Number of top products to include in report")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        data_path  = args.data,
        output_dir = args.output,
        top_n      = args.top,
    )
