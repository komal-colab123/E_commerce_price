#!/usr/bin/env python3
# main.py
# ─────────────────────────────────────────────────────────
# Entry point for the E-commerce AI Intelligence Pipeline.
#
# Run:
#     python main.py
#
# What it does:
#   1. Load raw product data (CSV)
#   2. Clean & normalise product names, extract brand/category
#   3. Compute popularity scores
#   4. Compute trend scores (if baseline data exists)
#   5. Match same products across platforms (TF-IDF)
#   6. Generate reports (CSV + console summary)
#   7. (Optional) Train the category ML classifier
# ─────────────────────────────────────────────────────────

import os
import sys
import pandas as pd

from config import (SAMPLE_DATA_PATH, BASELINE_DATA_PATH,
                    OUTPUT_DIR, TOP_N_PRODUCTS)
from utils.cleaner  import clean_dataframe
from utils.scorer   import compute_popularity, compute_trend, compute_price_stats
from utils.matcher  import match_products, best_deal
from utils.reporter import (save_trending_report, save_price_comparison,
                             save_full_report, print_summary)


def run_pipeline(data_path: str = SAMPLE_DATA_PATH,
                 baseline_path: str = None,
                 train_model: bool = False) -> dict:
    """
    Run the full AI pipeline.

    Args:
        data_path:     Path to the current scraped product CSV.
        baseline_path: Path to an older scraped CSV for trend comparison.
                       If None or file not found, trend step is skipped.
        train_model:   Whether to train the Naive Bayes category classifier.

    Returns:
        dict with keys: df, trend_df, matches_df
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "═" * 60)
    print("  AI-POWERED E-COMMERCE INTELLIGENCE PIPELINE")
    print("═" * 60)

    # ──────────────────────────────────────────────────────
    # STAGE 1: Load Data
    # ──────────────────────────────────────────────────────
    print(f"\n[1/6] Loading data from: {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df_raw = pd.read_csv(data_path)
    print(f"      Loaded {len(df_raw)} rows.")

    # ──────────────────────────────────────────────────────
    # STAGE 2: Clean & Normalise
    # ──────────────────────────────────────────────────────
    print("\n[2/6] Cleaning and normalising data...")
    df = clean_dataframe(df_raw)

    # ──────────────────────────────────────────────────────
    # STAGE 3: Popularity Score
    # ──────────────────────────────────────────────────────
    print("\n[3/6] Computing popularity scores...")
    df = compute_popularity(df)

    # ──────────────────────────────────────────────────────
    # STAGE 4: Trend Score (requires baseline data)
    # ──────────────────────────────────────────────────────
    trend_df = pd.DataFrame()
    bl = baseline_path or BASELINE_DATA_PATH

    if bl and os.path.exists(bl):
        print(f"\n[4/6] Computing trend scores using baseline: {bl}")
        df_baseline = clean_dataframe(pd.read_csv(bl))
        trend_df = compute_trend(df, df_baseline)

        # Merge trend scores back into main df
        trend_map = (
            trend_df.set_index(["clean_name", "platform"])["trend_score"]
            .to_dict()
        )
        df["trend_score"] = df.apply(
            lambda r: trend_map.get((r["clean_name"], r["platform"]), 0.0), axis=1
        )
        df["trend_label"] = df.apply(
            lambda r: trend_df[
                (trend_df["clean_name"] == r["clean_name"]) &
                (trend_df["platform"] == r["platform"])
            ]["trend_label"].values[0]
            if not trend_df[
                (trend_df["clean_name"] == r["clean_name"]) &
                (trend_df["platform"] == r["platform"])
            ].empty else "Stable",
            axis=1
        )
    else:
        print("\n[4/6] Skipping trend score (no baseline data found).")
        df["trend_score"] = 0.0
        df["trend_label"] = "Unknown"

    # ──────────────────────────────────────────────────────
    # STAGE 5: Cross-Platform Product Matching
    # ──────────────────────────────────────────────────────
    print("\n[5/6] Matching products across platforms...")
    matches_df = match_products(df)

    # ──────────────────────────────────────────────────────
    # STAGE 6: Generate Reports
    # ──────────────────────────────────────────────────────
    print("\n[6/6] Generating reports...")
    save_trending_report(df, top_n=TOP_N_PRODUCTS)
    save_full_report(df)

    if not matches_df.empty:
        save_price_comparison(matches_df)

    # Price stats CSV
    price_stats = compute_price_stats(df)
    stats_path = os.path.join(OUTPUT_DIR, "price_stats_by_platform.csv")
    price_stats.to_csv(stats_path, index=False)
    print(f"[Reporter] Price stats → {stats_path}")

    # Console summary
    print_summary(df, matches_df, trend_df if not trend_df.empty else None)

    # ── Optional: Train ML model ──
    if train_model:
        print("[Optional] Training category classifier...")
        try:
            from models.trend_model import train_category_classifier
            train_category_classifier(df, save=True)
        except Exception as e:
            print(f"[Model] Training skipped: {e}")

    return {
        "df":         df,
        "trend_df":   trend_df,
        "matches_df": matches_df,
    }


# ── CLI Entry Point ───────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="E-commerce AI Intelligence Pipeline"
    )
    parser.add_argument(
        "--data", default=SAMPLE_DATA_PATH,
        help="Path to current product CSV (default: data/sample_products.csv)"
    )
    parser.add_argument(
        "--baseline", default=None,
        help="Path to baseline product CSV for trend detection"
    )
    parser.add_argument(
        "--train-model", action="store_true",
        help="Train and save the Naive Bayes category classifier"
    )
    args = parser.parse_args()

    results = run_pipeline(
        data_path     = args.data,
        baseline_path = args.baseline,
        train_model   = args.train_model,
    )

    print("✅ Pipeline complete. Check the output/ folder for all reports.")
