"""
api.py
======
Lightweight Flask REST API — exposes the intelligence pipeline
so n8n (or any tool) can trigger it via HTTP.

Endpoints:
    GET  /health                 → system health check
    GET  /trending?top=15        → top trending products
    GET  /matches?threshold=0.72 → cross-platform matches
    GET  /categories             → category insights
    GET  /platforms              → platform insights
    POST /analyse                → run full pipeline on posted JSON data

Usage:
    pip install flask
    python api.py
"""

import os
import sys

import pandas as pd
from flask import Flask, jsonify, request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_cleaner        import clean_dataframe
from feature_engineering import build_features
from product_matcher     import match_products
from trend_detector      import (
    detect_trending_rule_based,
    category_trend_summary,
    platform_summary,
)
from report_generator    import generate_full_report


app = Flask(__name__)

# ── Load data at startup (hot cache) ──────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "products.csv")


def load_and_process(data_path: str = DATA_PATH,
                     threshold: float = 0.72) -> dict:
    raw        = pd.read_csv(data_path)
    clean      = clean_dataframe(raw)
    featured   = build_features(clean)
    matches    = match_products(featured, threshold=threshold)
    trending   = detect_trending_rule_based(featured)
    cat_sum    = category_trend_summary(featured)
    plat_sum   = platform_summary(featured)
    return {
        "featured":  featured,
        "matches":   matches,
        "trending":  trending,
        "cat_sum":   cat_sum,
        "plat_sum":  plat_sum,
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.route("/health")
def health():
    return jsonify({"status": "ok", "message": "E-commerce AI API is running"})


@app.route("/trending")
def trending():
    top_n = int(request.args.get("top", 15))
    try:
        data     = load_and_process()
        trending = detect_trending_rule_based(data["featured"], top_n=top_n)
        cols     = ["product_name", "brand", "category", "platform",
                    "price", "rating", "review_count",
                    "popularity_score", "trending_score"]
        result   = trending[cols].to_dict("records")
        return jsonify({"count": len(result), "trending_products": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/matches")
def matches():
    threshold = float(request.args.get("threshold", 0.72))
    try:
        data    = load_and_process(threshold=threshold)
        matches = data["matches"]
        if matches.empty:
            return jsonify({"count": 0, "matches": []})
        cols = ["brand", "category",
                "product_a", "platform_a", "price_a",
                "product_b", "platform_b", "price_b",
                "similarity", "price_diff", "savings_pct", "cheaper_on"]
        return jsonify({
            "count":   len(matches),
            "matches": matches[cols].to_dict("records"),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/categories")
def categories():
    try:
        data = load_and_process()
        return jsonify({
            "category_insights": data["cat_sum"].to_dict("records")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/platforms")
def platforms():
    try:
        data = load_and_process()
        return jsonify({
            "platform_insights": data["plat_sum"].to_dict("records")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/analyse", methods=["POST"])
def analyse():
    """
    Accept JSON product data, run full pipeline, return report.

    POST body example:
    [
        {"product_name": "boAt Rockerz 450", "price": 1499,
         "rating": 4.1, "review_count": 85000, "platform": "Amazon"},
        ...
    ]
    """
    try:
        products = request.get_json(force=True)
        if not products or not isinstance(products, list):
            return jsonify({"error": "Send a JSON array of products"}), 400

        raw      = pd.DataFrame(products)
        clean    = clean_dataframe(raw)
        featured = build_features(clean)
        matches  = match_products(featured)
        trending = detect_trending_rule_based(featured)
        cat_sum  = category_trend_summary(featured)
        plat_sum = platform_summary(featured)

        report = generate_full_report(
            df               = featured,
            matches_df       = matches,
            trending_df      = trending,
            category_summary = cat_sum,
            platform_summary = plat_sum,
            output_dir       = "output",
        )
        return jsonify(report)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Run ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n E-commerce AI API")
    print(" Endpoints:")
    print("   GET  http://localhost:5000/health")
    print("   GET  http://localhost:5000/trending?top=15")
    print("   GET  http://localhost:5000/matches?threshold=0.72")
    print("   GET  http://localhost:5000/categories")
    print("   GET  http://localhost:5000/platforms")
    print("   POST http://localhost:5000/analyse\n")
    app.run(debug=True, port=5000)
