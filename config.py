# config.py
# ─────────────────────────────────────────────────────────
# Central configuration for the E-commerce AI Pipeline.
# Change values here — nothing else needs to be touched.
# ─────────────────────────────────────────────────────────

import os

# ── Paths ────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
OUTPUT_DIR  = os.path.join(BASE_DIR, "output")

SAMPLE_DATA_PATH    = os.path.join(DATA_DIR,   "sample_products.csv")
BASELINE_DATA_PATH  = os.path.join(DATA_DIR,   "baseline_products.csv")  # for trend calc
TRENDING_OUT        = os.path.join(OUTPUT_DIR, "trending_products.csv")
MATCHES_OUT         = os.path.join(OUTPUT_DIR, "price_comparison.csv")
REPORT_OUT          = os.path.join(OUTPUT_DIR, "full_report.csv")

# ── Popularity Score Weights ─────────────────────────────
# Must sum to 1.0
WEIGHT_RATING  = 0.5   # how much star rating contributes
WEIGHT_REVIEWS = 0.5   # how much (log) review count contributes

# ── Trend Detection ──────────────────────────────────────
TREND_TOP_N = 20        # how many trending products to surface

# ── Product Matching ─────────────────────────────────────
MATCH_THRESHOLD   = 0.75   # cosine similarity threshold (0–1)
                            # raise to 0.85 for stricter matching
TFIDF_NGRAM_RANGE = (1, 2)  # unigrams + bigrams

# ── Reporting ────────────────────────────────────────────
TOP_N_PRODUCTS = 20    # rows in the trending report

# ── Supported Platforms ──────────────────────────────────
PLATFORMS = ["Amazon", "Flipkart", "Myntra", "Meesho"]

# ── Category Keyword Map ─────────────────────────────────
# Add more keywords to improve category detection
CATEGORY_KEYWORDS = {
    "Audio":      ["headphone", "earphone", "speaker", "earbud", "neckband"],
    "Phones":     ["mobile", "phone", "smartphone", "iphone", "redmi", "realme"],
    "Computers":  ["laptop", "tablet", "monitor", "keyboard", "mouse", "ssd"],
    "Fashion":    ["shirt", "dress", "jeans", "shoes", "kurta", "saree",
                   "jacket", "top", "legging", "sandal", "watch", "bag"],
    "Beauty":     ["cream", "serum", "lipstick", "shampoo", "conditioner",
                   "moisturiser", "sunscreen", "perfume"],
    "Appliances": ["fan", "cooler", "ac", "refrigerator", "washing", "mixer",
                   "iron", "geyser", "air purifier"],
}

# ── Flask API ─────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 5000
API_DEBUG = True
