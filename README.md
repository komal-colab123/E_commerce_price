# AI-Powered E-commerce Trend & Product Intelligence System

Complete AI/ML pipeline for Amazon, Flipkart, Myntra & Meesho product analysis.

## Quick Start
```bash
pip install -r requirements.txt
python main_pipeline.py
python api.py   # Flask API for n8n
```

## Project Structure
```
ecommerce_ai/
├── data/products.csv           ← Input data
├── src/
│   ├── data_cleaner.py         ← Clean & normalise
│   ├── feature_engineering.py  ← Popularity & value scores
│   ├── product_matcher.py      ← TF-IDF cross-platform matching
│   ├── trend_detector.py       ← Trend detection + ML ranker
│   └── report_generator.py     ← JSON + CSV output
├── output/
│   ├── intelligence_report.json
│   ├── trending_products.csv
│   ├── price_comparison.csv
│   ├── category_summary.csv
│   ├── platform_summary.csv
│   └── dashboard.html          ← Interactive dashboard
├── main_pipeline.py            ← Run everything
├── api.py                      ← Flask REST API
└── requirements.txt
```

## Key Formulas
popularity_score = 0.5 x norm_rating + 0.5 x norm_log_reviews
trending_score   = 0.45 x norm_reviews + 0.35 x norm_popularity + 0.20 x norm_rating
value_score      = popularity_score / price_ratio

## API Endpoints
GET  /trending?top=15
GET  /matches?threshold=0.72
GET  /categories
GET  /platforms
POST /analyse

## Tech Stack
pandas, numpy, scikit-learn (TF-IDF, LinearRegression), flask
