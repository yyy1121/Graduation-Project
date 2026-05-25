#!/usr/bin/env python3
"""
电商用户画像与购买预测系统 - FastAPI 后端服务
==============================================
轻量版：从 dashboard_cache 加载预计算数据，秒级启动。
预测结果来自 prepare_dashboard_data.py 生成的离线特征与训练时保存的预处理管线。
启动: uvicorn backend_api:app --host 0.0.0.0 --port 8000
"""

import json
import math
from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "parquet"
MODEL_DIR = BASE_DIR / "data" / "models"
CACHE_DIR = BASE_DIR / "data" / "dashboard_cache"

# ── Global state ───────────────────────────────────────────────────────────
model = None
feature_names = None
preprocessor = None
calibrator = None
model_metadata = {}
user_features = None
predictions_lookup = None  # precomputed predictions for instant lookup
cache: dict = {}


def sanitize(obj):
    """Recursively replace NaN/Inf with None for safe JSON serialization."""
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def load_all():
    """Load model + user features + precomputed JSON cache."""
    global model, feature_names, preprocessor, calibrator, model_metadata, user_features, predictions_lookup, cache

    # Load precomputed cache (instant)
    for name in ["overview", "clusters", "propensity", "behavior_path", "products"]:
        path = CACHE_DIR / f"{name}.json"
        if path.exists():
            with open(path) as f:
                cache[name] = sanitize(json.load(f))
            size_kb = path.stat().st_size / 1024
            print(f"  [OK] {name}.json ({size_kb:.0f} KB)")
        else:
            print(f"  [!] Missing cache: {name}.json — run prepare_dashboard_data.py first")

    # Load precomputed predictions (instant lookup table)
    pred_path = CACHE_DIR / "user_predictions.parquet"
    if pred_path.exists():
        predictions_lookup = pd.read_parquet(pred_path)
        predictions_lookup["user_id"] = predictions_lookup["user_id"].astype(str)
        print(f"  [OK] predictions lookup: {predictions_lookup.shape}")
    else:
        print(f"  [!] No predictions cache — will use live inference")

    # Load user features (9 MB, needed for user lookup)
    user_features = pd.read_parquet(DATA_DIR / "user_features_raw_with_cluster.parquet")
    user_features["user_id"] = user_features["user_id"].astype(str)
    print(f"  [OK] user_features: {user_features.shape}")

    # Load model artifacts for metadata/health. Online feature construction is
    # intentionally not attempted here because the lightweight backend does not
    # load raw click/order logs.
    model_path = MODEL_DIR / "best_purchase_model.pkl"
    preprocessor_path = MODEL_DIR / "preprocessor.pkl"
    calibrator_path = MODEL_DIR / "calibrator.pkl"
    metadata_path = MODEL_DIR / "model_metadata.json"
    if model_path.exists():
        model = joblib.load(model_path)
    if preprocessor_path.exists():
        preprocessor = joblib.load(preprocessor_path)
        feature_names = getattr(preprocessor, "selected_feature_names", None)
    elif (MODEL_DIR / "feature_names.pkl").exists():
        feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")
    if calibrator_path.exists():
        calibrator = joblib.load(calibrator_path)
    if metadata_path.exists():
        with open(metadata_path) as f:
            model_metadata = json.load(f)
    print(f"  [OK] model artifacts loaded, features={len(feature_names) if feature_names else 0}")


def predict_for_users(user_ids: list[str]) -> dict:
    """Get predictions from the precomputed lookup table.

    The backend deliberately avoids fake online inference: constructing the
    model features requires raw 30/60-day click/order windows and the persisted
    training preprocessor, which is handled by prepare_dashboard_data.py.
    """
    if predictions_lookup is not None:
        mask = predictions_lookup["user_id"].isin(user_ids)
        subset = predictions_lookup[mask]
        found = set(subset["user_id"].tolist())
        return {
            "user_ids": subset["user_id"].tolist(),
            "probabilities": subset["purchase_probability"].tolist(),
            "predictions": subset["prediction_label"].tolist(),
            "missing_user_ids": [uid for uid in user_ids if uid not in found],
        }

    return {"user_ids": [], "probabilities": [], "predictions": [], "missing_user_ids": list(user_ids)}


def segment_for_probability(prob: float) -> str:
    """Use the dashboard cache's fitted segment cutoffs for consistent labels."""
    propensity = cache.get("propensity", {})
    cutoffs = propensity.get("segment_cutoffs", {})
    threshold = cutoffs.get("threshold", model_metadata.get("threshold", 0.5))
    high_cutoff = cutoffs.get("high_cutoff", max(float(threshold), 0.7))
    if prob >= float(high_cutoff):
        return "高倾向"
    if prob >= float(threshold):
        return "中倾向"
    return "低倾向"


# ── Lifespan ────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 60)
    load_all()
    print("Backend ready: http://0.0.0.0:8000")
    print("=" * 60)
    yield


app = FastAPI(title="电商用户画像与购买预测 API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files & dashboard
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def root():
    """Serve the main dashboard page."""
    return FileResponse(STATIC_DIR / "dashboard.html")


# ── Models ──────────────────────────────────────────────────────────────────


class UserIdsRequest(BaseModel):
    user_ids: list[str]


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Overview                                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

@app.get("/api/overview/kpis")
async def overview_kpis():
    return cache.get("overview", {}).get("kpis", {})

@app.get("/api/overview/gender")
async def overview_gender():
    return cache.get("overview", {}).get("gender_dist", [])

@app.get("/api/overview/age")
async def overview_age():
    return cache.get("overview", {}).get("age_dist", [])

@app.get("/api/overview/rfm/{feature}")
async def overview_rfm(feature: str):
    key = f"{feature}_dist"
    val = cache.get("overview", {}).get(key)
    if val is None:
        raise HTTPException(404, f"Unknown RFM feature: {feature}")
    return val

@app.get("/api/overview/hourly")
async def overview_hourly():
    ov = cache.get("overview", {})
    return {"orders": ov.get("hourly_orders", []), "clicks": ov.get("hourly_clicks", [])}


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Purchase Propensity                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

@app.get("/api/propensity/distribution")
async def propensity_distribution():
    return cache.get("propensity", {}).get("pred_distribution", [])

@app.get("/api/propensity/segments")
async def propensity_segments():
    return cache.get("propensity", {}).get("segments", [])

@app.get("/api/propensity/top-features")
async def propensity_top_features():
    return cache.get("propensity", {}).get("top_features", [])

@app.get("/api/propensity/strategy")
async def propensity_strategy():
    prop = cache.get("propensity", {})
    return {
        "summary": prop.get("strategy_summary", {}),
        "operations": prop.get("operation_strategy", []),
        "cutoffs": prop.get("segment_cutoffs", {}),
    }

@app.post("/api/propensity/predict")
async def propensity_predict(req: UserIdsRequest):
    result = predict_for_users(req.user_ids)
    out = []
    for uid, prob, pred in zip(result["user_ids"], result["probabilities"], result["predictions"]):
        seg = segment_for_probability(prob)
        out.append({"user_id": uid, "probability": round(prob, 4),
                     "prediction": pred, "segment": seg})
    return {"results": out, "missing_user_ids": result.get("missing_user_ids", [])}


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Behavior Path                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

@app.get("/api/path/click-to-order-time")
async def path_click_to_order():
    return cache.get("behavior_path", {}).get("click_to_order_dist", [])

@app.get("/api/path/category-funnel")
async def path_category_funnel():
    return cache.get("behavior_path", {}).get("category_funnel", [])

@app.get("/api/path/user-journey/{user_id}")
async def path_user_journey(user_id: str):
    """User journey: we don't load full click/order parquet in lightweight mode.
    Return basic stats from user_features instead."""
    uf_row = user_features[user_features["user_id"] == user_id]
    if len(uf_row) == 0:
        raise HTTPException(404, f"User {user_id} not found")

    row = uf_row.iloc[0]
    return {
        "user_id": user_id,
        "total_clicks": int(row["total_clicks"]),
        "total_orders": int(row["F_count"]),
        "has_order": int(row["has_order"]),
        "has_click": int(row["has_click"]),
        "R_days": int(row["R_days"]),
        "click_to_buy_ratio": round(float(row["click_to_buy_ratio"]), 4),
        "message": "轻量模式：显示统计数据。如需详细事件日志，请加载完整数据。",
        "clicks": [],
        "orders": [],
    }


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  User Segments                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

@app.get("/api/segments/clusters")
async def segments_clusters():
    return cache.get("clusters", {}).get("profile", [])

@app.get("/api/segments/cluster/{cluster_id}")
async def segments_cluster_detail(cluster_id: int):
    cluster_users = user_features[user_features["cluster"] == cluster_id]
    if len(cluster_users) == 0:
        raise HTTPException(404, f"Cluster {cluster_id} not found")

    stats = {
        "user_count": len(cluster_users),
        "avg_R": round(cluster_users["R_days"].mean(), 1),
        "avg_F": round(cluster_users["F_count"].mean(), 1),
        "avg_M": round(cluster_users["total_order_amount"].mean(), 1),
        "avg_clicks": round(cluster_users["total_clicks"].mean(), 1),
        "has_order_rate": round(cluster_users["has_order"].mean() * 100, 1),
    }
    gender = cluster_users["gender"].value_counts().to_dict()
    age = {str(k): int(v) for k, v in cluster_users["age_range"].value_counts().sort_index().items()}

    sample = cluster_users[["user_id", "gender", "age_range", "F_count",
                             "total_order_amount", "total_clicks", "R_days"]].head(50)
    sample["R_days"] = sample["R_days"].round(0).astype(int)
    sample["F_count"] = sample["F_count"].round(0).astype(int)
    sample["total_order_amount"] = sample["total_order_amount"].round(1)
    sample["total_clicks"] = sample["total_clicks"].round(0).astype(int)

    return {
        "cluster_id": cluster_id,
        "stats": stats,
        "gender": {str(k): v for k, v in gender.items()},
        "age": age,
        "sample_users": sample.to_dict(orient="records"),
    }

@app.get("/api/segments/cluster-category")
async def segments_cluster_category():
    return cache.get("clusters", {}).get("category", [])


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Hot Products                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

@app.get("/api/products/top")
async def products_top(limit: int = Query(20, le=50)):
    items = cache.get("products", {}).get("top_items", [])
    return items[:limit]

@app.get("/api/products/top-categories")
async def products_top_categories():
    return cache.get("products", {}).get("top_categories", [])

@app.get("/api/products/top-brands")
async def products_top_brands():
    return cache.get("products", {}).get("top_brands", [])

@app.get("/api/products/price-distribution")
async def products_price_dist():
    return cache.get("products", {}).get("price_distribution", [])


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  User Detail & Health                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

@app.get("/api/user/{user_id}")
async def user_detail(user_id: str):
    uf_row = user_features[user_features["user_id"] == user_id]
    if len(uf_row) == 0:
        raise HTTPException(404, f"User {user_id} not found")

    row = uf_row.iloc[0]
    user_info = {
        "user_id": str(row["user_id"]),
        "gender": str(row["gender"]),
        "age_range": str(row["age_range"]),
        "cluster": int(row["cluster"]),
        "F_count": int(row["F_count"]),
        "total_order_amount": round(float(row["total_order_amount"]), 1),
        "total_clicks": int(row["total_clicks"]),
        "R_days": int(row["R_days"]),
        "has_order": int(row["has_order"]),
    }

    pred_result = predict_for_users([user_id])
    if pred_result["user_ids"]:
        prob = pred_result["probabilities"][0]
        pred = pred_result["predictions"][0]
        seg = segment_for_probability(prob)
        user_info["prediction"] = {
            "probability": round(prob, 4),
            "prediction": pred,
            "segment": seg,
        }
    return user_info

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "calibrator_loaded": calibrator is not None,
        "users_loaded": user_features is not None,
        "prediction_cache_loaded": predictions_lookup is not None,
        "features": len(feature_names) if feature_names else 0,
        "threshold": model_metadata.get("threshold"),
        "model_name": model_metadata.get("best_model_name"),
        "cache_keys": list(cache.keys()),
    }


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
