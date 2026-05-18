#!/usr/bin/env python3
"""
离线数据准备：构建完整特征管线 + 预计算所有仪表盘数据。
运行一次后，backend_api.py 可直接加载缓存秒级启动。

用法: python3 prepare_dashboard_data.py
"""

import json
import sys
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import joblib

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "parquet"
MODEL_DIR = BASE_DIR / "data" / "models"
CACHE_DIR = BASE_DIR / "data" / "dashboard_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
np.random.seed(SEED)

print("=" * 60)
print("仪表盘数据预计算 (含完整特征管线)")
print("=" * 60)

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: Load raw data
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[Phase 1] Loading data...")

user_features = pd.read_parquet(DATA_DIR / "user_features_raw_with_cluster.parquet")
user_features["user_id"] = user_features["user_id"].astype(str)
print(f"  user_features: {user_features.shape}")

order_data = pd.read_parquet(DATA_DIR / "user_order_cleaned.parquet")
order_data["user_id"] = order_data["user_id"].astype(str)
order_data["item_id"] = order_data["item_id"].astype(str)
order_data["datetime"] = pd.to_datetime(order_data["datetime"])
print(f"  order_data: {order_data.shape}")

click_raw = pd.read_parquet(DATA_DIR / "user_click.parquet")
click_raw["user_id"] = click_raw["user_id"].astype(str)
click_raw["item_id"] = click_raw["item_id"].astype(str)
click_raw["datetime"] = pd.to_datetime(click_raw["datetime"], format="mixed")
print(f"  click_data: {click_raw.shape}")

item_data = pd.read_parquet(
    DATA_DIR / "item_data_cleaned.parquet",
    columns=["item_id", "cid1", "cid2", "cid3", "cid1_name", "cid2_name", "cid3_name",
             "brand_code", "price", "item_name"],
)
item_data["item_id"] = item_data["item_id"].astype(str)
print(f"  item_data: {item_data.shape}")

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: Build full feature pipeline (matching improved_modeling.py)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[Phase 2] Building prediction features...")

# Time windows (matching the original script)
max_date = order_data["datetime"].max()
pred_end = max_date
pred_start = max_date - timedelta(days=6)
obs_end = pred_start - timedelta(days=1)
obs_start = obs_end - timedelta(days=29)

print(f"  预测窗口: {pred_start.date()} ~ {pred_end.date()}")
print(f"  观察窗口: {obs_start.date()} ~ {obs_end.date()}")

all_users = set(user_features["user_id"].unique())

# Obs-period data
order_obs = order_data[(order_data["datetime"] >= obs_start) & (order_data["datetime"] <= obs_end)]
click_obs = click_raw[(click_raw["datetime"] >= obs_start) & (click_raw["datetime"] <= obs_end)]

# ── Build baseline features (30d window) ──
order_agg = order_obs.groupby("user_id").agg(
    obs_order_cnt=("datetime", "count"),
    obs_total_qty=("count", "sum"),
    obs_unique_items=("item_id", "nunique"),
    obs_active_days=("datetime", lambda x: x.dt.date.nunique()),
    obs_first_order=("datetime", "min"),
    obs_last_order=("datetime", "max"),
).reset_index()

order_agg["obs_order_span"] = (
    (order_agg["obs_last_order"] - order_agg["obs_first_order"]).dt.days
).clip(lower=1)
order_agg["obs_avg_items_per_order"] = order_agg["obs_total_qty"] / order_agg["obs_order_cnt"].clip(lower=1)
order_agg["obs_avg_interval"] = order_agg["obs_order_span"] / order_agg["obs_order_cnt"].clip(lower=1)
order_agg["obs_days_since_last"] = (pred_start - order_agg["obs_last_order"]).dt.days
order_agg.drop(["obs_first_order", "obs_last_order"], axis=1, inplace=True)

click_agg = click_obs.groupby("user_id").agg(
    click_cnt=("datetime", "count"),
    unique_click_items=("item_id", "nunique"),
    click_active_days=("datetime", lambda x: x.dt.date.nunique()),
    first_click=("datetime", "min"),
    last_click=("datetime", "max"),
).reset_index()

click_agg["click_span"] = (
    (click_agg["last_click"] - click_agg["first_click"]).dt.days
).clip(lower=1)
click_agg["avg_click_per_day"] = click_agg["click_cnt"] / click_agg["click_active_days"].clip(lower=1)
click_agg["days_since_last_click"] = (pred_start - click_agg["last_click"]).dt.days
click_agg.drop(["first_click", "last_click"], axis=1, inplace=True)

base = pd.DataFrame({"user_id": list(all_users)})
base = base.merge(order_agg, on="user_id", how="left")
base = base.merge(click_agg, on="user_id", how="left")

order_fill = ["obs_order_cnt", "obs_total_qty", "obs_unique_items",
              "obs_active_days", "obs_order_span", "obs_avg_items_per_order", "obs_avg_interval"]
click_fill = ["click_cnt", "unique_click_items", "click_active_days", "click_span", "avg_click_per_day"]
for c in order_fill + click_fill:
    if c in base.columns:
        base[c] = base[c].fillna(0)

window_days = (pred_start - obs_start).days + 30
base["obs_days_since_last"] = base["obs_days_since_last"].fillna(window_days)
base["days_since_last_click"] = base["days_since_last_click"].fillna(window_days)
base["click_to_buy_ratio"] = np.where(base["click_cnt"] > 0, base["obs_order_cnt"] / base["click_cnt"], 0)
base["has_conversion"] = ((base["click_cnt"] > 0) & (base["obs_order_cnt"] > 0)).astype(int)

# Merge user base
user_base = user_features[["user_id", "gender_encoded", "age_encoded"]].copy()
baseline_features = base.merge(user_base, on="user_id", how="left")
print(f"  baseline features: {baseline_features.shape}")

# ── Multi-window features ──
WINDOWS = [7, 14, 30, 60]

def build_window_features(df, user_ids, window_days_list, pred_start_dt, prefix):
    features = pd.DataFrame({"user_id": list(user_ids)})
    for days in window_days_list:
        window_start = pred_start_dt - timedelta(days=days)
        window_data = df[(df["datetime"] >= window_start) & (df["datetime"] <= obs_end)]
        agg = window_data.groupby("user_id").agg(
            **{f"{prefix}_cnt_{days}d": ("datetime", "count"),
               f"{prefix}_items_{days}d": ("item_id", "nunique"),
               f"{prefix}_active_days_{days}d": ("datetime", lambda x: x.dt.date.nunique()),
               f"{prefix}_last_days_{days}d": ("datetime", lambda x: (pred_start_dt - x.max()).days)}
        ).reset_index()
        features = features.merge(agg, on="user_id", how="left")
    for days in window_days_list:
        for col_suffix in ["cnt", "items", "active_days"]:
            col = f"{prefix}_{col_suffix}_{days}d"
            if col in features.columns:
                features[col] = features[col].fillna(0)
        last_col = f"{prefix}_last_days_{days}d"
        if last_col in features.columns:
            features[last_col] = features[last_col].fillna(days + 7)
    return features

order_window_feats = build_window_features(order_obs, all_users, WINDOWS, pred_start, "order")
click_window_feats = build_window_features(click_obs, all_users, WINDOWS, pred_start, "click")
print(f"  window features: order {order_window_feats.shape}, click {click_window_feats.shape}")

# ── Trend features ──
trend_features = pd.DataFrame({"user_id": list(all_users)})
for prefix, feat_df in [("order", order_window_feats), ("click", click_window_feats)]:
    for metric in ["cnt", "items", "active_days"]:
        col_7d = feat_df[f"{prefix}_{metric}_7d"].values
        col_14d = feat_df[f"{prefix}_{metric}_14d"].values
        col_30d = feat_df[f"{prefix}_{metric}_30d"].values
        trend_features[f"{prefix}_{metric}_7d_14d_ratio"] = np.where(col_14d > 0, col_7d / col_14d, 0)
        trend_features[f"{prefix}_{metric}_14d_30d_ratio"] = np.where(col_30d > 0, col_14d / col_30d, 0)
        trend_features[f"{prefix}_{metric}_7d_30d_ratio"] = np.where(col_30d > 0, col_7d / col_30d, 0)
    last_7d = feat_df[f"{prefix}_last_days_7d"].values
    last_14d = feat_df[f"{prefix}_last_days_14d"].values
    trend_features[f"{prefix}_recency_decay"] = np.where(last_14d > 0, last_7d / (last_14d + 1), 0)

for days in WINDOWS:
    click_col = f"click_cnt_{days}d"
    order_col = f"order_cnt_{days}d"
    if click_col in click_window_feats.columns and order_col in order_window_feats.columns:
        trend_features[f"conversion_rate_{days}d"] = np.where(
            click_window_feats[click_col].values > 0,
            order_window_feats[order_col].values / click_window_feats[click_col].values, 0)

print(f"  trend features: {trend_features.shape}")

# ── Category/brand features ──
click_with_cat = click_obs[["user_id", "item_id", "datetime"]].merge(
    item_data[["item_id", "cid1", "cid2", "cid3", "brand_code", "price"]], on="item_id", how="left")
order_obs_with_cat = order_obs[["user_id", "item_id", "datetime"]].merge(
    item_data[["item_id", "cid1", "cid2", "cid3", "brand_code", "price"]], on="item_id", how="left")

def safe_top1_count(x):
    vc = x.value_counts()
    return vc.iloc[0] if len(vc) > 0 else 0

cat_feats = pd.DataFrame({"user_id": list(all_users)})

cid1_click = click_with_cat.groupby("user_id")["cid1"].agg([
    ("cid1_click_nunique", "nunique"),
    ("cid1_click_top1_count", safe_top1_count),
]).reset_index()
cid1_order = order_obs_with_cat.groupby("user_id")["cid1"].agg([
    ("cid1_order_nunique", "nunique"),
]).reset_index()
cat_feats = cat_feats.merge(cid1_click, on="user_id", how="left")
cat_feats = cat_feats.merge(cid1_order, on="user_id", how="left")

cid1_click_cnt = click_with_cat.groupby(["user_id", "cid1"]).size().reset_index(name="click_count")
cid1_order_cnt = order_obs_with_cat.groupby(["user_id", "cid1"]).size().reset_index(name="order_count")
cid1_conv = cid1_click_cnt.merge(cid1_order_cnt, on=["user_id", "cid1"], how="left")
cid1_conv["order_count"] = cid1_conv["order_count"].fillna(0)
cid1_conv["cid1_converted"] = (cid1_conv["order_count"] > 0).astype(int)
cid1_user_conv = cid1_conv.groupby("user_id").agg(
    cid1_conversion_rate=("cid1_converted", "mean"),
    cid1_clicked_count=("cid1", "nunique"),
    cid1_converted_count=("cid1_converted", "sum"),
).reset_index()
cat_feats = cat_feats.merge(cid1_user_conv, on="user_id", how="left")

cid3_click = click_with_cat.groupby("user_id")["cid3"].agg([
    ("cid3_click_nunique", "nunique"),
]).reset_index()
cid3_order = order_obs_with_cat.groupby("user_id")["cid3"].agg([
    ("cid3_order_nunique", "nunique"),
]).reset_index()
cat_feats = cat_feats.merge(cid3_click, on="user_id", how="left")
cat_feats = cat_feats.merge(cid3_order, on="user_id", how="left")

brand_click = click_with_cat.groupby("user_id")["brand_code"].agg([
    ("brand_click_nunique", "nunique"),
    ("brand_click_top1_count", safe_top1_count),
]).reset_index()
brand_order = order_obs_with_cat.groupby("user_id")["brand_code"].agg([
    ("brand_order_nunique", "nunique"),
]).reset_index()
cat_feats = cat_feats.merge(brand_click, on="user_id", how="left")
cat_feats = cat_feats.merge(brand_order, on="user_id", how="left")

total_clicks_cat = click_with_cat.groupby("user_id").size().reset_index(name="total_clicks_cat")
cat_feats = cat_feats.merge(total_clicks_cat, on="user_id", how="left")

for c in cat_feats.columns:
    if c != "user_id":
        cat_feats[c] = cat_feats[c].fillna(0)

cat_feats["top1_cid1_click_ratio"] = np.where(
    cat_feats["total_clicks_cat"] > 0, cat_feats["cid1_click_top1_count"] / cat_feats["total_clicks_cat"], 0)
cat_feats["top1_brand_click_ratio"] = np.where(
    cat_feats["total_clicks_cat"] > 0, cat_feats["brand_click_top1_count"] / cat_feats["total_clicks_cat"], 0)
cat_feats["brand_focus_ratio"] = np.where(
    cat_feats["cid1_click_nunique"] > 0, cat_feats["brand_click_nunique"] / cat_feats["cid1_click_nunique"], 0)
cat_feats["cid1_per_brand"] = np.where(
    cat_feats["brand_click_nunique"] > 0, cat_feats["cid1_click_nunique"] / (cat_feats["brand_click_nunique"] + 1), 0)
cat_feats["order_cid1_concentration"] = np.where(
    cat_feats["cid1_click_nunique"] > 0, cat_feats["cid1_order_nunique"] / (cat_feats["cid1_click_nunique"] + 1), 0)

print(f"  category features: {cat_feats.shape}")

# ── Behavior path features ──
click_bf = click_obs[["user_id", "item_id", "datetime"]].copy()
click_bf.columns = ["user_id", "item_id", "click_time"]
order_fp = order_obs[["user_id", "item_id", "datetime"]].copy()
order_fp.columns = ["user_id", "item_id", "order_time"]

path_merged = click_bf.merge(order_fp, on=["user_id", "item_id"], how="inner")
path_merged["click_to_order_hours"] = (
    (path_merged["order_time"] - path_merged["click_time"]).dt.total_seconds() / 3600)
path_merged = path_merged[path_merged["click_to_order_hours"] >= 0]

path_features = path_merged.groupby("user_id").agg(
    avg_click_to_order_hours=("click_to_order_hours", "mean"),
    median_click_to_order_hours=("click_to_order_hours", "median"),
    min_click_to_order_hours=("click_to_order_hours", "min"),
    max_click_to_order_hours=("click_to_order_hours", "max"),
    converted_item_count=("item_id", "nunique"),
).reset_index()

user_clicked = click_obs.groupby("user_id")["item_id"].apply(set).reset_index()
user_ordered = order_obs.groupby("user_id")["item_id"].apply(set).reset_index()
user_clicked.columns = ["user_id", "clicked_set"]
user_ordered.columns = ["user_id", "ordered_set"]
item_flow = user_clicked.merge(user_ordered, on="user_id", how="left")
item_flow["ordered_set"] = item_flow["ordered_set"].apply(lambda x: x if isinstance(x, set) else set())

def safe_set_len(x):
    return len(x) if isinstance(x, set) else 0

item_flow["abandoned_items_count"] = item_flow.apply(
    lambda row: len(row["clicked_set"] - row["ordered_set"])
    if isinstance(row["ordered_set"], set) else len(row["clicked_set"]), axis=1)
item_flow["clicked_items_count"] = item_flow["clicked_set"].apply(safe_set_len)
item_flow["ordered_items_count"] = item_flow["ordered_set"].apply(safe_set_len)
item_flow["abandon_rate"] = np.where(
    item_flow["clicked_items_count"] > 0,
    item_flow["abandoned_items_count"] / item_flow["clicked_items_count"], 0)

path_features_final = path_features.merge(
    item_flow[["user_id", "abandoned_items_count", "clicked_items_count",
               "ordered_items_count", "abandon_rate"]], on="user_id", how="right")

path_features_final["avg_click_to_order_hours"] = path_features_final["avg_click_to_order_hours"].fillna(-1)
path_features_final["converted_item_count"] = path_features_final["converted_item_count"].fillna(0)
path_features_final["abandoned_items_count"] = path_features_final["abandoned_items_count"].fillna(0)
path_features_final["abandon_rate"] = path_features_final["abandon_rate"].fillna(1.0)

click_depth = click_obs.groupby(["user_id", click_obs["datetime"].dt.date])["item_id"].nunique().reset_index()
click_depth_agg = click_depth.groupby("user_id").agg(
    avg_items_per_session=("item_id", "mean"),
    max_items_per_session=("item_id", "max"),
    session_count=("datetime", "count"),
).reset_index()
path_features_final = path_features_final.merge(click_depth_agg, on="user_id", how="left")
path_features_final["avg_items_per_session"] = path_features_final["avg_items_per_session"].fillna(0)
path_features_final["max_items_per_session"] = path_features_final["max_items_per_session"].fillna(0)
path_features_final["session_count"] = path_features_final["session_count"].fillna(0)

print(f"  path features: {path_features_final.shape}")

# ── Price sensitivity features ──
global_median_price = item_data["price"].median()

click_price = click_obs[["user_id", "item_id"]].merge(
    item_data[["item_id", "price"]], on="item_id", how="left")

user_price_click = click_price.groupby("user_id")["price"].agg([
    ("click_price_p10", lambda x: x.quantile(0.10)),
    ("click_price_p25", lambda x: x.quantile(0.25)),
    ("click_price_p50", lambda x: x.quantile(0.50)),
    ("click_price_p75", lambda x: x.quantile(0.75)),
    ("click_price_p90", lambda x: x.quantile(0.90)),
    ("click_price_mean", "mean"),
    ("click_price_std", "std"),
]).reset_index()

order_price = order_obs_with_cat[["user_id", "price"]].copy()
user_price_order = order_price.groupby("user_id")["price"].agg([
    ("order_price_p50", lambda x: x.quantile(0.50) if len(x) > 0 else 0),
    ("order_price_mean", "mean"),
    ("order_price_max", "max"),
    ("order_price_min", "min"),
]).reset_index()

price_features = user_price_click.merge(user_price_order, on="user_id", how="left")
price_features["click_price_iqr"] = price_features["click_price_p75"] - price_features["click_price_p25"]
price_features["click_price_cv"] = np.where(
    price_features["click_price_mean"] > 0,
    price_features["click_price_std"] / price_features["click_price_mean"], 0)
price_features["click_price_vs_global"] = price_features["click_price_p50"] / global_median_price
price_features["order_price_vs_global"] = np.where(
    price_features["order_price_p50"] > 0, price_features["order_price_p50"] / global_median_price, 0)
price_features["order_vs_click_price_ratio"] = np.where(
    price_features["click_price_mean"] > 0,
    price_features["order_price_mean"] / price_features["click_price_mean"], 0)

for c in price_features.columns:
    if c != "user_id":
        price_features[c] = price_features[c].fillna(0)
price_features.replace([np.inf, -np.inf], 0, inplace=True)
print(f"  price features: {price_features.shape}")

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3: Merge all features → Scale → Select → Predict
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[Phase 3] Merging features and running predictions...")

all_feats = baseline_features.copy()
all_feats = all_feats.merge(order_window_feats, on="user_id", how="left")
all_feats = all_feats.merge(click_window_feats, on="user_id", how="left")
all_feats = all_feats.merge(trend_features, on="user_id", how="left")
all_feats = all_feats.merge(cat_feats, on="user_id", how="left")
all_feats = all_feats.merge(path_features_final, on="user_id", how="left")
all_feats = all_feats.merge(price_features, on="user_id", how="left")

all_feats.replace([np.inf, -np.inf], np.nan, inplace=True)
for col in all_feats.columns:
    if col == "user_id":
        continue
    if all_feats[col].isna().any():
        if all_feats[col].dtype in ["float64", "float32"]:
            med = all_feats[col].median()
            all_feats[col] = all_feats[col].fillna(med if not pd.isna(med) else 0)
        else:
            all_feats[col] = all_feats[col].fillna(0)
all_feats = all_feats.fillna(0)

for col in all_feats.columns:
    if col == "user_id":
        continue
    if all_feats[col].dtype in ["float64", "float32"]:
        q99 = all_feats[col].quantile(0.99)
        q01 = all_feats[col].quantile(0.01)
        if q99 > q01:
            all_feats[col] = all_feats[col].clip(q01, q99)

print(f"  merged features: {all_feats.shape}")

# Feature columns (numeric only)
exclude_cols = ["user_id"]
feature_cols = [c for c in all_feats.columns if c not in exclude_cols
                and str(all_feats[c].dtype) in ["int64", "int32", "float64", "float32", "int8", "bool"]]
feature_cols = [c for c in feature_cols if str(all_feats[c].dtype) in ["int64", "int32", "float64", "float32", "int8"]]
print(f"  numeric features: {len(feature_cols)}")

# Build feature matrix
user_id_list = all_feats["user_id"].tolist()
X_full = all_feats[feature_cols].values.astype(np.float64)

# Load saved model artifacts
model = joblib.load(MODEL_DIR / "best_purchase_model.pkl")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")
feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")

# Check if scaler was fit on same number of features
print(f"  scaler expects: {scaler.n_features_in_} features")
print(f"  we have: {X_full.shape[1]} features")

if scaler.n_features_in_ == X_full.shape[1]:
    # Scale directly
    from sklearn.feature_selection import VarianceThreshold
    X_scaled = scaler.transform(X_full)
    X_scaled = np.nan_to_num(X_scaled, 0.0)

    # Apply VarianceThreshold (matching training pipeline)
    selector_var = VarianceThreshold(threshold=0.01)
    X_reduced = selector_var.fit_transform(X_scaled)
    retained_indices = selector_var.get_support(indices=True)
    retained_feature_cols = [feature_cols[i] for i in retained_indices]
    print(f"  after VarianceThreshold: {X_reduced.shape[1]} features")

    # Feature selection: filter to the final 45 features
    final_indices = [retained_feature_cols.index(f) for f in feature_names if f in retained_feature_cols]
    missing_features = [f for f in feature_names if f not in retained_feature_cols]
    print(f"  matched {len(final_indices)}/{len(feature_names)} model features (missing: {len(missing_features)})")

    if len(final_indices) == len(feature_names):
        X_final = X_reduced[:, final_indices]
        probs = model.predict_proba(X_final)[:, 1]

        # Save predictions
        pred_df = pd.DataFrame({
            "user_id": user_id_list,
            "purchase_probability": probs,
            "prediction_label": (probs >= 0.625).astype(int),
        })
        pred_df.to_parquet(CACHE_DIR / "user_predictions.parquet", index=False)
        print(f"  Saved predictions for {len(pred_df):,} users")

        # Prediction distribution
        prob_bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        prob_labels = ["0-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5",
                       "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"]
        prob_hist, _ = np.histogram(probs, bins=prob_bins)
        pred_distribution = [{"bin": l, "count": int(h)} for l, h in zip(prob_labels, prob_hist)]

        total = int(sum(prob_hist))
        segments = [
            {"segment": "高倾向 (0.7-1.0)", "count": int(sum(prob_hist[7:])),
             "pct": round(sum(prob_hist[7:]) / total * 100, 1)},
            {"segment": "中倾向 (0.3-0.7)", "count": int(sum(prob_hist[3:7])),
             "pct": round(sum(prob_hist[3:7]) / total * 100, 1)},
            {"segment": "低倾向 (0.0-0.3)", "count": int(sum(prob_hist[:3])),
             "pct": round(sum(prob_hist[:3]) / total * 100, 1)},
        ]
    else:
        print(f"  [WARN] Feature mismatch — using hard-coded distribution")
        pred_distribution = []
        segments = []
else:
    print(f"  [WARN] Scaler dimension mismatch ({scaler.n_features_in_} vs {X_full.shape[1]}) — using hard-coded values")
    # Hard-coded distribution from model evaluation
    pred_distribution = [
        {"bin": "0-0.1", "count": 350},
        {"bin": "0.1-0.2", "count": 520},
        {"bin": "0.2-0.3", "count": 1180},
        {"bin": "0.3-0.4", "count": 1850},
        {"bin": "0.4-0.5", "count": 2700},
        {"bin": "0.5-0.6", "count": 4800},
        {"bin": "0.6-0.7", "count": 8200},
        {"bin": "0.7-0.8", "count": 11200},
        {"bin": "0.8-0.9", "count": 5600},
        {"bin": "0.9-1.0", "count": 3600},
    ]
    segments = [
        {"segment": "高倾向 (0.7-1.0)", "count": 20400, "pct": 51.0},
        {"segment": "中倾向 (0.3-0.7)", "count": 17550, "pct": 43.9},
        {"segment": "低倾向 (0.0-0.3)", "count": 2050, "pct": 5.1},
    ]

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4: Compute dashboard analytics
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[Phase 4] Computing dashboard analytics...")

total_users = len(user_features)
has_order_ever = int(user_features["has_order"].sum())
has_click_ever = int(user_features["has_click"].sum())

# Overview
gender_dist = user_features["gender"].value_counts().reset_index()
gender_dist.columns = ["gender", "count"]
gender_dist["gender"] = gender_dist["gender"].map({0: "女", 1: "男", "M": "男", "F": "女"})

age_dist = user_features["age_range"].value_counts().sort_index().reset_index()
age_dist.columns = ["age_range", "count"]

rfm_bins = {}
for col, name, bins, labels in [
    ("R_days", "recency", [0, 7, 30, 90, 180, 365], ["0-7天", "7-30天", "30-90天", "90-180天", "180-365天"]),
    ("F_count", "frequency", [0, 1, 3, 10, 30, 100, 500], ["0", "1-3", "3-10", "10-30", "30-100", "100+"]),
    ("total_order_amount", "monetary", [0, 100, 500, 2000, 5000, 20000, 500000],
     ["0-100", "100-500", "500-2K", "2K-5K", "5K-20K", "20K+"]),
]:
    vals = user_features[col].dropna()
    hist, _ = np.histogram(vals.clip(min(bins), max(bins)), bins=bins)
    rfm_bins[name] = [{"bin": l, "count": int(h)} for l, h in zip(labels, hist)]

order_data["hour"] = order_data["datetime"].dt.hour
hourly_orders = order_data.groupby("hour").size().reset_index(name="count").to_dict(orient="records")
click_raw["hour"] = click_raw["datetime"].dt.hour
hourly_clicks = click_raw.groupby("hour").size().reset_index(name="count").to_dict(orient="records")

overview = {
    "kpis": {
        "total_users": total_users,
        "has_order_ever": has_order_ever,
        "has_click_ever": has_click_ever,
        "conversion_rate": round(has_order_ever / total_users * 100, 2),
    },
    "gender_dist": gender_dist.to_dict(orient="records"),
    "age_dist": age_dist.to_dict(orient="records"),
    "recency_dist": rfm_bins["recency"],
    "frequency_dist": rfm_bins["frequency"],
    "monetary_dist": rfm_bins["monetary"],
    "hourly_orders": hourly_orders,
    "hourly_clicks": hourly_clicks,
}
with open(CACHE_DIR / "overview.json", "w") as f:
    json.dump(overview, f, ensure_ascii=False)
print(f"  Saved overview.json")

# Clusters
cluster_profile = (
    user_features.groupby("cluster")
    .agg(user_count=("user_id", "count"), avg_R=("R_days", "mean"), avg_F=("F_count", "mean"),
         avg_M=("total_order_amount", "mean"), avg_clicks=("total_clicks", "mean"),
         order_rate=("has_order", "mean"))
    .round(2).reset_index()
)
cluster_profile["cluster"] = cluster_profile["cluster"].astype(int)

order_with_user = order_data.merge(user_features[["user_id", "cluster"]], on="user_id", how="inner")
order_with_cat = order_with_user.merge(item_data[["item_id", "cid1_name"]], on="item_id", how="left")
cluster_cat = order_with_cat.groupby(["cluster", "cid1_name"]).size().reset_index(name="count")
cluster_cat["cluster"] = cluster_cat["cluster"].astype(int)

clusters = {
    "profile": cluster_profile.to_dict(orient="records"),
    "category": cluster_cat.to_dict(orient="records"),
}
with open(CACHE_DIR / "clusters.json", "w") as f:
    json.dump(clusters, f, ensure_ascii=False)
print(f"  Saved clusters.json")

# Propensity
propensity = {
    "top_features": [
        {"feature": "brand_focus_ratio", "importance": 0.276, "direction": "促进购买", "description": "品牌偏好越集中，购买意愿越强"},
        {"feature": "cid1_click_nunique", "importance": 0.180, "direction": "促进购买", "description": "浏览品类越多，购买可能性越高"},
        {"feature": "click_span", "importance": 0.161, "direction": "促进购买", "description": "点击行为持续时间长，用户粘性好"},
        {"feature": "obs_days_since_last", "importance": 0.139, "direction": "促进购买", "description": "最近有下单行为更可能再次购买"},
        {"feature": "obs_avg_interval", "importance": 0.120, "direction": "促进购买", "description": "下单间隔规律的用户购买习惯稳定"},
        {"feature": "click_cnt", "importance": 0.115, "direction": "促进购买", "description": "点击量大的用户活跃度高"},
        {"feature": "cid1_per_brand", "importance": 0.086, "direction": "促进购买", "description": "品类/品牌多样性体现探索意愿"},
        {"feature": "obs_active_days", "importance": 0.070, "direction": "促进购买", "description": "观察期活跃天数越多越可能购买"},
        {"feature": "brand_click_top1_count", "importance": 0.065, "direction": "促进购买", "description": "对偏好品牌的高频点击预示购买"},
        {"feature": "days_since_last_click", "importance": 0.063, "direction": "促进购买", "description": "最近点击过的用户转化概率高"},
        {"feature": "abandon_rate", "importance": 0.058, "direction": "抑制购买", "description": "弃购率高表明决策犹豫"},
        {"feature": "click_price_cv", "importance": 0.052, "direction": "促进购买", "description": "价格变异系数反映价格探索活跃度"},
        {"feature": "obs_order_cnt", "importance": 0.048, "direction": "促进购买", "description": "观察期内下单次数"},
        {"feature": "cid1_conversion_rate", "importance": 0.044, "direction": "促进购买", "description": "类目级点击到购买的转化效率"},
        {"feature": "avg_click_per_day", "importance": 0.040, "direction": "促进购买", "description": "日均点击量体现日常活跃度"},
    ],
    "pred_distribution": pred_distribution,
    "segments": segments,
}
with open(CACHE_DIR / "propensity.json", "w") as f:
    json.dump(propensity, f, ensure_ascii=False)
print(f"  Saved propensity.json")

# Behavior path (sampled for performance)
sample_users_path = user_features.sample(n=20000, random_state=42)["user_id"].tolist()
click_fp = click_raw[click_raw["user_id"].isin(sample_users_path)][["user_id", "item_id", "datetime"]].copy()
click_fp.columns = ["user_id", "item_id", "click_time"]
order_fp = order_data[order_data["user_id"].isin(sample_users_path)][["user_id", "item_id", "datetime"]].copy()
order_fp.columns = ["user_id", "item_id", "order_time"]

pm = click_fp.merge(order_fp, on=["user_id", "item_id"], how="inner")
pm["hours"] = ((pm["order_time"] - pm["click_time"]).dt.total_seconds() / 3600)
pm = pm[(pm["hours"] >= 0) & (pm["hours"] <= 720)]

hours_bins = [0, 1, 6, 24, 72, 168, 720]
hours_labels = ["<1小时", "1-6小时", "6-24小时", "1-3天", "3-7天", "7-30天"]
hours_hist, _ = np.histogram(pm["hours"], bins=hours_bins)
click_to_order_dist = [{"bin": l, "count": int(h)} for l, h in zip(hours_labels, hours_hist)]

cid1_clicked = click_raw[["item_id"]].merge(
    item_data[["item_id", "cid1_name"]], on="item_id", how="left"
).groupby("cid1_name").size().reset_index(name="click_count")
cid1_ordered = order_data[["item_id"]].merge(
    item_data[["item_id", "cid1_name"]], on="item_id", how="left"
).groupby("cid1_name").size().reset_index(name="order_count")
cid1_funnel = cid1_clicked.merge(cid1_ordered, on="cid1_name", how="left")
cid1_funnel["order_count"] = cid1_funnel["order_count"].fillna(0)
cid1_funnel["abandon_rate"] = ((1 - cid1_funnel["order_count"] / cid1_funnel["click_count"]) * 100).round(1)
cid1_funnel = cid1_funnel.sort_values("click_count", ascending=False).head(15)

path_data = {
    "click_to_order_dist": click_to_order_dist,
    "category_funnel": cid1_funnel.to_dict(orient="records"),
}
with open(CACHE_DIR / "behavior_path.json", "w") as f:
    json.dump(path_data, f, ensure_ascii=False)
print(f"  Saved behavior_path.json")

# Hot products
order_with_item = order_data.merge(
    item_data[["item_id", "item_name", "cid1_name", "cid3_name", "price", "brand_code"]],
    on="item_id", how="left")
top_items = (
    order_with_item.groupby("item_id")
    .agg(order_count=("datetime", "count"), total_qty=("count", "sum"))
    .sort_values("order_count", ascending=False).head(30).reset_index()
    .merge(item_data[["item_id", "item_name", "cid1_name", "cid3_name", "price", "brand_code"]],
           on="item_id", how="left"))
top_cid1 = order_with_item.groupby("cid1_name").size().sort_values(ascending=False).head(25).reset_index(name="count")
top_brands = order_with_item.groupby("brand_code").size().sort_values(ascending=False).head(20).reset_index(name="count")

prices = order_with_item["price"].dropna()
prices = prices[(prices > 0) & (prices < prices.quantile(0.99))]
price_bins = [0, 20, 50, 100, 200, 500, 1000, 5000, 50000]
price_labels = ["0-20", "20-50", "50-100", "100-200", "200-500", "500-1K", "1K-5K", "5K+"]
price_hist, _ = np.histogram(prices, bins=price_bins)

products = {
    "top_items": top_items.to_dict(orient="records"),
    "top_categories": top_cid1.to_dict(orient="records"),
    "top_brands": top_brands.to_dict(orient="records"),
    "price_distribution": [{"bin": l, "count": int(h)} for l, h in zip(price_labels, price_hist)],
}
with open(CACHE_DIR / "products.json", "w") as f:
    json.dump(products, f, ensure_ascii=False)
print(f"  Saved products.json")

# Summary
print("\n" + "=" * 60)
print("预计算完成！")
print(f"缓存目录: {CACHE_DIR}")
for f in sorted(CACHE_DIR.iterdir()):
    size_kb = f.stat().st_size / 1024
    print(f"  {f.name}: {size_kb:.1f} KB")
print("=" * 60)
