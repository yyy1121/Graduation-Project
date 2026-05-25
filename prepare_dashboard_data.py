#!/usr/bin/env python3
"""Prepare dashboard cache with the persisted training pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from purchase_pipeline import build_features_for_window, day_end


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "parquet"
MODEL_DIR = BASE_DIR / "data" / "models"
CACHE_DIR = BASE_DIR / "data" / "dashboard_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def dump_json(name: str, data) -> None:
    with open(CACHE_DIR / f"{name}.json", "w") as f:
        json.dump(data, f, ensure_ascii=False, allow_nan=False)
    print(f"  saved {name}.json")


def sanitize_records(df: pd.DataFrame) -> list[dict]:
    clean = df.replace([np.inf, -np.inf], np.nan)
    return json.loads(clean.to_json(orient="records", force_ascii=False))


print("=" * 70)
print("仪表盘数据预计算：复用训练时保存的完整预处理管线")
print("=" * 70)

print("\n[1] Loading data and artifacts...")
user_features = pd.read_parquet(DATA_DIR / "user_features_raw_with_cluster.parquet")
order = pd.read_parquet(DATA_DIR / "user_order_cleaned.parquet")
click = pd.read_parquet(DATA_DIR / "user_click.parquet")
item = pd.read_parquet(
    DATA_DIR / "item_data_cleaned.parquet",
    columns=["item_id", "cid1", "cid2", "cid3", "cid1_name", "cid2_name", "cid3_name", "brand_code", "price", "item_name"],
)

for df in [user_features, order, click]:
    df["user_id"] = df["user_id"].astype(str)
for df in [order, click, item]:
    df["item_id"] = df["item_id"].astype(str)
order["datetime"] = pd.to_datetime(order["datetime"])
click["datetime"] = pd.to_datetime(click["datetime"], format="mixed")

model_path = MODEL_DIR / "best_purchase_model.pkl"
preprocessor_path = MODEL_DIR / "preprocessor.pkl"
if not preprocessor_path.exists():
    raise FileNotFoundError(
        f"{preprocessor_path} not found. Run `conda run -n ml python improved_modeling.py` "
        "first so dashboard predictions reuse the exact fitted preprocessing pipeline."
    )
model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)
calibrator_path = MODEL_DIR / "calibrator.pkl"
calibrator = joblib.load(calibrator_path) if calibrator_path.exists() else None
metadata_path = MODEL_DIR / "model_metadata.json"
metadata = json.load(open(metadata_path)) if metadata_path.exists() else {}
threshold = float(metadata.get("threshold", 0.625))
print(
    f"  model={metadata.get('best_model_name', type(model).__name__)}, "
    f"threshold={threshold:.3f}, calibrated={calibrator is not None}"
)

print("\n[2] Building latest-window features and predictions...")
pred_end = day_end(order["datetime"].max())
user_ids = sorted(user_features["user_id"].unique().tolist())
features = build_features_for_window(user_features, order, click, item, pred_end, user_ids=user_ids)
x_full = preprocessor.transform(features)
probs = model.predict_proba(x_full)[:, 1]
if calibrator is not None:
    probs = calibrator.transform(probs)
pred_df = pd.DataFrame({
    "user_id": features["user_id"].tolist(),
    "purchase_probability": probs,
    "prediction_label": (probs >= threshold).astype(int),
})
pred_df.to_parquet(CACHE_DIR / "user_predictions.parquet", index=False)
print(f"  saved predictions for {len(pred_df):,} users")

hist_upper = float(max(np.quantile(probs, 0.995), threshold * 4, 0.2))
hist_upper = min(max(hist_upper, 0.01), 1.0)
prob_bins = np.unique(np.r_[np.linspace(0, hist_upper, 11), 1.0])
prob_labels = [f"{prob_bins[i]:.3f}-{prob_bins[i + 1]:.3f}" for i in range(len(prob_bins) - 1)]
prob_hist, _ = np.histogram(probs, bins=prob_bins)
pred_distribution = [
    {"bin": l, "left": float(prob_bins[i]), "right": float(prob_bins[i + 1]), "count": int(h)}
    for i, (l, h) in enumerate(zip(prob_labels, prob_hist))
]

high_cutoff = float(max(threshold, np.quantile(probs, 0.90)))
predicted_positive = probs >= threshold
high_mask = probs >= high_cutoff
medium_mask = predicted_positive & ~high_mask
low_mask = probs < threshold
total = max(len(probs), 1)
segments = [
    {"segment": f"高倾向 (Top10% ≥ {high_cutoff:.3f})", "count": int(high_mask.sum()), "pct": round(float(high_mask.sum() / total * 100), 1)},
    {"segment": f"中倾向 (阈值-{high_cutoff:.3f})", "count": int(medium_mask.sum()), "pct": round(float(medium_mask.sum() / total * 100), 1)},
    {"segment": f"低倾向 (< {threshold:.3f})", "count": int(low_mask.sum()), "pct": round(float(low_mask.sum() / total * 100), 1)},
]
segment_cutoffs = {
    "threshold": threshold,
    "high_cutoff": high_cutoff,
    "strategy": "low_below_threshold_medium_above_threshold_high_top10",
}
metrics = metadata.get("test_metrics", {})
topk_rows = metadata.get("business_topk_test") or []
top10 = next((row for row in topk_rows if abs(float(row.get("top_rate", 0)) - 0.10) < 1e-9), {})
top20 = next((row for row in topk_rows if abs(float(row.get("top_rate", 0)) - 0.20) < 1e-9), {})
strategy_summary = {
    "model_name": metadata.get("best_model_name", type(model).__name__),
    "threshold": threshold,
    "high_cutoff": high_cutoff,
    "predicted_positive_count": int(predicted_positive.sum()),
    "predicted_positive_pct": round(float(predicted_positive.sum() / total * 100), 1),
    "auc": round(float(metrics.get("auc", 0)), 4) if metrics else None,
    "ap": round(float(metrics.get("ap", 0)), 4) if metrics else None,
    "precision": round(float(metrics.get("precision", 0)), 4) if metrics else None,
    "recall": round(float(metrics.get("recall", 0)), 4) if metrics else None,
    "f2": round(float(metrics.get("f2", 0)), 4) if metrics else None,
    "top10_precision": round(float(top10.get("precision", 0)), 4) if top10 else None,
    "top10_lift": round(float(top10.get("lift", 0)), 2) if top10 else None,
    "top10_profit": round(float(top10.get("expected_profit", 0)), 1) if top10 else None,
    "top20_precision": round(float(top20.get("precision", 0)), 4) if top20 else None,
    "top20_lift": round(float(top20.get("lift", 0)), 2) if top20 else None,
}
operation_strategy = [
    {
        "segment": "高倾向",
        "rule": f"概率 ≥ {high_cutoff:.3f}，约 Top10%",
        "count": int(high_mask.sum()),
        "pct": round(float(high_mask.sum() / total * 100), 1),
        "priority": "优先触达",
        "action": "会员权益、限时券、购物车/浏览商品提醒",
        "expected": f"测试 Top10 Precision {strategy_summary['top10_precision'] * 100:.1f}%，Lift {strategy_summary['top10_lift']:.2f}",
        "risk": "控制补贴强度，避免把优惠浪费在本会自然购买的用户",
    },
    {
        "segment": "中倾向",
        "rule": f"{threshold:.3f} ≤ 概率 < {high_cutoff:.3f}",
        "count": int(medium_mask.sum()),
        "pct": round(float(medium_mask.sum() / total * 100), 1),
        "priority": "培育转化",
        "action": "低成本站内信、相似商品推荐、价格/评价信息补全",
        "expected": f"阈值以上总覆盖 {strategy_summary['predicted_positive_pct']:.1f}%，Recall {strategy_summary['recall'] * 100:.1f}%",
        "risk": "假阳性成本较高，不建议直接发高额券",
    },
    {
        "segment": "低倾向",
        "rule": f"概率 < {threshold:.3f}",
        "count": int(low_mask.sum()),
        "pct": round(float(low_mask.sum() / total * 100), 1),
        "priority": "低频维护",
        "action": "品牌曝光、内容种草、节日大促统一触达",
        "expected": "不进入本轮强转化名单，保留低成本触达",
        "risk": "避免因模型分数低而永久排除，需定期刷新窗口",
    },
]

top_features = metadata.get("shap_top_features") or []
if not top_features:
    model_importance = getattr(model, "feature_importances_", None)
    if model_importance is not None and len(model_importance) == len(preprocessor.selected_feature_names):
        imp = np.asarray(model_importance, dtype=float)
        denom = float(imp.sum()) or 1.0
        order_idx = np.argsort(imp)[::-1][:15]
        top_features = [
            {
                "feature": preprocessor.selected_feature_names[i],
                "importance": round(float(imp[i] / denom), 6),
                "direction": "模型重要性",
                "description": "未运行 SHAP，使用模型内置特征重要性作为可视化回退",
            }
            for i in order_idx
        ]
    else:
        top_features = [
            {"feature": f, "importance": 0.0, "direction": "待重新训练后生成", "description": "请运行 improved_modeling.py 生成 SHAP 元数据"}
            for f in preprocessor.selected_feature_names[:15]
        ]
for row in top_features:
    if row.get("importance", 0) and not row.get("direction"):
        row["direction"] = "SHAP重要性"
    row.setdefault("direction", "")
    row.setdefault("description", "基于测试集抽样 SHAP，重要性为 mean(|SHAP|)")
dump_json(
    "propensity",
    {
        "top_features": top_features,
        "pred_distribution": pred_distribution,
        "segments": segments,
        "segment_cutoffs": segment_cutoffs,
        "strategy_summary": strategy_summary,
        "operation_strategy": operation_strategy,
    },
)

print("\n[3] Computing overview cache...")
gender_dist = user_features["gender"].value_counts().reset_index()
gender_dist.columns = ["gender", "count"]
gender_dist["gender"] = gender_dist["gender"].map({0: "女", 1: "男", "M": "男", "F": "女"}).fillna(gender_dist["gender"].astype(str))
age_dist = user_features["age_range"].value_counts().sort_index().reset_index()
age_dist.columns = ["age_range", "count"]

rfm_bins = {}
for col, name, bins, labels in [
    ("R_days", "recency", [0, 7, 30, 90, 180, 365], ["0-7天", "7-30天", "30-90天", "90-180天", "180-365天"]),
    ("F_count", "frequency", [0, 1, 3, 10, 30, 100, 500], ["0", "1-3", "3-10", "10-30", "30-100", "100+"]),
    ("total_order_amount", "monetary", [0, 100, 500, 2000, 5000, 20000, 500000], ["0-100", "100-500", "500-2K", "2K-5K", "5K-20K", "20K+"]),
]:
    vals = user_features[col].dropna().clip(min(bins), max(bins))
    hist, _ = np.histogram(vals, bins=bins)
    rfm_bins[name] = [{"bin": l, "count": int(h)} for l, h in zip(labels, hist)]

order["hour"] = order["datetime"].dt.hour
click["hour"] = click["datetime"].dt.hour
overview = {
    "kpis": {
        "total_users": int(len(user_features)),
        "has_order_ever": int(user_features["has_order"].sum()),
        "has_click_ever": int(user_features["has_click"].sum()),
        "conversion_rate": round(float(user_features["has_order"].mean() * 100), 2),
    },
    "gender_dist": sanitize_records(gender_dist),
    "age_dist": sanitize_records(age_dist),
    "recency_dist": rfm_bins["recency"],
    "frequency_dist": rfm_bins["frequency"],
    "monetary_dist": rfm_bins["monetary"],
    "hourly_orders": sanitize_records(order.groupby("hour").size().reset_index(name="count")),
    "hourly_clicks": sanitize_records(click.groupby("hour").size().reset_index(name="count")),
}
dump_json("overview", overview)

print("\n[4] Computing cluster cache...")
cluster_profile = (
    user_features.groupby("cluster")
    .agg(
        user_count=("user_id", "count"),
        avg_R=("R_days", "mean"),
        avg_F=("F_count", "mean"),
        avg_M=("total_order_amount", "mean"),
        avg_clicks=("total_clicks", "mean"),
        order_rate=("has_order", "mean"),
    )
    .round(2)
    .reset_index()
)
cluster_profile["cluster"] = cluster_profile["cluster"].astype(int)
order_with_user = order.merge(user_features[["user_id", "cluster"]], on="user_id", how="inner")
order_with_cat = order_with_user.merge(item[["item_id", "cid1_name"]], on="item_id", how="left")
cluster_cat = order_with_cat.groupby(["cluster", "cid1_name"]).size().reset_index(name="count")
cluster_cat["cluster"] = cluster_cat["cluster"].astype(int)
dump_json("clusters", {"profile": sanitize_records(cluster_profile), "category": sanitize_records(cluster_cat)})

print("\n[5] Computing behavior path cache...")
sample_users = user_features.sample(n=min(20000, len(user_features)), random_state=42)["user_id"].tolist()
click_fp = click[click["user_id"].isin(sample_users)][["user_id", "item_id", "datetime"]].rename(columns={"datetime": "click_time"})
order_fp = order[order["user_id"].isin(sample_users)][["user_id", "item_id", "datetime"]].rename(columns={"datetime": "order_time"})
pm = click_fp.merge(order_fp, on=["user_id", "item_id"], how="inner")
pm["hours"] = (pm["order_time"] - pm["click_time"]).dt.total_seconds() / 3600
pm = pm[(pm["hours"] >= 0) & (pm["hours"] <= 720)]
hours_bins = [0, 1, 6, 24, 72, 168, 720]
hours_labels = ["<1小时", "1-6小时", "6-24小时", "1-3天", "3-7天", "7-30天"]
hours_hist, _ = np.histogram(pm["hours"], bins=hours_bins)
cid1_clicked = click[["item_id"]].merge(item[["item_id", "cid1_name"]], on="item_id", how="left").groupby("cid1_name").size().reset_index(name="click_count")
cid1_ordered = order[["item_id"]].merge(item[["item_id", "cid1_name"]], on="item_id", how="left").groupby("cid1_name").size().reset_index(name="order_count")
cid1_funnel = cid1_clicked.merge(cid1_ordered, on="cid1_name", how="left")
cid1_funnel["order_count"] = cid1_funnel["order_count"].fillna(0)
cid1_funnel["abandon_rate"] = ((1 - cid1_funnel["order_count"] / cid1_funnel["click_count"]) * 100).round(1)
cid1_funnel = cid1_funnel.sort_values("click_count", ascending=False).head(15)
dump_json("behavior_path", {
    "click_to_order_dist": [{"bin": l, "count": int(h)} for l, h in zip(hours_labels, hours_hist)],
    "category_funnel": sanitize_records(cid1_funnel),
})

print("\n[6] Computing product cache...")
order_with_item = order.merge(
    item[["item_id", "item_name", "cid1_name", "cid3_name", "price", "brand_code"]],
    on="item_id",
    how="left",
)
top_items = (
    order_with_item.groupby("item_id")
    .agg(order_count=("datetime", "count"), total_qty=("count", "sum"))
    .sort_values("order_count", ascending=False)
    .head(30)
    .reset_index()
    .merge(item[["item_id", "item_name", "cid1_name", "cid3_name", "price", "brand_code"]], on="item_id", how="left")
)
top_cid1 = order_with_item.groupby("cid1_name").size().sort_values(ascending=False).head(25).reset_index(name="count")
top_brands = order_with_item.groupby("brand_code").size().sort_values(ascending=False).head(20).reset_index(name="count")
prices = order_with_item["price"].dropna()
prices = prices[(prices > 0) & (prices < prices.quantile(0.99))]
price_bins = [0, 20, 50, 100, 200, 500, 1000, 5000, 50000]
price_labels = ["0-20", "20-50", "50-100", "100-200", "200-500", "500-1K", "1K-5K", "5K+"]
price_hist, _ = np.histogram(prices, bins=price_bins)
dump_json("products", {
    "top_items": sanitize_records(top_items),
    "top_categories": sanitize_records(top_cid1),
    "top_brands": sanitize_records(top_brands),
    "price_distribution": [{"bin": l, "count": int(h)} for l, h in zip(price_labels, price_hist)],
})

print("\n预计算完成")
for path in sorted(CACHE_DIR.iterdir()):
    print(f"  {path.name}: {path.stat().st_size / 1024:.1f} KB")
