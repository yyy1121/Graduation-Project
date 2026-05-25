#!/usr/bin/env python3
"""Shared temporal feature and preprocessing utilities for purchase propensity."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.preprocessing import StandardScaler


OBS_DAYS = 30
PRED_DAYS = 7
WINDOWS = [7, 14, 30, 60]
ID_COLS = {"user_id", "label", "snapshot", "pred_start", "pred_end"}


def day_end(ts: pd.Timestamp) -> pd.Timestamp:
    return ts.normalize() + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)


def window_bounds(pred_end: pd.Timestamp, obs_days: int = OBS_DAYS, pred_days: int = PRED_DAYS) -> dict:
    pred_end = day_end(pd.Timestamp(pred_end))
    pred_start = pred_end.normalize() - pd.Timedelta(days=pred_days - 1)
    obs_end = pred_start - pd.Timedelta(microseconds=1)
    obs_start = pred_start - pd.Timedelta(days=obs_days)
    return {
        "pred_start": pred_start,
        "pred_end": pred_end,
        "obs_start": obs_start,
        "obs_end": obs_end,
    }


def temporal_snapshot_ends(max_dt: pd.Timestamp, n_train_windows: int = 4) -> dict:
    """Return non-overlapping weekly train/validation/test prediction windows."""
    test_end = day_end(pd.Timestamp(max_dt))
    test_start = window_bounds(test_end)["pred_start"]
    val_end = test_start - pd.Timedelta(microseconds=1)
    val_start = window_bounds(val_end)["pred_start"]
    train_ends = []
    cur_end = val_start - pd.Timedelta(microseconds=1)
    for _ in range(n_train_windows):
        train_ends.append(cur_end)
        cur_end = window_bounds(cur_end)["pred_start"] - pd.Timedelta(microseconds=1)
    train_ends.reverse()
    return {"train": train_ends, "val": val_end, "test": test_end}


def build_labels(order: pd.DataFrame, user_ids: Iterable[str], pred_start: pd.Timestamp, pred_end: pd.Timestamp) -> pd.DataFrame:
    pred_orders = order[(order["datetime"] >= pred_start) & (order["datetime"] <= pred_end)]
    pred_users = set(pred_orders["user_id"].unique())
    labels = pd.DataFrame({"user_id": list(user_ids)})
    labels["label"] = labels["user_id"].isin(pred_users).astype(int)
    return labels


def _safe_top1_count(x: pd.Series) -> int:
    vc = x.dropna().value_counts()
    return int(vc.iloc[0]) if len(vc) else 0


def _safe_ratio(num, den, default=0.0):
    return np.divide(num, den, out=np.full_like(np.asarray(num, dtype=float), default), where=np.asarray(den) != 0)


def build_window_features(
    df: pd.DataFrame,
    user_ids: Iterable[str],
    pred_start: pd.Timestamp,
    obs_end: pd.Timestamp,
    prefix: str,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    windows = windows or WINDOWS
    features = pd.DataFrame({"user_id": list(user_ids)})

    for days in windows:
        window_start = pred_start - pd.Timedelta(days=days)
        window_data = df[(df["datetime"] >= window_start) & (df["datetime"] <= obs_end)]
        agg = window_data.groupby("user_id").agg(
            **{
                f"{prefix}_cnt_{days}d": ("datetime", "count"),
                f"{prefix}_items_{days}d": ("item_id", "nunique"),
                f"{prefix}_active_days_{days}d": ("datetime", lambda x: x.dt.date.nunique()),
                f"{prefix}_last_days_{days}d": ("datetime", lambda x: (pred_start - x.max()).days),
            }
        ).reset_index()
        features = features.merge(agg, on="user_id", how="left")

    for days in windows:
        for suffix in ["cnt", "items", "active_days"]:
            col = f"{prefix}_{suffix}_{days}d"
            if col in features:
                features[col] = features[col].fillna(0)
        last_col = f"{prefix}_last_days_{days}d"
        if last_col in features:
            features[last_col] = features[last_col].fillna(days + PRED_DAYS)
    return features


def build_features_for_window(
    user_features: pd.DataFrame,
    order: pd.DataFrame,
    click_raw: pd.DataFrame,
    item: pd.DataFrame,
    pred_end: pd.Timestamp,
    user_ids: Iterable[str] | None = None,
    obs_days: int = OBS_DAYS,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Build leakage-safe features as of one prediction window.

    Baseline/category/path/price features use the 30-day observation window.
    Multi-window features use the raw logs and can therefore genuinely cover 60d.
    """
    windows = windows or WINDOWS
    bounds = window_bounds(pred_end, obs_days=obs_days)
    pred_start, obs_start, obs_end = bounds["pred_start"], bounds["obs_start"], bounds["obs_end"]
    user_ids = list(user_ids if user_ids is not None else user_features["user_id"].unique())

    order_obs = order[(order["datetime"] >= obs_start) & (order["datetime"] <= obs_end)]
    click_obs = click_raw[(click_raw["datetime"] >= obs_start) & (click_raw["datetime"] <= obs_end)]

    order_agg = order_obs.groupby("user_id").agg(
        obs_order_cnt=("datetime", "count"),
        obs_total_qty=("count", "sum"),
        obs_unique_items=("item_id", "nunique"),
        obs_active_days=("datetime", lambda x: x.dt.date.nunique()),
        obs_first_order=("datetime", "min"),
        obs_last_order=("datetime", "max"),
    ).reset_index()
    order_agg["obs_order_span"] = (order_agg["obs_last_order"] - order_agg["obs_first_order"]).dt.days.clip(lower=1)
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
    click_agg["click_span"] = (click_agg["last_click"] - click_agg["first_click"]).dt.days.clip(lower=1)
    click_agg["avg_click_per_day"] = click_agg["click_cnt"] / click_agg["click_active_days"].clip(lower=1)
    click_agg["days_since_last_click"] = (pred_start - click_agg["last_click"]).dt.days
    click_agg.drop(["first_click", "last_click"], axis=1, inplace=True)

    base = pd.DataFrame({"user_id": user_ids})
    base = base.merge(order_agg, on="user_id", how="left").merge(click_agg, on="user_id", how="left")
    for c in [
        "obs_order_cnt", "obs_total_qty", "obs_unique_items", "obs_active_days", "obs_order_span",
        "obs_avg_items_per_order", "obs_avg_interval", "click_cnt", "unique_click_items",
        "click_active_days", "click_span", "avg_click_per_day",
    ]:
        if c in base:
            base[c] = base[c].fillna(0)
    no_activity_days = obs_days + PRED_DAYS
    base["obs_days_since_last"] = base["obs_days_since_last"].fillna(no_activity_days)
    base["days_since_last_click"] = base["days_since_last_click"].fillna(no_activity_days)
    base["click_to_buy_ratio"] = np.where(base["click_cnt"] > 0, base["obs_order_cnt"] / base["click_cnt"], 0)
    base["has_conversion"] = ((base["click_cnt"] > 0) & (base["obs_order_cnt"] > 0)).astype(int)
    base = base.merge(user_features[["user_id", "gender_encoded", "age_encoded"]], on="user_id", how="left")

    history_start = pred_start - pd.Timedelta(days=max(windows))
    order_hist = order[(order["datetime"] >= history_start) & (order["datetime"] <= obs_end)]
    click_hist = click_raw[(click_raw["datetime"] >= history_start) & (click_raw["datetime"] <= obs_end)]
    order_windows = build_window_features(order_hist, user_ids, pred_start, obs_end, "order", windows)
    click_windows = build_window_features(click_hist, user_ids, pred_start, obs_end, "click", windows)

    trend = pd.DataFrame({"user_id": user_ids})
    for prefix, feat_df in [("order", order_windows), ("click", click_windows)]:
        for metric in ["cnt", "items", "active_days"]:
            c7 = feat_df[f"{prefix}_{metric}_7d"].to_numpy(dtype=float)
            c14 = feat_df[f"{prefix}_{metric}_14d"].to_numpy(dtype=float)
            c30 = feat_df[f"{prefix}_{metric}_30d"].to_numpy(dtype=float)
            trend[f"{prefix}_{metric}_7d_14d_ratio"] = _safe_ratio(c7, c14)
            trend[f"{prefix}_{metric}_14d_30d_ratio"] = _safe_ratio(c14, c30)
            trend[f"{prefix}_{metric}_7d_30d_ratio"] = _safe_ratio(c7, c30)
        last_7d = feat_df[f"{prefix}_last_days_7d"].to_numpy(dtype=float)
        last_14d = feat_df[f"{prefix}_last_days_14d"].to_numpy(dtype=float)
        trend[f"{prefix}_recency_decay"] = last_7d / (last_14d + 1)

    for days in windows:
        click_val = click_windows[f"click_cnt_{days}d"].to_numpy(dtype=float)
        order_val = order_windows[f"order_cnt_{days}d"].to_numpy(dtype=float)
        trend[f"conversion_rate_{days}d"] = _safe_ratio(order_val, click_val)

    item_cols = ["item_id", "cid1", "cid2", "cid3", "brand_code", "price"]
    click_cat = click_obs[["user_id", "item_id", "datetime"]].merge(item[item_cols], on="item_id", how="left")
    order_cat = order_obs[["user_id", "item_id", "datetime"]].merge(item[item_cols], on="item_id", how="left")
    cat = pd.DataFrame({"user_id": user_ids})
    cid1_click = click_cat.groupby("user_id")["cid1"].agg(
        cid1_click_nunique="nunique", cid1_click_top1_count=_safe_top1_count
    ).reset_index()
    cid1_order = order_cat.groupby("user_id")["cid1"].agg(cid1_order_nunique="nunique").reset_index()
    cid3_click = click_cat.groupby("user_id")["cid3"].agg(cid3_click_nunique="nunique").reset_index()
    cid3_order = order_cat.groupby("user_id")["cid3"].agg(cid3_order_nunique="nunique").reset_index()
    brand_click = click_cat.groupby("user_id")["brand_code"].agg(
        brand_click_nunique="nunique", brand_click_top1_count=_safe_top1_count
    ).reset_index()
    brand_order = order_cat.groupby("user_id")["brand_code"].agg(brand_order_nunique="nunique").reset_index()
    total_clicks_cat = click_cat.groupby("user_id").size().reset_index(name="total_clicks_cat")
    total_orders_cat = order_cat.groupby("user_id").size().reset_index(name="total_orders_cat")
    for df in [cid1_click, cid1_order, cid3_click, cid3_order, brand_click, brand_order, total_clicks_cat, total_orders_cat]:
        cat = cat.merge(df, on="user_id", how="left")
    for c in cat.columns:
        if c != "user_id":
            cat[c] = cat[c].fillna(0)
    cat["top1_cid1_click_ratio"] = np.where(cat["total_clicks_cat"] > 0, cat["cid1_click_top1_count"] / cat["total_clicks_cat"], 0)
    cat["top1_brand_click_ratio"] = np.where(cat["total_clicks_cat"] > 0, cat["brand_click_top1_count"] / cat["total_clicks_cat"], 0)
    cat["brand_focus_ratio"] = cat["top1_brand_click_ratio"]
    cat["brand_diversity_per_category"] = np.where(cat["cid1_click_nunique"] > 0, cat["brand_click_nunique"] / cat["cid1_click_nunique"], 0)
    cat["cid1_per_brand"] = np.where(cat["brand_click_nunique"] > 0, cat["cid1_click_nunique"] / cat["brand_click_nunique"], 0)
    cat["order_cid1_concentration"] = np.where(cat["cid1_click_nunique"] > 0, cat["cid1_order_nunique"] / cat["cid1_click_nunique"], 0)
    cat["cid1_conversion_rate"] = np.where(cat["total_clicks_cat"] > 0, cat["total_orders_cat"] / cat["total_clicks_cat"], 0)

    click_bf = click_obs[["user_id", "item_id", "datetime"]].rename(columns={"datetime": "click_time"})
    order_fp = order_obs[["user_id", "item_id", "datetime"]].rename(columns={"datetime": "order_time"})
    path_merged = click_bf.merge(order_fp, on=["user_id", "item_id"], how="inner")
    path_merged["click_to_order_hours"] = (path_merged["order_time"] - path_merged["click_time"]).dt.total_seconds() / 3600
    path_merged = path_merged[path_merged["click_to_order_hours"] >= 0]
    path = path_merged.groupby("user_id").agg(
        avg_click_to_order_hours=("click_to_order_hours", "mean"),
        median_click_to_order_hours=("click_to_order_hours", "median"),
        min_click_to_order_hours=("click_to_order_hours", "min"),
        max_click_to_order_hours=("click_to_order_hours", "max"),
        converted_item_count=("item_id", "nunique"),
    ).reset_index()
    user_clicked = click_obs.groupby("user_id")["item_id"].apply(set).reset_index(name="clicked_set")
    user_ordered = order_obs.groupby("user_id")["item_id"].apply(set).reset_index(name="ordered_set")
    item_flow = user_clicked.merge(user_ordered, on="user_id", how="left")
    item_flow["ordered_set"] = item_flow["ordered_set"].apply(lambda x: x if isinstance(x, set) else set())
    item_flow["abandoned_items_count"] = item_flow.apply(lambda r: len(r["clicked_set"] - r["ordered_set"]), axis=1)
    item_flow["clicked_items_count"] = item_flow["clicked_set"].apply(len)
    item_flow["ordered_items_count"] = item_flow["ordered_set"].apply(len)
    item_flow["abandon_rate"] = np.where(item_flow["clicked_items_count"] > 0, item_flow["abandoned_items_count"] / item_flow["clicked_items_count"], 0)
    path_final = pd.DataFrame({"user_id": user_ids}).merge(path, on="user_id", how="left").merge(
        item_flow[["user_id", "abandoned_items_count", "clicked_items_count", "ordered_items_count", "abandon_rate"]],
        on="user_id", how="left",
    )
    path_fill = {
        "avg_click_to_order_hours": -1, "median_click_to_order_hours": -1,
        "min_click_to_order_hours": -1, "max_click_to_order_hours": -1,
        "converted_item_count": 0, "abandoned_items_count": 0,
        "clicked_items_count": 0, "ordered_items_count": 0, "abandon_rate": 1,
    }
    path_final = path_final.fillna(path_fill)
    click_depth = click_obs.groupby(["user_id", click_obs["datetime"].dt.date])["item_id"].nunique().reset_index()
    click_depth = click_depth.groupby("user_id").agg(
        avg_items_per_session=("item_id", "mean"),
        max_items_per_session=("item_id", "max"),
        session_count=("datetime", "count"),
    ).reset_index()
    path_final = path_final.merge(click_depth, on="user_id", how="left")
    for c in ["avg_items_per_session", "max_items_per_session", "session_count"]:
        path_final[c] = path_final[c].fillna(0)

    global_median_price = item["price"].median()
    click_price = click_obs[["user_id", "item_id"]].merge(item[["item_id", "price"]], on="item_id", how="left")
    user_price_click = click_price.groupby("user_id")["price"].agg(
        click_price_p10=lambda x: x.quantile(0.10),
        click_price_p25=lambda x: x.quantile(0.25),
        click_price_p50=lambda x: x.quantile(0.50),
        click_price_p75=lambda x: x.quantile(0.75),
        click_price_p90=lambda x: x.quantile(0.90),
        click_price_mean="mean",
        click_price_std="std",
    ).reset_index()
    user_price_order = order_cat.groupby("user_id")["price"].agg(
        order_price_p50=lambda x: x.quantile(0.50),
        order_price_mean="mean",
        order_price_max="max",
        order_price_min="min",
    ).reset_index()
    price = pd.DataFrame({"user_id": user_ids}).merge(user_price_click, on="user_id", how="left").merge(user_price_order, on="user_id", how="left")
    price["click_price_iqr"] = price["click_price_p75"] - price["click_price_p25"]
    price["click_price_cv"] = np.where(price["click_price_mean"] > 0, price["click_price_std"] / price["click_price_mean"], 0)
    price["click_price_vs_global"] = price["click_price_p50"] / global_median_price
    price["order_price_vs_global"] = np.where(price["order_price_p50"] > 0, price["order_price_p50"] / global_median_price, 0)
    price["order_vs_click_price_ratio"] = np.where(price["click_price_mean"] > 0, price["order_price_mean"] / price["click_price_mean"], 0)
    price = price.replace([np.inf, -np.inf], np.nan).fillna(0)

    features = base
    for block in [order_windows, click_windows, trend, cat, path_final, price]:
        features = features.merge(block, on="user_id", how="left")
    meta = pd.DataFrame({
        "pred_start": [pred_start] * len(features),
        "pred_end": [bounds["pred_end"]] * len(features),
    }, index=features.index)
    return pd.concat([features.copy(), meta], axis=1)


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if c not in ID_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]


@dataclass
class FeaturePreprocessor:
    random_state: int = 42
    corr_threshold: float = 0.90
    variance_threshold: float = 0.01
    top_n: int = 50
    feature_cols: list[str] = field(default_factory=list)
    fill_values: dict[str, float] = field(default_factory=dict)
    clip_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    scaler: StandardScaler | None = None
    variance_selector: VarianceThreshold | None = None
    variance_feature_names: list[str] = field(default_factory=list)
    corr_keep_indices: list[int] = field(default_factory=list)
    filtered_feature_names: list[str] = field(default_factory=list)
    selected_feature_names: list[str] = field(default_factory=list)
    selected_indices: list[int] = field(default_factory=list)

    def _prepare_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        x = df.reindex(columns=self.feature_cols).copy()
        x = x.replace([np.inf, -np.inf], np.nan)
        for col in self.feature_cols:
            x[col] = x[col].fillna(self.fill_values.get(col, 0.0))
            if col in self.clip_bounds:
                lo, hi = self.clip_bounds[col]
                x[col] = x[col].clip(lo, hi)
        return x

    def fit_transform(self, df: pd.DataFrame, y: np.ndarray, feature_cols: list[str]) -> np.ndarray:
        self.feature_cols = list(feature_cols)
        x = df[self.feature_cols].replace([np.inf, -np.inf], np.nan).copy()
        for col in self.feature_cols:
            med = x[col].median()
            self.fill_values[col] = 0.0 if pd.isna(med) else float(med)
            x[col] = x[col].fillna(self.fill_values[col])
            lo, hi = x[col].quantile([0.01, 0.99])
            if pd.notna(lo) and pd.notna(hi) and hi > lo:
                self.clip_bounds[col] = (float(lo), float(hi))
                x[col] = x[col].clip(lo, hi)

        self.scaler = StandardScaler()
        x_scaled = np.nan_to_num(self.scaler.fit_transform(x.to_numpy(dtype=np.float64)), 0.0)
        self.variance_selector = VarianceThreshold(threshold=self.variance_threshold)
        x_var = self.variance_selector.fit_transform(x_scaled)
        self.variance_feature_names = [self.feature_cols[i] for i in self.variance_selector.get_support(indices=True)]

        sample_size = min(20000, x_var.shape[0])
        sample_idx = np.random.RandomState(self.random_state).choice(x_var.shape[0], sample_size, replace=False)
        corr = np.corrcoef(x_var[sample_idx].T)
        drop = set()
        for i in range(len(self.variance_feature_names)):
            for j in range(i + 1, len(self.variance_feature_names)):
                if abs(corr[i, j]) > self.corr_threshold:
                    drop.add(j)
        self.corr_keep_indices = [i for i in range(len(self.variance_feature_names)) if i not in drop]
        x_filtered = x_var[:, self.corr_keep_indices]
        self.filtered_feature_names = [self.variance_feature_names[i] for i in self.corr_keep_indices]

        mi_scores = mutual_info_classif(x_filtered, y, random_state=self.random_state)
        mi_order = np.argsort(mi_scores)[::-1][: min(self.top_n, len(mi_scores))]
        lgb_model = lgb.LGBMClassifier(n_estimators=150, class_weight="balanced", random_state=self.random_state, verbose=-1)
        lgb_model.fit(x_filtered, y)
        lgb_order = np.argsort(lgb_model.feature_importances_)[::-1][: min(self.top_n, x_filtered.shape[1])]
        selected_set = set(mi_order.tolist()) | set(lgb_order.tolist())
        self.selected_indices = [i for i in range(len(self.filtered_feature_names)) if i in selected_set]
        self.selected_feature_names = [self.filtered_feature_names[i] for i in self.selected_indices]
        return x_filtered[:, self.selected_indices]

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.scaler is None or self.variance_selector is None:
            raise RuntimeError("FeaturePreprocessor is not fitted")
        x = self._prepare_frame(df)
        x_scaled = np.nan_to_num(self.scaler.transform(x.to_numpy(dtype=np.float64)), 0.0)
        x_var = self.variance_selector.transform(x_scaled)
        x_filtered = x_var[:, self.corr_keep_indices]
        return x_filtered[:, self.selected_indices]
