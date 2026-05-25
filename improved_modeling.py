#!/usr/bin/env python3
"""Temporal purchase propensity training pipeline.

This version fixes the previous serious methodological issues:
- train/validation/test are separated by non-overlapping prediction windows;
- preprocessing, clipping, variance filtering and feature selection are fitted on
  training windows only;
- thresholds and model choice are tuned on validation only;
- the final test window is evaluated once;
- the full fitted preprocessing object is persisted for dashboard inference.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-codex"))

import catboost as cb
import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.isotonic import IsotonicRegression

from purchase_pipeline import (
    FeaturePreprocessor,
    build_features_for_window,
    build_labels,
    day_end,
    numeric_feature_columns,
    temporal_snapshot_ends,
    window_bounds,
)


SEED = 42
np.random.seed(SEED)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train purchase propensity models with temporal validation.")
    parser.add_argument("--data-dir", default="data/parquet", help="Directory containing parquet data files.")
    parser.add_argument("--model-dir", default="data/models", help="Directory for model artifacts.")
    parser.add_argument("--train-windows", type=int, default=4, help="Number of weekly windows before validation used for training.")
    parser.add_argument("--contact-cost", type=float, default=1.0, help="Assumed cost for contacting one predicted-positive user.")
    parser.add_argument("--conversion-value", type=float, default=10.0, help="Assumed gross value for one true conversion.")
    parser.add_argument("--skip-shap", action="store_true", help="Skip SHAP plots for faster runs.")
    return parser.parse_args()


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    user_features = pd.read_parquet(data_dir / "user_features_raw_with_cluster.parquet")
    order = pd.read_parquet(data_dir / "user_order_cleaned.parquet")
    click = pd.read_parquet(data_dir / "user_click.parquet")
    item = pd.read_parquet(
        data_dir / "item_data_cleaned.parquet",
        columns=["item_id", "cid1", "cid2", "cid3", "brand_code", "price"],
    )

    for df in [user_features, order, click]:
        df["user_id"] = df["user_id"].astype(str)
    for df in [order, click, item]:
        df["item_id"] = df["item_id"].astype(str)
    order["datetime"] = pd.to_datetime(order["datetime"])
    click["datetime"] = pd.to_datetime(click["datetime"], format="mixed")
    return user_features, order, click, item


def make_snapshot(
    name: str,
    pred_end: pd.Timestamp,
    user_features: pd.DataFrame,
    order: pd.DataFrame,
    click: pd.DataFrame,
    item: pd.DataFrame,
    user_ids: list[str],
) -> pd.DataFrame:
    bounds = window_bounds(pred_end)
    print(
        f"  {name}: obs {bounds['obs_start'].date()} ~ {bounds['obs_end'].date()}, "
        f"pred {bounds['pred_start'].date()} ~ {bounds['pred_end'].date()}"
    )
    features = build_features_for_window(user_features, order, click, item, pred_end, user_ids=user_ids)
    labels = build_labels(order, user_ids, bounds["pred_start"], bounds["pred_end"])
    snapshot = features.merge(labels, on="user_id", how="left")
    snapshot = snapshot.assign(snapshot=name)
    print(f"    rows={len(snapshot):,}, positive_rate={snapshot['label'].mean():.4f}")
    return snapshot


def evaluate_at_threshold(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "auc": float(roc_auc_score(y_true, proba)),
        "ap": float(average_precision_score(y_true, proba)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, pred, beta=2, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
    }


def best_f2_threshold(y_true: np.ndarray, proba: np.ndarray) -> tuple[float, dict]:
    thresholds = np.arange(0.05, 0.95, 0.025)
    scores = [fbeta_score(y_true, (proba >= t).astype(int), beta=2, zero_division=0) for t in thresholds]
    best_t = float(thresholds[int(np.argmax(scores))])
    return best_t, evaluate_at_threshold(y_true, proba, best_t)


def fit_probability_calibrator(y_true: np.ndarray, proba: np.ndarray) -> IsotonicRegression:
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(proba, y_true)
    return calibrator


def topk_report(
    y_true: np.ndarray,
    proba: np.ndarray,
    contact_cost: float = 1.0,
    conversion_value: float = 10.0,
    rates: tuple[float, ...] = (0.01, 0.05, 0.10, 0.20, 0.30),
) -> list[dict]:
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    order = np.argsort(proba)[::-1]
    base_rate = float(y_true.mean())
    positives = max(int(y_true.sum()), 1)
    rows = []
    for rate in rates:
        k = max(1, int(len(y_true) * rate))
        idx = order[:k]
        tp = int(y_true[idx].sum())
        fp = int(k - tp)
        precision = tp / k
        recall = tp / positives
        rows.append({
            "top_rate": rate,
            "users": k,
            "tp": tp,
            "fp": fp,
            "precision": precision,
            "recall": recall,
            "lift": precision / base_rate if base_rate else 0.0,
            "expected_profit": tp * conversion_value - k * contact_cost,
        })
    return rows


def threshold_profit_report(
    y_true: np.ndarray,
    proba: np.ndarray,
    contact_cost: float = 1.0,
    conversion_value: float = 10.0,
) -> dict:
    best = None
    for threshold in np.arange(0.05, 0.95, 0.025):
        pred = proba >= threshold
        selected = int(pred.sum())
        tp = int(y_true[pred].sum())
        fp = selected - tp
        profit = tp * conversion_value - selected * contact_cost
        row = {
            "threshold": float(threshold),
            "selected": selected,
            "tp": tp,
            "fp": fp,
            "precision": float(tp / selected) if selected else 0.0,
            "recall": float(tp / max(int(y_true.sum()), 1)),
            "expected_profit": float(profit),
        }
        if best is None or row["expected_profit"] > best["expected_profit"]:
            best = row
    return best or {}


def train_default_models(x_train: np.ndarray, y_train: np.ndarray) -> dict:
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    return {
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=250,
            learning_rate=0.04,
            num_leaves=31,
            max_depth=6,
            min_child_samples=30,
            class_weight="balanced",
            random_state=SEED,
            verbose=-1,
        ).fit(x_train, y_train),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=250,
            learning_rate=0.04,
            max_depth=4,
            min_child_weight=3,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            random_state=SEED,
            eval_metric="logloss",
            verbosity=0,
        ).fit(x_train, y_train),
        "CatBoost": cb.CatBoostClassifier(
            iterations=250,
            learning_rate=0.04,
            depth=6,
            l2_leaf_reg=3.0,
            auto_class_weights="Balanced",
            random_seed=SEED,
            verbose=False,
        ).fit(x_train, y_train),
    }


def rolling_backtest(train_parts: list[pd.DataFrame], val_df: pd.DataFrame, feature_cols: list[str]) -> list[dict]:
    """Backtest XGBoost over successive future windows for stability diagnostics."""
    windows = train_parts + [val_df]
    rows = []
    for val_idx in range(2, len(windows)):
        fold_train = pd.concat(windows[:val_idx], ignore_index=True)
        fold_val = windows[val_idx]
        fold_y_train = fold_train["label"].to_numpy(dtype=int)
        fold_y_val = fold_val["label"].to_numpy(dtype=int)
        fold_preprocessor = FeaturePreprocessor(random_state=SEED)
        fold_x_train = fold_preprocessor.fit_transform(fold_train, fold_y_train, feature_cols)
        fold_x_val = fold_preprocessor.transform(fold_val)
        model = train_default_models(fold_x_train, fold_y_train)["XGBoost"]
        proba = model.predict_proba(fold_x_val)[:, 1]
        rows.append({
            "fold": len(rows) + 1,
            "train_windows": val_idx,
            "validation_window_end": str(day_end(fold_val["pred_end"].iloc[0]).date()),
            "positive_rate": float(fold_y_val.mean()),
            "auc": float(roc_auc_score(fold_y_val, proba)),
            "ap": float(average_precision_score(fold_y_val, proba)),
        })
    return rows


def plot_model_eval(y_test: np.ndarray, model_results: dict, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    for idx, (name, res) in enumerate(model_results.items()):
        sns.heatmap(
            np.array(res["test"]["confusion_matrix"]),
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=axes[0, idx],
            xticklabels=["未下单", "下单"],
            yticklabels=["未下单", "下单"],
        )
        axes[0, idx].set_title(f"{name} 混淆矩阵\n验证阈值={res['threshold']:.3f}")
        axes[0, idx].set_xlabel("预测")
        axes[0, idx].set_ylabel("真实")

    ax_roc = axes[1, 0]
    ax_pr = axes[1, 1]
    for name, res in model_results.items():
        fpr, tpr, _ = roc_curve(y_test, res["test_proba"])
        precision, recall, _ = precision_recall_curve(y_test, res["test_proba"])
        ax_roc.plot(fpr, tpr, label=f"{name} AUC={res['test']['auc']:.3f}")
        ax_pr.plot(recall, precision, label=f"{name} AP={res['test']['ap']:.3f}")
    ax_roc.plot([0, 1], [0, 1], "k--")
    ax_roc.set_title("Test ROC")
    ax_roc.legend()
    ax_pr.set_title("Test PR")
    ax_pr.legend()

    names = list(model_results)
    axes[1, 2].bar(np.arange(len(names)) - 0.2, [model_results[n]["test"]["auc"] for n in names], width=0.4, label="AUC")
    axes[1, 2].bar(np.arange(len(names)) + 0.2, [model_results[n]["test"]["precision"] for n in names], width=0.4, label="Precision")
    axes[1, 2].set_xticks(np.arange(len(names)))
    axes[1, 2].set_xticklabels(names)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].set_title("Test 指标")
    axes[1, 2].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams["font.sans-serif"] = ["STHeiti", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    print("=" * 70)
    print("0. 数据加载")
    print("=" * 70)
    user_features, order, click, item = load_data(data_dir)
    user_ids = sorted(user_features["user_id"].unique().tolist())
    print(f"users={len(user_ids):,}, orders={len(order):,}, clicks={len(click):,}, items={len(item):,}")

    print("\n" + "=" * 70)
    print("1. 严格时间窗口划分")
    print("=" * 70)
    split = temporal_snapshot_ends(order["datetime"].max(), n_train_windows=args.train_windows)
    train_parts = [
        make_snapshot(f"train_{i + 1}", pred_end, user_features, order, click, item, user_ids)
        for i, pred_end in enumerate(split["train"])
    ]
    val_df = make_snapshot("validation", split["val"], user_features, order, click, item, user_ids)
    test_df = make_snapshot("test", split["test"], user_features, order, click, item, user_ids)
    train_df = pd.concat(train_parts, ignore_index=True)

    print("\n" + "=" * 70)
    print("2. 训练集拟合预处理与特征选择")
    print("=" * 70)
    feature_cols = numeric_feature_columns(train_df)
    y_train = train_df["label"].to_numpy(dtype=int)
    y_val = val_df["label"].to_numpy(dtype=int)
    y_test = test_df["label"].to_numpy(dtype=int)

    preprocessor = FeaturePreprocessor(random_state=SEED)
    x_train = preprocessor.fit_transform(train_df, y_train, feature_cols)
    x_val = preprocessor.transform(val_df)
    x_test = preprocessor.transform(test_df)
    print(f"raw_features={len(feature_cols)}, selected_features={len(preprocessor.selected_feature_names)}")
    print(f"train={x_train.shape}, val={x_val.shape}, test={x_test.shape}")

    print("\nTop selected features:")
    for feature in preprocessor.selected_feature_names[:30]:
        print(f"  {feature}")

    print("\n" + "=" * 70)
    print("2.5 滚动窗口稳定性回测（XGBoost, AUC/AP）")
    print("=" * 70)
    rolling_metrics = rolling_backtest(train_parts, val_df, feature_cols)
    for row in rolling_metrics:
        print(
            f"  fold={row['fold']}, train_windows={row['train_windows']}, "
            f"val_end={row['validation_window_end']}, pos={row['positive_rate']:.4f}, "
            f"AUC={row['auc']:.4f}, AP={row['ap']:.4f}"
        )

    print("\n" + "=" * 70)
    print("3. 模型训练与验证集阈值选择")
    print("=" * 70)
    models = train_default_models(x_train, y_train)
    model_results = {}
    for name, model in models.items():
        val_proba_raw = model.predict_proba(x_val)[:, 1]
        calibrator = fit_probability_calibrator(y_val, val_proba_raw)
        val_proba = calibrator.transform(val_proba_raw)
        threshold, val_metrics = best_f2_threshold(y_val, val_proba)
        test_proba_raw = model.predict_proba(x_test)[:, 1]
        test_proba = calibrator.transform(test_proba_raw)
        test_metrics = evaluate_at_threshold(y_test, test_proba, threshold)
        model_results[name] = {
            "threshold": threshold,
            "calibrator": calibrator,
            "validation": val_metrics,
            "test": test_metrics,
            "test_proba": test_proba,
            "test_proba_raw": test_proba_raw,
        }
        print(f"\n{name}")
        print(f"  validation AUC={val_metrics['auc']:.4f}, AP={val_metrics['ap']:.4f}, threshold={threshold:.3f}, F2={val_metrics['f2']:.4f}")
        print(f"  test       AUC={test_metrics['auc']:.4f}, AP={test_metrics['ap']:.4f}, Precision={test_metrics['precision']:.4f}, Recall={test_metrics['recall']:.4f}, F2={test_metrics['f2']:.4f}")

    best_model_name = max(model_results, key=lambda n: (model_results[n]["validation"]["ap"], model_results[n]["validation"]["auc"]))
    best_model = models[best_model_name]
    best_calibrator = model_results[best_model_name]["calibrator"]
    best_threshold = model_results[best_model_name]["threshold"]
    best_test_proba = model_results[best_model_name]["test_proba"]
    best_test_pred = (best_test_proba >= best_threshold).astype(int)
    print(f"\n最佳模型（按验证集 AP/AUC）: {best_model_name}, threshold={best_threshold:.3f}")
    print(classification_report(y_test, best_test_pred, target_names=["未下单", "下单"], zero_division=0))

    print("\n业务 Top-K / Lift / 收益评估（测试窗口）:")
    business_topk = topk_report(
        y_test,
        best_test_proba,
        contact_cost=args.contact_cost,
        conversion_value=args.conversion_value,
    )
    for row in business_topk:
        print(
            f"  Top {row['top_rate']:.0%}: users={row['users']:,}, "
            f"precision={row['precision']:.4f}, recall={row['recall']:.4f}, "
            f"lift={row['lift']:.2f}, profit={row['expected_profit']:.1f}"
        )
    validation_profit_threshold = threshold_profit_report(
        y_val,
        best_calibrator.transform(best_model.predict_proba(x_val)[:, 1]),
        contact_cost=args.contact_cost,
        conversion_value=args.conversion_value,
    )
    test_profit_at_validation_profit_threshold = evaluate_at_threshold(
        y_test,
        best_test_proba,
        validation_profit_threshold["threshold"],
    )
    print(
        f"  验证集利润最优阈值={validation_profit_threshold['threshold']:.3f}, "
        f"测试集 precision={test_profit_at_validation_profit_threshold['precision']:.4f}, "
        f"recall={test_profit_at_validation_profit_threshold['recall']:.4f}"
    )

    print("\n" + "=" * 70)
    print("4. 测试集图表与解释")
    print("=" * 70)
    plot_model_eval(y_test, model_results, Path("model_evaluation_plots.png"))
    print("保存 model_evaluation_plots.png")

    shap_top = []
    if not args.skip_shap:
        sample_size = min(2000, x_test.shape[0])
        sample_idx = np.random.RandomState(SEED).choice(x_test.shape[0], sample_size, replace=False)
        x_shap = x_test[sample_idx]
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(x_shap)
        if isinstance(shap_values, list):
            shap_values = shap_values[-1]
        shap.summary_plot(shap_values, x_shap, feature_names=preprocessor.selected_feature_names, max_display=15, show=False)
        plt.tight_layout()
        plt.savefig("shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
        shap.summary_plot(shap_values, x_shap, feature_names=preprocessor.selected_feature_names, plot_type="bar", max_display=15, show=False)
        plt.tight_layout()
        plt.savefig("shap_importance.png", dpi=150, bbox_inches="tight")
        plt.close()
        shap_importance = np.abs(shap_values).mean(axis=0)
        shap_mean = shap_values.mean(axis=0)
        shap_top = [
            {
                "feature": f,
                "importance": float(abs_mean),
                "direction": "促进购买" if mean_value >= 0 else "抑制购买",
                "description": "基于测试集抽样 SHAP 均值方向，重要性为 mean(|SHAP|)",
            }
            for f, abs_mean, mean_value in sorted(
                zip(preprocessor.selected_feature_names, shap_importance, shap_mean),
                key=lambda x: x[1],
                reverse=True,
            )[:15]
        ]
        print("保存 shap_summary.png / shap_importance.png")

    print("\n" + "=" * 70)
    print("5. 业务分层（仅基于最终测试窗口）")
    print("=" * 70)
    prediction_df = test_df[["user_id", "label"]].copy()
    prediction_df["pred_proba"] = best_test_proba
    prediction_df["pred_label"] = best_test_pred
    prediction_df["prob_bin"] = pd.cut(
        prediction_df["pred_proba"],
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=["很低(0-0.2)", "低(0.2-0.4)", "中(0.4-0.6)", "高(0.6-0.8)", "很高(0.8-1.0)"],
        include_lowest=True,
    )
    for bin_name, data in prediction_df.groupby("prob_bin", observed=False):
        if len(data):
            print(f"  {bin_name}: {len(data):,} 用户, 实际下单率={data['label'].mean():.4f}")

    print("\n" + "=" * 70)
    print("6. 保存模型工件")
    print("=" * 70)
    metadata = {
        "best_model_name": best_model_name,
        "threshold": best_threshold,
        "selected_feature_names": preprocessor.selected_feature_names,
        "train_windows": [str(day_end(x).date()) for x in split["train"]],
        "validation_window_end": str(day_end(split["val"]).date()),
        "test_window_end": str(day_end(split["test"]).date()),
        "validation_metrics": model_results[best_model_name]["validation"],
        "test_metrics": model_results[best_model_name]["test"],
        "rolling_backtest": rolling_metrics,
        "business_assumptions": {
            "contact_cost": args.contact_cost,
            "conversion_value": args.conversion_value,
        },
        "business_topk_test": business_topk,
        "validation_profit_threshold": validation_profit_threshold,
        "test_metrics_at_validation_profit_threshold": test_profit_at_validation_profit_threshold,
        "shap_top_features": shap_top,
    }
    joblib.dump(best_model, model_dir / "best_purchase_model.pkl")
    joblib.dump(best_calibrator, model_dir / "calibrator.pkl")
    joblib.dump(preprocessor, model_dir / "preprocessor.pkl")
    joblib.dump(preprocessor.selected_feature_names, model_dir / "feature_names.pkl")
    joblib.dump(preprocessor.scaler, model_dir / "scaler.pkl")
    with open(model_dir / "model_metadata.json", "w") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"模型工件已保存到 {model_dir}")


if __name__ == "__main__":
    main()
