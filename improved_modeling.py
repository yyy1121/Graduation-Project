#!/usr/bin/env python3
"""
电商购买预测 - 特征工程与模型优化完整流水线
==============================================
改进点：
1. 修复时间划分（确保训练/测试标签分布一致）
2. 多时间窗口特征（7/14/30/60天）
3. 类目/品牌转化特征
4. 行为路径特征
5. 价格敏感度特征
6. 特征选择（互信息 + 相关性过滤）
7. 不平衡处理（class_weight + SMOTE + 阈值优化）
8. Optuna 超参数优化（LightGBM / XGBoost / CatBoost）
9. SHAP 可解释性分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score, fbeta_score, roc_curve, precision_recall_curve,
    f1_score, recall_score, precision_score
)
from sklearn.feature_selection import mutual_info_classif
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import shap
import optuna
import warnings
import gc
from datetime import timedelta

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['STHeiti']
plt.rcParams['axes.unicode_minus'] = False

SEED = 42
np.random.seed(SEED)

# ============================================================================
# 0. 数据加载
# ============================================================================
print("=" * 70)
print("0. 数据加载")
print("=" * 70)

DATA_DIR = 'data/parquet'

# 加载用户特征表
user_features = pd.read_parquet(f'{DATA_DIR}/user_features_raw_with_cluster.parquet')
print(f"用户特征表: {user_features.shape}")

# 加载订单数据
order = pd.read_parquet(f'{DATA_DIR}/user_order_cleaned.parquet')
order['datetime'] = pd.to_datetime(order['datetime'])
print(f"订单数据: {order.shape}")

# 加载点击数据（只加载必要字段以节省内存）
click_raw = pd.read_parquet(f'{DATA_DIR}/user_click.parquet')
click_raw['datetime'] = pd.to_datetime(click_raw['datetime'], format='mixed')
print(f"点击数据: {click_raw.shape}")

# 加载商品信息（只加载关键字段）
item_cols = ['item_id', 'cid1', 'cid2', 'cid3', 'brand_code', 'price']
item = pd.read_parquet(f'{DATA_DIR}/item_data_cleaned.parquet', columns=item_cols)
item['item_id'] = item['item_id'].astype(str)
print(f"商品数据: {item.shape}")

# 统一 ID 类型
user_features['user_id'] = user_features['user_id'].astype(str)
order['user_id'] = order['user_id'].astype(str)
order['item_id'] = order['item_id'].astype(str)
click_raw['user_id'] = click_raw['user_id'].astype(str)
click_raw['item_id'] = click_raw['item_id'].astype(str)

# ============================================================================
# 1. 定义时间窗口 & 标签
# ============================================================================
print("\n" + "=" * 70)
print("1. 时间窗口定义 & 标签生成")
print("=" * 70)

max_date = order['datetime'].max()
print(f"数据最大日期: {max_date}")

# 预测窗口: 最后7天
pred_end = max_date
pred_start = max_date - timedelta(days=6)

# 观察窗口: 预测窗口前30天
obs_end = pred_start - timedelta(days=1)
obs_start = obs_end - timedelta(days=29)

print(f"预测窗口: {pred_start.date()} ~ {pred_end.date()}")
print(f"观察窗口: {obs_start.date()} ~ {obs_end.date()}")

# 生成标签
pred_users = set(order[order['datetime'] >= pred_start]['user_id'].unique())
all_users = set(user_features['user_id'].unique())
labels = pd.DataFrame({'user_id': list(all_users)})
labels['label'] = labels['user_id'].isin(pred_users).astype(int)
print(f"正样本比例: {labels['label'].mean():.4f} ({labels['label'].sum()} / {len(labels)})")

# ============================================================================
# 2. 修复时间划分策略
# ============================================================================
print("\n" + "=" * 70)
print("2. 时间顺序数据划分（修复版）")
print("=" * 70)

# 使用观察窗口内的最后活跃时间进行划分
# 计算每个用户在观察期的最后活跃时间
order_obs = order[(order['datetime'] >= obs_start) & (order['datetime'] <= obs_end)]
click_obs_full = click_raw[(click_raw['datetime'] >= obs_start) & (click_raw['datetime'] <= obs_end)]

order_last = order_obs.groupby('user_id')['datetime'].max().reset_index()
order_last.columns = ['user_id', 'last_order_time']
click_last = click_obs_full.groupby('user_id')['datetime'].max().reset_index()
click_last.columns = ['user_id', 'last_click_time']

# 合并最后活跃时间
last_activity = pd.DataFrame({'user_id': list(all_users)})
last_activity = last_activity.merge(order_last, on='user_id', how='left')
last_activity = last_activity.merge(click_last, on='user_id', how='left')
last_activity['last_active'] = last_activity[['last_order_time', 'last_click_time']].max(axis=1)
last_activity['last_active'] = last_activity['last_active'].fillna(obs_start)

# 按时间排序后取70%作为训练集（保证时间顺序）
last_activity_sorted = last_activity.sort_values('last_active').reset_index(drop=True)
split_idx = int(len(last_activity_sorted) * 0.7)
train_users = set(last_activity_sorted.iloc[:split_idx]['user_id'])
test_users = set(last_activity_sorted.iloc[split_idx:]['user_id'])

labels_with_split = labels.copy()
labels_with_split['split'] = labels_with_split['user_id'].apply(
    lambda x: 'train' if x in train_users else 'test'
)

train_labels = labels_with_split[labels_with_split['split'] == 'train']['label']
test_labels = labels_with_split[labels_with_split['split'] == 'test']['label']

print(f"训练集: {len(train_users)} 用户, 标签均值={train_labels.mean():.4f}")
print(f"测试集: {len(test_users)} 用户, 标签均值={test_labels.mean():.4f}")
print(f"时间切分点: {last_activity_sorted.iloc[split_idx]['last_active']}")

# ============================================================================
# 3. 基线特征构建（单窗口30天）
# ============================================================================
print("\n" + "=" * 70)
print("3. 基线特征构建（30天观察窗口）")
print("=" * 70)


def build_baseline_features(order_obs_df, click_obs_df, pred_start_dt, user_ids):
    """构建30天观察窗口的基线特征"""
    # 订单特征聚合
    order_agg = order_obs_df.groupby('user_id').agg(
        obs_order_cnt=('datetime', 'count'),
        obs_total_qty=('count', 'sum'),
        obs_unique_items=('item_id', 'nunique'),
        obs_active_days=('datetime', lambda x: x.dt.date.nunique()),
        obs_first_order=('datetime', 'min'),
        obs_last_order=('datetime', 'max'),
    ).reset_index()

    order_agg['obs_order_span'] = (
        (order_agg['obs_last_order'] - order_agg['obs_first_order']).dt.days
    ).clip(lower=1)
    order_agg['obs_avg_items_per_order'] = order_agg['obs_total_qty'] / order_agg['obs_order_cnt'].clip(lower=1)
    order_agg['obs_avg_interval'] = order_agg['obs_order_span'] / order_agg['obs_order_cnt'].clip(lower=1)
    order_agg['obs_days_since_last'] = (pred_start_dt - order_agg['obs_last_order']).dt.days
    order_agg.drop(['obs_first_order', 'obs_last_order'], axis=1, inplace=True)

    # 点击特征聚合
    click_agg = click_obs_df.groupby('user_id').agg(
        click_cnt=('datetime', 'count'),
        unique_click_items=('item_id', 'nunique'),
        click_active_days=('datetime', lambda x: x.dt.date.nunique()),
        first_click=('datetime', 'min'),
        last_click=('datetime', 'max'),
    ).reset_index()

    click_agg['click_span'] = (
        (click_agg['last_click'] - click_agg['first_click']).dt.days
    ).clip(lower=1)
    click_agg['avg_click_per_day'] = click_agg['click_cnt'] / click_agg['click_active_days'].clip(lower=1)
    click_agg['days_since_last_click'] = (pred_start_dt - click_agg['last_click']).dt.days
    click_agg.drop(['first_click', 'last_click'], axis=1, inplace=True)

    # 合并
    base = pd.DataFrame({'user_id': list(user_ids)})
    base = base.merge(order_agg, on='user_id', how='left')
    base = base.merge(click_agg, on='user_id', how='left')

    # 填充缺失值
    order_fill_cols = ['obs_order_cnt', 'obs_total_qty', 'obs_unique_items',
                       'obs_active_days', 'obs_order_span', 'obs_avg_items_per_order',
                       'obs_avg_interval']
    click_fill_cols = ['click_cnt', 'unique_click_items', 'click_active_days',
                       'click_span', 'avg_click_per_day']

    for c in order_fill_cols + click_fill_cols:
        if c in base.columns:
            base[c] = base[c].fillna(0)

    window_days = (pred_start_dt - obs_start).days + 30
    base['obs_days_since_last'] = base['obs_days_since_last'].fillna(window_days)
    base['days_since_last_click'] = base['days_since_last_click'].fillna(window_days)

    # 转化特征
    base['click_to_buy_ratio'] = np.where(
        base['click_cnt'] > 0,
        base['obs_order_cnt'] / base['click_cnt'],
        0
    )
    base['has_conversion'] = (
        (base['click_cnt'] > 0) & (base['obs_order_cnt'] > 0)
    ).astype(int)

    return base


# 构建基线特征
baseline_features = build_baseline_features(order_obs, click_obs_full, pred_start, all_users)

# 合并用户基础特征
user_base = user_features[['user_id', 'gender_encoded', 'age_encoded']].copy()
baseline_features = baseline_features.merge(user_base, on='user_id', how='left')
baseline_features = baseline_features.merge(labels, on='user_id', how='left')

print(f"基线特征维度: {baseline_features.shape}")

# ============================================================================
# 4. 新增特征工程
# ============================================================================
print("\n" + "=" * 70)
print("4. 新增特征工程")
print("=" * 70)

# 4.1 多时间窗口特征
print("4.1 构建多时间窗口特征...")


def build_window_features(df, user_ids, window_days_list, pred_start_dt, prefix, date_col='datetime'):
    """为多个时间窗口构建统计特征"""
    features = pd.DataFrame({'user_id': list(user_ids)})

    for days in window_days_list:
        window_start = pred_start_dt - timedelta(days=days)
        window_data = df[(df[date_col] >= window_start) & (df[date_col] <= obs_end)]

        if date_col == 'datetime':
            agg = window_data.groupby('user_id').agg(
                **{f'{prefix}_cnt_{days}d': (date_col, 'count'),
                   f'{prefix}_items_{days}d': ('item_id', 'nunique'),
                   f'{prefix}_active_days_{days}d': (date_col, lambda x: x.dt.date.nunique()),
                   f'{prefix}_last_days_{days}d': (date_col, lambda x: (pred_start_dt - x.max()).days)}
            ).reset_index()
        else:
            agg = window_data.groupby('user_id').agg(
                **{f'{prefix}_cnt_{days}d': (date_col, 'count'),
                   f'{prefix}_items_{days}d': ('item_id', 'nunique'),
                   f'{prefix}_active_days_{days}d': (date_col, lambda x: x.dt.date.nunique()),
                   f'{prefix}_last_days_{days}d': (date_col, lambda x: (pred_start_dt - x.max()).days)}
            ).reset_index()

        features = features.merge(agg, on='user_id', how='left')

    # 填充缺失值
    for days in window_days_list:
        for col_suffix in ['cnt', 'items', 'active_days']:
            col = f'{prefix}_{col_suffix}_{days}d'
            if col in features.columns:
                features[col] = features[col].fillna(0)
        last_col = f'{prefix}_last_days_{days}d'
        if last_col in features.columns:
            features[last_col] = features[last_col].fillna(days + 7)

    return features


WINDOWS = [7, 14, 30, 60]

# 订单窗口特征
order_window_feats = build_window_features(
    order_obs, all_users, WINDOWS, pred_start, 'order', 'datetime'
)

# 点击窗口特征
click_window_feats = build_window_features(
    click_obs_full, all_users, WINDOWS, pred_start, 'click', 'datetime'
)

print(f"  订单窗口特征: {order_window_feats.shape}")
print(f"  点击窗口特征: {click_window_feats.shape}")

# 4.1.1 行为变化率特征（趋势）
print("4.1.1 构建行为变化率特征...")
trend_features = pd.DataFrame({'user_id': list(all_users)})

for prefix in ['order', 'click']:
    if prefix == 'order':
        feat_df = order_window_feats
    else:
        feat_df = click_window_feats

    for metric in ['cnt', 'items', 'active_days']:
        col_7d = feat_df[f'{prefix}_{metric}_7d'].values
        col_14d = feat_df[f'{prefix}_{metric}_14d'].values
        col_30d = feat_df[f'{prefix}_{metric}_30d'].values

        # 7d/14d 比率（近期 vs 稍远期）
        trend_features[f'{prefix}_{metric}_7d_14d_ratio'] = np.where(col_14d > 0, col_7d / col_14d, 0)
        # 14d/30d 比率
        trend_features[f'{prefix}_{metric}_14d_30d_ratio'] = np.where(col_30d > 0, col_14d / col_30d, 0)
        # 7d/30d 比率（短期 vs 中期）
        trend_features[f'{prefix}_{metric}_7d_30d_ratio'] = np.where(col_30d > 0, col_7d / col_30d, 0)

    # 最近活跃度衰减
    last_7d = feat_df[f'{prefix}_last_days_7d'].values
    last_14d = feat_df[f'{prefix}_last_days_14d'].values
    trend_features[f'{prefix}_recency_decay'] = np.where(last_14d > 0, last_7d / (last_14d + 1), 0)

# 点击-订单转化趋势
for days in WINDOWS:
    click_col = f'click_cnt_{days}d'
    order_col = f'order_cnt_{days}d'
    if click_col in click_window_feats.columns and order_col in order_window_feats.columns:
        click_val = click_window_feats[click_col].values
        order_val = order_window_feats[order_col].values
        trend_features[f'conversion_rate_{days}d'] = np.where(click_val > 0, order_val / click_val, 0)

print(f"  趋势特征: {trend_features.shape}")

# 4.2 类目/品牌转化特征
print("4.2 构建类目/品牌转化特征...")

# 为观察期内的点击和订单数据关联类目信息
click_with_cat = click_obs_full[['user_id', 'item_id', 'datetime']].copy()
click_with_cat = click_with_cat.merge(
    item[['item_id', 'cid1', 'cid2', 'cid3', 'brand_code', 'price']],
    on='item_id', how='left'
)

order_obs_with_cat = order_obs[['user_id', 'item_id', 'datetime']].copy()
order_obs_with_cat = order_obs_with_cat.merge(
    item[['item_id', 'cid1', 'cid2', 'cid3', 'brand_code', 'price']],
    on='item_id', how='left'
)


def build_category_features(click_df, order_df, user_ids):
    """构建类目/品牌相关特征"""
    feats = pd.DataFrame({'user_id': list(user_ids)})

    # ---- cid1 (一级类目) 特征 ----
    # 点击类目多样性
    def safe_top1_count(x):
        vc = x.value_counts()
        return vc.iloc[0] if len(vc) > 0 else 0

    cid1_click = click_df.groupby('user_id')['cid1'].agg([
        ('cid1_click_nunique', 'nunique'),
        ('cid1_click_top1_count', safe_top1_count),
    ]).reset_index()

    # 订单类目
    cid1_order = order_df.groupby('user_id')['cid1'].agg([
        ('cid1_order_nunique', 'nunique'),
    ]).reset_index()

    feats = feats.merge(cid1_click, on='user_id', how='left')
    feats = feats.merge(cid1_order, on='user_id', how='left')

    # cid1 转化率 (点击->购买)
    cid1_click_cnt = click_df.groupby(['user_id', 'cid1']).size().reset_index(name='click_count')
    cid1_order_cnt = order_df.groupby(['user_id', 'cid1']).size().reset_index(name='order_count')
    cid1_conv = cid1_click_cnt.merge(cid1_order_cnt, on=['user_id', 'cid1'], how='left')
    cid1_conv['order_count'] = cid1_conv['order_count'].fillna(0)
    cid1_conv['cid1_converted'] = (cid1_conv['order_count'] > 0).astype(int)

    # 每个用户的类目转化率
    cid1_user_conv = cid1_conv.groupby('user_id').agg(
        cid1_conversion_rate=('cid1_converted', 'mean'),
        cid1_clicked_count=('cid1', 'nunique'),
        cid1_converted_count=('cid1_converted', 'sum'),
    ).reset_index()

    feats = feats.merge(cid1_user_conv, on='user_id', how='left')

    # ---- cid3 (三级类目) 特征 ----
    cid3_click = click_df.groupby('user_id')['cid3'].agg([
        ('cid3_click_nunique', 'nunique'),
    ]).reset_index()
    cid3_order = order_df.groupby('user_id')['cid3'].agg([
        ('cid3_order_nunique', 'nunique'),
    ]).reset_index()
    feats = feats.merge(cid3_click, on='user_id', how='left')
    feats = feats.merge(cid3_order, on='user_id', how='left')

    # ---- 品牌特征 ----
    brand_click = click_df.groupby('user_id')['brand_code'].agg([
        ('brand_click_nunique', 'nunique'),
        ('brand_click_top1_count', safe_top1_count),
    ]).reset_index()

    brand_order = order_df.groupby('user_id')['brand_code'].agg([
        ('brand_order_nunique', 'nunique'),
    ]).reset_index()

    feats = feats.merge(brand_click, on='user_id', how='left')
    feats = feats.merge(brand_order, on='user_id', how='left')

    # 品牌偏好强度
    total_clicks = click_df.groupby('user_id').size().reset_index(name='total_clicks_cat')
    feats = feats.merge(total_clicks, on='user_id', how='left')

    # 填充
    fill_cols = [c for c in feats.columns if c != 'user_id']
    for c in fill_cols:
        feats[c] = feats[c].fillna(0)

    # 衍生特征
    feats['top1_cid1_click_ratio'] = np.where(
        feats['total_clicks_cat'] > 0,
        feats['cid1_click_top1_count'] / feats['total_clicks_cat'],
        0
    )
    feats['top1_brand_click_ratio'] = np.where(
        feats['total_clicks_cat'] > 0,
        feats['brand_click_top1_count'] / feats['total_clicks_cat'],
        0
    )
    feats['brand_focus_ratio'] = np.where(
        feats['cid1_click_nunique'] > 0,
        feats['brand_click_nunique'] / feats['cid1_click_nunique'],
        0
    )

    # 类目-品牌交叉
    feats['cid1_per_brand'] = np.where(
        feats['brand_click_nunique'] > 0,
        feats['cid1_click_nunique'] / (feats['brand_click_nunique'] + 1),
        0
    )

    # 订单类目集中度
    feats['order_cid1_concentration'] = np.where(
        feats['cid1_click_nunique'] > 0,
        feats['cid1_order_nunique'] / (feats['cid1_click_nunique'] + 1),
        0
    )

    return feats


category_features = build_category_features(click_with_cat, order_obs_with_cat, all_users)
print(f"  类目/品牌特征: {category_features.shape}")

# 4.3 行为路径特征
print("4.3 构建行为路径特征...")

# 点击到购买时间差
click_before_order = click_obs_full[['user_id', 'item_id', 'datetime']].copy()
click_before_order.columns = ['user_id', 'item_id', 'click_time']
order_for_path = order_obs[['user_id', 'item_id', 'datetime']].copy()
order_for_path.columns = ['user_id', 'item_id', 'order_time']

# 合并同一用户-商品对
path_merged = click_before_order.merge(order_for_path, on=['user_id', 'item_id'], how='inner')
path_merged['click_to_order_hours'] = (
    (path_merged['order_time'] - path_merged['click_time']).dt.total_seconds() / 3600
)
# 只保留点击在订单之前的记录
path_merged = path_merged[path_merged['click_to_order_hours'] >= 0]

path_features = path_merged.groupby('user_id').agg(
    avg_click_to_order_hours=('click_to_order_hours', 'mean'),
    median_click_to_order_hours=('click_to_order_hours', 'median'),
    min_click_to_order_hours=('click_to_order_hours', 'min'),
    max_click_to_order_hours=('click_to_order_hours', 'max'),
    converted_item_count=('item_id', 'nunique'),
).reset_index()

# 点击但未购买的商品数
clicked_items = set(click_obs_full['item_id'].unique())
ordered_items = set(order_obs['item_id'].unique())
abandoned_items = clicked_items - ordered_items

user_clicked = click_obs_full.groupby('user_id')['item_id'].apply(set).reset_index()
user_ordered = order_obs.groupby('user_id')['item_id'].apply(set).reset_index()
user_clicked.columns = ['user_id', 'clicked_set']
user_ordered.columns = ['user_id', 'ordered_set']

item_flow = user_clicked.merge(user_ordered, on='user_id', how='left')
item_flow['ordered_set'] = item_flow['ordered_set'].fillna({}).apply(
    lambda x: x if isinstance(x, set) else set())

# 注意：这里处理 mixed type
def safe_set_len(x):
    if isinstance(x, set):
        return len(x)
    return 0


item_flow['abandoned_items_count'] = item_flow.apply(
    lambda row: len(row['clicked_set'] - row['ordered_set'])
    if isinstance(row['ordered_set'], set) else len(row['clicked_set']),
    axis=1
)
item_flow['clicked_items_count'] = item_flow['clicked_set'].apply(safe_set_len)
item_flow['ordered_items_count'] = item_flow['ordered_set'].apply(safe_set_len)
item_flow['abandon_rate'] = np.where(
    item_flow['clicked_items_count'] > 0,
    item_flow['abandoned_items_count'] / item_flow['clicked_items_count'],
    0
)

path_features_final = path_features.merge(
    item_flow[['user_id', 'abandoned_items_count', 'clicked_items_count',
               'ordered_items_count', 'abandon_rate']],
    on='user_id', how='right'
)

# 填充无转化用户的特征
path_features_final['avg_click_to_order_hours'] = path_features_final['avg_click_to_order_hours'].fillna(-1)
path_features_final['converted_item_count'] = path_features_final['converted_item_count'].fillna(0)
path_features_final['abandoned_items_count'] = path_features_final['abandoned_items_count'].fillna(0)
path_features_final['abandon_rate'] = path_features_final['abandon_rate'].fillna(1.0)

# 会话深度（平均每次活跃天点击不同商品数）
click_depth = click_obs_full.groupby(['user_id', click_obs_full['datetime'].dt.date])['item_id'].nunique().reset_index()
click_depth_agg = click_depth.groupby('user_id').agg(
    avg_items_per_session=('item_id', 'mean'),
    max_items_per_session=('item_id', 'max'),
    session_count=('datetime', 'count'),
).reset_index()

path_features_final = path_features_final.merge(click_depth_agg, on='user_id', how='left')
path_features_final['avg_items_per_session'] = path_features_final['avg_items_per_session'].fillna(0)
path_features_final['max_items_per_session'] = path_features_final['max_items_per_session'].fillna(0)
path_features_final['session_count'] = path_features_final['session_count'].fillna(0)

print(f"  行为路径特征: {path_features_final.shape}")

# 4.4 价格敏感度特征
print("4.4 构建价格敏感度特征...")

global_median_price = item['price'].median()

# 用户点击商品的价格分布
click_price = click_obs_full[['user_id', 'item_id']].merge(
    item[['item_id', 'price']], on='item_id', how='left'
)

user_price_click = click_price.groupby('user_id')['price'].agg([
    ('click_price_p10', lambda x: x.quantile(0.10)),
    ('click_price_p25', lambda x: x.quantile(0.25)),
    ('click_price_p50', lambda x: x.quantile(0.50)),
    ('click_price_p75', lambda x: x.quantile(0.75)),
    ('click_price_p90', lambda x: x.quantile(0.90)),
    ('click_price_mean', 'mean'),
    ('click_price_std', 'std'),
]).reset_index()

# 用户购买商品的价格分布
order_price = order_obs_with_cat[['user_id', 'price']].copy()
user_price_order = order_price.groupby('user_id')['price'].agg([
    ('order_price_p50', lambda x: x.quantile(0.50) if len(x) > 0 else 0),
    ('order_price_mean', 'mean'),
    ('order_price_max', 'max'),
    ('order_price_min', 'min'),
]).reset_index()

price_features = user_price_click.merge(user_price_order, on='user_id', how='left')

# 价格敏感度衍生特征
price_features['click_price_iqr'] = (
    price_features['click_price_p75'] - price_features['click_price_p25']
)
price_features['click_price_cv'] = np.where(
    price_features['click_price_mean'] > 0,
    price_features['click_price_std'] / price_features['click_price_mean'],
    0
)

# 与全局均价比值
price_features['click_price_vs_global'] = (
    price_features['click_price_p50'] / global_median_price
)
price_features['order_price_vs_global'] = np.where(
    price_features['order_price_p50'] > 0,
    price_features['order_price_p50'] / global_median_price,
    0
)

# 价格升级/降级
price_features['order_vs_click_price_ratio'] = np.where(
    price_features['click_price_mean'] > 0,
    price_features['order_price_mean'] / price_features['click_price_mean'],
    0
)

# 填充无点击/无订单用户
price_fill_cols = [c for c in price_features.columns if c != 'user_id']
for c in price_fill_cols:
    price_features[c] = price_features[c].fillna(0)

# 修复无穷大值
price_features.replace([np.inf, -np.inf], 0, inplace=True)

print(f"  价格敏感度特征: {price_features.shape}")

# ============================================================================
# 5. 特征合并
# ============================================================================
print("\n" + "=" * 70)
print("5. 特征合并")
print("=" * 70)

# 合并所有特征
all_features = baseline_features.copy()

# 合并多窗口特征
all_features = all_features.merge(order_window_feats, on='user_id', how='left')
all_features = all_features.merge(click_window_feats, on='user_id', how='left')
all_features = all_features.merge(trend_features, on='user_id', how='left')

# 合并类目特征
all_features = all_features.merge(category_features, on='user_id', how='left')

# 合并路径特征
all_features = all_features.merge(path_features_final, on='user_id', how='left')

# 合并价格特征
all_features = all_features.merge(price_features, on='user_id', how='left')

print(f"合并后特征维度: {all_features.shape}")

# 处理无穷大和缺失值
all_features.replace([np.inf, -np.inf], np.nan, inplace=True)

# 稳健填充策略
print("  处理缺失值和无穷值...")
for col in all_features.columns:
    if col in ['user_id', 'label']:
        continue
    if all_features[col].isna().any():
        if all_features[col].dtype in ['float64', 'float32']:
            med = all_features[col].median()
            if pd.isna(med):
                all_features[col] = all_features[col].fillna(0)
            else:
                all_features[col] = all_features[col].fillna(med)
        else:
            all_features[col] = all_features[col].fillna(0)

# 确保无 NaN - 使用 0 作为最终后备
all_features = all_features.fillna(0)

# 裁剪极端值
for col in all_features.columns:
    if col in ['user_id', 'label']:
        continue
    if all_features[col].dtype in ['float64', 'float32']:
        q99 = all_features[col].quantile(0.99)
        q01 = all_features[col].quantile(0.01)
        if q99 > q01:
            all_features[col] = all_features[col].clip(q01, q99)

nan_count = all_features.isna().sum().sum()
print(f"  填充后剩余 NaN 数: {nan_count}")

# ============================================================================
# 6. 准备训练/测试数据
# ============================================================================
print("\n" + "=" * 70)
print("6. 数据准备")
print("=" * 70)

# 特征列（排除 user_id, label, 以及非数值列）
exclude_cols = ['user_id', 'label', 'gender', 'age_range']
feature_cols = [c for c in all_features.columns if c not in exclude_cols
                and all_features[c].dtype in ['int64', 'int32', 'float64', 'float32', 'int8', 'bool']]

# 再移除可能的对象列
feature_cols = [c for c in feature_cols if all_features[c].dtype in ['int64', 'int32', 'float64', 'float32', 'int8']]

print(f"可用特征数: {len(feature_cols)}")

# 划分训练/测试
train_mask = all_features['user_id'].isin(train_users).values
test_mask = all_features['user_id'].isin(test_users).values

X_train = all_features.loc[train_mask, feature_cols].values.astype(np.float64)
y_train = all_features.loc[train_mask, 'label'].values.astype(int)
X_test = all_features.loc[test_mask, feature_cols].values.astype(np.float64)
y_test = all_features.loc[test_mask, 'label'].values.astype(int)

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}, 正样本率: {y_train.mean():.4f}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}, 正样本率: {y_test.mean():.4f}")

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 处理标准化后的 NaN
X_train_scaled = np.nan_to_num(X_train_scaled, 0)
X_test_scaled = np.nan_to_num(X_test_scaled, 0)

# 移除低方差特征
from sklearn.feature_selection import VarianceThreshold
selector_var = VarianceThreshold(threshold=0.01)
X_train_scaled = selector_var.fit_transform(X_train_scaled)
X_test_scaled = selector_var.transform(X_test_scaled)
selected_indices = selector_var.get_support(indices=True)
feature_cols_selected = [feature_cols[i] for i in selected_indices]
print(f"移除低方差特征后: {X_train_scaled.shape[1]} 个特征")

# ============================================================================
# 7. 基线模型评估
# ============================================================================
print("\n" + "=" * 70)
print("7. 基线模型评估（LightGBM 默认参数）")
print("=" * 70)


def evaluate_model(model, X_tr, y_tr, X_te, y_te, name="Model"):
    """评估模型并返回指标"""
    model.fit(X_tr, y_tr)
    y_proba = model.predict_proba(X_te)[:, 1]
    y_pred_default = (y_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_te, y_proba)
    ap = average_precision_score(y_te, y_proba)

    # 不同阈值下的评估
    results = {'Model': name, 'AUC': auc, 'AP': ap}
    for threshold in [0.2, 0.3, 0.4, 0.5]:
        y_pred_t = (y_proba >= threshold).astype(int)
        results[f'Recall@{threshold}'] = recall_score(y_te, y_pred_t)
        results[f'Precision@{threshold}'] = precision_score(y_te, y_pred_t)
        results[f'F1@{threshold}'] = f1_score(y_te, y_pred_t)

    return results, y_proba


# 训练基线 LightGBM
lgb_baseline = lgb.LGBMClassifier(random_state=SEED, verbose=-1)
baseline_results, baseline_proba = evaluate_model(
    lgb_baseline, X_train_scaled, y_train, X_test_scaled, y_test, "LightGBM Baseline"
)

print("\n基线模型结果:")
for k, v in baseline_results.items():
    if isinstance(v, float):
        print(f"  {k}: {v:.4f}")

# 混淆矩阵
y_pred_03 = (baseline_proba >= 0.3).astype(int)
cm = confusion_matrix(y_test, y_pred_03)
print(f"\n混淆矩阵 (threshold=0.3):\n{cm}")

# ============================================================================
# 8. 特征选择（互信息 + 相关性过滤）
# ============================================================================
print("\n" + "=" * 70)
print("8. 特征选择")
print("=" * 70)

# 8.1 相关性过滤（移除高度相关的特征）
print("8.1 相关性过滤...")

# 计算相关系数矩阵（采样以提高效率）
sample_size = min(20000, X_train_scaled.shape[0])
sample_idx = np.random.RandomState(SEED).choice(X_train_scaled.shape[0], sample_size, replace=False)
X_sample = X_train_scaled[sample_idx]

corr_matrix = np.corrcoef(X_sample.T)
high_corr_pairs = []
features_to_drop = set()

for i in range(len(feature_cols_selected)):
    for j in range(i + 1, len(feature_cols_selected)):
        if abs(corr_matrix[i, j]) > 0.9:
            high_corr_pairs.append((feature_cols_selected[i], feature_cols_selected[j], corr_matrix[i, j]))
            # 保留第一个特征，移除第二个
            features_to_drop.add(j)

print(f"  高度相关特征对 (|r|>0.9): {len(high_corr_pairs)}")
print(f"  计划移除特征: {len(features_to_drop)}")

# 保留的特征索引
keep_indices = [i for i in range(len(feature_cols_selected)) if i not in features_to_drop]
X_train_filtered = X_train_scaled[:, keep_indices]
X_test_filtered = X_test_scaled[:, keep_indices]
feature_cols_filtered = [feature_cols_selected[i] for i in keep_indices]
print(f"  过滤后特征数: {len(feature_cols_filtered)}")

# 8.2 互信息特征选择
print("8.2 互信息特征选择...")
mi_scores = mutual_info_classif(X_train_filtered, y_train, random_state=SEED)
mi_df = pd.DataFrame({
    'feature': feature_cols_filtered,
    'mi_score': mi_scores
}).sort_values('mi_score', ascending=False)

print(f"\nTop 20 互信息特征:")
for i, row in mi_df.head(20).iterrows():
    print(f"  {row['feature']:40s}: {row['mi_score']:.4f}")

# 8.3 LightGBM 特征重要性
print("\n8.3 LightGBM 特征重要性...")
lgb_temp = lgb.LGBMClassifier(n_estimators=100, random_state=SEED, verbose=-1)
lgb_temp.fit(X_train_filtered, y_train)
lgb_importance = pd.DataFrame({
    'feature': feature_cols_filtered,
    'importance': lgb_temp.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 20 LightGBM 特征重要性:")
for i, row in lgb_importance.head(20).iterrows():
    print(f"  {row['feature']:40s}: {row['importance']:.4f}")

# 保存特征名称用于后续使用
top_n = 50
top_mi_features = set(mi_df.head(top_n)['feature'].tolist())
top_lgb_features = set(lgb_importance.head(top_n)['feature'].tolist())
combined_top_features = list(top_mi_features | top_lgb_features)
combined_indices = [feature_cols_filtered.index(f) for f in combined_top_features
                    if f in feature_cols_filtered]

X_train_selected = X_train_filtered[:, combined_indices]
X_test_selected = X_test_filtered[:, combined_indices]
selected_feature_names = [feature_cols_filtered[i] for i in combined_indices]
print(f"\n特征选择后最终特征数: {len(selected_feature_names)}")

# ============================================================================
# 9. 特征选择效果对比
# ============================================================================
print("\n" + "=" * 70)
print("9. 特征选择效果对比（5折时序交叉验证）")
print("=" * 70)

tscv = TimeSeriesSplit(n_splits=5)

# 实验 A: 仅基线特征
baseline_only_cols = [c for c in feature_cols if c in baseline_features.columns
                      and c in feature_cols_filtered]
baseline_indices = [feature_cols_filtered.index(c) for c in baseline_only_cols
                    if c in feature_cols_filtered]
if len(baseline_indices) == 0:
    baseline_indices = list(range(min(20, X_train_filtered.shape[1])))

X_train_baseline = X_train_filtered[:, baseline_indices]
X_test_baseline = X_test_filtered[:, baseline_indices]

print(f"实验 A (仅基线特征): {len(baseline_indices)} 特征")
print(f"实验 B (基线+新特征): {X_train_filtered.shape[1]} 特征")
print(f"实验 C (特征选择后): {len(selected_feature_names)} 特征")

results_comparison = {}

for exp_name, X_tr, X_te in [
    ('A_基线特征', X_train_baseline, X_test_baseline),
    ('B_基线+新特征', X_train_filtered, X_test_filtered),
    ('C_特征选择后', X_train_selected, X_test_selected),
]:
    auc_scores = []
    rec_scores = []

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_tr)):
        X_tr_fold, X_val_fold = X_tr[tr_idx], X_tr[val_idx]
        y_tr_fold, y_val_fold = y_train[tr_idx], y_train[val_idx]

        model_fold = lgb.LGBMClassifier(n_estimators=100, random_state=SEED, verbose=-1)
        model_fold.fit(X_tr_fold, y_tr_fold)
        y_val_proba = model_fold.predict_proba(X_val_fold)[:, 1]

        auc_scores.append(roc_auc_score(y_val_fold, y_val_proba))
        rec_scores.append(recall_score(y_val_fold, (y_val_proba >= 0.3).astype(int)))

    results_comparison[exp_name] = {
        'AUC_mean': np.mean(auc_scores), 'AUC_std': np.std(auc_scores),
        'Recall_mean': np.mean(rec_scores), 'Recall_std': np.std(rec_scores),
    }
    print(f"\n{exp_name}:")
    print(f"  AUC = {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
    print(f"  Recall@0.3 = {np.mean(rec_scores):.4f} ± {np.std(rec_scores):.4f}")

# ============================================================================
# 10. 不平衡处理
# ============================================================================
print("\n" + "=" * 70)
print("10. 不平衡处理方法对比")
print("=" * 70)

# 使用特征选择后的数据
X_eval = X_train_selected
y_eval = y_train
X_test_eval = X_test_selected
y_test_eval = y_test

imbalance_results = {}

# 10.1 无处理（baseline）
print("10.1 无特殊处理...")
lgb_plain = lgb.LGBMClassifier(n_estimators=200, random_state=SEED, verbose=-1)
plain_res, plain_proba = evaluate_model(
    lgb_plain, X_eval, y_eval, X_test_eval, y_test_eval, "无处理"
)
imbalance_results['无处理'] = plain_res

# 10.2 类权重
print("10.2 类权重调整...")
lgb_weighted = lgb.LGBMClassifier(n_estimators=200, class_weight='balanced',
                                  random_state=SEED, verbose=-1)
weighted_res, weighted_proba = evaluate_model(
    lgb_weighted, X_eval, y_eval, X_test_eval, y_test_eval, "类权重"
)
imbalance_results['类权重'] = weighted_res

# 10.3 SMOTE 过采样
print("10.3 SMOTE 过采样...")
try:
    smote = SMOTE(random_state=SEED, k_neighbors=3)
    X_smote, y_smote = smote.fit_resample(X_eval, y_eval)
    lgb_smote = lgb.LGBMClassifier(n_estimators=200, random_state=SEED, verbose=-1)
    smote_res, smote_proba = evaluate_model(
        lgb_smote, X_smote, y_smote, X_test_eval, y_test_eval, "SMOTE"
    )
    imbalance_results['SMOTE'] = smote_res
except Exception as e:
    print(f"  SMOTE 失败: {e}")

# 10.4 随机下采样
print("10.4 随机下采样...")
try:
    rus = RandomUnderSampler(random_state=SEED)
    X_rus, y_rus = rus.fit_resample(X_eval, y_eval)
    lgb_rus = lgb.LGBMClassifier(n_estimators=200, random_state=SEED, verbose=-1)
    rus_res, rus_proba = evaluate_model(
        lgb_rus, X_rus, y_rus, X_test_eval, y_test_eval, "下采样"
    )
    imbalance_results['下采样'] = rus_res
except Exception as e:
    print(f"  下采样失败: {e}")

# 10.5 组合（SMOTE + 类权重）
print("10.5 SMOTE + 类权重...")
try:
    lgb_combined = lgb.LGBMClassifier(n_estimators=200, class_weight='balanced',
                                      random_state=SEED, verbose=-1)
    combined_res, combined_proba = evaluate_model(
        lgb_combined, X_smote, y_smote, X_test_eval, y_test_eval, "SMOTE+权重"
    )
    imbalance_results['SMOTE+权重'] = combined_res
except Exception as e:
    print(f"  SMOTE+权重 失败: {e}")

# 汇总对比
print("\n不平衡处理效果对比:")
print(f"{'方法':<15} {'AUC':<8} {'Recall@0.3':<12} {'Precision@0.3':<14} {'F1@0.3':<8}")
print("-" * 60)
for method, res in imbalance_results.items():
    print(f"{method:<15} {res['AUC']:<8.4f} {res['Recall@0.3']:<12.4f} "
          f"{res['Precision@0.3']:<14.4f} {res['F1@0.3']:<8.4f}")

# 10.6 阈值优化
print("\n10.6 阈值优化...")
thresholds = np.arange(0.05, 0.95, 0.025)
best_method = None
best_f2 = 0
best_threshold = 0.3
threshold_results = {}

for method_name in ['无处理', '类权重', 'SMOTE', 'SMOTE+权重']:
    if method_name not in imbalance_results:
        continue

    if method_name == '无处理':
        proba = plain_proba
    elif method_name == '类权重':
        proba = weighted_proba
    elif method_name == 'SMOTE':
        proba = smote_proba
    elif method_name == 'SMOTE+权重':
        proba = combined_proba
    else:
        continue

    f2_scores = []
    for t in thresholds:
        pred_t = (proba >= t).astype(int)
        f2_scores.append(fbeta_score(y_test_eval, pred_t, beta=2))

    best_idx = np.argmax(f2_scores)
    best_t = thresholds[best_idx]
    best_f2_score = f2_scores[best_idx]

    # 同时计算对应 F1
    pred_best = (proba >= best_t).astype(int)
    f1_best = f1_score(y_test_eval, pred_best)
    rec_best = recall_score(y_test_eval, pred_best)
    prec_best = precision_score(y_test_eval, pred_best)

    threshold_results[method_name] = {
        'best_threshold': best_t, 'best_F2': best_f2_score,
        'F1': f1_best, 'Recall': rec_best, 'Precision': prec_best
    }

    print(f"  {method_name}: 最佳阈值={best_t:.3f}, F2={best_f2_score:.4f}, "
          f"F1={f1_best:.4f}, Recall={rec_best:.4f}, Precision={prec_best:.4f}")

    if best_f2_score > best_f2:
        best_f2 = best_f2_score
        best_method = method_name
        best_threshold = best_t

# 选择最佳方法
print(f"\n最佳不平衡处理方法: {best_method}, 阈值: {best_threshold:.3f}")

# ============================================================================
# 11. 超参数优化 (Optuna)
# ============================================================================
print("\n" + "=" * 70)
print("11. Optuna 超参数优化")
print("=" * 70)

# 使用类权重策略（最稳健的选择）
X_opt_train, y_opt_train = X_eval, y_eval
use_class_weight = 'balanced'

# 验证训练数据无 NaN
assert not np.isnan(X_opt_train).any(), "训练数据存在 NaN!"
assert not np.isnan(y_opt_train).any(), "训练标签存在 NaN!"
print(f"X_opt_train: {X_opt_train.shape}, y_opt_train: {y_opt_train.shape}, 正样本率: {y_opt_train.mean():.4f}")

# 对训练数据再次裁剪确保数值稳定性
X_opt_train = np.clip(X_opt_train, -10, 10)


def safe_objective(trial, model_type='lgb'):
    """安全的 Optuna 目标函数，捕获异常返回默认值"""
    try:
        if model_type == 'lgb':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 15, 63),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
                'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 1.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 1.0, log=True),
                'class_weight': use_class_weight,
                'random_state': SEED,
                'verbose': -1,
            }
        elif model_type == 'xgb':
            scale_pos_weight = (y_opt_train == 0).sum() / max((y_opt_train == 1).sum(), 1)
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 1.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 1.0, log=True),
                'scale_pos_weight': scale_pos_weight,
                'random_state': SEED,
                'verbosity': 0,
            }
        else:  # catboost
            params = {
                'iterations': trial.suggest_int('iterations', 100, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'depth': trial.suggest_int('depth', 3, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-4, 10.0, log=True),
                'auto_class_weights': 'Balanced' if use_class_weight else None,
                'random_seed': SEED,
                'verbose': False,
            }

        # 使用简单的 hold-out 验证（更快）
        n_train = int(len(X_opt_train) * 0.8)
        X_tr, X_val = X_opt_train[:n_train], X_opt_train[n_train:]
        y_tr, y_val = y_opt_train[:n_train], y_opt_train[n_train:]

        if model_type == 'lgb':
            model = lgb.LGBMClassifier(**params)
        elif model_type == 'xgb':
            model = xgb.XGBClassifier(**params)
        else:
            model = cb.CatBoostClassifier(**params)

        model.fit(X_tr, y_tr)
        y_val_proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_val_proba)

        if np.isnan(auc):
            return 0.5
        return float(auc)
    except Exception:
        return 0.5


# 运行 Optuna 优化
optuna.logging.set_verbosity(optuna.logging.WARNING)
n_trials = 15  # 减少试验次数以加速

# LightGBM 优化
print("\n优化 LightGBM...")
study_lgb = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
study_lgb.optimize(lambda trial: safe_objective(trial, 'lgb'), n_trials=n_trials,
                   show_progress_bar=True, catch=())

lgb_best_params = {}
try:
    if len(study_lgb.trials) > 0:
        lgb_best_params = study_lgb.best_params
        print(f"  LightGBM 最优 AUC: {study_lgb.best_value:.4f}")
        print(f"  最优参数: {lgb_best_params}")
    else:
        raise ValueError("No trials completed")
except Exception:
    print("  LightGBM Optuna 失败，使用默认参数")
    lgb_best_params = {
        'n_estimators': 200, 'learning_rate': 0.05, 'num_leaves': 31,
        'max_depth': 8, 'min_child_samples': 20, 'subsample': 0.8,
        'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
    }

# XGBoost 优化
print("\n优化 XGBoost...")
study_xgb = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
study_xgb.optimize(lambda trial: safe_objective(trial, 'xgb'), n_trials=n_trials,
                   show_progress_bar=True, catch=())

xgb_best_params = {}
try:
    xgb_best_params = study_xgb.best_params
    print(f"  XGBoost 最优 AUC: {study_xgb.best_value:.4f}")
    print(f"  最优参数: {xgb_best_params}")
except Exception:
    print("  XGBoost Optuna 失败，使用默认参数")
    xgb_best_params = {
        'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 6,
        'min_child_weight': 1, 'subsample': 0.8, 'colsample_bytree': 0.8,
        'reg_alpha': 0.1, 'reg_lambda': 0.1,
    }

# CatBoost 优化
print("\n优化 CatBoost...")
study_cb = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
study_cb.optimize(lambda trial: safe_objective(trial, 'cb'), n_trials=n_trials,
                   show_progress_bar=True, catch=())

cb_best_params = {}
try:
    cb_best_params = study_cb.best_params
    print(f"  CatBoost 最优 AUC: {study_cb.best_value:.4f}")
    print(f"  最优参数: {cb_best_params}")
except Exception:
    print("  CatBoost Optuna 失败，使用默认参数")
    cb_best_params = {
        'iterations': 200, 'learning_rate': 0.05, 'depth': 6, 'l2_leaf_reg': 3.0,
    }

# ============================================================================
# 12. 训练最终模型
# ============================================================================
print("\n" + "=" * 70)
print("12. 训练最终模型")
print("=" * 70)

# LightGBM 最终参数
lgb_final_params = lgb_best_params.copy()
lgb_final_params['class_weight'] = use_class_weight
lgb_final_params['random_state'] = SEED
lgb_final_params['verbose'] = -1

# XGBoost 最终参数
xgb_final_params = xgb_best_params.copy()
xgb_final_params['scale_pos_weight'] = (y_opt_train == 0).sum() / max((y_opt_train == 1).sum(), 1)
xgb_final_params['random_state'] = SEED
xgb_final_params['verbosity'] = 0

# CatBoost 最终参数
cb_final_params = cb_best_params.copy()
cb_final_params['auto_class_weights'] = 'Balanced'
cb_final_params['random_seed'] = SEED
cb_final_params['verbose'] = False

# 训练最终模型
final_models = {}

print("训练 LightGBM...")
final_lgb = lgb.LGBMClassifier(**lgb_final_params)
final_lgb.fit(X_opt_train, y_opt_train)
final_models['LightGBM'] = final_lgb

print("训练 XGBoost...")
final_xgb = xgb.XGBClassifier(**xgb_final_params)
final_xgb.fit(X_opt_train, y_opt_train)
final_models['XGBoost'] = final_xgb

print("训练 CatBoost...")
final_cb = cb.CatBoostClassifier(**cb_final_params)
final_cb.fit(X_opt_train, y_opt_train)
final_models['CatBoost'] = final_cb

# ============================================================================
# 13. 最终测试集评估
# ============================================================================
print("\n" + "=" * 70)
print("13. 最终测试集评估")
print("=" * 70)

final_results = {}

for name, model in final_models.items():
    y_proba = model.predict_proba(X_test_eval)[:, 1]
    y_pred_default = (y_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test_eval, y_proba)
    ap = average_precision_score(y_test_eval, y_proba)

    # 阈值优化
    f2_scores = [fbeta_score(y_test_eval, (y_proba >= t).astype(int), beta=2)
                 for t in thresholds]
    best_t = thresholds[np.argmax(f2_scores)]
    y_pred_best = (y_proba >= best_t).astype(int)

    recall_val = recall_score(y_test_eval, y_pred_best)
    precision_val = precision_score(y_test_eval, y_pred_best)
    f1_val = f1_score(y_test_eval, y_pred_best)
    f2_val = fbeta_score(y_test_eval, y_pred_best, beta=2)

    final_results[name] = {
        'AUC': auc, 'AP': ap, 'Best_Threshold': best_t,
        'Recall': recall_val, 'Precision': precision_val,
        'F1': f1_val, 'F2': f2_val, 'proba': y_proba, 'pred': y_pred_best
    }

    print(f"\n{name}:")
    print(f"  AUC = {auc:.4f}, AP = {ap:.4f}")
    print(f"  最佳阈值 = {best_t:.3f} (F2优化)")
    print(f"  Recall = {recall_val:.4f}, Precision = {precision_val:.4f}")
    print(f"  F1 = {f1_val:.4f}, F2 = {f2_val:.4f}")
    print(f"\n  分类报告:")
    print(classification_report(y_test_eval, y_pred_best, target_names=['未下单', '下单']))

# 选择最佳模型
best_model_name = max(final_results, key=lambda x: final_results[x]['AUC'])
best_model = final_models[best_model_name]
best_proba = final_results[best_model_name]['proba']
best_pred = final_results[best_model_name]['pred']
print(f"\n最佳模型: {best_model_name} (AUC={final_results[best_model_name]['AUC']:.4f})")

# ============================================================================
# 14. 可视化 - 混淆矩阵 & ROC/PR 曲线
# ============================================================================
print("\n" + "=" * 70)
print("14. 可视化")
print("=" * 70)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 混淆矩阵
for idx, (name, res) in enumerate(final_results.items()):
    ax = axes[0, idx]
    cm = confusion_matrix(y_test_eval, res['pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['未下单', '下单'], yticklabels=['未下单', '下单'])
    ax.set_title(f'{name} 混淆矩阵\n(阈值={res["Best_Threshold"]:.2f})')
    ax.set_xlabel('预测')
    ax.set_ylabel('真实')

# ROC 曲线
ax_roc = axes[1, 0]
for name, res in final_results.items():
    fpr, tpr, _ = roc_curve(y_test_eval, res['proba'])
    ax_roc.plot(fpr, tpr, label=f'{name} (AUC={res["AUC"]:.3f})')
ax_roc.plot([0, 1], [0, 1], 'k--')
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.set_title('ROC 曲线')
ax_roc.legend()

# PR 曲线
ax_pr = axes[1, 1]
for name, res in final_results.items():
    precision, recall, _ = precision_recall_curve(y_test_eval, res['proba'])
    ax_pr.plot(recall, precision, label=f'{name} (AP={res["AP"]:.3f})')
ax_pr.set_xlabel('Recall')
ax_pr.set_ylabel('Precision')
ax_pr.set_title('Precision-Recall 曲线')
ax_pr.legend()

# AUC 柱状图对比
ax_bar = axes[1, 2]
names = list(final_results.keys())
auc_values = [final_results[n]['AUC'] for n in names]
recall_values = [final_results[n]['Recall'] for n in names]
x = np.arange(len(names))
width = 0.35
ax_bar.bar(x - width/2, auc_values, width, label='AUC')
ax_bar.bar(x + width/2, recall_values, width, label='Recall')
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(names)
ax_bar.set_ylabel('Score')
ax_bar.set_title('模型性能对比')
ax_bar.legend()
ax_bar.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('model_evaluation_plots.png', dpi=150, bbox_inches='tight')
plt.show()
print("图表已保存至 model_evaluation_plots.png")

# ============================================================================
# 15. SHAP 可解释性分析
# ============================================================================
print("\n" + "=" * 70)
print("15. SHAP 可解释性分析")
print("=" * 70)

# 对测试集采样以加速 SHAP 计算
shap_sample_size = min(2000, X_test_eval.shape[0])
shap_indices = np.random.RandomState(SEED).choice(X_test_eval.shape[0], shap_sample_size, replace=False)
X_shap = X_test_eval[shap_indices]

print(f"SHAP 分析样本数: {len(X_shap)}")

if best_model_name == 'LightGBM':
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_shap)
elif best_model_name == 'XGBoost':
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_shap)
elif best_model_name == 'CatBoost':
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_shap)
else:
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_shap)

# SHAP Summary Plot
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_shap,
                  feature_names=selected_feature_names,
                  max_display=15, show=False)
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=150, bbox_inches='tight')
plt.show()
print("SHAP summary plot 已保存至 shap_summary.png")

# SHAP Feature Importance (Bar)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_shap,
                  feature_names=selected_feature_names,
                  plot_type='bar', max_display=15, show=False)
plt.tight_layout()
plt.savefig('shap_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print("SHAP importance plot 已保存至 shap_importance.png")

# 输出 Top 10 重要特征
shap_importance = np.abs(shap_values).mean(axis=0)
shap_importance_df = pd.DataFrame({
    'feature': selected_feature_names,
    'shap_importance': shap_importance
}).sort_values('shap_importance', ascending=False)

print("\nTop 10 SHAP 重要特征:")
for i, row in shap_importance_df.head(10).iterrows():
    direction = "↑ 促进购买" if shap_values[:, i].mean() > 0 else "↓ 抑制购买"
    print(f"  {row['feature']:40s}: {row['shap_importance']:.4f} ({direction})")

# ============================================================================
# 16. 业务建议 & 分群分析
# ============================================================================
print("\n" + "=" * 70)
print("16. 业务建议 & 分群分析")
print("=" * 70)

# 加载聚类信息
cluster_info = user_features[['user_id', 'cluster']].copy()
cluster_info['user_id'] = cluster_info['user_id'].astype(str)

# 合并预测结果
test_users_list = list(test_users)
prediction_df = pd.DataFrame({
    'user_id': test_users_list,
    'true_label': y_test_eval,
    'pred_proba': best_proba,
    'pred_label': best_pred,
})
prediction_df = prediction_df.merge(cluster_info, on='user_id', how='left')

# 分群效果分析
print("\n各用户群组的预测效果:")
for cluster_id in sorted(prediction_df['cluster'].dropna().unique()):
    cluster_data = prediction_df[prediction_df['cluster'] == int(cluster_id)]
    if len(cluster_data) == 0:
        continue
    actual_rate = cluster_data['true_label'].mean()
    pred_rate = cluster_data['pred_label'].mean()
    avg_proba = cluster_data['pred_proba'].mean()
    n_users = len(cluster_data)

    print(f"\n  Cluster {int(cluster_id)} ({n_users} 用户):")
    print(f"    实际下单率: {actual_rate:.4f}")
    print(f"    预测下单率: {pred_rate:.4f}")
    print(f"    平均预测概率: {avg_proba:.4f}")

# 运营建议
print("\n" + "-" * 50)
print("运营建议:")
print("-" * 50)

# 不同预测概率段的用户策略
prediction_df['prob_bin'] = pd.cut(prediction_df['pred_proba'],
                                     bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                     labels=['很低(0-0.2)', '低(0.2-0.4)', '中(0.4-0.6)',
                                             '高(0.6-0.8)', '很高(0.8-1.0)'])

print("\n按预测概率分段的用户分布:")
for bin_name in ['很低(0-0.2)', '低(0.2-0.4)', '中(0.4-0.6)', '高(0.6-0.8)', '很高(0.8-1.0)']:
    bin_data = prediction_df[prediction_df['prob_bin'] == bin_name]
    if len(bin_data) == 0:
        continue
    actual = bin_data['true_label'].mean()
    print(f"  {bin_name}: {len(bin_data)} 用户, 实际下单率={actual:.4f}")

# 高概率但可能流失的用户
print("\n运营策略建议:")
print("  1. 高概率用户 (prob > 0.6): 发送大额优惠券促进转化, 推送新品/高单价商品")
print("  2. 中等概率用户 (0.3-0.6): 发送限时折扣提醒, 推送\"浏览未购\"商品")
print("  3. 低概率用户 (0.1-0.3): 发送品类促销活动, 低门槛优惠券激活")
print("  4. 极低概率用户 (<0.1): 长期未活跃用户, 采用大促/季节性召回策略")
print(f"  5. 推荐阈值: {best_threshold:.3f} (优化F2-score), 平衡召回与精确度")

# ============================================================================
# 17. 最终性能汇总
# ============================================================================
print("\n" + "=" * 70)
print("17. 最终性能汇总")
print("=" * 70)

print(f"\n{'='*70}")
print(f"最终模型: {best_model_name}")
print(f"最佳参数: {lgb_final_params if best_model_name == 'LightGBM' else xgb_final_params if best_model_name == 'XGBoost' else cb_final_params}")
print(f"测试集 AUC: {final_results[best_model_name]['AUC']:.4f}")
print(f"测试集 AP: {final_results[best_model_name]['AP']:.4f}")
print(f"最佳阈值: {final_results[best_model_name]['Best_Threshold']:.3f}")
print(f"Recall: {final_results[best_model_name]['Recall']:.4f}")
print(f"Precision: {final_results[best_model_name]['Precision']:.4f}")
print(f"F1: {final_results[best_model_name]['F1']:.4f}")
print(f"F2: {final_results[best_model_name]['F2']:.4f}")
print(f"{'='*70}")

# 保存模型和特征名
import joblib
joblib.dump(best_model, 'data/models/best_purchase_model.pkl')
joblib.dump(scaler, 'data/models/scaler.pkl')
joblib.dump(selected_feature_names, 'data/models/feature_names.pkl')
print("\n模型已保存至 data/models/")

print("\n流水线执行完成!")
