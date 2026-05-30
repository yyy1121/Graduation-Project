#!/usr/bin/env python3
"""
电商用户画像与购买预测系统 - Streamlit 交互式可视化仪表盘
==========================================================
启动: streamlit run dashboard.py --server.port 8501

模块:
  1. 用户画像总览
  2. 购买倾向分析
  3. 行为路径追踪
  4. 用户分群与标签管理
  5. 热销商品分析
"""

import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime

# ── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="电商用户画像与购买预测系统",
    page_icon="◇",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* ── Typography ── */
    * {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC',
            'Hiragino Sans GB', 'Microsoft YaHei', 'Noto Sans CJK SC',
            'Source Han Sans SC', sans-serif;
    }

    h1, h2, h3, .big-number, .module-title {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC',
            'Hiragino Sans GB', 'Microsoft YaHei', 'Noto Sans CJK SC',
            'Source Han Sans SC', sans-serif;
        font-weight: 600;
    }

    /* ── Global Theme ── */
    :root {
        --bg-primary: #08080f;
        --bg-surface: #12121d;
        --bg-card: #1a1a2e;
        --gold: #c9a96e;
        --gold-light: #e0c78a;
        --gold-dark: #a6844a;
        --text-primary: #e8e4d9;
        --text-secondary: #9895a0;
        --accent-green: #5b8c5a;
        --accent-copper: #c97e4f;
        --border: #2a2a3a;
        --chart-1: #c9a96e;
        --chart-2: #8b7355;
        --chart-3: #5b8c5a;
        --chart-4: #c97e4f;
        --chart-5: #7b8ca8;
    }

    /* Override Streamlit defaults */
    .stApp {
        background-color: #08080f;
    }

    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 0;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0c0c18 0%, #0f0f1f 100%);
        border-right: 1px solid #2a2a3a;
    }
    [data-testid="stSidebar"] h1 {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC',
            'Microsoft YaHei', 'Noto Sans CJK SC', sans-serif;
        font-size: 1.4rem;
        font-weight: 600;
        color: #c9a96e;
        letter-spacing: 0.04em;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #c9a96e;
    }
    [data-testid="stSidebar"] .stRadio > div {
        gap: 0.25rem;
    }
    [data-testid="stSidebar"] .stRadio label {
        color: #9895a0;
        font-size: 0.9rem;
        padding: 0.5rem 0.75rem;
        border-radius: 6px;
        transition: all 0.2s ease;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        color: #e8e4d9;
        background: rgba(201,169,110,0.08);
    }

    /* ── Metric Cards ── */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16162a 100%);
        border: 1px solid #2a2a3a;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        border-color: #c9a96e;
        box-shadow: 0 4px 24px rgba(201,169,110,0.08);
    }
    .metric-value {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC',
            'Microsoft YaHei', 'Noto Sans CJK SC', sans-serif;
        font-size: 2rem;
        font-weight: 600;
        color: #c9a96e;
        letter-spacing: 0;
        font-variant-numeric: tabular-nums;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #9895a0;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 0.25rem;
    }

    /* ── Section headers ── */
    .section-title {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC',
            'Microsoft YaHei', 'Noto Sans CJK SC', sans-serif;
        font-size: 1.5rem;
        font-weight: 600;
        color: #e8e4d9;
        letter-spacing: 0.03em;
        margin-bottom: 0.25rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #2a2a3a;
    }

    /* ── DataFrames ── */
    [data-testid="stDataFrame"] {
        background: #12121d;
        border: 1px solid #2a2a3a;
        border-radius: 8px;
    }

    /* ── Expanders ── */
    .streamlit-expanderHeader {
        background: #1a1a2e;
        border: 1px solid #2a2a3a;
        border-radius: 8px;
        color: #e8e4d9;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: transparent;
        border-bottom: 1px solid #2a2a3a;
    }
    .stTabs [data-baseweb="tab"] {
        color: #9895a0;
        font-size: 0.9rem;
        padding: 0.6rem 1.25rem;
        border-radius: 8px 8px 0 0;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        color: #c9a96e;
        background: rgba(201,169,110,0.06);
        border-bottom: 2px solid #c9a96e;
    }

    /* ── Buttons ── */
    .stButton button {
        background: linear-gradient(135deg, #c9a96e, #a6844a);
        color: #08080f;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        letter-spacing: 0.04em;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background: linear-gradient(135deg, #e0c78a, #c9a96e);
        box-shadow: 0 4px 20px rgba(201,169,110,0.2);
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {background: transparent;}
</style>
""", unsafe_allow_html=True)

# ── Constants ───────────────────────────────────────────────────────────────

API_BASE = "http://localhost:8000"
FONT_FAMILY = (
    "-apple-system, BlinkMacSystemFont, Segoe UI, PingFang SC, "
    "Microsoft YaHei, Noto Sans CJK SC, Source Han Sans SC, sans-serif"
)
GOLD_SEQUENTIAL = [
    [0, "#1a1a2e"], [0.2, "#c9a96e"], [0.5, "#e0c78a"], [0.8, "#c9a96e"], [1, "#1a1a2e"]
]
CHART_COLORS = ["#c9a96e", "#8b7355", "#5b8c5a", "#c97e4f", "#7b8ca8", "#a6844a",
                "#6b9e6a", "#d4956a", "#5a7a9a", "#b8955a"]
CLUSTER_COLORS = ["#7b8ca8", "#c9a96e", "#c97e4f", "#5b8c5a", "#8b7d9e", "#6b95a8"]

CACHE_TTL = 300  # 5 minutes


def cluster_role(c: dict, clusters: list[dict]) -> str:
    """Derive business labels from metrics because K-Means ids are arbitrary."""
    order_rate = float(c.get("order_rate") or 0)
    avg_m = float(c.get("avg_M") or 0)
    avg_f = float(c.get("avg_F") or 0)
    avg_clicks = float(c.get("avg_clicks") or 0)
    max_m = max([float(x.get("avg_M") or 0) for x in clusters] + [1.0])
    max_f = max([float(x.get("avg_F") or 0) for x in clusters] + [1.0])

    if order_rate >= 0.8 and avg_m >= max_m * 0.6:
        return "高价值用户"
    if order_rate >= 0.8 and (avg_f >= max_f * 0.3 or avg_m > 0):
        return "活跃购买用户"
    if order_rate < 0.05 and avg_clicks <= 1 and avg_m < 1:
        return "沉睡/无行为用户"
    if order_rate < 0.05 and avg_clicks > 1:
        return "浏览未转化用户"
    return "低频维护用户"


def enrich_clusters(clusters: list[dict]) -> list[dict]:
    enriched = []
    for i, c in enumerate(clusters):
        row = dict(c)
        row["role"] = cluster_role(c, clusters)
        row["color"] = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        enriched.append(row)
    return enriched


# ── API helpers ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=CACHE_TTL)
def api_get(endpoint: str):
    try:
        r = requests.get(f"{API_BASE}{endpoint}", timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API 错误: {endpoint} — {e}")
        return None


@st.cache_data(ttl=60)
def api_post(endpoint: str, body: dict):
    try:
        r = requests.post(f"{API_BASE}{endpoint}", json=body, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API 错误: {endpoint} — {e}")
        return None


# ── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("# ◇ 洞察面板")
    st.markdown("")

    page = st.radio(
        "导航",
        [
            "🏠 用户画像总览",
            "🎯 购买倾向分析",
            "🔀 行为路径追踪",
            "👥 用户分群与标签管理",
            "🔥 热销商品分析",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        '<p style="color:#9895a0;font-size:0.75rem;">'
        '电商用户画像与购买倾向预测<br>'
        'Graduation Project · 2026'
        '</p>',
        unsafe_allow_html=True,
    )

    # User search
    st.markdown("---")
    st.markdown(
        '<p style="color:#c9a96e;font-size:0.8rem;font-weight:600;letter-spacing:0.04em;">'
        '🔍 用户快速查询</p>',
        unsafe_allow_html=True,
    )
    search_id = st.text_input("输入 User ID", placeholder="e.g. 0, 100, 99999", label_visibility="collapsed")
    if search_id:
        user_data = api_get(f"/api/user/{search_id.strip()}")
        if user_data:
            with st.container():
                st.markdown(f"""
                <div style="background:#1a1a2e;border:1px solid #2a2a3a;border-radius:8px;padding:0.75rem;font-size:0.8rem;">
                    <p style="color:#e8e4d9;margin:0;"><b>用户 {user_data['user_id']}</b></p>
                    <p style="color:#9895a0;margin:0;">性别: {user_data['gender']} · 年龄: {user_data['age_range']}</p>
                    <p style="color:#9895a0;margin:0;">群组: Cluster {user_data['cluster']}</p>
                    <p style="color:#9895a0;margin:0;">R: {user_data['R_days']}天 · F: {user_data['F_count']}次 · M: ¥{user_data['total_order_amount']:,.0f}</p>
                    {"<p style='color:#c9a96e;margin:0;'>购买倾向: " + str(user_data.get('prediction', {}).get('probability', 'N/A')) + " (" + user_data.get('prediction', {}).get('segment', '') + ")</p>" if 'prediction' in user_data else ""}
                </div>
                """, unsafe_allow_html=True)


# ── Shared chart helpers ────────────────────────────────────────────────────

def chart_layout(fig: go.Figure, title: str = "", height: int = 420) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color="#e8e4d9", family=FONT_FAMILY),
                   x=0.01),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#9895a0", size=12, family=FONT_FAMILY),
        margin=dict(l=20, r=20, t=50, b=20),
        height=height,
        xaxis=dict(gridcolor="#1e1e30", zeroline=False),
        yaxis=dict(gridcolor="#1e1e30", zeroline=False),
        legend=dict(font=dict(color="#9895a0"), orientation="h", yanchor="top", y=1.15, xanchor="center", x=0.5),
        hovermode="x unified",
    )
    return fig


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULE 1: 用户画像总览                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def render_overview():
    st.markdown('<p class="section-title">用户画像总览</p>', unsafe_allow_html=True)

    # KPI row
    kpis = api_get("/api/overview/kpis")
    if kpis:
        cols = st.columns(4)
        metrics = [
            ("总用户数", f"{kpis['total_users']:,}", "👥"),
            ("有订单用户", f"{kpis['has_order_ever']:,}", "🛒"),
            ("有浏览用户", f"{kpis['has_click_ever']:,}", "👁"),
            ("下单转化率", f"{kpis['conversion_rate']}%", "📈"),
        ]
        for col, (label, value, icon) in zip(cols, metrics):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size:1.5rem;margin-bottom:0.25rem;">{icon}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts row 1
    left, right = st.columns(2)

    with left:
        # Gender pie chart
        gender_data = api_get("/api/overview/gender")
        if gender_data:
            fig = go.Figure(go.Pie(
                labels=[d["gender"] for d in gender_data],
                values=[d["count"] for d in gender_data],
                hole=0.55,
                marker=dict(colors=["#c9a96e", "#7b8ca8"]),
                textinfo="percent+label",
                textfont=dict(color="#e8e4d9", size=13),
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=40, b=10),
                height=360,
                title=dict(text="性别分布", font=dict(size=16, color="#e8e4d9"), x=0.01),
                annotations=[dict(
                    text=f"<b>{sum(d['count'] for d in gender_data):,}</b><br>用户",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=16, color="#c9a96e"),
                )],
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with right:
        # Age bar chart
        age_data = api_get("/api/overview/age")
        if age_data:
            ages = [d["age_range"] for d in age_data]
            counts = [d["count"] for d in age_data]
            fig = go.Figure(go.Bar(
                x=ages, y=counts,
                marker=dict(
                    color=counts,
                    colorscale=GOLD_SEQUENTIAL,
                    line=dict(width=0),
                ),
                text=counts, textposition="outside",
                textfont=dict(color="#9895a0", size=11),
                hovertemplate="%{x}<br>%{y:,} 人<extra></extra>",
            ))
            fig = chart_layout(fig, "年龄分布", height=360)
            fig.update_xaxis(tickangle=30)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Charts row 2
    left2, right2 = st.columns(2)

    with left2:
        # Recency distribution
        rf = api_get("/api/overview/rfm/recency")
        if rf:
            fig = go.Figure(go.Bar(
                x=[d["bin"] for d in rf],
                y=[d["count"] for d in rf],
                marker=dict(
                    color=[d["count"] for d in rf],
                    colorscale=GOLD_SEQUENTIAL,
                    line=dict(width=0),
                ),
                hovertemplate="%{x}<br>%{y:,} 人<extra></extra>",
            ))
            fig = chart_layout(fig, "最近购买距今天数 (Recency)", height=340)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with right2:
        # Frequency distribution
        ff = api_get("/api/overview/rfm/frequency")
        if ff:
            fig = go.Figure(go.Bar(
                x=[d["bin"] for d in ff],
                y=[d["count"] for d in ff],
                marker=dict(
                    color=[d["count"] for d in ff],
                    colorscale=GOLD_SEQUENTIAL,
                    line=dict(width=0),
                ),
                hovertemplate="%{x}<br>%{y:,} 人<extra></extra>",
            ))
            fig = chart_layout(fig, "购买频次分布 (Frequency)", height=340)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Activity heatmap (hourly)
    st.markdown("<br>", unsafe_allow_html=True)
    hourly = api_get("/api/overview/hourly")
    if hourly:
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("订单时段分布", "点击时段分布"),
            horizontal_spacing=0.12,
        )
        fig.add_trace(
            go.Scatter(
                x=[d["hour"] for d in hourly["orders"]],
                y=[d["count"] for d in hourly["orders"]],
                mode="lines+markers",
                line=dict(color="#c9a96e", width=2.5, shape="spline"),
                marker=dict(color="#c9a96e", size=6),
                fill="tozeroy",
                fillcolor="rgba(201,169,110,0.08)",
                name="订单",
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[d["hour"] for d in hourly["clicks"]],
                y=[d["count"] for d in hourly["clicks"]],
                mode="lines+markers",
                line=dict(color="#7b8ca8", width=2.5, shape="spline"),
                marker=dict(color="#7b8ca8", size=6),
                fill="tozeroy",
                fillcolor="rgba(123,140,168,0.08)",
                name="点击",
            ),
            row=1, col=2,
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#9895a0"),
            height=320,
            margin=dict(l=20, r=20, t=50, b=20),
            showlegend=False,
            xaxis=dict(gridcolor="#1e1e30", dtick=2),
            xaxis2=dict(gridcolor="#1e1e30", dtick=2),
            yaxis=dict(gridcolor="#1e1e30"),
            yaxis2=dict(gridcolor="#1e1e30"),
        )
        for anno in fig.layout.annotations:
            anno.font = dict(color="#e8e4d9", size=14, family=FONT_FAMILY)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULE 2: 购买倾向分析                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def render_propensity():
    st.markdown('<p class="section-title">购买倾向分析</p>', unsafe_allow_html=True)
    health = api_get("/api/health") or {}
    threshold = health.get("threshold")
    threshold_label = f"阈值 {threshold:.3f}" if isinstance(threshold, (int, float)) else "验证集阈值"

    left, right = st.columns([1, 1])

    with left:
        # Prediction distribution
        dist = api_get("/api/propensity/distribution")
        if dist:
            threshold_x = 0
            if isinstance(threshold, (int, float)):
                for idx, row in enumerate(dist):
                    left_edge = row.get("left")
                    right_edge = row.get("right")
                    if isinstance(left_edge, (int, float)) and isinstance(right_edge, (int, float)) and left_edge <= threshold <= right_edge:
                        ratio = (threshold - left_edge) / (right_edge - left_edge) if right_edge > left_edge else 0
                        threshold_x = max(-0.5, min(len(dist) - 0.5, idx - 0.5 + ratio))
                        break
            fig = go.Figure(go.Bar(
                x=[d["bin"] for d in dist],
                y=[d["count"] for d in dist],
                marker=dict(
                    color=[d["count"] for d in dist],
                    colorscale=[
                        [0, "#5b8c5a"],
                        [0.25, "#c9a96e"],
                        [0.5, "#c97e4f"],
                        [0.75, "#8b7355"],
                        [1, "#7b8ca8"],
                    ],
                    line=dict(width=0),
                ),
                hovertemplate="概率区间: %{x}<br>用户数: %{y:,}<extra></extra>",
            ))
            fig = chart_layout(fig, "预测概率分布 (10K 用户样本)", height=380)
            fig.add_vline(
                x=threshold_x, line_dash="dash", line_color="#c9a96e", line_width=1.5,
                annotation=dict(text=threshold_label, font=dict(color="#c9a96e", size=11)),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with right:
        # Segments donut
        segs = api_get("/api/propensity/segments")
        if segs:
            fig = go.Figure(go.Pie(
                labels=[s["segment"] for s in segs],
                values=[s["count"] for s in segs],
                hole=0.6,
                marker=dict(colors=["#c9a96e", "#7b8ca8", "#c97e4f"]),
                textinfo="label+percent",
                textfont=dict(color="#e8e4d9", size=12),
                sort=False,
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=40, b=10),
                height=380,
                title=dict(text="购买倾向分层", font=dict(size=16, color="#e8e4d9"), x=0.01),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # SHAP Top Features
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#e8e4d9;font-size:1.1rem;font-weight:500;margin-bottom:0.5rem;">'
        '关键特征重要性 (SHAP)</p>',
        unsafe_allow_html=True,
    )

    top_feats = api_get("/api/propensity/top-features")
    if top_feats:
        feat_df = pd.DataFrame(top_feats).sort_values("importance", ascending=True)
        has_shap = feat_df["direction"].isin(["促进购买", "抑制购买"]).any()

        fig = go.Figure()
        colors = [
            "#c9a96e" if row.get("direction") == "促进购买" else ("#c97e4f" if has_shap else "#7b8ca8")
            for row in top_feats
        ]
        colors.reverse()

        fig.add_trace(go.Bar(
            y=feat_df["feature"].tolist(),
            x=feat_df["importance"].tolist(),
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{row.importance:.3f}  {row.direction}" for row in feat_df.itertuples()],
            textposition="outside",
            textfont=dict(color="#9895a0", size=11),
            hovertemplate="%{y}<br>SHAP: %{x:.4f}<br>%{text}<extra></extra>",
        ))
        fig = chart_layout(fig, "", height=460)
        fig.update_xaxes(title="SHAP Importance", gridcolor="#1e1e30")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Batch prediction interface
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("🔮 批量预测查询", expanded=False):
        user_input = st.text_area(
            "输入用户 ID（逗号或换行分隔）",
            placeholder="0, 100, 99999\n或每行一个 ID",
            height=80,
        )
        if st.button("执行预测", key="batch_predict"):
            ids = [u.strip() for u in user_input.replace("\n", ",").split(",") if u.strip()]
            if ids:
                with st.spinner("预测中..."):
                    payload = api_post("/api/propensity/predict", {"user_ids": ids})
                if payload:
                    results = payload if isinstance(payload, list) else payload.get("results", [])
                    missing = [] if isinstance(payload, list) else payload.get("missing_user_ids", [])
                    if missing:
                        st.warning("无预计算结果：" + ", ".join(missing))
                    df = pd.DataFrame(results)
                    if df.empty:
                        st.warning("未找到匹配用户")
                        return
                    df["probability"] = df["probability"].apply(lambda x: f"{x:.4f}")
                    st.dataframe(
                        df,
                        column_config={
                            "user_id": "用户 ID",
                            "probability": "购买概率",
                            "prediction": st.column_config.NumberColumn("预测标签", format="%d"),
                            "segment": "倾向分层",
                        },
                        use_container_width=True,
                        hide_index=True,
                    )
                    # Download button
                    csv = df.to_csv(index=False)
                    st.download_button("📥 导出 CSV", csv, f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULE 3: 行为路径追踪                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def render_behavior_path():
    st.markdown('<p class="section-title">行为路径追踪</p>', unsafe_allow_html=True)

    # Click to order time distribution
    time_dist = api_get("/api/path/click-to-order-time")
    if time_dist:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[d["bin"] for d in time_dist],
            y=[d["count"] for d in time_dist],
            marker=dict(
                color=[d["count"] for d in time_dist],
                colorscale=GOLD_SEQUENTIAL,
                line=dict(width=0),
            ),
            hovertemplate="%{x}<br>转化: %{y:,}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=[d["bin"] for d in time_dist],
            y=[d["count"] for d in time_dist],
            mode="lines+markers",
            line=dict(color="#c97e4f", width=2),
            marker=dict(color="#c97e4f", size=8, symbol="diamond"),
            name="趋势",
            showlegend=False,
        ))
        fig = chart_layout(fig, "点击到购买转化时间分布", height=380)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Category funnel (abandonment)
    st.markdown("<br>", unsafe_allow_html=True)
    funnel = api_get("/api/path/category-funnel")
    if funnel:
        funnel_df = pd.DataFrame(funnel).sort_values("click_count", ascending=True)
        st.markdown(
            '<p style="color:#e8e4d9;font-size:1rem;font-weight:500;">品类点击-购买漏斗 (弃购率)</p>',
            unsafe_allow_html=True,
        )

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=funnel_df["cid1_name"].tolist(),
            x=funnel_df["click_count"].tolist(),
            name="点击量",
            orientation="h",
            marker=dict(color="#7b8ca8", line=dict(width=0)),
            hovertemplate="点击: %{x:,}<extra></extra>",
        ))
        fig.add_trace(go.Bar(
            y=funnel_df["cid1_name"].tolist(),
            x=funnel_df["order_count"].tolist(),
            name="订单量",
            orientation="h",
            marker=dict(color="#c9a96e", line=dict(width=0)),
            hovertemplate="订单: %{x:,}<extra></extra>",
        ))
        fig = chart_layout(fig, "", height=420)
        fig.update_layout(barmode="group", bargap=0.25)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # User journey drill-down
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("🔍 单用户行为路径追踪", expanded=False):
        uid = st.text_input("用户 ID", key="journey_uid", placeholder="输入要追踪的用户 ID")
        if uid and st.button("追踪", key="track_btn"):
            journey = api_get(f"/api/path/user-journey/{uid.strip()}")
            if journey:
                st.markdown(f"""
                <div style="background:#1a1a2e;border:1px solid #2a2a3a;border-radius:8px;padding:1rem;margin-bottom:1rem;">
                    <span style="color:#c9a96e;font-weight:600;">用户 {journey['user_id']}</span>
                    <span style="color:#9895a0;margin-left:2rem;">浏览 {journey['total_clicks']} 次</span>
                    <span style="color:#9895a0;margin-left:1rem;">购买 {journey['total_orders']} 次</span>
                    <span style="color:#9895a0;margin-left:1rem;">
                        转化率 {round(journey['total_orders']/max(journey['total_clicks'],1)*100,1)}%
                    </span>
                </div>
                """, unsafe_allow_html=True)

                if journey["clicks"] or journey["orders"]:
                    events = []
                    for c in journey["clicks"][:50]:
                        events.append({
                            "时间": c["datetime"][:19],
                            "类型": "🔍 浏览",
                            "商品": c.get("item_name", c["item_id"])[:30],
                            "品类": c.get("category", ""),
                            "价格": c.get("price", 0),
                        })
                    for o in journey["orders"][:50]:
                        events.append({
                            "时间": o["datetime"][:19],
                            "类型": "🛒 购买",
                            "商品": o.get("item_name", o["item_id"])[:30],
                            "品类": o.get("category", ""),
                            "价格": o.get("price", 0),
                        })
                    events.sort(key=lambda x: x["时间"], reverse=True)
                    st.dataframe(pd.DataFrame(events).head(30), use_container_width=True, hide_index=True)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULE 4: 用户分群与标签管理                                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def render_segments():
    st.markdown('<p class="section-title">用户分群与标签管理</p>', unsafe_allow_html=True)

    clusters = api_get("/api/segments/clusters")
    if not clusters:
        return
    clusters = enrich_clusters(clusters)

    # Cluster KPI cards
    cols = st.columns(len(clusters))

    for col, c in zip(cols, clusters):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="border-left: 3px solid {c['color']};">
                <div style="color:{c['color']};font-weight:700;font-size:0.9rem;margin-bottom:0.5rem;">
                    Cluster {c['cluster']}<br>{c['role']}</div>
                <div style="font-size:1.5rem;font-weight:700;color:#e8e4d9;">{c['user_count']:,}</div>
                <div style="font-size:0.75rem;color:#9895a0;">用户</div>
                <div style="font-size:0.75rem;color:#9895a0;margin-top:0.25rem;">
                    订单率 {c['order_rate']*100:.0f}% · R {c['avg_R']:.0f}天</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Radar chart + Cluster profile table
    left, right = st.columns([1, 1])

    with left:
        # Radar chart
        radar_cols = ["avg_R", "avg_F", "avg_M", "avg_clicks", "order_rate"]
        radar_labels = ["活跃度(R↓)", "频次(F)", "消费(M)", "点击量", "订单率"]
        r_values = [float(c.get("avg_R", 0) or 0) for c in clusters]
        r_min, r_max = min(r_values), max(r_values)

        fig = go.Figure()
        for c in clusters:
            metric_cols = ["avg_F", "avg_M", "avg_clicks", "order_rate"]
            max_vals = [max(abs(c2.get(col, 0)) for c2 in clusters) for col in metric_cols]
            recency_score = (r_max - float(c.get("avg_R", 0) or 0)) / max(r_max - r_min, 1)
            norm_vals = [recency_score] + [
                float(c.get(col, 0) or 0) / max(mv, 1) for col, mv in zip(metric_cols, max_vals)
            ]

            fig.add_trace(go.Scatterpolar(
                r=norm_vals + [norm_vals[0]],
                theta=radar_labels + [radar_labels[0]],
                name=f"Cluster {c['cluster']} {c['role']}",
                fill="toself",
                line=dict(color=c["color"], width=2),
                fillcolor=f"rgba({','.join(str(int(c['color'].lstrip('#')[i:i+2], 16)) for i in (0, 2, 4))}, 0.12)",
            ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], gridcolor="#1e1e30", color="#9895a0"),
                angularaxis=dict(gridcolor="#1e1e30", color="#9895a0"),
                bgcolor="rgba(0,0,0,0)",
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=420,
            margin=dict(l=20, r=20, t=40, b=20),
            title=dict(text="用户分群画像雷达图 (归一化)", font=dict(size=14, color="#e8e4d9")),
            legend=dict(font=dict(color="#9895a0", size=11), yanchor="bottom", y=-0.15),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with right:
        # Cluster comparison table
        table_data = []
        for c in clusters:
            table_data.append({
                "群组": f"Cluster {c['cluster']}",
                "角色": c["role"],
                "用户数": f"{c['user_count']:,}",
                "均R(天)": f"{c['avg_R']:.0f}",
                "均F(次)": f"{c['avg_F']:.1f}",
                "均M(元)": f"{c['avg_M']:,.0f}",
                "均点击": f"{c['avg_clicks']:.0f}",
                "订单率": f"{c['order_rate']*100:.1f}%",
            })
        st.markdown(
            '<p style="color:#e8e4d9;font-size:1rem;font-weight:500;margin-bottom:0.5rem;">群组特征对比表</p>',
            unsafe_allow_html=True,
        )
        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

    # Cluster-category heatmap
    st.markdown("<br>", unsafe_allow_html=True)
    cat_data = api_get("/api/segments/cluster-category")
    if cat_data:
        cat_df = pd.DataFrame(cat_data)
        # Get top categories
        top_cats = cat_df.groupby("cid1_name")["count"].sum().nlargest(12).index.tolist()
        heatmap_data = cat_df[cat_df["cid1_name"].isin(top_cats)]

        pivot = heatmap_data.pivot_table(
            values="count", index="cluster", columns="cid1_name", aggfunc="sum"
        ).fillna(0)
        # Normalize per cluster
        pivot_norm = pivot.div(pivot.sum(axis=1), axis=0)

        fig = go.Figure(data=go.Heatmap(
            z=pivot_norm.values,
            x=pivot_norm.columns.tolist(),
            y=[f"Cluster {int(i)}" for i in pivot_norm.index.tolist()],
            colorscale=[
                [0, "#12121d"],
                [0.3, "#2a2a3a"],
                [0.5, "#c9a96e"],
                [0.7, "#e0c78a"],
                [1, "#c97e4f"],
            ],
            text=[[f"{v:.1%}" for v in row] for row in pivot_norm.values],
            texttemplate="%{text}",
            textfont=dict(color="#e8e4d9", size=10),
            hovertemplate="群组: %{y}<br>品类: %{x}<br>占比: %{z:.1%}<extra></extra>",
        ))
        fig = chart_layout(fig, "各群组 × 品类偏好热力图 (归一化)", height=320)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Cluster drill-down
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📋 群组详情与样本用户", expanded=False):
        selected_cluster = st.selectbox(
            "选择群组", [c["cluster"] for c in clusters],
            format_func=lambda x: f"Cluster {x} — {next((c['role'] for c in clusters if c['cluster'] == x), '')}"
        )
        detail = api_get(f"/api/segments/cluster/{selected_cluster}")
        if detail:
            st.markdown(f"""
            <div style="background:#1a1a2e;border:1px solid #2a2a3a;border-radius:8px;padding:1rem;margin-bottom:1rem;">
                <span style="color:#c9a96e;font-weight:600;">Cluster {detail['cluster_id']} 统计</span>
                <span style="color:#9895a0;margin-left:1.5rem;">用户数: {detail['stats']['user_count']:,}</span>
                <span style="color:#9895a0;margin-left:1rem;">R: {detail['stats']['avg_R']}天</span>
                <span style="color:#9895a0;margin-left:1rem;">F: {detail['stats']['avg_F']}</span>
                <span style="color:#9895a0;margin-left:1rem;">M: ¥{detail['stats']['avg_M']:,.0f}</span>
                <span style="color:#9895a0;margin-left:1rem;">订单率: {detail['stats']['has_order_rate']}%</span>
            </div>
            """, unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(detail["sample_users"]).head(30), use_container_width=True, hide_index=True)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULE 5: 热销商品分析                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def render_hot_products():
    st.markdown('<p class="section-title">热销商品分析</p>', unsafe_allow_html=True)

    top_items = api_get("/api/products/top?limit=15")
    top_cats = api_get("/api/products/top-categories")
    top_brands = api_get("/api/products/top-brands")
    price_dist = api_get("/api/products/price-distribution")

    # Row 1: Top items + Top categories
    left, right = st.columns([1, 1])

    with left:
        if top_items:
            items_df = pd.DataFrame(top_items)
            fig = go.Figure(go.Bar(
                x=items_df["order_count"].tolist(),
                y=items_df["item_name"].fillna(items_df["item_id"]).str[:20].tolist(),
                orientation="h",
                marker=dict(
                    color=items_df["order_count"].tolist(),
                    colorscale=GOLD_SEQUENTIAL,
                    line=dict(width=0),
                ),
                text=items_df["order_count"].tolist(),
                textposition="outside",
                textfont=dict(color="#9895a0", size=10),
                hovertemplate="%{y}<br>订单数: %{x:,}<br>价格: ¥%{customdata:.1f}<extra></extra>",
                customdata=items_df["price"].fillna(0).tolist(),
            ))
            fig = chart_layout(fig, "Top 15 热销商品", height=420)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with right:
        if top_cats:
            cats_df = pd.DataFrame(top_cats)
            fig = go.Figure(go.Treemap(
                labels=cats_df["cid1_name"].tolist(),
                values=cats_df["count"].tolist(),
                parents=["" for _ in range(len(cats_df))],
                marker=dict(
                    colors=cats_df["count"].tolist(),
                    colorscale=[
                        [0, "#1a1a2e"],
                        [0.3, "#8b7355"],
                        [0.6, "#c9a96e"],
                        [1, "#c97e4f"],
                    ],
                ),
                textfont=dict(color="#e8e4d9", size=13, family=FONT_FAMILY),
                hovertemplate="%{label}<br>订单量: %{value:,}<extra></extra>",
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=40, b=10),
                height=420,
                title=dict(text="品类销售分布", font=dict(size=16, color="#e8e4d9"), x=0.01),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Row 2: Top brands + Price distribution
    st.markdown("<br>", unsafe_allow_html=True)
    left2, right2 = st.columns([1, 1])

    with left2:
        if top_brands:
            brands_df = pd.DataFrame(top_brands)
            fig = go.Figure(go.Bar(
                x=brands_df["count"].tolist(),
                y=brands_df["brand_code"].fillna("unknown").str[:20].tolist(),
                orientation="h",
                marker=dict(
                    color=brands_df["count"].tolist(),
                    colorscale=[
                        [0, "#2a2a3a"],
                        [0.5, "#7b8ca8"],
                        [1, "#c9a96e"],
                    ],
                    line=dict(width=0),
                ),
                hovertemplate="%{y}<br>订单量: %{x:,}<extra></extra>",
            ))
            fig = chart_layout(fig, "Top 20 热销品牌", height=420)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with right2:
        if price_dist:
            fig = go.Figure(go.Bar(
                x=[d["bin"] for d in price_dist],
                y=[d["count"] for d in price_dist],
                marker=dict(
                    color=[d["count"] for d in price_dist],
                    colorscale=GOLD_SEQUENTIAL,
                    line=dict(width=0),
                ),
                hovertemplate="价格区间: ¥%{x}<br>订单量: %{y:,}<extra></extra>",
            ))
            fig = chart_layout(fig, "购买商品的价位分布", height=420)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Product detail modal
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📊 商品数据明细", expanded=False):
        if top_items:
            detail_df = pd.DataFrame(top_items)
            detail_df = detail_df.rename(columns={
                "item_name": "商品名称",
                "cid1_name": "一级类目",
                "cid3_name": "三级类目",
                "order_count": "订单数",
                "total_qty": "销售件数",
                "price": "单价",
                "brand_code": "品牌",
            })
            display_cols = ["商品名称", "一级类目", "品牌", "单价", "订单数", "销售件数"]
            available = [c for c in display_cols if c in detail_df.columns]
            st.dataframe(detail_df[available], use_container_width=True, hide_index=True)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Main Router                                                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

page_map = {
    "🏠 用户画像总览": render_overview,
    "🎯 购买倾向分析": render_propensity,
    "🔀 行为路径追踪": render_behavior_path,
    "👥 用户分群与标签管理": render_segments,
    "🔥 热销商品分析": render_hot_products,
}

render_fn = page_map.get(page)
if render_fn:
    render_fn()
else:
    st.error("未知页面")
