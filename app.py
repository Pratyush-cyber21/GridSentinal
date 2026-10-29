"""Streamlit field-verification dashboard for electricity-theft risk triage."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ARTIFACT_DIR = Path(__file__).parent / "artifacts"

st.set_page_config(
    page_title="GridSentinel | Field Verification",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    /* Ambient modern dark background */
    .stApp {
        background: radial-gradient(circle at 10% 8%, rgba(6, 182, 212, 0.09) 0%, transparent 35%),
                    radial-gradient(circle at 90% 15%, rgba(99, 102, 241, 0.08) 0%, transparent 40%),
                    radial-gradient(circle at 50% 90%, rgba(16, 185, 129, 0.05) 0%, transparent 45%),
                    #0B0F19 !important;
        color: #F1F5F9;
    }

    /* Sidebar glassmorphism */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.88) !important;
        backdrop-filter: blur(16px);
        border-right: 1px solid rgba(255, 255, 255, 0.07);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3.5rem;
        max-width: 1440px;
    }

    /* Header styling */
    .eyebrow-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: linear-gradient(135deg, rgba(6, 182, 212, 0.15), rgba(59, 130, 246, 0.15));
        border: 1px solid rgba(6, 182, 212, 0.35);
        color: #38BDF8;
        padding: 0.3rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }

    .hero-title {
        font-size: clamp(2.3rem, 4.2vw, 3.4rem);
        font-weight: 800;
        letter-spacing: -0.04em;
        line-height: 1.05;
        background: linear-gradient(135deg, #FFFFFF 20%, #E2E8F0 55%, #38BDF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.4rem 0 0.6rem 0;
    }

    .hero-copy {
        color: #94A3B8;
        max-width: 760px;
        font-size: 1.02rem;
        line-height: 1.65;
        margin-bottom: 1.2rem;
    }

    /* Pulsing live indicator */
    @keyframes pulse-glow {
        0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.6); }
        70% { box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #6EE7B7;
        padding: 0.35rem 0.8rem;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 600;
    }

    .live-dot {
        width: 8px;
        height: 8px;
        background: #10B981;
        border-radius: 50%;
        animation: pulse-glow 2s infinite;
    }

    /* Metric card upgrades */
    div[data-testid="stMetric"] {
        background: rgba(17, 24, 39, 0.7) !important;
        backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        transition: all 0.25s ease;
    }

    div[data-testid="stMetric"]:hover {
        border-color: rgba(56, 189, 248, 0.45);
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(6, 182, 212, 0.15);
    }

    div[data-testid="stMetric"] label {
        color: #94A3B8 !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em;
    }

    div[data-testid="stMetricValue"] {
        color: #F8FAFC !important;
        font-size: 2rem !important;
        font-weight: 750 !important;
        letter-spacing: -0.035em;
    }

    div[data-testid="stMetricDelta"] {
        background: rgba(30, 41, 59, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 999px;
        color: #38BDF8 !important;
        padding: 0.12rem 0.55rem;
        font-size: 0.78rem;
        font-weight: 600;
    }

    /* Notice banner */
    .notice {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.75), rgba(30, 41, 59, 0.5));
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-left: 4px solid #06B6D4;
        border-radius: 10px;
        padding: 0.95rem 1.2rem;
        color: #CBD5E1;
        font-size: 0.89rem;
        line-height: 1.55;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 4px;
        margin-bottom: 1rem;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 8px;
        color: #94A3B8;
        padding: 0.65rem 1.25rem;
        font-weight: 600;
        font-size: 0.9rem;
        transition: all 0.2s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: #F8FAFC;
        background: rgba(30, 41, 59, 0.75);
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(6, 182, 212, 0.22), rgba(59, 130, 246, 0.22)) !important;
        color: #38BDF8 !important;
        border: 1px solid rgba(56, 189, 248, 0.45) !important;
        box-shadow: 0 4px 14px rgba(6, 182, 212, 0.15);
    }

    /* Buttons */
    .stButton > button, .stDownloadButton > button {
        background: linear-gradient(135deg, #0284C7, #2563EB) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(255, 255, 255, 0.18) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.4rem !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35) !important;
        transition: all 0.2s ease !important;
    }

    .stButton > button:hover, .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #0369A1, #1D4ED8) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 22px rgba(37, 99, 235, 0.55) !important;
        border-color: rgba(255, 255, 255, 0.3) !important;
    }

    /* Table styling */
    [data-testid="stDataFrame"] {
        background: rgba(17, 24, 39, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }

    /* Select and input tags */
    [data-baseweb="tag"] {
        background: rgba(30, 41, 59, 0.8) !important;
        border: 1px solid rgba(56, 189, 248, 0.25) !important;
        color: #E2E8F0 !important;
        border-radius: 6px !important;
    }

    [data-baseweb="select"] > div {
        background: rgba(17, 24, 39, 0.75) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
    }

    hr {
        border-color: rgba(255, 255, 255, 0.08) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_artifacts() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    required = [
        "consumer_risk_ranking.csv",
        "model_comparison.csv",
        "precision_recall_curves.csv",
        "gains_curves.csv",
        "evaluation_summary.json",
        "priority_meter_readings.csv.gz",
    ]
    missing = [name for name in required if not (ARTIFACT_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing benchmark artifacts: {', '.join(missing)}. Run `python load_indian_dataset.py` first."
        )
    ranking = pd.read_csv(ARTIFACT_DIR / required[0])
    metrics = pd.read_csv(ARTIFACT_DIR / required[1])
    pr_curves = pd.read_csv(ARTIFACT_DIR / required[2])
    gains = pd.read_csv(ARTIFACT_DIR / required[3])
    summary = json.loads((ARTIFACT_DIR / required[4]).read_text(encoding="utf-8"))
    readings = pd.read_csv(ARTIFACT_DIR / required[5], parse_dates=["timestamp"])
    return ranking, metrics, pr_curves, gains, summary, readings


def manifest_pdf(rows: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        leftMargin=12 * mm,
        rightMargin=12 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
    )
    styles = getSampleStyleSheet()
    story = [
        Paragraph("GridSentinel — Field Inspection Manifest", styles["Title"]),
        Paragraph(
            "Targeted electricity risk ranking based on smart-meter anomaly detection. Verify findings on-site.",
            styles["BodyText"],
        ),
        Spacer(1, 7 * mm),
    ]
    columns = ["rank", "consumer_id", "risk_score", "risk_tier", "primary_signal", "evaluation_split"]
    labels = ["Rank", "Consumer ID", "Risk Score", "Risk Tier", "Primary Signal", "Split"]
    body = [labels] + [
        [
            str(int(row["rank"])),
            row["consumer_id"],
            f"{row['risk_score']:.3f}",
            row["risk_tier"],
            row["primary_signal"],
            row["evaluation_split"],
        ]
        for _, row in rows[columns].iterrows()
    ]
    table = Table(body, repeatRows=1, colWidths=[16 * mm, 32 * mm, 22 * mm, 30 * mm, 55 * mm, 26 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#38BDF8")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(table)
    document.build(story)
    return buffer.getvalue()


def sparkline_data_url(values: list[float], width: int = 600, height: int = 46) -> str:
    """Render a modern electric-cyan sparkline with gradient fill as an inline SVG."""
    series = np.asarray(values, dtype=float)
    if series.size == 0:
        return ""
    lower, upper = float(np.min(series)), float(np.max(series))
    span = max(upper - lower, 1e-9)
    x_coords = np.linspace(4, width - 4, series.size)
    y_coords = height - 4 - ((series - lower) / span) * (height - 8)
    points = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(x_coords, y_coords))
    area_points = f"{x_coords[0]:.1f},{height} {points} {x_coords[-1]:.1f},{height}"
    
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        '<defs>'
        '<linearGradient id="cyanGlow" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0%" stop-color="#38BDF8" stop-opacity="0.38"/>'
        '<stop offset="100%" stop-color="#38BDF8" stop-opacity="0.0"/>'
        '</linearGradient>'
        '</defs>'
        f'<polygon points="{area_points}" fill="url(#cyanGlow)"/>'
        f'<polyline points="{points}" fill="none" stroke="#38BDF8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
        '</svg>'
    )
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def line_figure(consumer: pd.DataFrame, consumer_id: str) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=consumer.timestamp,
            y=consumer.expected_kwh,
            name="Expected baseline",
            line=dict(color="#64748B", width=1.6, dash="dash"),
            hovertemplate="%{x}<br>Expected: %{y:.2f} kWh<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=consumer.timestamp,
            y=consumer.consumption_kwh,
            name="Observed load",
            line=dict(color="#38BDF8", width=2),
            hovertemplate="%{x}<br>Observed: %{y:.2f} kWh<extra></extra>",
        )
    )
    abnormal = consumer[consumer.is_anomaly_period.astype(str).str.lower().isin(["true", "1"])]
    if not abnormal.empty:
        figure.add_trace(
            go.Scatter(
                x=abnormal.timestamp,
                y=abnormal.consumption_kwh,
                name="Abnormal period",
                mode="markers",
                marker=dict(color="#F43F5E", size=5, symbol="circle", line=dict(color="#FFFFFF", width=1)),
                hovertemplate="%{x}<br>Abnormal: %{y:.2f} kWh<extra></extra>",
            )
        )
    figure.update_layout(
        title=dict(
            text=f"Observed vs. Expected Load Curve · <b>{consumer_id}</b>",
            font=dict(color="#F1F5F9", size=15),
        ),
        height=460,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.65)",
        margin=dict(l=15, r=15, t=55, b=15),
        legend=dict(
            orientation="h",
            y=1.09,
            x=0,
            font=dict(color="#CBD5E1", size=12),
            bgcolor="rgba(15, 23, 42, 0.6)",
            bordercolor="rgba(255, 255, 255, 0.08)",
            borderwidth=1,
        ),
        xaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.06)",
            tickfont=dict(color="#94A3B8"),
            showgrid=True,
        ),
        yaxis=dict(
            title=dict(text="Consumption (kWh)", font=dict(color="#94A3B8")),
            gridcolor="rgba(255, 255, 255, 0.06)",
            tickfont=dict(color="#94A3B8"),
            showgrid=True,
        ),
        hovermode="x unified",
    )
    return figure


def signal_radar(consumer: pd.DataFrame) -> go.Figure:
    observed = consumer.consumption_kwh.to_numpy(float)
    expected = consumer.expected_kwh.to_numpy(float)
    hours = consumer.timestamp.dt.hour.to_numpy()
    peak = (hours >= 18) & (hours <= 23)
    recent = np.arange(len(observed)) >= int(len(observed) * 0.65)
    measures = {
        "Mean deviation": np.mean(np.abs(observed - expected)) / max(np.mean(expected), 1e-6),
        "Peak suppression": 1 - np.mean(observed[peak & recent]) / max(np.mean(expected[peak & recent]), 1e-6),
        "Low-load persistence": np.mean(observed < np.quantile(expected, 0.15) * 0.5),
        "Volatility shift": abs(np.std(observed[recent]) - np.std(expected[recent])) / max(np.std(expected[recent]), 1e-6),
        "Baseline drift": abs(np.mean(observed[recent]) - np.mean(expected[recent])) / max(np.mean(expected[recent]), 1e-6),
    }
    values = np.clip(list(measures.values()), 0, 1)
    categories = list(measures)
    figure = go.Figure(
        go.Scatterpolar(
            r=np.r_[values, values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            line=dict(color="#38BDF8", width=2),
            fillcolor="rgba(6, 182, 212, 0.22)",
            name="Signal strength",
        )
    )
    figure.update_layout(
        template="plotly_dark",
        height=360,
        paper_bgcolor="rgba(0,0,0,0)",
        polar=dict(
            bgcolor="rgba(15, 23, 42, 0.65)",
            radialaxis=dict(range=[0, 1], showticklabels=False, gridcolor="rgba(255, 255, 255, 0.08)"),
            angularaxis=dict(tickfont=dict(color="#CBD5E1", size=11), gridcolor="rgba(255, 255, 255, 0.08)"),
        ),
        margin=dict(l=45, r=45, t=35, b=35),
        showlegend=False,
    )
    return figure


try:
    ranking, metrics, pr_curves, gains, summary, readings = load_artifacts()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.code("python load_indian_dataset.py\nstreamlit run app.py", language="bash")
    st.stop()

benchmark = summary["benchmark"]
selected_model = summary["selected_model"]
top_priority = ranking[ranking.priority_top_10_percent.astype(str).str.lower().isin(["true", "1"])].copy()
top_ten = ranking.head(10).copy()

with st.sidebar:
    st.markdown("### ⚡ GridSentinel")
    st.markdown(
        '<span class="status-badge"><span class="live-dot"></span> System Live & Verified</span>',
        unsafe_allow_html=True,
    )
    st.caption("Distribution Intelligence & Field Triage")
    st.divider()
    tier_options = list(ranking.risk_tier.unique())
    tiers = st.multiselect("Risk tiers", tier_options, default=tier_options)
    signal_options = sorted(ranking.primary_signal.unique())
    signals = st.multiselect("Anomaly signals", signal_options, default=signal_options)
    cohort_options = ["train", "validation", "test"]
    cohorts = st.multiselect("Evaluation cohort", cohort_options, default=cohort_options)
    filtered = ranking[
        ranking.risk_tier.isin(tiers)
        & ranking.primary_signal.isin(signals)
        & ranking.evaluation_split.isin(cohorts)
    ].copy()
    st.divider()
    st.markdown("**Benchmark Governance**")
    st.caption(f"Selected Model: **{selected_model}**")
    st.caption("Data Source: **Indian Household Electricity Data (Kaggle)**")
    st.caption("Validation: Realistic cyber-physical attacks")

# Main Header Section
st.markdown('<div class="eyebrow-badge">⚡ Smart Grid Intelligence · Field Operations</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Electricity Risk Triage Console</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-copy">Automated abnormal consumption detection and consumer-level risk prioritization for Indian power distribution companies. Powered by calibrated ensemble ML & sequential deep learning.</div>',
    unsafe_allow_html=True,
)

# Modern 4 KPI Metric Cards with top color accents
kpi_cols = st.columns(4)

with kpi_cols[0]:
    st.markdown(
        f"""
        <div style="background: rgba(17, 24, 39, 0.75); border: 1px solid rgba(255,255,255,0.08); border-top: 3px solid #06B6D4; border-radius: 12px; padding: 1.1rem 1.25rem;">
            <div style="color: #94A3B8; font-size: 0.82rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Meters Inspected</div>
            <div style="color: #F8FAFC; font-size: 2.1rem; font-weight: 800; letter-spacing: -0.04em; margin: 0.2rem 0;">{summary['dataset']['consumers']:,}</div>
            <div style="color: #38BDF8; font-size: 0.78rem; font-weight: 550;">16 Indian Cities · Discom Benchmark</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_cols[1]:
    st.markdown(
        f"""
        <div style="background: rgba(17, 24, 39, 0.75); border: 1px solid rgba(255,255,255,0.08); border-top: 3px solid #F43F5E; border-radius: 12px; padding: 1.1rem 1.25rem;">
            <div style="color: #94A3B8; font-size: 0.82rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Flagged High Risk</div>
            <div style="color: #FDA4AF; font-size: 2.1rem; font-weight: 800; letter-spacing: -0.04em; margin: 0.2rem 0;">{len(top_priority):,}</div>
            <div style="color: #FB7185; font-size: 0.78rem; font-weight: 550;">Top 10% Priority Triage Band</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_cols[2]:
    st.markdown(
        f"""
        <div style="background: rgba(17, 24, 39, 0.75); border: 1px solid rgba(255,255,255,0.08); border-top: 3px solid #10B981; border-radius: 12px; padding: 1.1rem 1.25rem;">
            <div style="color: #94A3B8; font-size: 0.82rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Benchmark Recall</div>
            <div style="color: #6EE7B7; font-size: 2.1rem; font-weight: 800; letter-spacing: -0.04em; margin: 0.2rem 0;">{benchmark['recall']:.1%}</div>
            <div style="color: #34D399; font-size: 0.78rem; font-weight: 550;">Exceeds Target (≥ {summary['config']['recall_target']:.0%})</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_cols[3]:
    st.markdown(
        f"""
        <div style="background: rgba(17, 24, 39, 0.75); border: 1px solid rgba(255,255,255,0.08); border-top: 3px solid #8B5CF6; border-radius: 12px; padding: 1.1rem 1.25rem;">
            <div style="color: #94A3B8; font-size: 0.82rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Top 20% Capture</div>
            <div style="color: #C4B5FD; font-size: 2.1rem; font-weight: 800; letter-spacing: -0.04em; margin: 0.2rem 0;">{benchmark['top20_capture']:.1%}</div>
            <div style="color: #A78BFA; font-size: 0.78rem; font-weight: 550;">Exceeds Target (≥ {summary['config']['top20_target']:.0%})</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

overview_tab, inspector_tab, comparison_tab, operations_tab = st.tabs(
    ["🎯 Priority Queue", "🔍 Load Inspector", "📊 Model Evidence", "📋 Field Operations"]
)

with overview_tab:
    st.markdown("#### Top 10 High-Risk Field Priorities")
    st.caption("Inspect the 168-hour consumption trend sparklines and stage priority consumers for field inspection.")
    trend_map = {
        cid: group.sort_values("timestamp").consumption_kwh.tail(168).round(3).tolist()
        for cid, group in readings.groupby("consumer_id")
    }
    display = top_ten[["rank", "consumer_id", "risk_score", "risk_tier", "primary_signal"]].copy()
    display["trend_7d"] = display.consumer_id.map(trend_map).map(sparkline_data_url)
    display["schedule_inspection"] = display.consumer_id.isin(st.session_state.get("scheduled_ids", []))
    display = display[
        ["schedule_inspection", "rank", "consumer_id", "risk_score", "trend_7d", "risk_tier", "primary_signal"]
    ]
    edited = st.data_editor(
        display,
        hide_index=True,
        width="stretch",
        disabled=["rank", "consumer_id", "risk_score", "risk_tier", "primary_signal", "trend_7d"],
        column_config={
            "rank": st.column_config.NumberColumn("Rank", format="#%d", width="small"),
            "consumer_id": st.column_config.TextColumn("Consumer ID"),
            "risk_score": st.column_config.ProgressColumn("Calibrated Risk", min_value=0, max_value=1, format="%.3f"),
            "risk_tier": st.column_config.TextColumn("Risk Category"),
            "primary_signal": st.column_config.TextColumn("Primary Anomaly Signal", width="medium"),
            "trend_7d": st.column_config.ImageColumn("7-Day Trend Sparkline", width="medium"),
            "schedule_inspection": st.column_config.CheckboxColumn(
                "Inspect", help="Tick to add this consumer to the field manifest", width="small"
            ),
        },
        key="priority_editor",
    )
    st.session_state["scheduled_ids"] = edited.loc[edited.schedule_inspection, "consumer_id"].tolist()
    scheduled_count = len(st.session_state["scheduled_ids"])
    if scheduled_count:
        st.success(f"✓ {scheduled_count} consumer{'s' if scheduled_count != 1 else ''} staged for field inspection manifest.")
    else:
        st.info("No consumers staged yet. Tick the 'Inspect' box to build the inspection manifest.")
    st.markdown(
        '<div class="notice"><b>Field Notice:</b> Risk scores prioritize investigations based on statistical anomalies and cyber-physical signatures; they do not constitute legal proof of theft. Confirm meter integrity and physical seals on-site.</div>',
        unsafe_allow_html=True,
    )

with inspector_tab:
    choices = [cid for cid in filtered.consumer_id if cid in set(readings.consumer_id)]
    if not choices:
        st.warning("No priority consumers match the current sidebar filters.")
    else:
        selected_id = st.selectbox("Select Consumer for In-Depth Triage", choices, index=0)
        selected_row = ranking.set_index("consumer_id").loc[selected_id]
        consumer = readings[readings.consumer_id == selected_id].sort_values("timestamp")
        left, right = st.columns([2.4, 1])
        with left:
            st.plotly_chart(
                line_figure(consumer, selected_id), width="stretch", config={"displaylogo": False}
            )
        with right:
            st.metric("Calibrated Risk Score", f"{selected_row.risk_score:.3f}")
            st.metric("Priority Rank", f"#{int(selected_row['rank'])}")
            st.markdown(f"**Anomaly Signature:** `{selected_row.primary_signal}`")
            st.plotly_chart(
                signal_radar(consumer), width="stretch", config={"displayModeBar": False}
            )

with comparison_tab:
    st.markdown("#### Model Performance on Shared Indian Test Cohort")
    st.caption(
        f"Both models were evaluated on the same {summary['dataset']['test_consumers']}-consumer holdout cohort. "
        f"Model selection used validation AUC-PR to eliminate test split leakage."
    )
    formatted_metrics = metrics.copy()
    metric_cols = ["accuracy", "recall", "precision", "f1", "auc_roc", "auc_pr", "top10_capture", "top20_capture"]
    formatted_metrics[metric_cols] = formatted_metrics[metric_cols].map(lambda value: f"{value:.1%}")
    st.dataframe(formatted_metrics, hide_index=True, width="stretch")
    chart_left, chart_right = st.columns(2)
    with chart_left:
        pr_figure = px.line(
            pr_curves,
            x="recall",
            y="precision",
            color="model",
            title="Precision–Recall Curves",
            color_discrete_sequence=["#06B6D4", "#A855F7"],
        )
        pr_figure.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15, 23, 42, 0.65)",
            height=410,
            xaxis=dict(gridcolor="rgba(255,255,255,0.06)", title="Recall"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)", title="Precision"),
        )
        st.plotly_chart(pr_figure, width="stretch", config={"displaylogo": False})
    with chart_right:
        gains_figure = px.line(
            gains,
            x="population_fraction",
            y="anomaly_capture",
            color="model",
            title="Cumulative Gains (Lift Curve)",
            color_discrete_sequence=["#06B6D4", "#A855F7"],
        )
        gains_figure.add_vline(x=0.2, line_dash="dash", line_color="#38BDF8", annotation_text="Top 20%")
        gains_figure.add_hline(y=0.6, line_dash="dot", line_color="#10B981", annotation_text="60% target")
        gains_figure.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15, 23, 42, 0.65)",
            height=410,
            xaxis=dict(gridcolor="rgba(255,255,255,0.06)", title="Fraction of Consumers Inspected"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)", title="Anomaly Capture Rate"),
        )
        st.plotly_chart(gains_figure, width="stretch", config={"displaylogo": False})

with operations_tab:
    st.markdown("#### Field Inspection Manifest")
    manifest_mode = st.radio("Manifest Source", ["Scheduled selections", "Filtered priority list"], horizontal=True)
    if manifest_mode == "Scheduled selections":
        selected_ids = st.session_state.get("scheduled_ids", [])
        manifest = ranking[ranking.consumer_id.isin(selected_ids)].copy()
    else:
        manifest = filtered.head(100).copy()
    st.dataframe(
        manifest[["rank", "consumer_id", "risk_score", "risk_tier", "primary_signal", "evaluation_split"]],
        hide_index=True,
        width="stretch",
    )
    if manifest.empty:
        st.warning("No consumers staged yet. Select rows in the Priority Queue or switch to 'Filtered priority list'.")
    else:
        csv_bytes = manifest.to_csv(index=False).encode("utf-8")
        pdf_bytes = manifest_pdf(manifest)
        export_left, export_right = st.columns(2)
        export_left.download_button(
            "📥 Download Field Manifest (CSV)",
            csv_bytes,
            "field_inspection_manifest.csv",
            "text/csv",
            width="stretch",
        )
        export_right.download_button(
            "📄 Download Field Manifest (PDF)",
            pdf_bytes,
            "field_inspection_manifest.pdf",
            "application/pdf",
            width="stretch",
        )
