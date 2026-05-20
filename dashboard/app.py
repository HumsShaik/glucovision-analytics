# ============================================================
# GlucoVision Analytics Streamlit Dashboard
# ============================================================

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="GlucoVision Analytics Dashboard",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# Custom Styling
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #08111f 0%, #102a43 45%, #1b4332 100%);
        color: #f8fafc;
    }

    h1, h2, h3 {
        color: #ffffff;
    }

    .metric-card {
        background: linear-gradient(135deg, #0f766e, #2563eb);
        padding: 22px;
        border-radius: 18px;
        color: white;
        text-align: center;
        box-shadow: 0px 6px 18px rgba(0,0,0,0.35);
    }

    .metric-value {
        font-size: 34px;
        font-weight: 800;
    }

    .metric-label {
        font-size: 15px;
        opacity: 0.9;
    }

    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.10);
        padding: 18px;
        border-radius: 16px;
        box-shadow: 0px 5px 15px rgba(0,0,0,0.25);
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #020617, #0f172a);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255,255,255,0.12);
        border-radius: 14px;
        color: white;
        padding: 12px 18px;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #22c55e, #06b6d4);
        color: black;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# File Paths
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

CLEANED_DATA_PATH = os.path.join(DATA_DIR, "glucovision_cleaned_preprocessed.csv")
PATIENT_SUMMARY_PATH = os.path.join(DATA_DIR, "patient_summary.csv")

PATIENT_RISK_PATH = os.path.join(OUTPUT_DIR, "patient_risk_segmentation.csv")
HOURLY_RISK_PATH = os.path.join(OUTPUT_DIR, "hourly_risk_recommendations.csv")
PRESCRIPTIVE_SUMMARY_PATH = os.path.join(OUTPUT_DIR, "prescriptive_dashboard_summary.csv")

PREDICTIVE_EVAL_PATH = os.path.join(OUTPUT_DIR, "predictive_model_evaluation.csv")
ACTUAL_VS_PRED_PATH = os.path.join(OUTPUT_DIR, "actual_vs_predicted_glucose.csv")
FEATURE_IMPORTANCE_PATH = os.path.join(OUTPUT_DIR, "predictive_feature_importance.csv")
PREDICTIVE_SUMMARY_PATH = os.path.join(OUTPUT_DIR, "predictive_dashboard_summary.csv")


# ============================================================
# Data Loading
# ============================================================

@st.cache_data
def load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


df = load_csv(CLEANED_DATA_PATH)
patient_summary = load_csv(PATIENT_SUMMARY_PATH)
patient_risk = load_csv(PATIENT_RISK_PATH)
hourly_risk = load_csv(HOURLY_RISK_PATH)
prescriptive_summary = load_csv(PRESCRIPTIVE_SUMMARY_PATH)
predictive_eval = load_csv(PREDICTIVE_EVAL_PATH)
actual_vs_pred = load_csv(ACTUAL_VS_PRED_PATH)
feature_importance = load_csv(FEATURE_IMPORTANCE_PATH)
predictive_summary = load_csv(PREDICTIVE_SUMMARY_PATH)


# ============================================================
# Data Preparation
# ============================================================

if not df.empty:
    df["time"] = pd.to_datetime(df["time"], errors="coerce")

    if "date" not in df.columns:
        df["date"] = df["time"].dt.date

    if "hour" not in df.columns:
        df["hour"] = df["time"].dt.hour


# ============================================================
# Helper Functions
# ============================================================

VIBRANT_COLORS = [
    "#00E5FF", "#FF4ECD", "#FFD166", "#06D6A0",
    "#EF476F", "#A78BFA", "#F97316", "#22C55E"
]

CATEGORY_COLORS = {
    "Normal": "#22C55E",
    "High": "#F97316",
    "Low": "#38BDF8"
}


def style_plotly(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        title_font=dict(size=22, color="white"),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="white")
        ),
        margin=dict(l=30, r=30, t=60, b=30)
    )
    return fig


def kpi_card(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# Sidebar Filters
# ============================================================

st.sidebar.title("🩸 GlucoVision Filters")

if df.empty:
    st.error("Main cleaned dataset not found. Please check data/processed/glucovision_cleaned_preprocessed.csv")
    st.stop()

patient_options = sorted(df["patient_id"].dropna().unique())

selected_patients = st.sidebar.multiselect(
    "Select Patient ID",
    options=patient_options,
    default=patient_options[:5] if len(patient_options) > 5 else patient_options
)

category_options = sorted(df["glucose_category"].dropna().unique())

selected_categories = st.sidebar.multiselect(
    "Select Glucose Category",
    options=category_options,
    default=category_options
)

min_date = df["time"].min().date()
max_date = df["time"].max().date()

selected_date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

filtered_df = df.copy()

if selected_patients:
    filtered_df = filtered_df[filtered_df["patient_id"].isin(selected_patients)]

if selected_categories:
    filtered_df = filtered_df[filtered_df["glucose_category"].isin(selected_categories)]

if len(selected_date_range) == 2:
    start_date, end_date = selected_date_range
    filtered_df = filtered_df[
        (filtered_df["time"].dt.date >= start_date) &
        (filtered_df["time"].dt.date <= end_date)
    ]


# ============================================================
# Dashboard Header
# ============================================================

st.title("🩸 GlucoVision Analytics Dashboard")
st.markdown(
    """
    Interactive dashboard for descriptive, prescriptive, and limited predictive glucose analytics.
    Use the sidebar filters to explore patient-level glucose behavior, risk patterns, and model results.
    """
)


# ============================================================
# Tabs
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📌 Overview",
    "📊 Descriptive Analysis",
    "🧑‍⚕️ Patient Insights",
    "⚠️ Prescriptive Analysis",
    "🤖 Predictive Analysis",
    "📁 Data Explorer"
])


# ============================================================
# Tab 1: Overview
# ============================================================

with tab1:
    st.header("Project Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        kpi_card("Total Records", f"{len(filtered_df):,}")

    with col2:
        kpi_card("Patients Selected", filtered_df["patient_id"].nunique())

    with col3:
        kpi_card("Average Glucose", f"{filtered_df['glucose'].mean():.1f}")

    with col4:
        high_pct = (filtered_df["glucose_category"].eq("High").mean() * 100)
        kpi_card("High Glucose %", f"{high_pct:.1f}%")

    st.subheader("Glucose Category Distribution")

    category_counts = (
        filtered_df["glucose_category"]
        .value_counts()
        .reset_index()
    )
    category_counts.columns = ["glucose_category", "count"]

    fig = px.pie(
        category_counts,
        names="glucose_category",
        values="count",
        hole=0.45,
        color="glucose_category",
        color_discrete_map=CATEGORY_COLORS,
        title="Normal vs High vs Low Glucose Records"
    )
    st.plotly_chart(style_plotly(fig), use_container_width=True)

    st.subheader("Average Glucose by Hour")

    hourly_avg = (
        filtered_df.groupby("hour", as_index=False)["glucose"]
        .mean()
        .sort_values("hour")
    )

    fig = px.line(
        hourly_avg,
        x="hour",
        y="glucose",
        markers=True,
        title="Average Glucose Trend by Hour",
        color_discrete_sequence=["#00E5FF"]
    )
    st.plotly_chart(style_plotly(fig), use_container_width=True)


# ============================================================
# Tab 2: Descriptive Analysis
# ============================================================

with tab2:
    st.header("Descriptive Analysis")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            filtered_df,
            x="glucose",
            nbins=40,
            title="Glucose Distribution",
            color_discrete_sequence=["#FF4ECD"]
        )
        st.plotly_chart(style_plotly(fig), use_container_width=True)

    with col2:
        fig = px.box(
            filtered_df,
            x="glucose_category",
            y="glucose",
            color="glucose_category",
            color_discrete_map=CATEGORY_COLORS,
            title="Glucose by Category"
        )
        st.plotly_chart(style_plotly(fig), use_container_width=True)

    st.subheader("Glucose Category by Hour")

    hourly_category = (
        filtered_df.groupby(["hour", "glucose_category"])
        .size()
        .reset_index(name="count")
    )

    fig = px.bar(
        hourly_category,
        x="hour",
        y="count",
        color="glucose_category",
        color_discrete_map=CATEGORY_COLORS,
        barmode="group",
        title="Hourly Glucose Category Counts"
    )
    st.plotly_chart(style_plotly(fig), use_container_width=True)

    st.subheader("Glucose Trend Over Time")

    trend_df = (
        filtered_df.sort_values("time")
        .groupby("time", as_index=False)["glucose"]
        .mean()
    )

    fig = px.line(
        trend_df,
        x="time",
        y="glucose",
        title="Average Glucose Over Time",
        color_discrete_sequence=["#FFD166"]
    )
    st.plotly_chart(style_plotly(fig), use_container_width=True)


# ============================================================
# Tab 3: Patient Insights
# ============================================================

with tab3:
    st.header("Patient-Level Insights")

    if not patient_summary.empty:
        st.subheader("Patient Summary Table")
        st.dataframe(patient_summary, use_container_width=True)

        glucose_col = "avg_glucose" if "avg_glucose" in patient_summary.columns else "mean_glucose"

        if glucose_col in patient_summary.columns:
            fig = px.bar(
                patient_summary.sort_values(glucose_col, ascending=False),
                x="patient_id",
                y=glucose_col,
                title="Average Glucose by Patient",
                color=glucose_col,
                color_continuous_scale="Turbo"
            )
            st.plotly_chart(style_plotly(fig), use_container_width=True)

    st.subheader("Selected Patient Glucose Timeline")

    patient_for_timeline = st.selectbox(
        "Choose one patient for timeline view",
        options=selected_patients if selected_patients else patient_options
    )

    patient_timeline = filtered_df[
        filtered_df["patient_id"] == patient_for_timeline
    ].sort_values("time")

    fig = px.line(
        patient_timeline,
        x="time",
        y="glucose",
        color="glucose_category",
        color_discrete_map=CATEGORY_COLORS,
        title=f"Glucose Timeline for {patient_for_timeline}"
    )
    st.plotly_chart(style_plotly(fig), use_container_width=True)


# ============================================================
# Tab 4: Prescriptive Analysis
# ============================================================

with tab4:
    st.header("Prescriptive Analysis")

    if not patient_risk.empty:
        st.subheader("Patient Risk Segmentation")

        st.dataframe(patient_risk, use_container_width=True)

        risk_col = "risk_segment" if "risk_segment" in patient_risk.columns else None

        if risk_col:
            risk_counts = (
                patient_risk[risk_col]
                .value_counts()
                .reset_index()
            )
            risk_counts.columns = ["risk_segment", "count"]

            fig = px.bar(
                risk_counts,
                x="risk_segment",
                y="count",
                title="Patient Risk Segment Distribution",
                color="risk_segment",
                color_discrete_sequence=VIBRANT_COLORS
            )
            st.plotly_chart(style_plotly(fig), use_container_width=True)

    if not hourly_risk.empty:
        st.subheader("Hourly Risk Recommendations")
        st.dataframe(hourly_risk, use_container_width=True)

        if "hour" in hourly_risk.columns:
            numeric_cols = hourly_risk.select_dtypes(include=np.number).columns.tolist()
            numeric_cols = [col for col in numeric_cols if col != "hour"]

            if numeric_cols:
                selected_risk_metric = st.selectbox(
                    "Select hourly risk metric",
                    options=numeric_cols
                )

                fig = px.line(
                    hourly_risk,
                    x="hour",
                    y=selected_risk_metric,
                    markers=True,
                    title=f"{selected_risk_metric} by Hour",
                    color_discrete_sequence=["#06D6A0"]
                )
                st.plotly_chart(style_plotly(fig), use_container_width=True)

    if not prescriptive_summary.empty:
        st.subheader("Prescriptive Dashboard Summary")
        st.dataframe(prescriptive_summary, use_container_width=True)


# ============================================================
# Tab 5: Predictive Analysis
# ============================================================

with tab5:
    st.header("Limited Predictive Analysis")

    if not predictive_eval.empty:
        st.subheader("Model Evaluation")

        st.dataframe(predictive_eval, use_container_width=True)

        metric_cols = [
            col for col in predictive_eval.columns
            if col.lower() in ["mae", "rmse", "r2", "r2_score"]
        ]

        if metric_cols:
            selected_metric = st.selectbox(
                "Select Evaluation Metric",
                options=metric_cols
            )

            model_col = "model" if "model" in predictive_eval.columns else predictive_eval.columns[0]

            fig = px.bar(
                predictive_eval,
                x=model_col,
                y=selected_metric,
                color=model_col,
                title=f"Model Comparison by {selected_metric}",
                color_discrete_sequence=VIBRANT_COLORS
            )
            st.plotly_chart(style_plotly(fig), use_container_width=True)

    if not actual_vs_pred.empty:
        st.subheader("Actual vs Predicted Glucose")

        st.dataframe(actual_vs_pred.head(1000), use_container_width=True)

        actual_col = "actual_glucose" if "actual_glucose" in actual_vs_pred.columns else None
        pred_col = "predicted_glucose" if "predicted_glucose" in actual_vs_pred.columns else None

        if actual_col and pred_col:
            fig = px.scatter(
                actual_vs_pred,
                x=actual_col,
                y=pred_col,
                title="Actual vs Predicted Glucose",
                color_discrete_sequence=["#FF4ECD"]
            )

            min_val = min(actual_vs_pred[actual_col].min(), actual_vs_pred[pred_col].min())
            max_val = max(actual_vs_pred[actual_col].max(), actual_vs_pred[pred_col].max())

            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode="lines",
                    name="Perfect Prediction"
                )
            )

            st.plotly_chart(style_plotly(fig), use_container_width=True)

    if not feature_importance.empty:
        st.subheader("Feature Importance")

        feature_col = "feature" if "feature" in feature_importance.columns else feature_importance.columns[0]
        importance_col = "importance" if "importance" in feature_importance.columns else feature_importance.columns[-1]

        top_features = feature_importance.sort_values(
            importance_col,
            ascending=False
        ).head(15)

        fig = px.bar(
            top_features,
            x=importance_col,
            y=feature_col,
            orientation="h",
            title="Top Predictive Features",
            color=importance_col,
            color_continuous_scale="Plasma"
        )

        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(style_plotly(fig), use_container_width=True)

    if not predictive_summary.empty:
        st.subheader("Predictive Dashboard Summary")
        st.dataframe(predictive_summary, use_container_width=True)


# ============================================================
# Tab 6: Data Explorer
# ============================================================

with tab6:
    st.header("Data Explorer")

    st.subheader("Filtered Main Dataset")
    st.dataframe(filtered_df, use_container_width=True)

    csv = filtered_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Filtered Dataset",
        data=csv,
        file_name="filtered_glucovision_data.csv",
        mime="text/csv"
    )

    st.subheader("Dataset Columns")
    st.write(filtered_df.columns.tolist())

    st.subheader("Summary Statistics")
    st.dataframe(filtered_df.describe(include="all"), use_container_width=True)