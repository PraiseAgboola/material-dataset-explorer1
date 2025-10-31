# ================================================================
# 🌍 MATERIAL DATASET EXPLORER
# A polished, interactive dashboard for analyzing material properties
# Built by Praise Agboola | https://github.com/PraiseAgboola
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------------------------------
# Page Configuration
# ------------------------------------------------
st.set_page_config(
    page_title="Material Dataset Explorer",
    layout="wide"
)

# ------------------------------------------------
# Header (Hero Section)
# ------------------------------------------------
st.markdown("""
<div style='text-align:center; padding:1.5rem 0;'>
    <h1 style='color:#2563eb; font-size:2.5rem; font-weight:700;'> Material Dataset Explorer</h1>
    <p style='color:#475569; font-size:1.1rem;'>
        Discover insights, optimize material choices, and visualize trade-offs 
        in next-generation energy materials.
    </p>
    <a href='https://github.com/PraiseAgboola/material-dataset-explorer1' target='_blank'>
        <button style='background-color:#2563eb;color:white;border:none;
        padding:0.7rem 1.5rem;border-radius:8px;cursor:pointer;font-size:1rem;'>
            View Project on GitHub
        </button>
    </a>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------
# Load Data
# ------------------------------------------------
@st.cache_data
def load_sample_data():
    np.random.seed(42)
    materials = ['LFP', 'NMC-811', 'NMC-622', 'NCA', 'LCO', 'LMO', 'LNMO']
    data = {
        'Material': materials,
        'Energy_Density_Wh_kg': [150, 250, 200, 240, 190, 120, 160],
        'Cycle_Life': [3000, 1500, 2000, 1200, 1000, 2500, 2000],
        'Cost_per_kWh_USD': [80, 140, 120, 150, 130, 90, 110],
        'Thermal_Stability_C': [270, 210, 230, 180, 195, 250, 240],
        'Safety_Rating': [9.5, 6.5, 7.5, 6.0, 6.5, 8.0, 7.5],
        'Cobalt_Content_pct': [0, 80, 60, 80, 100, 0, 0],
        'Voltage_V': [3.2, 3.8, 3.7, 3.7, 3.9, 3.8, 4.7]
    }
    return pd.DataFrame(data)

uploaded_file = st.sidebar.file_uploader("📂 Upload your CSV dataset", type=['csv'])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ Dataset loaded successfully")
else:
    df = load_sample_data()
    st.sidebar.info("Using sample dataset. Upload your own to customize.")

# ------------------------------------------------
# Sidebar Metrics
# ------------------------------------------------
st.sidebar.markdown("### 📊 Dataset Summary")
st.sidebar.metric("Total Materials", len(df))
st.sidebar.metric("Tracked Properties", len(df.columns) - 1)

# ------------------------------------------------
# Tabs
# ------------------------------------------------
tabs = st.tabs([
    "📋 Data Explorer",
    "📊 Comparative Analysis",
    "🎯 Material Selector",
    "📈 Insights"
])

# ------------------------------------------------
# TAB 1 — Data Explorer
# ------------------------------------------------
with tabs[0]:
    st.subheader("Dataset Overview")
    col1, col2 = st.columns([3, 1])

    with col1:
        st.dataframe(df, use_container_width=True, height=400)

    with col2:
        st.subheader("Quick Stats")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            selected_col = st.selectbox("Select property:", numeric_cols)
            st.metric("Mean", f"{df[selected_col].mean():.2f}")
            st.metric("Max", f"{df[selected_col].max():.2f}")
            st.metric("Min", f"{df[selected_col].min():.2f}")

    st.divider()
    st.subheader("📉 Property Correlation Matrix")
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 1:
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', center=0, ax=ax)
        st.pyplot(fig)
    else:
        st.info("Add more numeric properties for correlation analysis.")

# ------------------------------------------------
# TAB 2 — Comparative Analysis
# ------------------------------------------------
with tabs[1]:
    st.subheader("Property Comparison")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) >= 2:
        x_axis = st.selectbox("X-axis:", numeric_cols, index=0)
        y_axis = st.selectbox("Y-axis:", numeric_cols, index=1)

        fig = px.scatter(
            df, x=x_axis, y=y_axis, color="Material",
            size_max=20, hover_name="Material",
            title=f"{y_axis} vs {x_axis}",
            color_discrete_sequence=px.colors.qualitative.Vivid
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🔹 Multi-Property Comparison")
        selected_props = st.multiselect(
            "Select properties:", numeric_cols, default=numeric_cols[:3]
        )

        if selected_props:
            df_norm = df.copy()
            for prop in selected_props:
                df_norm[f'{prop}_norm'] = (df[prop] - df[prop].min()) / (df[prop].max() - df[prop].min())

            df_melted = df_norm.melt(id_vars=['Material'], 
                                     value_vars=[f"{p}_norm" for p in selected_props], 
                                     var_name="Property", value_name="Normalized Value")

            fig = px.bar(df_melted, x="Material", y="Normalized Value",
                         color="Property", barmode="group",
                         title="Normalized Multi-Property Comparison")
            st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------
# TAB 3 — Material Selector
# ------------------------------------------------
with tabs[2]:
    st.subheader("Smart Material Selector")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        st.markdown("Adjust the sliders below to weight each property based on importance.")
        weights = {col: st.slider(col, 0.0, 1.0, 0.5, 0.1) for col in numeric_cols}

        df_score = df.copy()
        for col in numeric_cols:
            col_min, col_max = df[col].min(), df[col].max()
            df_score[f'{col}_norm'] = (df[col] - col_min) / (col_max - col_min) if col_max > col_min else 0.5

        df_score['Total_Score'] = sum(df_score[f'{col}_norm'] * w for col, w in weights.items())
        df_score['Rank'] = df_score['Total_Score'].rank(ascending=False)
        st.dataframe(df_score[['Material', 'Total_Score', 'Rank']].sort_values('Rank'), use_container_width=True)

        fig = px.bar(df_score.sort_values('Total_Score', ascending=True),
                     x='Total_Score', y='Material', orientation='h',
                     color='Total_Score', color_continuous_scale='viridis',
                     title='Material Rankings by Weighted Score')
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------
# TAB 4 — Insights
# ------------------------------------------------
with tabs[3]:
    st.subheader("Statistical Insights")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if numeric_cols:
        selected_col = st.selectbox("Select property to analyze:", numeric_cols)
        fig = px.histogram(df, x=selected_col, nbins=15, title=f"Distribution of {selected_col}",
                           color_discrete_sequence=["#2563eb"])
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Summary Statistics")
        st.dataframe(df[numeric_cols].describe().T, use_container_width=True)

# ------------------------------------------------
# Footer
# ------------------------------------------------
st.markdown("""
---

""", unsafe_allow_html=True)
