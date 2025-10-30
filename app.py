import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

st.markdown("""
<div style='text-align:center; padding:2rem 0;'>
    <h1 style='color:#2563eb; font-size:2.5rem;'>🔋 Material Dataset Explorer</h1>
    <p style='color:#555; font-size:1.1rem;'>
        Discover insights, optimize materials, and visualize trade-offs in next-generation battery research.
    </p>
    <a href='https://github.com/PraiseAgboola/material-dataset-explorer1' target='_blank' style='text-decoration:none;'>
        <button style='background-color:#2563eb;color:white;border:none;padding:0.7rem 1.5rem;border-radius:8px;cursor:pointer;'>
            ⭐ View on GitHub
        </button>
    </a>
</div>
""", unsafe_allow_html=True)


# Page configuration
st.set_page_config(page_title="Materials Dataset Explorer", layout="wide", page_icon="🔋")

# Custom CSS
st.markdown("""
    <style>
    .main-header {font-size: 2.5rem; font-weight: bold; color: #1f77b4;}
    .sub-header {font-size: 1.2rem; color: #666;}
    col1, col2 = st.columns([3, 1])
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header"> Materials Dataset Explorer</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Analyze, Compare, and Optimize Material Selection</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("📊 Data Controls")

# Sample data generator (replace with your Kaggle dataset)
@st.cache_data
def load_sample_data():
    """Generate sample battery materials data - replace with actual Kaggle CSV"""
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

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload your CSV", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ Dataset loaded!")
else:
    df = load_sample_data()
    st.sidebar.info("📝 Using sample data. Upload your dataset to begin.")

# Display dataset info
st.sidebar.metric("Total Materials", len(df))
st.sidebar.metric("Properties Tracked", len(df.columns) - 1)

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs(["📋 Data Explorer", "📊 Comparative Analysis", "🎯 Material Selector", "📈 Insights"])

# TAB 1: Data Explorer
with tab1:
    st.header("Dataset Overview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.dataframe(df, use_container_width=True, height=400)
    
    with col2:
        st.subheader("Quick Stats")
        # Get numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            selected_stat_col = st.selectbox("Select property:", numeric_cols)
            st.metric("Mean", f"{df[selected_stat_col].mean():.2f}")
            st.metric("Max", f"{df[selected_stat_col].max():.2f}")
            st.metric("Min", f"{df[selected_stat_col].min():.2f}")
            st.metric("Std Dev", f"{df[selected_stat_col].std():.2f}")
    
    # Correlation heatmap
    st.subheader("Property Correlations")
    numeric_df = df.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', center=0, ax=ax)
        plt.title("Material Properties Correlation Matrix")
        st.pyplot(fig)
    else:
        st.warning("Not enough numeric columns for correlation analysis")

# TAB 2: Comparative Analysis
with tab2:
    st.header("Material Comparison")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) >= 2:
        col1, col2 = st.columns(2)
        
        with col1:
            x_axis = st.selectbox("X-axis property:", numeric_cols, index=0)
        with col2:
            y_axis = st.selectbox("Y-axis property:", numeric_cols, index=min(1, len(numeric_cols)-1))
        
        # Scatter plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Use first column as labels if it's text
        label_col = df.columns[0] if df[df.columns[0]].dtype == 'object' else None
        
        if label_col:
            scatter = ax.scatter(df[x_axis], df[y_axis], s=200, alpha=0.6, c=range(len(df)), cmap='viridis')
            
            for idx, row in df.iterrows():
                ax.annotate(row[label_col], (row[x_axis], row[y_axis]), 
                           fontsize=9, ha='center', va='bottom')
        else:
            scatter = ax.scatter(df[x_axis], df[y_axis], s=200, alpha=0.6)
        
        ax.set_xlabel(x_axis, fontsize=12)
        ax.set_ylabel(y_axis, fontsize=12)
        ax.set_title(f"{y_axis} vs {x_axis}", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        # Multi-property bar chart
        st.subheader("Multi-Property Comparison")
        
        selected_props = st.multiselect(
            "Select properties to compare:",
            numeric_cols,
            default=numeric_cols[:min(3, len(numeric_cols))]
        )
        
        if selected_props and label_col:
            # Normalize data for comparison
            df_norm = df.copy()
            for prop in selected_props:
                df_norm[f'{prop}_norm'] = (df[prop] - df[prop].min()) / (df[prop].max() - df[prop].min())
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            x = np.arange(len(df))
            width = 0.8 / len(selected_props)
            
            for i, prop in enumerate(selected_props):
                offset = width * i - (width * len(selected_props) / 2)
                ax.bar(x + offset, df_norm[f'{prop}_norm'], width, label=prop, alpha=0.8)
            
            ax.set_xlabel('Materials', fontsize=12)
            ax.set_ylabel('Normalized Value (0-1)', fontsize=12)
            ax.set_title('Normalized Property Comparison', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(df[label_col], rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            st.pyplot(fig)
    else:
        st.warning("Dataset needs at least 2 numeric columns for comparison")

# TAB 3: Material Selector
with tab3:
    st.header("Smart Material Selection")
    st.write("Set your requirements and find the best matching materials")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if numeric_cols:
        # Weight sliders
        st.subheader("Property Importance Weights")
        weights = {}
        
        cols = st.columns(3)
        for idx, col in enumerate(numeric_cols):
            with cols[idx % 3]:
                weights[col] = st.slider(f"{col}", 0.0, 1.0, 0.5, 0.1)
        
        # Calculate weighted scores
        df_scored = df.copy()
        
        # Normalize each property
        for col in numeric_cols:
            col_min = df[col].min()
            col_max = df[col].max()
            if col_max != col_min:
                df_scored[f'{col}_norm'] = (df[col] - col_min) / (col_max - col_min)
            else:
                df_scored[f'{col}_norm'] = 0.5
        
        # Calculate weighted score
        df_scored['Total_Score'] = 0
        for col in numeric_cols:
            df_scored['Total_Score'] += df_scored[f'{col}_norm'] * weights[col]
        
        # Normalize total score
        total_weight = sum(weights.values())
        if total_weight > 0:
            df_scored['Total_Score'] = df_scored['Total_Score'] / total_weight * 100
        
        # Sort by score
        df_scored = df_scored.sort_values('Total_Score', ascending=False)
        
        # Display results
        st.subheader("🏆 Ranking Results")
        
        # Show ranking with first column as identifier
        label_col = df.columns[0]
        result_df = df_scored[[label_col, 'Total_Score']].copy()
        result_df['Rank'] = range(1, len(result_df) + 1)
        result_df = result_df[['Rank', label_col, 'Total_Score']]
        result_df['Total_Score'] = result_df['Total_Score'].round(2)
        
        st.dataframe(result_df, use_container_width=True, height=300)
        
        # Visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(df_scored[label_col], df_scored['Total_Score'], color=plt.cm.viridis(np.linspace(0, 1, len(df_scored))))
        ax.set_xlabel('Weighted Score', fontsize=12)
        ax.set_title('Material Rankings Based on Your Criteria', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, df_scored['Total_Score'])):
            ax.text(score + 1, i, f'{score:.1f}', va='center', fontsize=10)
        
        st.pyplot(fig)
    else:
        st.warning("No numeric columns available for scoring")

# TAB 4: Insights
with tab4:
    st.header("Dataset Insights & Analytics")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if numeric_cols:
        # Distribution plots
        st.subheader("Property Distributions")
        
        selected_dist = st.selectbox("Select property to analyze:", numeric_cols)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(df[selected_dist], bins=15, color='skyblue', edgecolor='black', alpha=0.7)
            ax.set_xlabel(selected_dist, fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title(f'Distribution of {selected_dist}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.boxplot(df[selected_dist], vert=True)
            ax.set_ylabel(selected_dist, fontsize=12)
            ax.set_title(f'Box Plot - {selected_dist}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            st.pyplot(fig)
        
        # Summary statistics
        st.subheader("Statistical Summary")
        st.dataframe(df[numeric_cols].describe(), use_container_width=True)
        
        # Pareto frontier (if applicable)
        if len(numeric_cols) >= 2:
            st.subheader("Pareto Frontier Analysis")
            st.write("Identify materials that represent optimal trade-offs")
            
            col1, col2 = st.columns(2)
            with col1:
                pareto_x = st.selectbox("Property to maximize (X):", numeric_cols, key='pareto_x')
            with col2:
                pareto_y = st.selectbox("Property to maximize (Y):", numeric_cols, key='pareto_y', index=min(1, len(numeric_cols)-1))
            
            # Simple Pareto frontier identification
            df_pareto = df.copy()
            is_pareto = np.ones(len(df_pareto), dtype=bool)
            
            for i, row_i in df_pareto.iterrows():
                for j, row_j in df_pareto.iterrows():
                    if i != j:
                        if (row_j[pareto_x] >= row_i[pareto_x] and row_j[pareto_y] >= row_i[pareto_y] and
                            (row_j[pareto_x] > row_i[pareto_x] or row_j[pareto_y] > row_i[pareto_y])):
                            is_pareto[i] = False
                            break
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot all points
            ax.scatter(df_pareto[pareto_x], df_pareto[pareto_y], s=100, alpha=0.5, label='All Materials', color='lightblue')
            
            # Highlight Pareto frontier
            pareto_points = df_pareto[is_pareto]
            ax.scatter(pareto_points[pareto_x], pareto_points[pareto_y], s=200, alpha=0.9, 
                      label='Pareto Optimal', color='red', edgecolors='black', linewidth=2)
            
            # Annotate
            label_col = df.columns[0] if df[df.columns[0]].dtype == 'object' else None
            if label_col:
                for idx, row in pareto_points.iterrows():
                    ax.annotate(row[label_col], (row[pareto_x], row[pareto_y]), 
                               fontsize=9, ha='center', va='bottom')
            
            ax.set_xlabel(pareto_x, fontsize=12)
            ax.set_ylabel(pareto_y, fontsize=12)
            ax.set_title(f'Pareto Frontier: {pareto_y} vs {pareto_x}', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            
            st.info(f"🎯 {sum(is_pareto)} materials are on the Pareto frontier (optimal trade-offs)")
    else:
        st.warning("No numeric data available for insights")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 About")
st.sidebar.info("""
This MVP tool helps you explore material datasets with:
- Interactive data exploration
- Multi-property comparisons
- Smart material selection
- Statistical insights
""")

st.sidebar.markdown("### 💡 Tips")
st.sidebar.markdown("""
- Upload your Kaggle CSV in the sidebar
- Ensure first column is material names
- Numeric columns = properties to analyze
- Use sliders to weight importance
""")
