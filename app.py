import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Material Dataset Explorer", layout="wide")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")
    return df

df = load_data()

# Title
st.title("🌍 Material Dataset Explorer")
st.write("Explore and visualize material demand trends interactively.")

# Show Data Preview
st.write("### Dataset Preview")
st.dataframe(df.head())

# Basic Visualization
st.write("### Simple Material Demand Plot")
if "Material" in df.columns and "Year" in df.columns and "Demand" in df.columns:
    fig, ax = plt.subplots()
    for material, group in df.groupby("Material"):
        ax.plot(group["Year"], group["Demand"], label=material)
    ax.legend()
    st.pyplot(fig)
else:
    st.warning("Make sure your data has 'Material', 'Year', and 'Demand' columns.")
