import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Global Land Temperatures", layout="wide")

st.title("🌍 Global Land Temperatures Dashboard")

# Sidebar
st.sidebar.header("Settings")

dataset = st.sidebar.selectbox("Choose Dataset:", ["city", "country", "major_city", "state"])

file_path = os.path.join('processed_data', f'{dataset}.csv')

# Check if file exists
if not os.path.exists(file_path):
    st.error(f"Processed file '{dataset}.csv' not found! Please run preprocessing.py first.")
    st.stop()

# Load data
df = pd.read_csv(file_path)

# Fix: If 'Year' column missing, recreate it
if 'Year' not in df.columns:
    df['dt'] = pd.to_datetime(df['dt'], errors='coerce')
    df['Year'] = df['dt'].dt.year

# Show dataframe
st.subheader(f"Showing Data: {dataset.capitalize()}")
st.dataframe(df, use_container_width=True)

# Basic stats
st.subheader("Summary Statistics")
st.write(df.describe())

# Temperature Trend Over Years
st.subheader("Temperature Trend Over Years")
temp_by_year = df.groupby('Year')['AverageTemperature'].mean()

st.line_chart(temp_by_year)

# Show saved static plot if exists
plot_path = os.path.join('processed_data', f'{dataset}_plot.png')
if os.path.exists(plot_path):
    st.subheader("Saved Static Plot")
    st.image(plot_path, caption=f"Temperature Over Years ({dataset.capitalize()})", use_column_width=True)
else:
    st.warning("No static plot available. Please check preprocessing.")
