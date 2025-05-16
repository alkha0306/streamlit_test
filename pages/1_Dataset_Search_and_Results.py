import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

st.title("Correlated datasets and tables")

# Load data
df = pd.read_csv('merged_output_final.csv')
df = df.fillna('')

# Encode categorical columns
encode_cols = [
    'granularity', 'paid/unpaid', 'region', 'hierarchy', 'category',
    'Being used', 'Utilised in Project'
]
for col in encode_cols:
    df[col] = df[col].astype(str)
    df[col + '_enc'] = LabelEncoder().fit_transform(df[col])

# Clustering
features = df[[col + '_enc' for col in encode_cols]]
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(features)

# Sidebar filters
region = st.sidebar.selectbox('Region', ['All'] + sorted(df['region'].unique()))
category = st.sidebar.selectbox('Category', ['All'] + sorted(df['category'].unique()))
being_used = st.sidebar.selectbox('Being used', ['All'] + sorted(df['Being used'].unique()))
utilised = st.sidebar.selectbox('Utilised in Project', ['All'] + sorted(df['Utilised in Project'].unique()))

# Filtered data
filtered = df.copy()
if region != 'All':
    filtered = filtered[filtered['region'] == region]
if category != 'All':
    filtered = filtered[filtered['category'] == category]
if being_used != 'All':
    filtered = filtered[filtered['Being used'] == being_used]
if utilised != 'All':
    filtered = filtered[filtered['Utilised in Project'] == utilised]

st.subheader("Dataset and Table Information")

# Display 3 cards per row using Streamlit columns and HTML for card effect
card_style = """
    <div style="
        background-color: #fff;
        border-radius: 12px;
        border: 1.5px solid #e0e4f0;
        padding: 18px 16px 12px 16px;
        margin-bottom: 18px;
        box-shadow: 0 2px 8px rgba(44,56,126,0.07);
        min-height: 270px;
        max-height: 340px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        font-family: 'Segoe UI', 'Arial', sans-serif;
        transition: box-shadow 0.2s;
        overflow: hidden;
    ">
        <div style="flex:1; overflow-wrap: break-word; word-break: break-all;">
            <h5 style="color:#1a237e; margin-bottom:8px; font-size: clamp(1rem, 2vw, 1.1rem); overflow-wrap: break-word; word-break: break-all;">{dataset_name}</h5>
            <div style="font-size: clamp(13px, 1.8vw, 15px); color:#333; overflow-wrap: break-word; word-break: break-all;">
                <b>Table:</b> {table_name}<br>
                <b>Granularity:</b> {granularity}<br>
                <b>Paid/Unpaid:</b> {paid_unpaid}<br>
                <b>Region:</b> {region}<br>
                <b>Hierarchy:</b> {hierarchy}<br>
                <b>Category:</b> {category}<br>
                <b>DateTime Column:</b> {dateTimeColumn}<br>
            </div>
        </div>
        <div style="font-size: 13px; color:#444; margin-top:10px; overflow-wrap: break-word; word-break: break-all;">
            <b>Being Used:</b> {being_used} &nbsp;|&nbsp;
            <b>Utilised:</b> {utilised} &nbsp;|&nbsp;
            <b>Cluster:</b> {cluster}
        </div>
    </div>
"""

rows = list(filtered.iterrows())
for i in range(0, len(rows), 3):
    cols = st.columns(3)
    for j, (idx, row) in enumerate(rows[i:i+3]):
        with cols[j]:
            st.markdown(
                card_style.format(
                    dataset_name=row.get('dataset_name', ''),
                    table_name=row.get('table_name', ''),
                    granularity=row.get('granularity', ''),
                    paid_unpaid=row.get('paid/unpaid', ''),
                    region=row.get('region', ''),
                    hierarchy=row.get('hierarchy', ''),
                    category=row.get('category', ''),
                    dateTimeColumn=row.get('dateTimeColumn', ''),
                    being_used=row.get('Being used', ''),
                    utilised=row.get('Utilised in Project', ''),
                    cluster=row.get('cluster', ''),
                ),
                unsafe_allow_html=True
            )