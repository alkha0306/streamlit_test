import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import plotly.express as px

st.set_page_config(page_title="Hackathon Dashboard", layout="wide")

st.title("Dataset & Table Insights Dashboard")


# Load data
import os
df = pd.read_csv(os.path.join(os.path.dirname(__file__), "merged_output_final.csv"))
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
kmeans = KMeans(n_clusters=5, random_state=42)
df['cluster'] = kmeans.fit_predict(features)

# # Charts (one per row)
# st.subheader("Cluster Distribution")
# cluster_counts = df['cluster'].value_counts().reset_index()
# cluster_counts.columns = ['cluster', 'count']
# st.plotly_chart(
#     px.bar(
#         cluster_counts,
#         x='cluster', y='count',
#         labels={'cluster': 'Cluster', 'count': 'Count'},
#         color='cluster',
#         color_discrete_sequence=px.colors.qualitative.Pastel,
#         title="Cluster Distribution"
#     )
# )

st.subheader("Paid vs Unpaid Distribution")
paid_counts = df['paid/unpaid'].value_counts().reset_index()
paid_counts.columns = ['paid/unpaid', 'count']
st.plotly_chart(
    px.pie(
        names=paid_counts['paid/unpaid'],
        values=paid_counts['count'],
        title="Paid/Unpaid",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
)

st.subheader("Datasets by Region")
region_counts = df['region'].value_counts().reset_index()
region_counts.columns = ['region', 'count']
st.plotly_chart(
    px.bar(
        region_counts,
        x='region', y='count',
        labels={'region': 'Region', 'count': 'Count'},
        color='region',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
)

st.subheader("Datasets by Category")
category_counts = df['category'].value_counts().reset_index()
category_counts.columns = ['category', 'count']
st.plotly_chart(
    px.bar(
        category_counts,
        x='category', y='count',
        labels={'category': 'Category', 'count': 'Count'},
        color='category',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
)

st.subheader("Granularity Distribution")
granularity_counts = df['granularity'].value_counts().reset_index()
granularity_counts.columns = ['granularity', 'count']
st.plotly_chart(
    px.bar(
        granularity_counts,
        x='granularity', y='count',
        labels={'granularity': 'Granularity', 'count': 'Count'},
        color='granularity',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
)

st.subheader("Being Used Distribution (Pie)")
being_used_counts = df['Being used'].value_counts().reset_index()
being_used_counts.columns = ['Being used', 'count']
st.plotly_chart(
    px.pie(
        names=being_used_counts['Being used'],
        values=being_used_counts['count'],
        title="Being Used",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
)

st.subheader("Utilised in Project Distribution")
utilised_counts = df['Utilised in Project'].value_counts().reset_index()
utilised_counts.columns = ['Utilised in Project', 'count']
st.plotly_chart(
    px.bar(
        utilised_counts,
        x='Utilised in Project', y='count',
        labels={'Utilised in Project': 'Utilised in Project', 'count': 'Count'},
        color='Utilised in Project',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
)

# Navigation button to search/cards page
st.markdown("---")
# st.markdown("### Go to the [Search & Cards Page](./1_Search_and_Cards)")
# st.button("Open Search & Cards Page", on_click=lambda: st.experimental_rerun())