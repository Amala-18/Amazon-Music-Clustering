import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score,davies_bouldin_score
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

st.set_page_config(page_title="Amazon Music Cluster", layout="wide", page_icon="Music")

st.title(":green[Amazon Music Clustering Dashboard]")

@st.cache_data
def load_data():
    df = pd.read_csv("C:/Users/Sri Advikam/Desktop/Amala/Amazon Music Clustering/single_genre_artists.csv")

    #Drop unnecessary columns
    df = df.drop(columns=['id_songs','name_song','name_artists','id_artists','genres'])
    df = df.drop(columns=['key','mode','release_date'])
    df = df.drop_duplicates()
    return df

df = load_data()
st.sidebar.header("Controls")

# Sidebar Controls
scaler_option = st.sidebar.selectbox("Choose Scaling Method", ["StandardScaler", "MinMaxScaler", "None"])
k = st.sidebar.slider("Number of clusters (k)", 2, 10, 5)
show_heatmap = st.sidebar.checkbox("Show Cluster Heatmap", value=True)
download = st.sidebar.checkbox("Enable CSV Download")

numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
X = df[numeric_features].copy()

# ---- Scaling ----
if scaler_option == "StandardScaler":
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
elif scaler_option == "MinMaxScaler":
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
else:
    X_scaled = X.values

kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)
df["Cluster"] = labels

#Finding out Metrics
sil_score = silhouette_score(X_scaled, labels)
db_index = davies_bouldin_score(X_scaled, labels)

st.subheader(" Clustering Evaluation Metrics")
col1, col2 = st.columns(2)
col1.metric("Silhouette Score", f"{sil_score:.3f}")
col2.metric("Davies-Bouldin Index", f"{db_index:.3f}")

# ---- Cluster Profiling ----
st.subheader(" Cluster Profiling")
profile = df.groupby("Cluster")[numeric_features].mean()

if show_heatmap:
    fig, ax = plt.subplots(figsize=(14,6))
    sns.heatmap(profile, annot=True, fmt=".2f", cmap="Greens")
    plt.title("Cluster Feature Heatmap")
    st.pyplot(fig)

st.write("### Cluster Summary Table")
st.dataframe(profile)


# ---- Feature Distribution ----
st.subheader(" Feature Distribution by Cluster")
feature_choice = st.selectbox("Select Feature to Compare", numeric_features)
fig, ax = plt.subplots()
sns.boxplot(x="Cluster", y=feature_choice, data=df, ax=ax, palette="Set2")
st.pyplot(fig)

if download:
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Clustered Data", csv, "amazon_music_clusters.csv", "text/csv")