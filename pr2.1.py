from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap

from IPython.display import display

import kagglehub

from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
)
from sklearn.preprocessing import StandardScaler


# =========================
# SETTINGS
# =========================

plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.35

np.random.seed(42)


# =========================
# LOAD DATASET
# =========================

path = kagglehub.dataset_download(
    "shambhurajejagadale/student-performance-prediction-dataset"
)

print("Path to dataset files:", path)

data_path = Path(path)

csv_files = list(data_path.glob("*.csv"))

print("CSV files:", csv_files)

df = pd.read_csv(csv_files[0])

print("\nFIRST ROWS")
display(df.head())

print("\nDATASET SHAPE")
print(df.shape)


# =========================
# PREPROCESSING
# =========================

print("\nINFO")
print(df.info())

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nDUPLICATES:", df.duplicated().sum())

# remove duplicates
df = df.drop_duplicates()

# remove missing values
df = df.dropna()

print("\nSHAPE AFTER CLEANING")
print(df.shape)


# =========================
# ENCODE CATEGORICAL FEATURES
# =========================

categorical_columns = df.select_dtypes(
    include=["object", "string"]
).columns

print("\nCATEGORICAL COLUMNS")
print(categorical_columns)

for column in categorical_columns:
    df[column] = df[column].astype("category").cat.codes

print("\nENCODED DATA")
display(df.head())


# =========================
# CORRELATION ANALYSIS
# =========================

corr = df.corr(numeric_only=True)

plt.figure(figsize=(14, 10))

sns.heatmap(
    corr,
    cmap="coolwarm",
    annot=True,
    fmt=".2f"
)

plt.title("Correlation Matrix")

plt.show()


# =========================
# FEATURE SELECTION
# =========================

cluster_features = [
    "study_hours",
    "attendance",
    "sleep_hours",
    "internet_usage",
    "assignments_completed",
    "previous_score",
    "exam_score"
]

cluster_frame = df[cluster_features].copy()

print("\nFEATURE STATISTICS")
display(cluster_frame.describe().round(2))


# =========================
# FEATURE VISUALIZATION
# =========================

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

axes[0].scatter(
    df["study_hours"],
    df["exam_score"],
    alpha=0.7
)

axes[0].set_title("Study Hours vs Exam Score")
axes[0].set_xlabel("Study Hours")
axes[0].set_ylabel("Exam Score")

axes[1].scatter(
    df["attendance"],
    df["previous_score"],
    alpha=0.7
)

axes[1].set_title("Attendance vs Previous Score")
axes[1].set_xlabel("Attendance")
axes[1].set_ylabel("Previous Score")

plt.tight_layout()
plt.show()


# =========================
# SCALING
# =========================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(cluster_frame)


# =========================
# PCA VISUALIZATION
# =========================

pca = PCA(
    n_components=2,
    random_state=42
)

coords = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))

plt.scatter(
    coords[:, 0],
    coords[:, 1],
    alpha=0.7
)

plt.title("PCA Projection")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")

plt.show()


# =========================
# KMEANS QUALITY SEARCH
# =========================

quality_rows = []

for k in range(2, 11):

    model = KMeans(
        n_clusters=k,
        n_init=20,
        random_state=42
    )

    labels = model.fit_predict(X_scaled)

    quality_rows.append({
        "k": k,
        "inertia": round(model.inertia_, 2),
        "silhouette": round(
            silhouette_score(X_scaled, labels),
            4
        ),
        "davies_bouldin": round(
            davies_bouldin_score(X_scaled, labels),
            4
        )
    })

quality_df = pd.DataFrame(quality_rows)

print("\nKMEANS QUALITY")
display(quality_df)


# =========================
# ELBOW METHOD
# =========================

fig, axes = plt.subplots(1, 3, figsize=(20, 5))

axes[0].plot(
    quality_df["k"],
    quality_df["inertia"],
    marker="o"
)

axes[0].set_title("Elbow Method")
axes[0].set_xlabel("Clusters k")
axes[0].set_ylabel("Inertia")

axes[1].plot(
    quality_df["k"],
    quality_df["silhouette"],
    marker="o"
)

axes[1].set_title("Silhouette Score")
axes[1].set_xlabel("Clusters k")

axes[2].plot(
    quality_df["k"],
    quality_df["davies_bouldin"],
    marker="o"
)

axes[2].set_title("Davies-Bouldin Score")
axes[2].set_xlabel("Clusters k")

plt.tight_layout()
plt.show()


# =========================
# KMEANS CLUSTERING
# =========================

kmeans = KMeans(
    n_clusters=3,
    n_init=20,
    random_state=42
)

df["cluster_kmeans"] = kmeans.fit_predict(X_scaled)

print("\nKMEANS LABELS")
print(df["cluster_kmeans"].value_counts())


# =========================
# KMEANS VISUALIZATION
# =========================

plt.figure(figsize=(10, 6))

scatter = plt.scatter(
    coords[:, 0],
    coords[:, 1],
    c=df["cluster_kmeans"],
    cmap="tab10",
    alpha=0.8
)

plt.colorbar(scatter)

plt.title("KMeans Clusters")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")

plt.show()


# =========================
# KMEANS CLUSTER PROFILE
# =========================

cluster_profile = (
    df.groupby("cluster_kmeans")[cluster_features]
    .mean()
    .round(2)
)

print("\nKMEANS CLUSTER PROFILE")
display(cluster_profile)


# =========================
# TSNE VISUALIZATION
# =========================

tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate="auto",
    init="pca",
    random_state=42
)

tsne_coords = tsne.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))

scatter = plt.scatter(
    tsne_coords[:, 0],
    tsne_coords[:, 1],
    c=df["cluster_kmeans"],
    cmap="tab10",
    alpha=0.8
)

plt.colorbar(scatter)

plt.title("KMeans t-SNE Projection")

plt.show()


# =========================
# UMAP VISUALIZATION
# =========================

reducer = umap.UMAP(
    n_components=2,
    n_neighbors=20,
    min_dist=0.1,
    random_state=42
)

umap_coords = reducer.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))

scatter = plt.scatter(
    umap_coords[:, 0],
    umap_coords[:, 1],
    c=df["cluster_kmeans"],
    cmap="tab10",
    alpha=0.8
)

plt.colorbar(scatter)

plt.title("KMeans UMAP Projection")

plt.show()


# =========================
# DBSCAN PARAMETER SEARCH
# =========================

dbscan_results = []

for eps in np.arange(0.5, 3.1, 0.1):

    for min_samples in range(3, 10):

        model = DBSCAN(
            eps=eps,
            min_samples=min_samples
        )

        labels = model.fit_predict(X_scaled)

        unique_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        if unique_clusters < 2:
            continue

        mask = labels != -1

        if mask.sum() < 10:
            continue

        try:

            silhouette = silhouette_score(
                X_scaled[mask],
                labels[mask]
            )

            db_score = davies_bouldin_score(
                X_scaled[mask],
                labels[mask]
            )

            noise_percent = (
                (labels == -1).sum() / len(labels)
            ) * 100

            dbscan_results.append({
                "eps": round(eps, 2),
                "min_samples": min_samples,
                "clusters": unique_clusters,
                "noise_percent": round(noise_percent, 2),
                "silhouette": round(silhouette, 4),
                "davies_bouldin": round(db_score, 4)
            })

        except:
            pass


dbscan_df = pd.DataFrame(dbscan_results)

dbscan_df = dbscan_df.sort_values(
    ["davies_bouldin", "silhouette"],
    ascending=[True, False]
)

print("\nBEST DBSCAN PARAMETERS")
display(dbscan_df.head(10))


# =========================
# BEST DBSCAN MODEL
# =========================

best_row = dbscan_df.iloc[0]

best_eps = float(best_row["eps"])
best_min_samples = int(best_row["min_samples"])

print("\nBEST EPS:", best_eps)
print("BEST MIN SAMPLES:", best_min_samples)

dbscan = DBSCAN(
    eps=best_eps,
    min_samples=best_min_samples
)

df["cluster_dbscan"] = dbscan.fit_predict(X_scaled)

print("\nDBSCAN LABELS")
print(df["cluster_dbscan"].value_counts())


# =========================
# DBSCAN VISUALIZATION
# =========================

plt.figure(figsize=(10, 6))

scatter = plt.scatter(
    coords[:, 0],
    coords[:, 1],
    c=df["cluster_dbscan"],
    cmap="tab10",
    alpha=0.8
)

plt.colorbar(scatter)

plt.title("DBSCAN Clusters")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")

plt.show()


# =========================
# DBSCAN METRICS
# =========================

mask = df["cluster_dbscan"] != -1

labels_valid = df.loc[mask, "cluster_dbscan"]

X_valid = X_scaled[mask]

dbscan_silhouette = silhouette_score(
    X_valid,
    labels_valid
)

dbscan_db = davies_bouldin_score(
    X_valid,
    labels_valid
)

noise_points = (df["cluster_dbscan"] == -1).sum()

noise_percent = (
    noise_points / len(df)
) * 100

print("\nDBSCAN METRICS")

print("Silhouette:", round(dbscan_silhouette, 4))

print("Davies-Bouldin:", round(dbscan_db, 4))

print("Noise points:", noise_points)

print("Noise percent:", round(noise_percent, 2), "%")


# =========================
# DBSCAN CLUSTER PROFILE
# =========================

dbscan_profile = (
    df[df["cluster_dbscan"] != -1]
    .groupby("cluster_dbscan")[cluster_features]
    .mean()
    .round(2)
)

print("\nDBSCAN CLUSTER PROFILE")
display(dbscan_profile)


# =========================
# METHOD COMPARISON
# =========================

comparison_df = pd.DataFrame([
    {
        "method": "KMeans",
        "silhouette": round(
            silhouette_score(
                X_scaled,
                df["cluster_kmeans"]
            ),
            4
        ),
        "davies_bouldin": round(
            davies_bouldin_score(
                X_scaled,
                df["cluster_kmeans"]
            ),
            4
        ),
        "noise_percent": 0
    },
    {
        "method": "DBSCAN",
        "silhouette": round(
            dbscan_silhouette,
            4
        ),
        "davies_bouldin": round(
            dbscan_db,
            4
        ),
        "noise_percent": round(
            noise_percent,
            2
        )
    }
])

print("\nMETHOD COMPARISON")
display(comparison_df)


# =========================
# COMPARISON VISUALIZATION
# =========================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].bar(
    comparison_df["method"],
    comparison_df["silhouette"]
)

axes[0].set_title("Silhouette Score")

axes[1].bar(
    comparison_df["method"],
    comparison_df["davies_bouldin"]
)

axes[1].set_title("Davies-Bouldin Score")

axes[2].bar(
    comparison_df["method"],
    comparison_df["noise_percent"]
)

axes[2].set_title("Noise Percent")

plt.tight_layout()
plt.show()


# =========================
# FINAL CONCLUSION
# =========================

best_method = comparison_df.sort_values(
    "davies_bouldin"
).iloc[0]

print("\nBEST METHOD")

print(
    f"Method: {best_method['method']}\n"
    f"Davies-Bouldin: {best_method['davies_bouldin']}\n"
    f"Silhouette: {best_method['silhouette']}"
)