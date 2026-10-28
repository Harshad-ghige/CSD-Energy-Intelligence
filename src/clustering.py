"""
CSD Energy Intelligence - Usage Patterns & KMeans Clustering
Implements behavioral pattern clustering using StandardScaler, KMeans, and PCA.
Provides transparent cluster profiling and 2D projections.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def run_room_clustering(
    room_features_df: pd.DataFrame,
    n_clusters: int = 3,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Applies StandardScaler, KMeans clustering, and PCA 2D reduction on room behavioral features.
    Features: [mean_kwh, peak_kwh, median_kwh, night_ratio, weekend_ratio, working_mean_kwh, peak_to_avg_ratio]
    """
    if room_features_df.empty:
        return pd.DataFrame(), pd.DataFrame(), {}

    clustering_cols = [
        "mean_kwh",
        "peak_kwh",
        "median_kwh",
        "night_ratio",
        "weekend_ratio",
        "working_mean_kwh",
        "peak_to_avg_ratio",
    ]

    df = room_features_df.copy()
    X = df[clustering_cols].values

    # Step 1: Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 2: KMeans Clustering
    # Cap n_clusters at number of samples
    k = min(n_clusters, len(df))
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    df["cluster"] = cluster_labels

    # Step 3: PCA 2D Projection for visualization
    pca = PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(X_scaled)
    df["pca_x"] = np.round(coords[:, 0], 3)
    df["pca_y"] = np.round(coords[:, 1], 3)

    var_explained = [round(float(v) * 100, 2) for v in pca.explained_variance_ratio_]

    # Step 4: Cluster Characteristics Summary
    cluster_summary = (
        df.groupby("cluster")
        .agg(
            rooms_count=("room_name", "count"),
            rooms_list=("room_name", lambda x: ", ".join(x)),
            avg_mean_kwh=("mean_kwh", "mean"),
            avg_peak_kwh=("peak_kwh", "mean"),
            avg_night_ratio=("night_ratio", "mean"),
            avg_weekend_ratio=("weekend_ratio", "mean"),
            avg_working_kwh=("working_mean_kwh", "mean"),
        )
        .reset_index()
    )

    # Assign objective educational interpretation labels based on average load
    def label_cluster(row):
        avg_load = row["avg_mean_kwh"]
        if avg_load >= 2.5:
            return "Heavy Intensive Laboratory (Dual AC + High Workstation Computing)"
        elif avg_load >= 1.2:
            return "Moderate Operational Spaces (Single AC Labs / Faculty Office)"
        else:
            return "Standard Lecture Halls (Intermittent Fan & Projector Load)"

    cluster_summary["profile_description"] = cluster_summary.apply(label_cluster, axis=1)

    metadata = {
        "n_clusters": k,
        "inertia": round(float(kmeans.inertia_), 4),
        "pca_variance_ratio": var_explained,
        "features_used": clustering_cols,
    }

    return df, cluster_summary, metadata
