import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ---------------------------------------------------
# ✅ HELPERS
# ---------------------------------------------------


def _attach_labels(df, labels):
    temp = df.copy()

    # ✅ force alignment
    temp["cluster"] = pd.Series(labels).astype(str).values

    return temp


# ---------------------------------------------------
# ✅ PCA PLOT
# ---------------------------------------------------

def plot_pca(X_processed):

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_processed)

    return pd.DataFrame(X_pca, columns=["PC1", "PC2"])


def plot_pca_scatter(pca_df):

    return px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        title="PCA Projection"
    )


# ---------------------------------------------------
# ✅ KMEANS
# ---------------------------------------------------

def plot_kmeans(pca_df, labels):

    if labels is None:
        return px.scatter(title="KMeans (no labels)")

    df = _attach_labels(pca_df, labels)

    return px.scatter(
        df,
        x="PC1",
        y="PC2",
        color="cluster",
        title="KMeans Clusters"
    )


# ---------------------------------------------------
# ✅ DBSCAN
# ---------------------------------------------------

def plot_dbscan(pca_df, labels):

    if labels is None:
        return px.scatter(title="DBSCAN (no labels)")

    df = _attach_labels(pca_df, labels)

    return px.scatter(
        df,
        x="PC1",
        y="PC2",
        color="cluster",
        title="DBSCAN Clusters"
    )


# ---------------------------------------------------
# ✅ ELBOW
# ---------------------------------------------------

def plot_elbow(X_processed):

    inertia = []

    for k in range(2, 8):
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X_processed)
        inertia.append(km.inertia_)

    return px.line(
        x=list(range(2, 8)),
        y=inertia,
        title="Elbow Curve"
    )


# ---------------------------------------------------
# ✅ METRICS
# ---------------------------------------------------

def plot_metrics(df):

    if "silhouette_score" not in df.columns:
        return px.scatter(title="Metric not available")

    df2 = df[df["silhouette_score"].notna()]

    if df2.empty:
        return px.scatter(title="No valid silhouette scores")

    return px.bar(
        df2,
        x="model",
        y="silhouette_score",
        title="Clustering Performance"
    )


# ---------------------------------------------------
# ✅ MAIN ENTRY (NOW MATCHES CLASSIFICATION ✅)
# ---------------------------------------------------
def plot_all(results, artifacts=None):

    import pandas as pd
    from sklearn.decomposition import PCA

    df = pd.DataFrame(results)

    if artifacts is None:
        return {}

    X_processed = artifacts.get("X_processed")
    clusters = artifacts.get("clusters", {})
    reducers = artifacts.get("reducers", {})  # ✅ NEW

    if X_processed is None or clusters is None:
        return {}

    # ======================================================
    # ✅ USE SAME REDUCER USED DURING TRAINING ✅
    # ======================================================
    exp_id = "KMeans | unsupervised"  # reference model

    reducer = reducers.get(exp_id)

    if reducer:
        X_reduced = reducer.transform(X_processed)
    else:
        # fallback if reducer missing
        fallback = PCA(n_components=10, random_state=42)
        X_reduced = fallback.fit_transform(X_processed)

    # ✅ FINAL 2D PROJECTION (consistent)
    plot_pca2 = PCA(n_components=2, random_state=42)
    X_2d = plot_pca2.fit_transform(X_reduced)

    pca_df = pd.DataFrame(X_2d, columns=["PC1", "PC2"]).reset_index(drop=True)

    # ======================================================
    # ✅ CLUSTER PLOTS (ALIGNED WITH SAME SPACE ✅)
    # ======================================================
    return {
        "pca": plot_pca_scatter(pca_df),

        "kmeans": plot_kmeans(
            pca_df,
            clusters.get("KMeans")
        ),

        "dbscan": plot_dbscan(
            pca_df,
            clusters.get("DBSCAN")
        ),

        "elbow": plot_elbow(X_processed),

        "metrics": plot_metrics(df),
    }
