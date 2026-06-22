import pandas as pd
import plotly.express as px

# ---------------------------------------------------
# ✅ PCA 2D PLOT
# ---------------------------------------------------


def plot_pca_2d(X_reduced, labels=None):

    df = pd.DataFrame(X_reduced, columns=["PC1", "PC2"])

    if labels is not None:
        df["label"] = labels

    return px.scatter(
        df,
        x="PC1",
        y="PC2",
        color="label" if labels is not None else None,
        title="PCA 2D Projection"
    )


# ---------------------------------------------------
# ✅ PCA 3D
# ---------------------------------------------------

def plot_pca_3d(X_reduced, labels=None):

    df = pd.DataFrame(X_reduced, columns=["PC1", "PC2", "PC3"])

    if labels is not None:
        df["label"] = labels

    return px.scatter_3d(
        df,
        x="PC1",
        y="PC2",
        z="PC3",
        color="label" if labels is not None else None,
        title="PCA 3D Projection"
    )


# ---------------------------------------------------
# ✅ TSNE 2D
# ---------------------------------------------------

def plot_tsne_2d(X_embedded, labels=None):

    df = pd.DataFrame(X_embedded, columns=["Dim1", "Dim2"])

    if labels is not None:
        df["label"] = labels

    return px.scatter(
        df,
        x="Dim1",
        y="Dim2",
        color="label" if labels is not None else None,
        title="t-SNE Projection"
    )
