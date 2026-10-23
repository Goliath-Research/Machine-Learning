# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "pydataset",
#     "scikit-learn",
#     "seaborn",
# ]
# ///

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns

    plt.rcParams["figure.max_open_warning"] = 0
    sns.set_theme(style="whitegrid")

    return mo, np, pd, plt, sns


@app.cell
def _():
    from pydataset import data
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.datasets import make_blobs
    from sklearn.metrics import (
        calinski_harabasz_score,
        davies_bouldin_score,
        silhouette_score,
    )
    from sklearn.preprocessing import StandardScaler

    return (
        KMeans,
        PCA,
        StandardScaler,
        calinski_harabasz_score,
        data,
        davies_bouldin_score,
        make_blobs,
        silhouette_score,
    )


@app.cell
def _(np, pd, sns):
    def draw_clusters(axes, x, y, labels, title, centers=None):
        frame = pd.DataFrame(
            {"x": np.asarray(x), "y": np.asarray(y), "cluster": np.asarray(labels).astype(str)}
        )
        sns.scatterplot(
            data=frame, x="x", y="y", hue="cluster", s=28, ax=axes, legend=False
        )
        if centers is not None:
            axes.scatter(centers[:, 0], centers[:, 1], c="black", marker="x", s=70)
        axes.set_title(title)

    def knee_k(ks, values):
        x = (np.asarray(ks, dtype=float) - ks[0]) / (ks[-1] - ks[0])
        y = (np.asarray(values, dtype=float) - np.min(values)) / (
            np.max(values) - np.min(values)
        )
        start = np.array([x[0], y[0]])
        line = np.array([x[-1], y[-1]]) - start
        distances = np.abs(
            line[0] * (start[1] - y) - line[1] * (start[0] - x)
        ) / np.linalg.norm(line)
        return int(np.asarray(ks)[int(np.argmax(distances))])

    return draw_clusters, knee_k


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # k-Means Clustering

    ## Objectives

    - Partition a cloud of points into the number of groups you ask for.
    - Compare that choice with inertia, the silhouette score, the Davies–Bouldin score, and the Calinski–Harabasz score.
    - Standardize the US arrest rates, draw them with two principal components, and cluster the standardized columns.

    ## Background

    k-means places k centers and assigns each point to the nearest center. The centers then move to the mean of their points, and the assignment repeats. Inertia is the sum of squared distances from each point to its center. It falls as k grows, because a point can always get a closer center.

    The silhouette score is high when points sit inside their own group and away from the others. The Davies–Bouldin score is low in that same situation. The Calinski–Harabasz score is high when the groups are tight and far apart.

    ## Datasets Used

    - **Blobs.** 300 points in four round groups, from `make_blobs`, with `random_state=0`.
    - **USArrests.** Arrest rates for murder, assault, and rape per 100,000 residents, and the percent of the population living in urban areas, for 50 states in 1973. The table comes from `pydataset`.
    """)
    return


@app.cell
def _(draw_clusters, make_blobs, np, plt):
    blobs, _unused_blob_labels = make_blobs(
        n_samples=300, centers=4, cluster_std=0.5, random_state=0
    )
    print(f"{len(blobs)} points")
    _figure, _axes = plt.subplots(figsize=(5, 4))
    draw_clusters(_axes, blobs[:, 0], blobs[:, 1], np.zeros(len(blobs)), "Four blobs")
    return _figure


@app.cell
def _(KMeans, blobs, draw_clusters, np, plt):
    blob_fits = {}
    _figure, _axes = plt.subplots(2, 2, figsize=(9, 8))
    for _ax, _k in zip(_axes.ravel(), (2, 3, 4, 5)):
        _model = KMeans(n_clusters=_k, random_state=10, n_init="auto").fit(blobs)
        blob_fits[_k] = _model
        draw_clusters(
            _ax,
            blobs[:, 0],
            blobs[:, 1],
            _model.labels_,
            f"k = {_k}",
            centers=_model.cluster_centers_,
        )
    for _k in (6, 7, 8):
        blob_fits[_k] = KMeans(n_clusters=_k, random_state=10, n_init="auto").fit(blobs)
    _figure.tight_layout()
    print("Centers for k = 4")
    print(np.round(blob_fits[4].cluster_centers_, 2))
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    k-means returns as many groups as you request. With three centers, two of the visible blobs share a center. With five, one blob is split.

    Inertia keeps falling after the blobs are already separated. The bend in that curve is one way to stop. The knee below is the k farthest from the straight line that joins the first and last points of the curve.
    """)
    return


@app.cell
def _(KMeans, blobs, knee_k, np, plt):
    ks = list(range(2, 21))
    inertia = np.array(
        [
            KMeans(n_clusters=_k, random_state=10, n_init="auto").fit(blobs).inertia_
            for _k in ks
        ]
    )
    elbow_k = knee_k(ks, inertia)
    _figure, _axes = plt.subplots(figsize=(6, 4))
    _axes.plot(ks, inertia, marker="o", color="firebrick")
    _axes.axvline(elbow_k, color="gray", linestyle="--")
    _axes.set_xlabel("k")
    _axes.set_ylabel("Inertia")
    _axes.set_title(f"Elbow, knee at k = {elbow_k}")
    return _figure


@app.cell
def _(
    blobs,
    blob_fits,
    calinski_harabasz_score,
    davies_bouldin_score,
    pd,
    plt,
    silhouette_score,
    sns,
):
    blob_rows = []
    for _k, _model in blob_fits.items():
        blob_rows.append(
            {
                "k": _k,
                "silhouette": silhouette_score(blobs, _model.labels_),
                "davies_bouldin": davies_bouldin_score(blobs, _model.labels_),
                "calinski_harabasz": calinski_harabasz_score(blobs, _model.labels_),
            }
        )
    blob_scores = pd.DataFrame(blob_rows).set_index("k").sort_index()
    silhouette_k = int(blob_scores["silhouette"].idxmax())
    davies_k = int(blob_scores["davies_bouldin"].idxmin())
    calinski_k = int(blob_scores["calinski_harabasz"].idxmax())
    _figure, _axes = plt.subplots(1, 3, figsize=(12, 3.5))
    sns.barplot(x=blob_scores.index, y=blob_scores["silhouette"], color="steelblue", ax=_axes[0])
    _axes[0].set_title(f"Silhouette, highest at k = {silhouette_k}")
    sns.barplot(x=blob_scores.index, y=blob_scores["davies_bouldin"], color="steelblue", ax=_axes[1])
    _axes[1].set_title(f"Davies–Bouldin, lowest at k = {davies_k}")
    sns.barplot(
        x=blob_scores.index, y=blob_scores["calinski_harabasz"], color="steelblue", ax=_axes[2]
    )
    _axes[2].set_title(f"Calinski–Harabasz, highest at k = {calinski_k}")
    _figure.tight_layout()
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Which attribute of a fitted `KMeans` model holds the sum of squared distances to the centers? Replace `None` with its name.
    """)
    return


@app.cell
def _():
    student_inertia = None
    print(student_inertia)
    return (student_inertia,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r'''
```python
student_inertia = "inertia_"
```
**inertia_.** The trailing underscore marks a value computed during `fit`. It falls as k grows.
''')
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## USArrests

    `pydataset` supplies the table. Assault counts are much larger than murder rates, so a distance that uses the raw columns is mostly a distance in assaults. `StandardScaler` puts each column in training units of its own standard deviation. Here every state is used, because there is no holdout.

    The four standardized columns are what k-means sees. The scatter is the first two principal components of those columns, so the picture is a view and not the data the centers were fit on. The scores below use the four standardized columns.
    """)
    return


@app.cell
def _(
    KMeans,
    PCA,
    StandardScaler,
    calinski_harabasz_score,
    data,
    davies_bouldin_score,
    draw_clusters,
    pd,
    plt,
    silhouette_score,
    sns,
):
    arrests = data("USArrests")
    arrest_columns = ["Murder", "Assault", "UrbanPop", "Rape"]
    arrests_z = pd.DataFrame(
        StandardScaler().fit_transform(arrests[arrest_columns]),
        columns=arrest_columns,
        index=arrests.index,
    )
    arrest_pca = PCA(n_components=2, random_state=0).fit(arrests_z)
    arrest_plane = arrest_pca.transform(arrests_z)
    arrest_models = {
        _k: KMeans(n_clusters=_k, random_state=22, n_init="auto").fit(arrests_z)
        for _k in (2, 3, 4, 5)
    }
    arrest_rows = []
    for _k, _model in arrest_models.items():
        arrest_rows.append(
            {
                "k": _k,
                "silhouette": silhouette_score(arrests_z, _model.labels_),
                "davies_bouldin": davies_bouldin_score(arrests_z, _model.labels_),
                "calinski_harabasz": calinski_harabasz_score(arrests_z, _model.labels_),
            }
        )
    arrest_scores = pd.DataFrame(arrest_rows).set_index("k")
    arrest_silhouette_k = int(arrest_scores["silhouette"].idxmax())
    arrest_davies_k = int(arrest_scores["davies_bouldin"].idxmin())
    arrest_calinski_k = int(arrest_scores["calinski_harabasz"].idxmax())
    chosen = arrest_models[arrest_silhouette_k]
    arrests_z["cluster"] = chosen.labels_.astype(str)
    _figure, _axes = plt.subplots(1, 2, figsize=(10, 4))
    draw_clusters(
        _axes[0],
        arrest_plane[:, 0],
        arrest_plane[:, 1],
        chosen.labels_,
        f"k = {arrest_silhouette_k} in the first two components",
    )
    _long = arrests_z.melt(id_vars="cluster", var_name="measure", value_name="z")
    sns.boxplot(data=_long, x="cluster", y="z", hue="measure", ax=_axes[1])
    _axes[1].set_title("Standardized rates by cluster")
    _axes[1].legend(fontsize=8)
    _figure.tight_layout()
    print(f"{len(arrests)} states")
    print(
        "Share of variance in the picture: "
        f"{arrest_pca.explained_variance_ratio_.sum():.3f}"
    )
    print(arrest_scores.round(3).to_string())
    return _figure


@app.cell(hide_code=True)
def _(
    arrest_calinski_k,
    arrest_davies_k,
    arrest_scores,
    arrest_silhouette_k,
    blob_scores,
    calinski_k,
    davies_k,
    elbow_k,
    mo,
    silhouette_k,
):
    mo.md(f"""
    ## Conclusions

    **Key takeaways:**

    - On the blobs, the knee of the inertia curve is k = {elbow_k}. The silhouette score is highest at k = {silhouette_k} ({blob_scores.loc[silhouette_k, "silhouette"]:.3f}). Davies–Bouldin is lowest at k = {davies_k}. Calinski–Harabasz is highest at k = {calinski_k}.
    - Asking for more clusters than the eye sees splits a blob. Asking for fewer merges two blobs. Inertia alone does not stop that.
    - On the arrest rates, silhouette prefers k = {arrest_silhouette_k} ({arrest_scores.loc[arrest_silhouette_k, "silhouette"]:.3f}). Davies–Bouldin prefers k = {arrest_davies_k}. Calinski–Harabasz prefers k = {arrest_calinski_k}.
    - The scatter is a two-component view. The centers were fit on the four standardized columns, and the scores use those same columns.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check your understanding

    1. What does k-means do with the centers after the assignment step?
    2. Why does inertia fall as k increases?
    3. Which of the three scores should be small rather than large?
    4. Why are the arrest columns standardized before the distance is computed?
    5. What is the picture of the states a view of?
    6. Why can the three scores choose different values of k?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
1. **They move to the mean:** Each center is replaced by the average of the points assigned to it.
2. **A point can get a closer center:** More centers shrink the squared distances. The drop slows once the real groups have their own centers.
3. **Davies–Bouldin:** A lower score means each cluster is less like its nearest neighbor. Silhouette and Calinski–Harabasz are read in the other direction.
4. **Assault would dominate:** Its raw numbers are much larger than the murder rate. Standardization gives each column a standard deviation of 1.
5. **The first two principal components:** k-means did not see only those two axes. It saw Murder, Assault, UrbanPop, and Rape after standardization.
6. **They reward different shapes:** A partition can be compact on one score and mixed on another. The table is the comparison.
""")
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - [KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
    - [Clustering](https://scikit-learn.org/stable/modules/clustering.html)
    - VanderPlas, J. (2017). *Python Data Science Handbook*. O'Reilly Media. Chapter 5.
    """)
    return


if __name__ == "__main__":
    app.run()
