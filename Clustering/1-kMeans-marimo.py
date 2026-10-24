# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "kneed",
#     "marimo",
#     "numpy",
#     "pandas",
#     "plotly",
#     "pydataset",
#     "scikit-learn",
# ]
# ///

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))

    import ClusterVisualizer as cv
    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.express as px
    from kneed import KneeLocator
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
        KneeLocator,
        KMeans,
        PCA,
        StandardScaler,
        calinski_harabasz_score,
        cv,
        data,
        davies_bouldin_score,
        make_blobs,
        mo,
        np,
        pd,
        px,
        silhouette_score,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # k-Means Clustering

    ## Objectives

    - Place centers for several values of k and read the clusters those centers produce.
    - Choose k with the elbow of the inertia curve, the silhouette score, the Davies–Bouldin score, and the Calinski–Harabasz score.
    - Standardize the US arrest rates, draw the states with two principal components, and cluster the four standardized columns.

    ## Background

    k-means places k centers and assigns each point to the nearest center. The centers then move to the mean of their points, and the assignment repeats. Inertia, stored as `inertia_`, is the sum of squared distances from each point to its center. It falls as k grows, because a point can always get a closer center.

    The silhouette score is high when points sit inside their own group and away from the others. The Davies–Bouldin score is low in that same situation. The Calinski–Harabasz score is high when the groups are tight and far apart.

    The scatters and the score bars come from `ClusterVisualizer`. That class scores the table it was given. On the blobs, that table is the data k-means saw. On the states, the visualizer is given the two principal components, so its bars score the picture. A second table below scores the four standardized columns the centers were fit on.

    ## Datasets Used

    - **Blobs.** 300 points in four round groups, from `make_blobs`, with `random_state=0` and `cluster_std=0.5`.
    - **USArrests.** Murder, assault, and rape arrests per 100,000 residents, and the percent urban, for 50 states in 1973. The table comes from `pydataset`.
    """)
    return


@app.cell
def _(cv, make_blobs, pd):
    blobs, _blob_labels = make_blobs(
        n_samples=300, centers=4, cluster_std=0.5, random_state=0
    )
    blob_frame = pd.DataFrame(blobs, columns=["x", "y"])
    blob_view = cv.ClusterVisualizer(blob_frame)
    print(f"{len(blobs)} points")
    blob_view.plot_data()
    return blobs, blob_view


@app.cell
def _(KMeans, blobs):
    blob_models = {
        _k: KMeans(n_clusters=_k, random_state=10, n_init="auto").fit(blobs)
        for _k in range(2, 9)
    }
    print("Centers for k = 4")
    print(blob_models[4].cluster_centers_.round(2))
    return (blob_models,)


@app.cell
def _(blob_models, blob_view):
    blob_view.plot_clusters(
        blob_models[4].labels_,
        blob_models[4].cluster_centers_,
        title="k-Means with 4 clusters",
    )
    return


@app.cell
def _(blob_models, blob_view):
    blob_view.plot_clusters(
        blob_models[3].labels_,
        blob_models[3].cluster_centers_,
        title="k-Means with 3 clusters",
    )
    return


@app.cell
def _(blob_models, blob_view):
    blob_view.plot_clusters(
        blob_models[2].labels_,
        blob_models[2].cluster_centers_,
        title="k-Means with 2 clusters",
    )
    return


@app.cell
def _(blob_models, blob_view):
    blob_view.plot_clusters(
        blob_models[5].labels_,
        blob_models[5].cluster_centers_,
        title="k-Means with 5 clusters",
    )
    return


@app.cell
def _(blob_models, blob_view):
    blob_view.plot_clusters(
        blob_models[8].labels_,
        blob_models[8].cluster_centers_,
        title="k-Means with 8 clusters",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    With three centers, two blobs share a center. With five or eight, a blob is split. Zoom the Plotly scatter if a center is sitting on top of its points.

    Inertia keeps falling after the four blobs have their own centers. `KneeLocator` marks the bend, with `curve="convex"` and `direction="decreasing"`.
    """)
    return


@app.cell
def _(KMeans, KneeLocator, blobs, px):
    ks = list(range(2, 21))
    inertia = [
        KMeans(n_clusters=_k, random_state=10, n_init="auto").fit(blobs).inertia_
        for _k in ks
    ]
    elbow = KneeLocator(ks, inertia, curve="convex", direction="decreasing")
    elbow_k = int(elbow.knee)
    print(f"Elbow point at: {elbow_k}")
    _fig = px.scatter(x=ks, y=inertia)
    _fig.update_traces(marker=dict(size=8, opacity=0.8))
    _fig.update_layout(
        width=600,
        height=400,
        title="Elbow curve",
        xaxis_title="k",
        yaxis_title="Sum of squared distances",
    )
    _fig.add_vline(x=elbow_k, line_width=2, line_dash="dash", line_color="grey")
    _fig
    return (elbow_k,)


@app.cell
def _(blob_models, blob_view, pd):
    blob_labels = pd.DataFrame(
        {f"k={_k}": _model.labels_ for _k, _model in blob_models.items()}
    )
    blob_view.plot_silhouette(blob_labels, title="Silhouette scores for k-means")
    return (blob_labels,)


@app.cell
def _(blob_labels, blob_view):
    blob_view.plot_davies_bouldin(blob_labels, title="Davies-Bouldin scores for k-means")
    return


@app.cell
def _(blob_labels, blob_view):
    blob_view.plot_calinski_harabasz(
        blob_labels, title="Calinski-Harabasz scores for k-means"
    )
    return


@app.cell
def _(
    blobs,
    blob_models,
    calinski_harabasz_score,
    davies_bouldin_score,
    pd,
    silhouette_score,
):
    blob_scores = pd.DataFrame(
        [
            {
                "k": _k,
                "silhouette": silhouette_score(blobs, _model.labels_),
                "davies_bouldin": davies_bouldin_score(blobs, _model.labels_),
                "calinski_harabasz": calinski_harabasz_score(blobs, _model.labels_),
            }
            for _k, _model in blob_models.items()
        ]
    ).set_index("k")
    silhouette_k = int(blob_scores["silhouette"].idxmax())
    davies_k = int(blob_scores["davies_bouldin"].idxmin())
    calinski_k = int(blob_scores["calinski_harabasz"].idxmax())
    print(blob_scores.round(3).to_string())
    print(f"Silhouette highest at k = {silhouette_k}")
    print(f"Davies-Bouldin lowest at k = {davies_k}")
    print(f"Calinski-Harabasz highest at k = {calinski_k}")
    return blob_scores, calinski_k, davies_k, silhouette_k


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
**inertia_.** The trailing underscore marks a value computed during `fit`. It is the vertical axis of the elbow curve, and it falls as k grows.
''')
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## USArrests

    Assault counts are much larger than murder rates, so a distance on the raw columns is mostly a distance in assaults. `StandardScaler` gives each column a standard deviation of 1. Every state is used, because there is no holdout.

    k-means is fit on the four standardized columns. `ClusterVisualizer` is given the two principal components, which is the picture in the notebook. Its score bars therefore score that picture. The printed table scores the four columns.
    """)
    return


@app.cell
def _(StandardScaler, data, pd, px):
    arrests = data("USArrests")
    arrest_columns = ["Murder", "Assault", "UrbanPop", "Rape"]
    print(f"{len(arrests)} states")
    _raw = arrests[arrest_columns].melt(var_name="measure", value_name="value")
    _fig = px.box(_raw, x="measure", y="value", color="measure")
    _fig.update_layout(
        width=600, height=400, title="USArrests, original units", showlegend=False
    )
    _fig
    return arrest_columns, arrests


@app.cell
def _(
    KMeans,
    PCA,
    StandardScaler,
    arrest_columns,
    arrests,
    cv,
    pd,
    px,
):
    arrests_z = pd.DataFrame(
        StandardScaler().fit_transform(arrests[arrest_columns]),
        columns=arrest_columns,
        index=arrests.index,
    )
    arrest_pca = PCA(n_components=2, random_state=0).fit(arrests_z)
    arrest_plane = pd.DataFrame(
        arrest_pca.transform(arrests_z), columns=["component 1", "component 2"]
    )
    arrest_view = cv.ClusterVisualizer(arrest_plane)
    arrest_models = {
        _k: KMeans(n_clusters=_k, random_state=22, n_init="auto").fit(arrests_z)
        for _k in (2, 3, 4, 5)
    }
    print(f"Share of variance in the picture: {arrest_pca.explained_variance_ratio_.sum():.3f}")
    _long = arrests_z.melt(var_name="measure", value_name="z")
    _fig = px.box(_long, x="measure", y="z", color="measure")
    _fig.update_layout(
        width=600, height=400, title="USArrests, standardized", showlegend=False
    )
    _fig
    return arrest_models, arrest_plane, arrest_view, arrests_z


@app.cell
def _(arrest_view):
    arrest_view.plot_data(title="PCA space (2 components)")
    return


@app.cell
def _(arrest_models, arrest_view, pd):
    arrest_picture_labels = pd.DataFrame(
        {f"k={_k}": _model.labels_ for _k, _model in arrest_models.items()}
    )
    arrest_view.plot_silhouette(
        arrest_picture_labels, title="Silhouette scores on the two components"
    )
    return (arrest_picture_labels,)


@app.cell
def _(arrest_picture_labels, arrest_view):
    arrest_view.plot_davies_bouldin(
        arrest_picture_labels, title="Davies-Bouldin scores on the two components"
    )
    return


@app.cell
def _(arrest_picture_labels, arrest_view):
    arrest_view.plot_calinski_harabasz(
        arrest_picture_labels, title="Calinski-Harabasz scores on the two components"
    )
    return


@app.cell
def _(
    arrest_models,
    arrests_z,
    calinski_harabasz_score,
    davies_bouldin_score,
    pd,
    silhouette_score,
):
    arrest_scores = pd.DataFrame(
        [
            {
                "k": _k,
                "silhouette": silhouette_score(arrests_z, _model.labels_),
                "davies_bouldin": davies_bouldin_score(arrests_z, _model.labels_),
                "calinski_harabasz": calinski_harabasz_score(arrests_z, _model.labels_),
            }
            for _k, _model in arrest_models.items()
        ]
    ).set_index("k")
    arrest_silhouette_k = int(arrest_scores["silhouette"].idxmax())
    arrest_davies_k = int(arrest_scores["davies_bouldin"].idxmin())
    arrest_calinski_k = int(arrest_scores["calinski_harabasz"].idxmax())
    print(arrest_scores.round(3).to_string())
    return arrest_calinski_k, arrest_davies_k, arrest_scores, arrest_silhouette_k


@app.cell
def _(arrest_models, arrest_silhouette_k, arrest_view):
    arrest_view.plot_clusters(
        arrest_models[arrest_silhouette_k].labels_,
        title=f"k = {arrest_silhouette_k} in the PCA space",
    )
    return


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

    - On the blobs, the elbow is k = {elbow_k}. The silhouette score is highest at k = {silhouette_k} ({blob_scores.loc[silhouette_k, "silhouette"]:.3f}). Davies–Bouldin is lowest at k = {davies_k}. Calinski–Harabasz is highest at k = {calinski_k}. Those bars score the same two columns k-means saw.
    - Asking for fewer than four clusters merges blobs. Asking for more splits one. Inertia alone does not stop that, which is why the elbow is read together with the three scores.
    - On the arrest rates, the four standardized columns give silhouette k = {arrest_silhouette_k} ({arrest_scores.loc[arrest_silhouette_k, "silhouette"]:.3f}), Davies–Bouldin k = {arrest_davies_k}, and Calinski–Harabasz k = {arrest_calinski_k}.
    - The bars drawn by `ClusterVisualizer` for the states score the two principal components, because that is the table the visualizer was given. The printed table scores the four columns the centers were fit on.
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
    5. What table do the state score bars use?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
1. **They move to the mean:** Each center is replaced by the average of the points assigned to it.
2. **A point can get a closer center:** More centers shrink the squared distances. The drop slows once the real groups have their own centers. That slowdown is the elbow.
3. **Davies–Bouldin:** A lower score means each cluster is less like its nearest neighbor. Silhouette and Calinski–Harabasz are read in the other direction.
4. **Assault would dominate:** Its raw numbers are much larger than the murder rate. Standardization gives each column a standard deviation of 1.
5. **The two principal components:** `ClusterVisualizer` scores the frame it stores. The centers were fit on Murder, Assault, UrbanPop, and Rape after standardization. The printed table is that fit.
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
    - [KneeLocator](https://kneed.readthedocs.io/)
    - VanderPlas, J. (2017). *Python Data Science Handbook*. O'Reilly Media. Chapter 5.
    """)
    return


if __name__ == "__main__":
    app.run()
