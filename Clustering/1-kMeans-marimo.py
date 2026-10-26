# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
#     "pandas",
#     "plotly",
#     "scikit-learn>=1.2",
# ]
# ///

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
async def _():
    import importlib.util as _import_util
    import sys as _sys

    # Setup: run this cell to install any missing packages.
    # You do not need to modify this code.
    _packages = {
        "numpy": "numpy",
        "pandas": "pandas",
        "plotly": "plotly",
        "sklearn": "scikit-learn>=1.2",
    }
    _missing = [
        _package for _module, _package in _packages.items()
        if _import_util.find_spec(_module) is None
    ]
    if _missing:
        if _sys.platform == "emscripten":
            # Browser Python uses micropip; it cannot launch subprocesses.
            import micropip as _micropip
            await _micropip.install(_missing)
        else:
            import subprocess as _subprocess
            _subprocess.check_call([_sys.executable, "-m", "pip", "install", *_missing])

    # The imports below wait until package installation has completed.
    packages_ready = True
    return (packages_ready,)


@app.cell
def _(packages_ready):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))

    import ClusterVisualizer as cv
    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.io as pio

    pio.templates.default = "plotly_white"
    px.defaults.color_discrete_sequence = px.colors.qualitative.G10
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.datasets import make_blobs
    from sklearn.metrics import (
        calinski_harabasz_score,
        davies_bouldin_score,
        silhouette_score,
    )
    from sklearn.preprocessing import StandardScaler

    import warnings

    warnings.filterwarnings(
        "ignore",
        message=r"JsProxy\.as_object_map\(\) is deprecated.*",
        category=RuntimeWarning,
        module=r"threadpoolctl",
    )

    return (
        KMeans,
        PCA,
        StandardScaler,
        calinski_harabasz_score,
        cv,
        Path,
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

    ## Goals

    By the end of this lesson, you should be able to:

    1. Explain how **k-means** assigns points to clusters and updates their centers.
    2. Compare different values of **k** using cluster plots, an elbow curve, and three internal validation scores.
    3. Explain why variables with different scales should be standardized before **k-means**.
    4. Use Principal Component Analysis (PCA) to display clusters fitted using all four standardized variables.
    5. Describe how the USArrests clusters differ using their original variables, without treating the groups as proven categories.

    ## Background

    Clustering groups observations with similar measured features. It is **unsupervised**: we do not supply group labels when fitting the model.

    **k-means** is an unsupervised learning algorithm that divides observations into **k** clusters based on how close they are to one another. You choose the number of clusters, **k**, before fitting the model.

    The letter **k** is the number of clusters we request. A **center** (or centroid) is the average position of the points assigned to a cluster. **k-means** repeats these steps:

    1. Assign each point to its nearest center using Euclidean distance.
    2. Move each center to the mean of the points assigned to it.
    3. Repeat until the solution stops changing appreciably or the iteration limit is reached.

    **k-means** seeks to minimize **inertia**, the sum of squared distances from points to their assigned centers. Different initial centers can lead to different solutions. We use fixed `random_state` values so rerunning the examples is reproducible, and `n_init="auto"`; with the default k-means++ initialization, `auto` uses one initialization. In other analyses, trying several initializations can improve stability.

    **k-means** works especially well for compact, roughly spherical groups of similar spread. Outliers and elongated or unequal groups can make the result misleading. Cluster numbers such as 0 and 1 are names, not ranks.

    ## Datasets used

    - **Synthetic blobs:** 300 points generated around four centers. We know how these data were generated, but do not use their true labels for fitting.
    - **USArrests:** murder, assault, and rape arrest rates per 100,000 residents, plus urban population percentage for 50 US states in 1973. Arrest rates are not direct measurements of crime rates.

    ## 1. Explore the synthetic data

    Each row is one point; `x` and `y` give its coordinates. `ClusterVisualizer` returns interactive Plotly charts. Hover over points or zoom to inspect them.
    """)
    return


@app.cell
def _(cv, make_blobs, pd):
    blobs, _blob_labels = make_blobs(
        n_samples=300, centers=4, cluster_std=0.5, random_state=0
    )
    blob_frame = pd.DataFrame(blobs, columns=["x", "y"])
    blob_view = cv.ClusterVisualizer(blob_frame)
    blob_view.plot_data()
    return blobs, blob_view


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Fit and compare k-means solutions

    Start with **k = 4**, then compare 3, 2, 5, 6, 7, and 8. Black markers show centers. `fit` learns the centers, `labels_` stores each point's assigned cluster, and `cluster_centers_` stores the centers' coordinates. The dictionary below stores one fitted model for each k from 2 to 8; `blob_models[4]` selects the four-cluster model.
    """)
    return


@app.cell
def _(blobs, pd):
    pd.DataFrame(blobs, columns=["x", "y"]).head().round(4)
    return


@app.cell
def _(KMeans, blobs):
    blob_models = {
        _k: KMeans(n_clusters=_k, random_state=10, n_init="auto").fit(blobs)
        for _k in range(2, 9)
    }
    blob_models[4].cluster_centers_.round(4)
    return (blob_models,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Start with k = 4

    We generated these data around four centers, so four clusters are a useful starting point. In a real dataset, we usually do not know the number of groups beforehand.

    Run the next cell and compare it with the uncolored scatter plot. Most points in each visible blob receive the same color, and each black center lies near the middle of its assigned points. This shows how a fitted four-cluster solution describes the data. It does not mean **k-means** discovered the number four: **we supplied k = 4**.
    """)
    return


@app.cell
def _(blob_models, blob_view):
    blob_view.plot_clusters(
        blob_models[4].labels_,
        blob_models[4].cluster_centers_,
        title="k-Means with 4 clusters",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reduce the number to k = 3

    The data have not changed; we now request only three clusters. Because there are four visible blobs but only three available centers, at least two blobs must share a cluster label.

    In the next plot, find the two blobs with the same color. Their shared center summarizes points from both blobs and can lie in the space between them. Reducing k gives a simpler description, but here it merges groups that looked separate.
    """)
    return


@app.cell
def _(blob_models, blob_view):
    blob_view.plot_clusters(
        blob_models[3].labels_,
        blob_models[3].cluster_centers_,
        title="k-Means with 3 clusters",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reduce the number again to k = 2

    With only two centers, the model must describe all 300 points using two groups. More of the visible structure is combined than with k = 3. Find which blobs share a color and notice where their centers move.

    Each value of k is fitted independently. **k-means** does not simply merge two clusters from the preceding plot, so do not assume that cluster boundaries or color labels will stay the same.
    """)
    return


@app.cell
def _(blob_models, blob_view):
    blob_view.plot_clusters(
        blob_models[2].labels_,
        blob_models[2].cluster_centers_,
        title="k-Means with 2 clusters",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Increase the number to k = 5

    We now request more clusters than the four visible blobs. To create five groups, **k-means** must split at least one blob into smaller parts. Look for a blob containing two colors and two nearby centers.

    The split can reduce distances to the assigned centers and therefore reduce inertia. However, a lower inertia does not automatically give a more useful description: the extra group may only subdivide a blob that we would prefer to keep together.
    """)
    return


@app.cell
def _(blob_models, blob_view):
    blob_view.plot_clusters(
        blob_models[5].labels_,
        blob_models[5].cluster_centers_,
        title="k-Means with 5 clusters",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Increase to k = 6

    Two more clusters are requested than the number of visible blobs. Compared with k = 4, the model uses additional centers to describe variation within the blobs.

    Inspect where the extra centers are placed and which blobs contain several colors. Their placement depends on the fitted solution; **k-means** does not promise an equal number of subgroups in each blob. Ask whether these subdivisions add useful information.
    """)
    return


@app.cell
def _(blob_models, blob_view):
    blob_view.plot_clusters(
        blob_models[6].labels_, blob_models[6].cluster_centers_,
        title="k-Means with 6 clusters",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Increase to k = 7

    With seven centers, the partition becomes more detailed. Some points that belonged together in the four-cluster solution now receive different labels, even though their coordinates have not changed.

    Compare this plot with k = 6. The goal is not to count colors as evidence of newly discovered natural groups. Instead, notice how requesting more clusters creates finer divisions of the same data.
    """)
    return


@app.cell
def _(blob_models, blob_view):
    blob_view.plot_clusters(
        blob_models[7].labels_, blob_models[7].cluster_centers_,
        title="k-Means with 7 clusters",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Increase to k = 8

    Eight centers give an even finer description of the four blobs. Find the centers within each blob and the boundaries between its subgroups. Do not assume that every blob must be split into exactly two groups: the model minimizes squared distances rather than enforcing that arrangement.

    Across these examples, too few clusters combine visible groups, while extra clusters divide them. **k-means** returns the number we request. We therefore need a reason for choosing k, rather than accepting a detailed plot simply because it has more colors. The elbow curve and validation scores in the next section help us compare the choices.
    """)
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
    ### Try it yourself

    Use the four-cluster model to find the number of points assigned to each cluster. Replace `None` with an expression using `pd.Series` and `value_counts()`. Do the counts add to 300?
    """)
    return


@app.cell
def _(blob_models, pd):
    student_cluster_counts = None
    student_cluster_counts
    return (student_cluster_counts,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
```python
student_cluster_counts = pd.Series(blob_models[4].labels_).value_counts().sort_index()
student_cluster_counts
```
The four counts sum to 300 because each point belongs to exactly one cluster. The labels identify groups; they do not indicate an ordering.
""")}, expanded=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    With three centers, two blobs share a center. With five or eight, a blob is split. Zoom the Plotly scatter if a center is sitting on top of its points.

    ## 3. Compare values of k

    ### The elbow method

    Inertia, stored in `inertia_`, measures within-cluster squared distances. A smaller value means a tighter fit, but adding centers can improve the fit even when it splits a useful group. For the globally best fit at each k, inertia cannot increase as k grows; individual fitted runs can occasionally depart from this because of local solutions.

    Look for an **elbow**: a bend after which extra clusters produce smaller improvements. This is a guide, not proof of a unique correct k. Some curves have no clear elbow.

    Inertia typically keeps falling after the four blobs have their own centers. Inspect the curve yourself: here the bend is near **k = 4**, where the decrease begins to flatten. No automatic elbow detector is used.
    """)
    return


@app.cell
def _(KMeans, blobs, px):
    ks = list(range(2, 21))
    inertia = [
        KMeans(n_clusters=_k, random_state=10, n_init="auto").fit(blobs).inertia_
        for _k in ks
    ]
    _fig = px.scatter(x=ks, y=inertia)
    _fig.update_traces(marker=dict(size=8, opacity=0.8), mode="lines+markers")
    _fig.update_layout(
        width=600,
        height=400,
        title="Elbow curve",
        xaxis_title="Number of clusters (k)",
        yaxis_title="Sum of squared distances",
    )
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Internal validation scores

    These scores use the feature values and assigned clusters; they do not require known correct labels. Compare them on the **same data representation**.

    | Score | Prefer | What it measures |
    | :--- | :--- | :--- |
    | Silhouette | Higher | How close points are to their own cluster compared with the nearest other cluster. Values range from −1 to 1; near zero suggests overlap. |
    | Davies–Bouldin | Lower | How similar each cluster is to its most similar other cluster, using within-cluster spread and separation. |
    | Calinski–Harabasz | Higher | Between-cluster dispersion relative to within-cluster dispersion, adjusted for the numbers of clusters and observations. |

    There is no universal good cutoff for all three. They can disagree, and they do not establish whether groups are useful in practice. The bars compare k = 2 through 8; the highlighted bar is the best score among these candidates.
    """)
    return


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
    blob_scores.round(4)
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
    student_inertia
    return (student_inertia,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r'''
```python
student_inertia = "inertia_"
```
**inertia_.** The trailing underscore marks a value computed during `fit`. It is the vertical axis of the elbow curve, and its best possible value cannot increase as k grows.
''')
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Find the k with the highest silhouette score and the k with the lowest Davies–Bouldin score in `blob_scores`. Use `idxmax()` and `idxmin()`, which return the index of the selected value.
    """)
    return


@app.cell
def _(blob_scores):
    student_silhouette_k = None
    student_davies_k = None
    (student_silhouette_k, student_davies_k)
    return (student_silhouette_k, student_davies_k)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
```python
student_silhouette_k = int(blob_scores["silhouette"].idxmax())
student_davies_k = int(blob_scores["davies_bouldin"].idxmin())
(student_silhouette_k, student_davies_k)
```
Both select k = 4 for these data. Inertia alone would favor adding still more centers, so we also inspect the elbow and validation scores.
""")}, expanded=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Standardize the USArrests variables

    The four columns have different units and spreads. Assault rates vary on a much larger numerical scale than murder rates. Without scaling, that variable can dominate Euclidean distances.

    `StandardScaler` subtracts each column's mean and divides by its population standard deviation. Each transformed column has mean approximately 0 and standard deviation 1. A standardized value of 1 means one standard deviation above that variable's mean; it is not an arrest rate.

    The first box plot shows the original scales; the next shows the standardized values. The shared raw axis combines different units only to illustrate the scale difference.

    We fit k-means on **all four standardized columns**. PCA will provide a two-dimensional view, rather than replace the four clustering inputs.
    """)
    return


@app.cell
def _(Path, pd, px):
    # Source and variable descriptions: https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/USArrests.html
    # Save USArrests.csv beside this notebook; in browser Marimo, upload it to the notebook's files.
    # CSV columns: State, Murder, Assault, UrbanPop, Rape (50 states, 1973).
    arrests = pd.read_csv(Path(__file__).resolve().parent / "USArrests.csv", index_col="State")
    arrest_columns = ["Murder", "Assault", "UrbanPop", "Rape"]
    _raw = arrests[arrest_columns].melt(var_name="measure", value_name="value")
    _fig = px.box(_raw, x="measure", y="value", color="measure")
    _fig.update_layout(
        width=600, height=400, title="USArrests, original units", showlegend=False
    )
    _fig
    return arrest_columns, arrests


@app.cell
def _(arrests):
    arrests.head().round(4)
    return


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
    pca_variance = float(arrest_pca.explained_variance_ratio_.sum())
    _long = arrests_z.melt(var_name="measure", value_name="z")
    _fig = px.box(_long, x="measure", y="z", color="measure")
    _fig.update_layout(
        width=600, height=400, title="USArrests, standardized", showlegend=False
    )
    _fig
    return arrest_models, arrest_plane, arrest_view, arrests_z, pca_variance


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Check the mean and population standard deviation of each standardized column. Use `mean()` and `std(ddof=0)`. Why do we specify `ddof=0` here?
    """)
    return


@app.cell
def _(arrests_z):
    student_standardized_means = None
    student_standardized_stds = None
    (student_standardized_means, student_standardized_stds)
    return (student_standardized_means, student_standardized_stds)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
```python
student_standardized_means = arrests_z.mean().round(4)
student_standardized_stds = arrests_z.std(ddof=0).round(4)
(student_standardized_means, student_standardized_stds)
```
The means are approximately 0 and the standard deviations are 1. `StandardScaler` uses the population standard deviation (`ddof=0`); pandas defaults to the sample version (`ddof=1`).
""")}, expanded=False)
    return


@app.cell(hide_code=True)
def _(mo, pca_variance):
    mo.md(f"""
    ## 5. Display four-variable clusters with PCA

    Principal Component Analysis (PCA) creates new axes that summarize variation in the standardized variables. The first two components retain **{pca_variance:.1%}** of the total variance. Some information is omitted, so apparent overlap in this picture need not mean overlap in all four dimensions.

    Models for k = 2, 3, 4, and 5 are fitted using all four standardized variables. The main score charts below also use those four variables. A later comparison shows how the scores change when evaluated on only the two displayed components.
    """)
    return


@app.cell
def _(arrest_view):
    arrest_view.plot_data(title="PCA space (2 components)")
    return


@app.cell
def _(arrest_models, arrests_z, cv, pd):
    arrest_score_view = cv.ClusterVisualizer(arrests_z)
    arrest_picture_labels = pd.DataFrame(
        {f"k={_k}": _model.labels_ for _k, _model in arrest_models.items()}
    )
    arrest_score_view.plot_silhouette(
        arrest_picture_labels, title="Silhouette scores: four standardized variables"
    )
    return arrest_picture_labels, arrest_score_view


@app.cell
def _(arrest_picture_labels, arrest_score_view):
    arrest_score_view.plot_davies_bouldin(
        arrest_picture_labels, title="Davies–Bouldin: four standardized variables"
    )
    return


@app.cell
def _(arrest_picture_labels, arrest_score_view):
    arrest_score_view.plot_calinski_harabasz(
        arrest_picture_labels, title="Calinski–Harabasz: four standardized variables"
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
    arrest_scores.round(4)
    return arrest_calinski_k, arrest_davies_k, arrest_scores, arrest_silhouette_k


@app.cell
def _(arrest_models, arrest_silhouette_k, arrest_view):
    arrest_view.plot_clusters(
        arrest_models[arrest_silhouette_k].labels_,
        title=f"k = {arrest_silhouette_k} in the PCA space",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### How the data representation changes the scores

    The table below evaluates the same cluster assignments using only the **two PCA components**. Compare it with the four-variable table above. Projecting the data changes distances and loses some information, so the scores can change even though the cluster labels stay the same. Use the four-variable scores to assess the fitted clustering in its input space; use this table to understand how a two-dimensional display can differ.
    """)
    return


@app.cell
def _(arrest_models, arrest_plane, calinski_harabasz_score, davies_bouldin_score, pd, silhouette_score):
    arrest_pca_scores = pd.DataFrame([
        {"k": _k,
         "silhouette": silhouette_score(arrest_plane, _model.labels_),
         "davies_bouldin": davies_bouldin_score(arrest_plane, _model.labels_),
         "calinski_harabasz": calinski_harabasz_score(arrest_plane, _model.labels_)}
        for _k, _model in arrest_models.items()
    ]).set_index("k")
    arrest_pca_scores.round(4)
    return (arrest_pca_scores,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Describe the clusters

    We now examine **k = 2** in more detail to describe what the fitted groups contain. The preceding scores support comparing this solution with the other candidates, but do not prove that states belong to two natural categories.

    Compare the distributions of the four standardized variables within each cluster. A box shows the middle 50% of values and its line marks the median. The table reports cluster means in the original units. Interpret the measured variables rather than calling one cluster simply “better” or “worse.”
    """)
    return


@app.cell
def _(arrest_models, arrest_view):
    arrest_view.plot_clusters(
        arrest_models[2].labels_, title="USArrests: two clusters displayed in PCA space"
    )
    return


@app.cell
def _(arrest_models, arrests_z, px):
    arrest_profiles = arrests_z.assign(Cluster=arrest_models[2].labels_.astype(str))
    _profile_long = arrest_profiles.melt(id_vars="Cluster", var_name="Variable", value_name="Standardized value")
    _profile_fig = px.box(
        _profile_long, x="Cluster", y="Standardized value", color="Variable",
        title="USArrests: variable distributions within the two clusters",
        color_discrete_sequence=px.colors.qualitative.G10,
    )
    _profile_fig.update_layout(width=800, height=450,
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.15))
    _profile_fig
    return (arrest_profiles,)


@app.cell
def _(arrest_columns, arrest_models, arrests):
    arrest_cluster_means = (
        arrests[arrest_columns].assign(Cluster=arrest_models[2].labels_)
        .groupby("Cluster").mean()
    )
    arrest_cluster_means.round(4)
    return (arrest_cluster_means,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Use `arrest_cluster_means` to identify which cluster has the higher mean assault arrest rate. Use `idxmax()`. Can its label alone tell you what the cluster means?
    """)
    return


@app.cell
def _(arrest_cluster_means):
    student_high_assault_cluster = None
    student_high_assault_cluster
    return (student_high_assault_cluster,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
```python
student_high_assault_cluster = int(arrest_cluster_means["Assault"].idxmax())
student_high_assault_cluster
```
Read that row of the table to describe its other variables. Cluster labels are arbitrary identifiers and may swap between fitted solutions. The label itself has no substantive meaning.
""")}, expanded=False)
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
    mo,
    silhouette_k,
):
    mo.md(f"""
    ## Conclusions

    1. **K-means** assigns observations to their nearest center and updates each center using the mean of its 
    assigned observations.
    2. The number of clusters must be specified beforehand. In the synthetic example, too few clusters merge 
    visible groups, while additional clusters divide them.
    3. The elbow curve and the three validation scores support choosing k = 4 for the synthetic data. These tools 
    help compare solutions but do not guarantee a single correct choice.
    4. Standardizing the USArrests variables prevents differences in numerical scale from dominating the 
    distances. **K-means** uses all four standardized variables; PCA provides a two-dimensional visualization.
    5. For USArrests, all three validation scores favor k = 2 among the evaluated solutions. The box plots and 
    original-unit means help describe how the fitted groups differ in arrest rates and urban population 
    percentage.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check your understanding

    1. After assigning points to their nearest centers, how does k-means update each center?
    2. Why is choosing the largest k just because it has the smallest inertia unhelpful?
    3. Which validation score should be small? What does a silhouette score near zero suggest?
    4. Why standardize the USArrests columns before fitting k-means?
    5. Are the states clustered using two PCA components or four standardized variables? Which data do the main score bars evaluate?
    6. If the scores favor different values of k, has the analysis failed?
    7. Does a cluster label such as 1 mean higher arrest rates? How should you describe that group?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
1. Replace each center with the mean of the points assigned to it.
2. Adding centers can reduce inertia by splitting useful groups. Compare the elbow, internal scores, plots, and the purpose of the analysis instead.
3. Davies–Bouldin should be small. A silhouette near zero suggests points lie close to a boundary between clusters.
4. Their units and numerical spreads differ; otherwise a large-scale variable can dominate distances.
5. Four standardized variables are used for fitting and for the main score bars. Two PCA components are used for display and the PCA comparison table.
6. No. The scores emphasize different aspects of separation and spread. Report disagreement and justify a practical choice; there may be no single compelling k.
7. No. Labels are arbitrary. Inspect the variable distributions and original-unit means to describe the group's measured characteristics.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - [KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
    - [USArrests dataset documentation](https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/USArrests.html)
    - VanderPlas, J. (2017). *Python Data Science Handbook*. O'Reilly Media. Chapter 5.
    """)
    return


if __name__ == "__main__":
    app.run()
