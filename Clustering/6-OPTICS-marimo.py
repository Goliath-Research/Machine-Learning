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
    from sklearn.cluster import OPTICS, cluster_optics_dbscan
    from sklearn.datasets import make_blobs
    from sklearn.preprocessing import StandardScaler

    return OPTICS, StandardScaler, cluster_optics_dbscan, data, make_blobs


@app.cell
def _(np, pd, sns):
    def cluster_census(labels):
        labels = np.asarray(labels)
        noise = int(np.sum(labels == -1))
        groups = len(set(labels.tolist())) - (1 if noise else 0)
        return groups, noise

    def draw_clusters(axes, x, y, labels, title):
        names = np.where(np.asarray(labels) == -1, "noise", np.asarray(labels).astype(str))
        frame = pd.DataFrame({"x": np.asarray(x), "y": np.asarray(y), "cluster": names})
        clusters = sorted(name for name in frame["cluster"].unique() if name != "noise")
        palette = dict(zip(clusters, sns.color_palette("deep", n_colors=max(len(clusters), 1))))
        palette["noise"] = (0.65, 0.65, 0.65)
        present = [*clusters, "noise"] if "noise" in set(names) else clusters
        sns.scatterplot(
            data=frame,
            x="x",
            y="y",
            hue="cluster",
            hue_order=present,
            palette=palette,
            s=22,
            ax=axes,
            legend=False,
        )
        axes.set_title(title)

    return cluster_census, draw_clusters


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # OPTICS and a DBSCAN Cut

    ## Objectives

    - Read a reachability plot as an ordering of the points.
    - Cut that ordering at several values of `eps` with `cluster_optics_dbscan`.
    - Repeat the cut on a cloud with four densities, then on earthquake depth and magnitude.

    ## Background

    OPTICS visits points from dense regions outward and records a reachability distance for each one. A valley in that sequence is a dense group. `cluster_optics_dbscan` does not refit the points. It cuts the stored reachability at one `eps`: points whose reachability is at most `eps`, and that can be connected through such points, share a cluster. A larger `eps` reaches across wider gaps.

    ## Datasets Used

    - **Three densities.** 100 points at (0, 0), (3, 10), and (12, 5), with standard deviations 0.6, 0.1, and 3.
    - **Four densities.** 200 points, `random_state=2`, with standard deviations 0.3, 0.05, 1.2, and 0.7.
    - **Quakes.** 1,000 earthquakes from `pydataset`, with latitude, longitude, depth, magnitude, and the number of reporting stations.
    """)
    return


@app.cell
def _(OPTICS, draw_clusters, make_blobs, np, plt):
    wide, _wide_labels = make_blobs(
        n_samples=100,
        centers=[(0, 0), (3, 10), (12, 5)],
        cluster_std=[0.6, 0.1, 3.0],
        random_state=0,
    )
    wide_optics = OPTICS(min_samples=10, min_cluster_size=0.05).fit(wide)
    reach = wide_optics.reachability_[wide_optics.ordering_].astype(float)
    reach[~np.isfinite(reach)] = np.nan
    _figure, _axes = plt.subplots(1, 2, figsize=(10, 4))
    draw_clusters(
        _axes[0], wide[:, 0], wide[:, 1], wide_optics.labels_, "OPTICS labels"
    )
    _axes[1].plot(reach, color="firebrick")
    _axes[1].set_title("Reachability, in visit order")
    _axes[1].set_ylabel("Reachability")
    _figure.tight_layout()
    return _figure


@app.cell
def _(cluster_census, cluster_optics_dbscan, draw_clusters, pd, plt, wide, wide_optics):
    cut_rows = []
    _figure, _axes = plt.subplots(1, 3, figsize=(11, 3.5))
    for _ax, _eps in zip(_axes, (0.5, 3, 5)):
        _labels = cluster_optics_dbscan(
            reachability=wide_optics.reachability_,
            core_distances=wide_optics.core_distances_,
            ordering=wide_optics.ordering_,
            eps=_eps,
        )
        _groups, _noise = cluster_census(_labels)
        cut_rows.append({"eps": _eps, "clusters": _groups, "noise": _noise})
        draw_clusters(_ax, wide[:, 0], wide[:, 1], _labels, f"eps = {_eps:g}")
    wide_cuts = pd.DataFrame(cut_rows).set_index("eps")
    _figure.tight_layout()
    print(wide_cuts.to_string())
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The same OPTICS ordering produces a different partition at each cut. A small `eps` leaves the sparse group as noise. A large `eps` can join groups that the reachability plot still shows as separate valleys.

    The next cloud has four densities. OPTICS is fit with `min_cluster_size=0.2`, so a cluster must hold at least a fifth of the points. The cuts use that same fit.
    """)
    return


@app.cell
def _(
    OPTICS,
    cluster_census,
    cluster_optics_dbscan,
    draw_clusters,
    make_blobs,
    pd,
    plt,
):
    four, _four_labels = make_blobs(
        n_samples=200,
        centers=[(-2, 3), (0, 2), (3, -2), (7, 3)],
        cluster_std=[0.3, 0.05, 1.2, 0.7],
        random_state=2,
    )
    four_optics = OPTICS(min_cluster_size=0.2).fit(four)
    four_groups, four_noise = cluster_census(four_optics.labels_)
    four_rows = [
        {"eps": "OPTICS", "clusters": four_groups, "noise": four_noise}
    ]
    _figure, _axes = plt.subplots(1, 3, figsize=(11, 3.5))
    draw_clusters(
        _axes[0],
        four[:, 0],
        four[:, 1],
        four_optics.labels_,
        f"OPTICS: {four_groups} clusters, {four_noise} noise",
    )
    for _ax, _eps in zip(_axes[1:], (0.5, 1.5)):
        _labels = cluster_optics_dbscan(
            reachability=four_optics.reachability_,
            core_distances=four_optics.core_distances_,
            ordering=four_optics.ordering_,
            eps=_eps,
        )
        _groups, _noise = cluster_census(_labels)
        four_rows.append({"eps": str(_eps), "clusters": _groups, "noise": _noise})
        draw_clusters(_ax, four[:, 0], four[:, 1], _labels, f"eps = {_eps:g}")
    four_cuts = pd.DataFrame(four_rows).set_index("eps")
    _figure.tight_layout()
    print(four_cuts.to_string())
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Earthquakes

    `pydataset` supplies the quakes table. Depth is in kilometres and magnitude is a score near 4 to 6, so the clustering uses standardized depth and magnitude. Latitude and longitude are only the map. The scaler uses every earthquake, because there is no holdout.

    DBSCAN uses `eps=0.5` and `min_samples=5` on the standardized pair. OPTICS uses `min_samples=5` and `min_cluster_size=0.2`.
    """)
    return


@app.cell
def _(
    OPTICS,
    StandardScaler,
    cluster_census,
    data,
    draw_clusters,
    pd,
    plt,
):
    from sklearn.cluster import DBSCAN

    quakes = data("quakes")
    quake_z = StandardScaler().fit_transform(quakes[["depth", "mag"]])
    quake_db = DBSCAN(eps=0.5, min_samples=5).fit_predict(quake_z)
    quake_op = OPTICS(min_samples=5, min_cluster_size=0.2).fit_predict(quake_z)
    quake_db_groups, quake_db_noise = cluster_census(quake_db)
    quake_op_groups, quake_op_noise = cluster_census(quake_op)
    _figure, _axes = plt.subplots(1, 2, figsize=(10, 4))
    draw_clusters(
        _axes[0],
        quakes["long"],
        quakes["lat"],
        quake_db,
        f"DBSCAN: {quake_db_groups} clusters, {quake_db_noise} noise",
    )
    draw_clusters(
        _axes[1],
        quakes["long"],
        quakes["lat"],
        quake_op,
        f"OPTICS: {quake_op_groups} clusters, {quake_op_noise} noise",
    )
    _axes[0].set_xlabel("long")
    _axes[1].set_xlabel("long")
    _axes[0].set_ylabel("lat")
    _figure.tight_layout()
    print(f"{len(quakes)} earthquakes")
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Which function cuts an OPTICS ordering at one `eps`? Replace `None` with its name.
    """)
    return


@app.cell
def _():
    student_extract = None
    print(student_extract)
    return (student_extract,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r'''
```python
student_extract = "cluster_optics_dbscan"
```
**cluster_optics_dbscan.** It uses the reachability, core distances, and ordering already stored on the OPTICS fit.
''')
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(
    four_cuts,
    four_groups,
    four_noise,
    mo,
    quake_db_groups,
    quake_db_noise,
    quake_op_groups,
    quake_op_noise,
    wide_cuts,
):
    mo.md(f"""
    ## Conclusions

    **Key takeaways:**

    - One OPTICS fit can be cut at many values of `eps`. On the three-density cloud, `eps=0.5` leaves {int(wide_cuts.loc[0.5, "clusters"])} clusters and {int(wide_cuts.loc[0.5, "noise"])} noise points. `eps=5` leaves {int(wide_cuts.loc[5, "clusters"])} clusters and {int(wide_cuts.loc[5, "noise"])} noise points.
    - On the four-density cloud, the OPTICS labels themselves contain {four_groups} clusters and {four_noise} noise points. The `eps=0.5` cut contains {int(four_cuts.loc["0.5", "clusters"])} clusters and {int(four_cuts.loc["0.5", "noise"])} noise points.
    - On the earthquakes, DBSCAN finds {quake_db_groups} clusters and {quake_db_noise} noise points. OPTICS finds {quake_op_groups} clusters and {quake_op_noise} noise points. Both fits used standardized depth and magnitude. The map is longitude and latitude.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check your understanding

    1. What is recorded in the reachability sequence?
    2. Does `cluster_optics_dbscan` fit a new model?
    3. What does a larger `eps` reach across?
    4. What does `min_cluster_size=0.2` require?
    5. Which two quake columns were standardized for the clustering?
    6. Why can a point be noise at one `eps` and a cluster member at another?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
1. **How far the visit had to reach for each next point:** Dense groups show up as low stretches of that sequence.
2. **No:** It reads the ordering, reachability, and core distances from the OPTICS fit and applies one distance cut.
3. **Wider gaps:** Points that were too far apart at a small `eps` can join the same cluster.
4. **At least 20 percent of the rows:** A fraction below 1 is a share of the sample. A value of 1 or more is a count.
5. **Depth and magnitude:** Latitude and longitude place the result on the map.
6. **The cut moves:** A point just outside a small neighborhood can fall inside a larger one.
""")
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - [OPTICS](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html)
    - [cluster_optics_dbscan](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.cluster_optics_dbscan.html)
    - VanderPlas, J. (2017). *Python Data Science Handbook*. O'Reilly Media. Chapter 5.
    """)
    return


if __name__ == "__main__":
    app.run()
