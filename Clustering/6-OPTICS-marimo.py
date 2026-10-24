# /// script
# requires-python = ">=3.10"
# dependencies = [
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
    import plotly.graph_objects as go
    from pydataset import data
    from sklearn.cluster import DBSCAN, OPTICS, cluster_optics_dbscan
    from sklearn.datasets import make_blobs
    from sklearn.preprocessing import StandardScaler

    return (
        DBSCAN,
        OPTICS,
        StandardScaler,
        cluster_optics_dbscan,
        cv,
        data,
        go,
        make_blobs,
        mo,
        np,
        pd,
        px,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # OPTICS Clustering

    ## Objectives

    - Read a reachability plot as an ordering, not as a finished partition.
    - Cut that same ordering at several values of `eps` with `cluster_optics_dbscan`.
    - Cluster earthquakes by standardized depth and magnitude, and map the labels on longitude and latitude.

    ## Background

    OPTICS visits points from dense regions toward sparse ones. The reachability of a point is how far that visit had to stretch. A valley in the reachability plot is a dense group. The first reachability in the ordering is infinite, because nothing came before it.

    `cluster_optics_dbscan` does not fit a new model. It cuts the stored ordering: a point joins the current cluster when its reachability is at most `eps`, and it becomes noise when the reachability is larger. `plot_optics_dbscan` draws that cut twice: the scatter, which you can zoom, and the reachability plot with a horizontal line at `eps`.

    A static scatter cannot show a very tight group next to a very wide one. Use the Plotly zoom.

    ## Datasets Used

    - **Three densities.** 100 points at (0, 0), (3, 10), and (12, 5), with standard deviations 0.6, 0.1, and 3.
    - **Four densities.** 200 points, `random_state=2`, with standard deviations 0.3, 0.05, 1.2, and 0.7.
    - **Quakes.** 1,000 earthquakes from `pydataset`, with latitude, longitude, depth, magnitude, and the number of reporting stations.
    """)
    return


@app.cell
def _(np):
    def cluster_census(labels):
        """Count clusters and noise points. The noise label is -1 and is not a cluster."""
        labels = np.asarray(labels)
        noise = int(np.sum(labels == -1))
        groups = len(set(labels.tolist())) - (1 if noise else 0)
        return groups, noise

    return (cluster_census,)


@app.cell
def _(go, pd, px):
    def plot_box_with_noise(data, cluster, title="Boxplots with Noise Points"):
        """Plot boxplots for each feature in the data, with the noise points highlighted.

        data: DataFrame with the features
        cluster: Array with the cluster labels
        """
        data = data.copy()
        data["cluster"] = cluster
        data_melted = data.melt(id_vars="cluster", var_name="feature", value_name="value")
        noise_data = data_melted[data_melted["cluster"] == -1]
        fig = px.box(data_melted, x="feature", y="value", notched=True)
        fig.update_layout(width=600, height=400, title=title, legend=dict(x=0.8, y=1.12))
        fig.add_trace(
            go.Scatter(
                x=noise_data["feature"],
                y=noise_data["value"],
                mode="markers",
                marker=dict(color="grey", size=7),
                name="Noise",
            )
        )
        return fig

    return (plot_box_with_noise,)


@app.cell
def _(OPTICS, cluster_census, cv, make_blobs, np, pd):
    np.random.seed(0)
    wide, _wide_labels = make_blobs(
        n_samples=100,
        centers=[(0, 0), (3, 10), (12, 5)],
        cluster_std=[0.6, 0.1, 3.0],
    )
    wide_view = cv.ClusterVisualizer(pd.DataFrame(wide, columns=["x", "y"]))
    wide_optics = OPTICS(min_samples=10, min_cluster_size=0.05).fit(wide)
    wide_groups, wide_noise = cluster_census(wide_optics.labels_)
    print(f"OPTICS labels: {wide_groups} clusters, {wide_noise} noise")
    print(
        "First reachability in the ordering:",
        wide_optics.reachability_[wide_optics.ordering_][0],
    )
    wide_view.plot_data(title="Three densities")
    return wide_groups, wide_noise, wide_optics, wide_view


@app.cell
def _(wide_optics, wide_view):
    wide_view.plot_clusters(wide_optics.labels_, title="Data - OPTICS Algorithm")
    return


@app.cell
def _(px, wide_optics):
    _reach = wide_optics.reachability_[wide_optics.ordering_]
    _fig = px.line(
        x=list(range(len(_reach))),
        y=_reach,
        title="Reachability Plot",
    )
    _fig.update_layout(width=600, height=400, xaxis_title="Order", yaxis_title="Reachability")
    _fig
    return


@app.cell
def _(cluster_census, cluster_optics_dbscan, pd, wide_optics):
    wide_rows = []
    for _eps in (0.5, 1, 3, 4, 5):
        _labels = cluster_optics_dbscan(
            reachability=wide_optics.reachability_,
            core_distances=wide_optics.core_distances_,
            ordering=wide_optics.ordering_,
            eps=_eps,
        )
        _groups, _noise = cluster_census(_labels)
        wide_rows.append({"eps": _eps, "clusters": _groups, "noise": _noise})
    wide_cuts = pd.DataFrame(wide_rows).set_index("eps")
    print(wide_cuts.to_string())
    return (wide_cuts,)


@app.cell
def _(wide_optics, wide_view):
    wide_view.plot_optics_dbscan(wide_optics, eps=0.5)
    return


@app.cell
def _(wide_optics, wide_view):
    wide_view.plot_optics_dbscan(wide_optics, eps=1)
    return


@app.cell
def _(wide_optics, wide_view):
    wide_view.plot_optics_dbscan(wide_optics, eps=3)
    return


@app.cell
def _(wide_optics, wide_view):
    wide_view.plot_optics_dbscan(wide_optics, eps=4)
    return


@app.cell
def _(wide_optics, wide_view):
    wide_view.plot_optics_dbscan(wide_optics, eps=5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Each cut uses the same OPTICS ordering. A small `eps` leaves the sparse group as noise. A larger `eps` reaches farther and can join valleys the reachability plot still shows as separate. The tight group, standard deviation 0.1, is a few pixels until you zoom.

    The next cloud has four densities, including one with standard deviation 0.05. OPTICS is fit with `min_cluster_size=0.2`, so a cluster must hold at least a fifth of the points.
    """)
    return


@app.cell
def _(OPTICS, cluster_census, cluster_optics_dbscan, cv, make_blobs, pd):
    four, _four_labels = make_blobs(
        n_samples=200,
        centers=[(-2, 3), (0, 2), (3, -2), (7, 3)],
        cluster_std=[0.3, 0.05, 1.2, 0.7],
        random_state=2,
    )
    four_view = cv.ClusterVisualizer(pd.DataFrame(four, columns=["x", "y"]))
    four_optics = OPTICS(min_cluster_size=0.2).fit(four)
    four_groups, four_noise = cluster_census(four_optics.labels_)
    four_rows = [{"eps": "OPTICS labels", "clusters": four_groups, "noise": four_noise}]
    for _eps in (0.5, 1, 1.5, 2, 4):
        _labels = cluster_optics_dbscan(
            reachability=four_optics.reachability_,
            core_distances=four_optics.core_distances_,
            ordering=four_optics.ordering_,
            eps=_eps,
        )
        _groups, _noise = cluster_census(_labels)
        four_rows.append({"eps": str(_eps), "clusters": _groups, "noise": _noise})
    four_cuts = pd.DataFrame(four_rows).set_index("eps")
    print(four_cuts.to_string())
    four_view.plot_data(title="Four densities")
    return four_cuts, four_groups, four_noise, four_optics, four_view


@app.cell
def _(four_optics, four_view):
    four_view.plot_density_based_clustering(
        four_optics.labels_, title="OPTICS labels"
    )
    return


@app.cell
def _(four_optics, four_view):
    four_view.plot_optics_dbscan(four_optics, eps=0.5)
    return


@app.cell
def _(four_optics, four_view):
    four_view.plot_optics_dbscan(four_optics, eps=1)
    return


@app.cell
def _(four_optics, four_view):
    four_view.plot_optics_dbscan(four_optics, eps=1.5)
    return


@app.cell
def _(four_optics, four_view):
    four_view.plot_optics_dbscan(four_optics, eps=2)
    return


@app.cell
def _(four_optics, four_view):
    four_view.plot_optics_dbscan(four_optics, eps=4)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Earthquakes

    Depth is in kilometres and magnitude is a score near 4 to 6, so the clustering uses standardized depth and magnitude. The scaler uses every earthquake, because there is no holdout.

    `ClusterVisualizer` is given the full quakes table. Its scatter uses the first two columns, latitude and longitude, which is the map. DBSCAN uses `eps=0.5` and `min_samples=5` on the standardized pair. OPTICS uses `min_samples=5` and `min_cluster_size=0.2`.
    """)
    return


@app.cell
def _(
    DBSCAN,
    OPTICS,
    StandardScaler,
    cluster_census,
    cv,
    data,
    pd,
):
    quakes = data("quakes")
    quake_columns = ["depth", "mag"]
    quake_z = pd.DataFrame(
        StandardScaler().fit_transform(quakes[quake_columns]),
        columns=quake_columns,
        index=quakes.index,
    )
    quake_view = cv.ClusterVisualizer(quakes)
    quake_db = DBSCAN(eps=0.5, min_samples=5).fit(quake_z)
    quake_op = OPTICS(min_samples=5, min_cluster_size=0.2).fit(quake_z)
    quake_db_groups, quake_db_noise = cluster_census(quake_db.labels_)
    quake_op_groups, quake_op_noise = cluster_census(quake_op.labels_)
    print(f"{len(quakes)} earthquakes, columns {list(quakes.columns)}")
    print(f"DBSCAN: {quake_db_groups} clusters, {quake_db_noise} noise")
    print(f"OPTICS: {quake_op_groups} clusters, {quake_op_noise} noise")
    quake_view.plot_data(title="Earthquakes")
    return quake_db, quake_db_groups, quake_db_noise, quake_op, quake_op_groups, quake_op_noise, quake_view, quake_z


@app.cell
def _(quake_db, quake_view):
    quake_view.plot_density_based_clustering(
        quake_db.labels_, title="Quakes - DBSCAN"
    )
    return


@app.cell
def _(quake_op, quake_view):
    quake_view.plot_density_based_clustering(
        quake_op.labels_, title="Quakes - OPTICS"
    )
    return


@app.cell
def _(plot_box_with_noise, quake_db, quake_z):
    plot_box_with_noise(
        quake_z, quake_db.labels_, title="Quakes - Boxplots with Noise Points"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Which function cuts an OPTICS ordering at one `eps` without fitting a new model? Replace `None` with its name.
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
**cluster_optics_dbscan.** It uses the reachability, core distances, and ordering already stored on the OPTICS fit. `plot_optics_dbscan` draws that cut, including the reachability plot and the `eps` line.
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
    wide_groups,
    wide_noise,
):
    mo.md(f"""
    ## Conclusions

    **Key takeaways:**

    - The OPTICS labels on the three-density cloud contain {wide_groups} clusters and {wide_noise} noise points. Cutting that same ordering at `eps=0.5` leaves {int(wide_cuts.loc[0.5, "clusters"])} clusters and {int(wide_cuts.loc[0.5, "noise"])} noise points. At `eps=3` there are {int(wide_cuts.loc[3.0, "clusters"])} clusters and {int(wide_cuts.loc[3.0, "noise"])} noise points. At `eps=5` there are {int(wide_cuts.loc[5.0, "clusters"])} clusters and {int(wide_cuts.loc[5.0, "noise"])} noise points.
    - On the four-density cloud, the OPTICS labels contain {four_groups} clusters and {four_noise} noise points. The `eps=0.5` cut contains {int(four_cuts.loc["0.5", "clusters"])} clusters and {int(four_cuts.loc["0.5", "noise"])} noise points. The `eps=1.5` cut contains {int(four_cuts.loc["1.5", "clusters"])} clusters and {int(four_cuts.loc["1.5", "noise"])} noise points. The standard deviation 0.05 group is readable after a zoom.
    - On the earthquakes, DBSCAN finds {quake_db_groups} clusters and {quake_db_noise} noise points. OPTICS finds {quake_op_groups} clusters and {quake_op_noise} noise points. Both fits used standardized depth and magnitude. The map is the latitude and longitude stored on the visualizer.
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
1. **How far the visit stretched to reach the next point:** Dense groups are valleys. The first value is infinite.
2. **No:** It cuts the reachability, core distances, and ordering from the OPTICS fit you already have.
3. **Higher ridges in the reachability plot:** Points that were noise at a small `eps` can join a cluster when the cut sits above their reachability.
4. **At least a fifth of the rows:** `0.2` is a fraction of the sample. A cluster smaller than that is not kept.
5. **Depth and magnitude:** Latitude and longitude are the map drawn by `ClusterVisualizer`, not the columns the distance was computed on.
6. **The cut moved:** The ordering stays the same. Only the horizontal line on the reachability plot changes.
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
    - Ankerst, M., Breunig, M. M., Kriegel, H.-P., and Sander, J. (1999). OPTICS: Ordering points to identify the clustering structure. *ACM SIGMOD Record*, 28(2), 49–60.
    """)
    return


if __name__ == "__main__":
    app.run()
