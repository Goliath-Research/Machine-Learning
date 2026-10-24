# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
#     "pandas",
#     "plotly",
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
    from sklearn.cluster import DBSCAN, OPTICS
    from sklearn.datasets import make_blobs, make_circles, make_moons
    from sklearn.metrics import adjusted_rand_score
    from sklearn.preprocessing import StandardScaler

    return (
        DBSCAN,
        OPTICS,
        StandardScaler,
        adjusted_rand_score,
        cv,
        make_blobs,
        make_circles,
        make_moons,
        mo,
        np,
        pd,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Density-Based Clustering

    ## Objectives

    - Mark core points, border points, and noise with DBSCAN.
    - Watch `eps` and `min_samples` merge groups or turn them into noise.
    - Use OPTICS on a cloud whose groups do not share one density, then on moons and rings.

    ## Background

    DBSCAN looks for points that have at least `min_samples` neighbors inside a ball of radius `eps`. Those points are core points. A border point is inside that ball of a core point but is not itself a core point. Anything else is noise, labeled `-1`.

    The number of clusters is a result, not an input. One `eps` is a single density. OPTICS orders the points by how far the algorithm has to reach to connect them, so groups of different densities can be read from the same ordering. `min_cluster_size` can be a fraction of the sample. `xi` decides how steep a drop in reachability has to be before a new cluster starts.

    Every picture is a Plotly figure from `ClusterVisualizer`. Zoom is part of the lesson: a tight group and a wide group cannot be read on one static set of axes. The default `eps` is 0.5 and the default `min_samples` is 5. Cluster counts leave the noise label out.

    ## Datasets Used

    - **Two tight blobs.** 50 points, `random_state=0`, `cluster_std=0.5`.
    - **Three densities.** 100 points at three centers, with standard deviations 0.5, 0.1, and 2.
    - **Half moons.** 500 points, noise 0.05, `random_state=0`.
    - **Concentric circles.** 1,000 points, noise 0.08, factor 0.2, `random_state=0`.
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
def _(DBSCAN, cluster_census, cv, make_blobs, pd):
    blobs, _blob_labels = make_blobs(
        random_state=0, centers=2, cluster_std=0.5, n_samples=50
    )
    blob_view = cv.ClusterVisualizer(pd.DataFrame(blobs, columns=["x", "y"]))
    blob_models = {
        _eps: DBSCAN(eps=_eps).fit(blobs) for _eps in (0.35, 0.5, 1, 2)
    }
    eps_table = pd.DataFrame(
        [
            {
                "eps": _eps,
                "clusters": cluster_census(_model.labels_)[0],
                "noise": cluster_census(_model.labels_)[1],
                "core": len(_model.core_sample_indices_),
            }
            for _eps, _model in blob_models.items()
        ]
    ).set_index("eps")
    print(eps_table.to_string())
    blob_view.plot_data(title="Two blobs")
    return blob_models, blob_view, eps_table


@app.cell
def _(blob_models, blob_view):
    blob_view.plot_density_based_clustering(
        blob_models[0.5].labels_, title="DBSCAN Clustering"
    )
    return


@app.cell
def _(blob_models, blob_view):
    blob_view.plot_DBSCAN_categories(
        blob_models[0.5].labels_,
        blob_models[0.5].core_sample_indices_,
        title="DBSCAN Clustering (eps=0.5)",
    )
    return


@app.cell
def _(blob_models, blob_view):
    blob_view.plot_DBSCAN_categories(
        blob_models[0.35].labels_,
        blob_models[0.35].core_sample_indices_,
        title="DBSCAN Clustering (eps=0.35)",
    )
    return


@app.cell
def _(blob_models, blob_view):
    blob_view.plot_DBSCAN_categories(
        blob_models[1].labels_,
        blob_models[1].core_sample_indices_,
        title="DBSCAN Clustering (eps=1)",
    )
    return


@app.cell
def _(blob_models, blob_view):
    blob_view.plot_DBSCAN_categories(
        blob_models[2].labels_,
        blob_models[2].core_sample_indices_,
        title="DBSCAN Clustering (eps=2)",
    )
    return


@app.cell
def _(DBSCAN, blobs, cluster_census, pd):
    sample_models = {
        _samples: DBSCAN(min_samples=_samples).fit(blobs) for _samples in (3, 5, 8)
    }
    sample_table = pd.DataFrame(
        [
            {
                "min_samples": _samples,
                "clusters": cluster_census(_model.labels_)[0],
                "noise": cluster_census(_model.labels_)[1],
                "core": len(_model.core_sample_indices_),
            }
            for _samples, _model in sample_models.items()
        ]
    ).set_index("min_samples")
    print("eps stays at the default 0.5")
    print(sample_table.to_string())
    return sample_models, sample_table


@app.cell
def _(blob_view, sample_models):
    blob_view.plot_DBSCAN_categories(
        sample_models[3].labels_,
        sample_models[3].core_sample_indices_,
        title="DBSCAN Clustering (min_samples=3)",
    )
    return


@app.cell
def _(blob_view, sample_models):
    blob_view.plot_DBSCAN_categories(
        sample_models[5].labels_,
        sample_models[5].core_sample_indices_,
        title="DBSCAN Clustering (min_samples=5)",
    )
    return


@app.cell
def _(blob_view, sample_models):
    blob_view.plot_DBSCAN_categories(
        sample_models[8].labels_,
        sample_models[8].core_sample_indices_,
        title="DBSCAN Clustering (min_samples=8)",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A smaller neighborhood leaves points outside every core, so the noise count rises. A larger neighborhood can join the two blobs into one cluster. Raising `min_samples` asks each core point for more neighbors, which also creates noise.

    The next cloud has a tight group, a medium group, and a wide group. Zoom until the tight group opens up. DBSCAN with one `eps` cannot be right for all three densities at once.
    """)
    return


@app.cell
def _(DBSCAN, OPTICS, cluster_census, cv, make_blobs, np, pd):
    np.random.seed(0)
    varied, _varied_labels = make_blobs(
        n_samples=100,
        centers=[(0, 0), (6, 3), (2, 10)],
        cluster_std=[0.5, 0.1, 2.0],
    )
    varied_view = cv.ClusterVisualizer(pd.DataFrame(varied, columns=["x", "y"]))
    varied_dbscan = DBSCAN(eps=0.5, min_samples=10).fit(varied)
    varied_optics = {
        "min_samples=10, xi=0.05": OPTICS(min_samples=10, xi=0.05).fit(varied),
        "min_samples=50": OPTICS(min_samples=50).fit(varied),
        "min_samples=5": OPTICS(min_samples=5).fit(varied),
        "xi=0.9": OPTICS(min_samples=10, xi=0.9).fit(varied),
        "xi=0.3": OPTICS(min_samples=10, xi=0.3).fit(varied),
        "min_cluster_size=0.3": OPTICS(min_cluster_size=0.3).fit(varied),
        "min_cluster_size=0.4": OPTICS(min_cluster_size=0.4).fit(varied),
        "min_cluster_size=0.1": OPTICS(min_cluster_size=0.1).fit(varied),
    }
    varied_db_groups, varied_db_noise = cluster_census(varied_dbscan.labels_)
    varied_rows = [
        {
            "fit": "DBSCAN eps=0.5, min_samples=10",
            "clusters": varied_db_groups,
            "noise": varied_db_noise,
        }
    ]
    for _name, _model in varied_optics.items():
        _groups, _noise = cluster_census(_model.labels_)
        varied_rows.append({"fit": _name, "clusters": _groups, "noise": _noise})
    varied_table = pd.DataFrame(varied_rows).set_index("fit")
    print(varied_table.to_string())
    varied_view.plot_data(title="Three densities")
    return varied_dbscan, varied_optics, varied_table, varied_view


@app.cell
def _(varied_dbscan, varied_view):
    varied_view.plot_DBSCAN_categories(
        varied_dbscan.labels_,
        varied_dbscan.core_sample_indices_,
        title="DBSCAN Clustering",
    )
    return


@app.cell
def _(varied_optics, varied_view):
    varied_view.plot_density_based_clustering(
        varied_optics["min_samples=10, xi=0.05"].labels_,
        title="OPTICS Clustering",
    )
    return


@app.cell
def _(varied_optics, varied_view):
    varied_view.plot_density_based_clustering(
        varied_optics["min_samples=50"].labels_,
        title="OPTICS Clustering (min_samples=50)",
    )
    return


@app.cell
def _(varied_optics, varied_view):
    varied_view.plot_density_based_clustering(
        varied_optics["min_samples=5"].labels_,
        title="OPTICS Clustering (min_samples=5)",
    )
    return


@app.cell
def _(varied_optics, varied_view):
    varied_view.plot_density_based_clustering(
        varied_optics["xi=0.9"].labels_, title="OPTICS Clustering (xi=0.9)"
    )
    return


@app.cell
def _(varied_optics, varied_view):
    varied_view.plot_density_based_clustering(
        varied_optics["xi=0.3"].labels_, title="OPTICS Clustering (xi=0.3)"
    )
    return


@app.cell
def _(varied_optics, varied_view):
    varied_view.plot_density_based_clustering(
        varied_optics["min_cluster_size=0.3"].labels_,
        title="OPTICS Clustering (min_cluster_size=0.3)",
    )
    return


@app.cell
def _(varied_optics, varied_view):
    varied_view.plot_density_based_clustering(
        varied_optics["min_cluster_size=0.4"].labels_,
        title="OPTICS Clustering (min_cluster_size=0.4)",
    )
    return


@app.cell
def _(varied_optics, varied_view):
    varied_view.plot_density_based_clustering(
        varied_optics["min_cluster_size=0.1"].labels_,
        title="OPTICS Clustering (min_cluster_size=0.1)",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Moons and rings

    The pictures use the original coordinates, so the shapes stay recognizable and you can zoom. DBSCAN and OPTICS are fit on the standardized coordinates, which is what the notebook does after drawing the raw cloud. Core indices are passed into `plot_DBSCAN_categories`, so core and border are different colors. Noise stays grey.
    """)
    return


@app.cell
def _(
    DBSCAN,
    OPTICS,
    StandardScaler,
    adjusted_rand_score,
    cluster_census,
    cv,
    make_circles,
    make_moons,
    pd,
):
    moons, moon_arms = make_moons(500, noise=0.05, random_state=0)
    circles, circle_rings = make_circles(
        n_samples=1000, random_state=0, noise=0.08, factor=0.2
    )
    moon_view = cv.ClusterVisualizer(pd.DataFrame(moons, columns=["x", "y"]))
    circle_view = cv.ClusterVisualizer(pd.DataFrame(circles, columns=["x", "y"]))
    moon_db = DBSCAN(min_samples=10).fit(StandardScaler().fit_transform(moons))
    moon_op = OPTICS(min_cluster_size=0.5).fit_predict(
        StandardScaler().fit_transform(moons)
    )
    circle_db = DBSCAN(min_samples=10).fit(StandardScaler().fit_transform(circles))
    circle_op = OPTICS(min_cluster_size=0.5).fit_predict(
        StandardScaler().fit_transform(circles)
    )
    moon_db_groups, moon_db_noise = cluster_census(moon_db.labels_)
    moon_op_groups, moon_op_noise = cluster_census(moon_op)
    circle_db_groups, circle_db_noise = cluster_census(circle_db.labels_)
    circle_op_groups, circle_op_noise = cluster_census(circle_op)
    moon_db_ari = adjusted_rand_score(moon_arms, moon_db.labels_)
    moon_op_ari = adjusted_rand_score(moon_arms, moon_op)
    circle_db_ari = adjusted_rand_score(circle_rings, circle_db.labels_)
    circle_op_ari = adjusted_rand_score(circle_rings, circle_op)
    print(
        f"Moons DBSCAN: {moon_db_groups} clusters, {moon_db_noise} noise, ARI {moon_db_ari:.3f}"
    )
    print(
        f"Moons OPTICS: {moon_op_groups} clusters, {moon_op_noise} noise, ARI {moon_op_ari:.3f}"
    )
    print(
        f"Rings DBSCAN: {circle_db_groups} clusters, {circle_db_noise} noise, ARI {circle_db_ari:.3f}"
    )
    print(
        f"Rings OPTICS: {circle_op_groups} clusters, {circle_op_noise} noise, ARI {circle_op_ari:.3f}"
    )
    moon_view.plot_data(title="Half Moons")
    return (
        circle_db,
        circle_db_ari,
        circle_db_groups,
        circle_db_noise,
        circle_op,
        circle_op_ari,
        circle_view,
        moon_db,
        moon_db_ari,
        moon_db_groups,
        moon_db_noise,
        moon_op,
        moon_op_ari,
        moon_view,
    )


@app.cell
def _(moon_db, moon_view):
    moon_view.plot_density_based_clustering(
        moon_db.labels_, title="Half Moons - DBSCAN Clustering"
    )
    return


@app.cell
def _(moon_db, moon_view):
    moon_view.plot_DBSCAN_categories(
        moon_db.labels_,
        moon_db.core_sample_indices_,
        title="Half Moons - DBSCAN Clustering",
    )
    return


@app.cell
def _(moon_op, moon_view):
    moon_view.plot_density_based_clustering(
        moon_op, title="Half Moons - OPTICS Clustering"
    )
    return


@app.cell
def _(circle_view):
    circle_view.plot_data(title="Concentric Circles")
    return


@app.cell
def _(circle_db, circle_view):
    circle_view.plot_density_based_clustering(
        circle_db.labels_, title="Concentric Circles - DBSCAN Clustering"
    )
    return


@app.cell
def _(circle_db, circle_view):
    circle_view.plot_DBSCAN_categories(
        circle_db.labels_,
        circle_db.core_sample_indices_,
        title="Concentric Circles - DBSCAN Clustering",
    )
    return


@app.cell
def _(circle_op, circle_view):
    circle_view.plot_density_based_clustering(
        circle_op, title="Concentric Circles - OPTICS Clustering"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Which label does DBSCAN give a point that is neither a core point nor a border point? Replace `None` with that number.
    """)
    return


@app.cell
def _():
    student_noise = None
    print(student_noise)
    return (student_noise,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r'''
```python
student_noise = -1
```
**-1.** Noise is not a cluster. In the Plotly scatters that label is drawn in grey, and the cluster counts leave it out. Zoom a tight group before you decide it is missing.
''')
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(
    circle_db_ari,
    circle_db_groups,
    circle_db_noise,
    circle_op_ari,
    eps_table,
    mo,
    moon_db_ari,
    moon_db_groups,
    moon_db_noise,
    moon_op_ari,
    sample_table,
    varied_table,
):
    mo.md(f"""
    ## Conclusions

    **Key takeaways:**

    - On the two blobs, `eps=0.5` and `min_samples=5` find {int(eps_table.loc[0.5, "clusters"])} clusters, {int(eps_table.loc[0.5, "core"])} core points, and {int(eps_table.loc[0.5, "noise"])} noise points.
    - At `eps=0.35` there are {int(eps_table.loc[0.35, "noise"])} noise points. At `eps=2` there are {int(eps_table.loc[2.0, "clusters"])} clusters and {int(eps_table.loc[2.0, "noise"])} noise points. At `min_samples=8`, with `eps` still 0.5, there are {int(sample_table.loc[8, "noise"])} noise points.
    - On the three densities, DBSCAN finds {int(varied_table.loc["DBSCAN eps=0.5, min_samples=10", "clusters"])} clusters and {int(varied_table.loc["DBSCAN eps=0.5, min_samples=10", "noise"])} noise points. OPTICS with `min_samples=10` and `xi=0.05` finds {int(varied_table.loc["min_samples=10, xi=0.05", "clusters"])} clusters and {int(varied_table.loc["min_samples=10, xi=0.05", "noise"])} noise points. The tight group is visible once the Plotly scatter is zoomed.
    - On the moons, DBSCAN finds {moon_db_groups} clusters and {moon_db_noise} noise points, with adjusted Rand index {moon_db_ari:.3f}. OPTICS has adjusted Rand index {moon_op_ari:.3f}.
    - On the rings, DBSCAN finds {circle_db_groups} clusters and {circle_db_noise} noise points, with adjusted Rand index {circle_db_ari:.3f}. OPTICS has adjusted Rand index {circle_op_ari:.3f}.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check your understanding

    1. What makes a point a core point?
    2. What is a border point?
    3. Why can one value of `eps` miss a wide group and keep a tight group?
    4. What does OPTICS order the points by?
    5. What does a `min_cluster_size` below 1 mean?
    6. Why is the adjusted Rand index useful on the moons when the labels are arbitrary?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
1. **Enough neighbors inside `eps`:** The count includes the point itself, and it must reach `min_samples`.
2. **A neighbor of a core point that is not a core point:** It joins the cluster, but it cannot recruit points on its own.
3. **The wide group has larger gaps:** A ball that is large enough for the tight group can be too small for the wide one, so those points become noise. Zoom the Plotly scatter to see the tight group.
4. **Reachability:** The order records how far the search had to go to connect each next point. Dense groups appear as valleys in that plot.
5. **A fraction of the rows:** `0.5` means a cluster must contain at least half the sample. A number of 1 or more is a count of points.
6. **It compares pairs, not names:** Two points on the same moon should share a cluster, whatever that cluster is called. Noise points are their own label, so they count as disagreements.
""")
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - [DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
    - [OPTICS](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html)
    - VanderPlas, J. (2017). *Python Data Science Handbook*. O'Reilly Media. Chapter 5.
    """)
    return


if __name__ == "__main__":
    app.run()
