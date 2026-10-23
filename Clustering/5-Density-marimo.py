# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "numpy",
#     "pandas",
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
    from sklearn.cluster import DBSCAN, OPTICS
    from sklearn.datasets import make_blobs, make_circles, make_moons
    from sklearn.metrics import adjusted_rand_score

    return DBSCAN, OPTICS, adjusted_rand_score, make_blobs, make_circles, make_moons


@app.cell
def _(np, pd, sns):
    def cluster_census(labels):
        labels = np.asarray(labels)
        noise = int(np.sum(labels == -1))
        groups = len(set(labels.tolist())) - (1 if noise else 0)
        return groups, noise

    def draw_clusters(axes, x, y, labels, title):
        names = np.asarray(labels).astype(str)
        names = np.where(np.asarray(labels) == -1, "noise", names)
        frame = pd.DataFrame({"x": np.asarray(x), "y": np.asarray(y), "cluster": names})
        clusters = sorted(name for name in frame["cluster"].unique() if name != "noise")
        palette = dict(zip(clusters, sns.color_palette("deep", n_colors=max(len(clusters), 1))))
        palette["noise"] = (0.65, 0.65, 0.65)
        sns.scatterplot(
            data=frame,
            x="x",
            y="y",
            hue="cluster",
            hue_order=[*clusters, "noise"] if "noise" in set(frame["cluster"]) else clusters,
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
    # Density-Based Clustering

    ## Objectives

    - Mark core points, border points, and noise with DBSCAN.
    - Watch `eps` and `min_samples` merge groups or turn them into noise.
    - Use OPTICS on a cloud whose groups do not share one density, then on moons and rings.

    ## Background

    DBSCAN looks for points that have at least `min_samples` neighbors inside a ball of radius `eps`. Those points are core points. A border point is inside that ball of a core point but is not itself a core point. Anything else is noise, labeled `-1`.

    The number of clusters is a result, not an input. One `eps` is a single density. OPTICS orders the points by how far the algorithm has to reach to connect them, so groups of different densities can be read from the same ordering. `min_cluster_size` can be a fraction of the sample. `xi` decides how steep a drop in reachability has to be before a new cluster starts.

    ## Datasets Used

    - **Two tight blobs.** 50 points, `random_state=0`, `cluster_std=0.5`.
    - **Three densities.** 100 points at three centers, with standard deviations 0.5, 0.1, and 2.
    - **Half moons.** 500 points, noise 0.05, `random_state=0`.
    - **Concentric circles.** 1,000 points, noise 0.08, factor 0.2, `random_state=0`.
    """)
    return


@app.cell
def _(DBSCAN, cluster_census, draw_clusters, make_blobs, pd, plt):
    blobs, _blob_labels = make_blobs(
        random_state=0, centers=2, cluster_std=0.5, n_samples=50
    )
    eps_rows = []
    _figure, _axes = plt.subplots(1, 4, figsize=(12, 3.2))
    for _ax, _eps in zip(_axes, (0.35, 0.5, 1, 2)):
        _model = DBSCAN(eps=_eps).fit(blobs)
        _groups, _noise = cluster_census(_model.labels_)
        _core = len(_model.core_sample_indices_)
        eps_rows.append(
            {"eps": _eps, "clusters": _groups, "noise": _noise, "core": _core}
        )
        draw_clusters(_ax, blobs[:, 0], blobs[:, 1], _model.labels_, f"eps = {_eps:g}")
    eps_table = pd.DataFrame(eps_rows).set_index("eps")
    _figure.tight_layout()
    print(eps_table.to_string())
    return _figure


@app.cell
def _(DBSCAN, blobs, cluster_census, pd):
    sample_rows = []
    for _samples in (3, 5, 8):
        _model = DBSCAN(min_samples=_samples).fit(blobs)
        _groups, _noise = cluster_census(_model.labels_)
        sample_rows.append(
            {
                "min_samples": _samples,
                "clusters": _groups,
                "noise": _noise,
                "core": len(_model.core_sample_indices_),
            }
        )
    sample_table = pd.DataFrame(sample_rows).set_index("min_samples")
    print("eps stays at the default 0.5")
    print(sample_table.to_string())


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A smaller neighborhood leaves points outside every core, so the noise count rises. A larger neighborhood can join the two blobs into one cluster. Raising `min_samples` asks each core point for more neighbors, which also creates noise.

    The next cloud has a tight group, a medium group, and a wide group. DBSCAN with `eps=0.5` and `min_samples=10` uses one density for all three. OPTICS uses `min_samples=10` and `xi=0.05`.
    """)
    return


@app.cell
def _(DBSCAN, OPTICS, cluster_census, draw_clusters, make_blobs, np, plt):
    varied, _varied_labels = make_blobs(
        n_samples=100,
        centers=[(0, 0), (6, 3), (2, 10)],
        cluster_std=[0.5, 0.1, 2.0],
        random_state=0,
    )
    varied_dbscan = DBSCAN(eps=0.5, min_samples=10).fit(varied)
    varied_optics = OPTICS(min_samples=10, xi=0.05).fit(varied)
    varied_db_groups, varied_db_noise = cluster_census(varied_dbscan.labels_)
    varied_op_groups, varied_op_noise = cluster_census(varied_optics.labels_)
    _figure, _axes = plt.subplots(1, 2, figsize=(9, 4))
    draw_clusters(
        _axes[0],
        varied[:, 0],
        varied[:, 1],
        varied_dbscan.labels_,
        f"DBSCAN: {varied_db_groups} clusters, {varied_db_noise} noise",
    )
    draw_clusters(
        _axes[1],
        varied[:, 0],
        varied[:, 1],
        varied_optics.labels_,
        f"OPTICS: {varied_op_groups} clusters, {varied_op_noise} noise",
    )
    _figure.tight_layout()
    return _figure


@app.cell
def _(
    DBSCAN,
    OPTICS,
    adjusted_rand_score,
    cluster_census,
    draw_clusters,
    make_circles,
    make_moons,
    plt,
):
    moons, moon_arms = make_moons(500, noise=0.05, random_state=0)
    circles, circle_rings = make_circles(
        n_samples=1000, random_state=0, noise=0.08, factor=0.2
    )
    moon_db = DBSCAN(min_samples=10).fit_predict(moons)
    moon_op = OPTICS(min_cluster_size=0.5).fit_predict(moons)
    circle_db = DBSCAN(min_samples=10).fit_predict(circles)
    circle_op = OPTICS(min_cluster_size=0.5).fit_predict(circles)
    moon_db_ari = adjusted_rand_score(moon_arms, moon_db)
    moon_op_ari = adjusted_rand_score(moon_arms, moon_op)
    circle_db_ari = adjusted_rand_score(circle_rings, circle_db)
    circle_op_ari = adjusted_rand_score(circle_rings, circle_op)
    moon_db_groups, moon_db_noise = cluster_census(moon_db)
    circle_db_groups, circle_db_noise = cluster_census(circle_db)
    _figure, _axes = plt.subplots(2, 2, figsize=(9, 8))
    draw_clusters(_axes[0, 0], moons[:, 0], moons[:, 1], moon_db, f"Moons, DBSCAN, ARI {moon_db_ari:.3f}")
    draw_clusters(_axes[0, 1], moons[:, 0], moons[:, 1], moon_op, f"Moons, OPTICS, ARI {moon_op_ari:.3f}")
    draw_clusters(
        _axes[1, 0], circles[:, 0], circles[:, 1], circle_db, f"Rings, DBSCAN, ARI {circle_db_ari:.3f}"
    )
    draw_clusters(
        _axes[1, 1], circles[:, 0], circles[:, 1], circle_op, f"Rings, OPTICS, ARI {circle_op_ari:.3f}"
    )
    for _ax in _axes.ravel():
        _ax.set_aspect("equal", adjustable="datalim")
    _figure.tight_layout()
    return _figure


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
**-1.** Noise is not a cluster. Counts of clusters leave that label out.
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
    varied_db_groups,
    varied_db_noise,
    varied_op_groups,
    varied_op_noise,
):
    mo.md(f"""
    ## Conclusions

    **Key takeaways:**

    - On the two blobs, `eps=0.5` and `min_samples=5` find {int(eps_table.loc[0.5, "clusters"])} clusters, {int(eps_table.loc[0.5, "core"])} core points, and {int(eps_table.loc[0.5, "noise"])} noise points.
    - At `eps=2` there are {int(eps_table.loc[2, "clusters"])} clusters and {int(eps_table.loc[2, "noise"])} noise points. At `min_samples=8`, with `eps` still 0.5, there are {int(sample_table.loc[8, "noise"])} noise points.
    - On the three densities, DBSCAN finds {varied_db_groups} clusters and {varied_db_noise} noise points. OPTICS finds {varied_op_groups} clusters and {varied_op_noise} noise points.
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
3. **The wide group has larger gaps:** A ball that is large enough for the tight group can be too small for the wide one, so those points become noise.
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
