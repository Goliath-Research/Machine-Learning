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
    import plotly.graph_objects as go
    import plotly.io as pio
    from sklearn.cluster import DBSCAN, OPTICS, cluster_optics_dbscan
    from sklearn.datasets import make_blobs
    from sklearn.metrics import adjusted_rand_score
    from sklearn.preprocessing import StandardScaler

    pio.templates.default = "plotly_white"
    px.defaults.color_discrete_sequence = px.colors.qualitative.G10

    return (
        DBSCAN,
        OPTICS,
        Path,
        StandardScaler,
        adjusted_rand_score,
        cluster_optics_dbscan,
        cv,
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

    By the end of this lesson, you should be able to:

    1. Explain what a reachability plot records, and read a valley as a dense group.
    2. Explain why the first reachability value is infinite, and why that is not an error.
    3. Cut one OPTICS ordering at several values of `eps` with `cluster_optics_dbscan`, and predict how the cluster count and noise count respond.
    4. Explain how a single `eps` can discard an entire real group while the adjusted Rand index still reports a perfect score.
    5. Cluster earthquakes on standardized depth and magnitude, then map the result on longitude and latitude.

    ## Background

    **OPTICS** stands for Ordering Points To Identify the Clustering Structure. It is closely related to DBSCAN, which the previous lesson covered, and the relationship is the whole point of this lesson.

    ### The problem OPTICS addresses

    DBSCAN needs one `eps`. That is fine when every group in your data is about equally dense, and it fails when they are not. A radius small enough to see a tight cluster is too small for a spread-out one, and a radius large enough for the spread-out one swallows everything nearby.

    OPTICS avoids committing to a radius up front. Instead of producing clusters directly, it **orders** the points and records a distance for each one.

    ### The reachability plot

    OPTICS starts somewhere, then repeatedly visits the nearest unvisited point from the current region, preferring dense areas. As it goes, it records the **reachability distance** of each newly visited point: roughly, how far the search had to stretch to reach it from the previous neighbourhood.

    The result is a sequence, not a partition. Plotted against visit order, it looks like a landscape.

    - **A valley** is a stretch of small reachability values. The search barely had to stretch, which means the points are packed together. **A valley is a dense group.**
    - **A peak or ridge** is a stretch of large values. The search had to jump a long way, which means it crossed empty space. **A gap between groups.**

    A picture worth keeping in mind: the reachability plot has the same relationship to the data that a mountain range has to a landscape. The peaks are the empty regions you walk over to get from one settlement to the next.

    ### Why the first value is infinite

    The very first point in the ordering has nothing before it. There is no previous neighbourhood to measure a distance from, so scikit-learn records `inf`. This is a convention, not a failure, and it is why the reachability plot starts with a spike you should ignore.

    ### Cutting the ordering

    OPTICS on its own gives you the landscape. To get actual cluster labels you cut that landscape at a height, and the height is a radius.

    `cluster_optics_dbscan` does exactly this. It does **not** fit a new model and does not look at the data again. It reads the reachability distances, core distances, and ordering already stored on the OPTICS fit, and applies a DBSCAN-style rule to them: a point joins the current cluster if its reachability is at most `eps`, and is labelled noise if it is larger.

    That is why the name is a bridge between the two algorithms. The **ordering** is OPTICS; the **threshold** is DBSCAN's. One expensive OPTICS fit can then be cut at many `eps` values for almost no extra cost, which is what the exercises below do.

    `plot_optics_dbscan` draws both views side by side: the scatter, coloured by cluster, and the reachability plot with a dashed line at the chosen `eps`. The line is the cut, and you can see exactly which valleys it clears.

    ### What `min_cluster_size` does

    OPTICS also takes `min_cluster_size`, the smallest group it will report. A value **below 1 is read as a fraction of the sample**, so `0.05` means "at least 5% of the points". An integer is a count.

    This parameter filters the result rather than changing the density model. Raising it drops small groups from the output instead of turning points into noise. That difference becomes important when comparing it with `eps`.

    ## Reading the score

    The synthetic clouds come with the labels that generated them, so the cuts can be scored with the **adjusted Rand index**: 1.0 means the partition matches exactly, about 0 means no better than chance, and cluster numbers are arbitrary.

    There is a trap here that this lesson demonstrates rather than merely warns about. **All noise points share one label**, so as far as the index is concerned they form a single group. If an entire real group is labelled noise while the other groups are found correctly, the index can still be **1.0**. The score cannot distinguish "found all three groups" from "found two and renamed the third as noise".

    So never read the index alone. Read it beside the **cluster count** and the **noise count**, as every table in this lesson does.

    ## Datasets used

    ### Three densities, and four densities

    Both are generated in the notebook by `sklearn.datasets.make_blobs`, so nothing is downloaded. The three-density cloud (100 points) uses standard deviations 0.6, 0.1, and 3.0; the four-density cloud (200 points, `random_state=2`) uses 0.3, 0.05, 1.2, and 0.7. The uneven spreads are the point: they create the situation that one radius cannot handle.

    ### Quakes

    **Source.** The `quakes` dataset, distributed with the R statistical software and documented at the [R datasets reference](https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/quakes.html). It originates from the **Harvard PRIM-H project**, which obtained it from **Dr John Woodhouse** of the Department of Geophysics at Harvard University. A copy, `quakes.csv`, sits beside this notebook.

    **What it holds.** 1,000 seismic events recorded in a cube of the earth's crust near the **Fiji islands** since 1964, all with magnitude above 4.0:

    | Column | Meaning | Range in this file |
    | :--- | :--- | ---: |
    | `lat` | Latitude of the event | −38.59 to −10.72 |
    | `long` | Longitude of the event | 165.67 to 188.13 |
    | `depth` | Depth below the surface, in km | 40 to 680 |
    | `mag` | Richter magnitude | 4.0 to 6.4 |
    | `stations` | Number of stations that reported the event | 10 to 132 |

    **Two details worth knowing.** The longitudes exceed 180 degrees, which is how the Pacific is often recorded: values above 180 wrap around to the western hemisphere rather than being negative. And the source documentation notes **two clear planes of seismic activity** in this region, one a major plate junction and the other the Tonga trench. That structure is a fact about the earth, not about the clustering.

    This file is a **subsample of a larger dataset** of 5,000 observations, so it is not the complete record. It is also from the 1960s onward and records only events above magnitude 4.0, so it says nothing about smaller or more recent earthquakes.

    **One caution that matters for this lesson.** The clustering uses `depth` and `mag`, not location. So when the map shows the earthquakes tracing an arc, you are seeing geography that the algorithm never used.

    A static scatter cannot show a very tight group next to a very wide one. **Use the Plotly zoom**: the group with standard deviation 0.1 is barely a few pixels wide at full scale, and reading it as absent is the mistake this lesson is designed to prevent.
    """)
    return


@app.cell
def _(np):
    def cluster_census(labels):
        """Count clusters and noise points.

        Parameters
        ----------
        labels : array-like
            Cluster labels as returned by OPTICS or `cluster_optics_dbscan`.
            Noise is labelled -1.

        Returns
        -------
        tuple of (int, int)
            The number of clusters, not counting noise, and the number of noise
            points.
        """
        labels = np.asarray(labels)
        noise = int(np.sum(labels == -1))
        groups = len(set(labels.tolist())) - (1 if noise else 0)
        return groups, noise

    return (cluster_census,)


@app.cell
def _(go, pd, px):
    def plot_box_with_noise(data, cluster, title="Boxplots with Noise Points"):
        """Draw boxplots per feature, with noise points overlaid as grey dots.

        Parameters
        ----------
        data : pandas.DataFrame
            Feature columns only. A "cluster" column is added internally.
        cluster : array-like
            Cluster label for each row, where -1 marks noise.
        title : str, default "Boxplots with Noise Points"
            Plot title.

        Returns
        -------
        plotly.graph_objects.Figure
            The boxplots, with one scatter trace for the noise points.
        """
        data = data.copy()
        data["cluster"] = cluster
        data_melted = data.melt(
            id_vars="cluster", var_name="feature", value_name="value"
        )
        noise_data = data_melted[data_melted["cluster"] == -1]
        fig = px.box(data_melted, x="feature", y="value", notched=True)
        fig.update_layout(
            width=650, height=400, title=title, showlegend=True
        )
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Three densities: the reachability landscape

    Three groups with standard deviations 0.6, 0.1, and 3.0. The middle one is ten times tighter than the first, and the last is thirty times wider than the tight one.

    OPTICS is fitted once with `min_samples=10` and `min_cluster_size=0.05` (at least 5% of the 100 points). Everything in the next two sections reuses that one fit.
    """)
    return


@app.cell
def _(OPTICS, cluster_census, cv, make_blobs, np, pd):
    np.random.seed(0)
    wide, wide_truth = make_blobs(
        n_samples=100,
        centers=[(0, 0), (3, 10), (12, 5)],
        cluster_std=[0.6, 0.1, 3.0],
    )
    wide_view = cv.ClusterVisualizer(pd.DataFrame(wide, columns=["x", "y"]))
    wide_optics = OPTICS(min_samples=10, min_cluster_size=0.05).fit(wide)
    wide_groups, wide_noise = cluster_census(wide_optics.labels_)
    wide_first_reach = wide_optics.reachability_[wide_optics.ordering_][0]
    wide_view.plot_data(title="Three groups with very different densities")
    return (
        wide,
        wide_first_reach,
        wide_groups,
        wide_noise,
        wide_optics,
        wide_truth,
        wide_view,
    )


@app.cell
def _(wide_groups, wide_noise, wide_optics, wide_view):
    wide_view.plot_clusters(
        wide_optics.labels_,
        title=f"OPTICS labels: {wide_groups} clusters, {wide_noise} noise",
    )
    return


@app.cell(hide_code=True)
def _(mo, wide_first_reach):
    mo.md(f"""
    OPTICS reports **{wide_groups} clusters and {wide_noise} noise points**, matching the three groups without being told how many there are.

    The tightest group (standard deviation 0.1) is at (3, 10) and appears as a small dot at this scale. **Zoom in before deciding it is missing**; it is by far the most useful thing to check in this plot.

    Note also the first reachability value in the ordering, which is **{wide_first_reach}**. That is the infinite value the background described: the first point has nothing before it to measure against.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The reachability plot

    This is the picture the section is named for. The x axis is the **order in which OPTICS visited the points**, not the original row order and not a coordinate. The y axis is the reachability distance recorded at each visit.
    """)
    return


@app.cell
def _(px, wide_optics):
    _reach = wide_optics.reachability_[wide_optics.ordering_]
    _fig = px.line(
        x=list(range(len(_reach))),
        y=_reach,
        title="Reachability plot for the three-density cloud",
    )
    _fig.update_layout(
        width=700,
        height=400,
        xaxis_title="Visit order",
        yaxis_title="Reachability distance",
    )
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Read it left to right as a landscape of three settlements separated by empty ground.

    - **The first spike is the infinite value.** Ignore it; it is the convention described above, not a real distance.
    - **Three shallow stretches** follow, each a run of small values. Those are the three dense groups. The search stayed low because the points kept being close.
    - **The ridges between them** are the empty space. When the search finished one group and had to travel to the next, the reachability jumped. The largest gap is the one before the widest group.

    This is why the plot is more informative than any single clustering. It shows **where the groups are** and **how dense each one is**, all at once, without you choosing a radius. The number of valleys is the number of groups the data actually contains.

    ### Try it yourself

    Replace `None` so that `student_valley_count` holds the number of reachability values below **1.0**, ignoring the infinite first entry. Use `wide_optics.reachability_`, which is in original row order.
    """)
    return


@app.cell
def _(np, wide_optics):
    student_valley_count = None
    print(student_valley_count)
    return (student_valley_count,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
```python
_reach = wide_optics.reachability_
student_valley_count = int(
    np.sum((_reach < 1.0) & np.isfinite(_reach))
)
```
The answer is **65**. 34 of the 100 values sit above 1.0, and one is infinite, which leaves 65 inside the valleys.

The `np.isfinite(...)` guard matters. Comparing `inf < 1.0` is simply `False`, so it would not break the count, but being explicit documents that the infinite entry is a convention rather than a distance. Roughly two thirds of the points are in dense regions and the rest are on the ridges between them.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Why does the first point in the ordering have an **infinite** reachability, and why is that not an error? Work out your answer, then reveal it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
Because **reachability is measured relative to something that came before**, and for the first point nothing did.

The reachability of a point is the distance the search had to stretch from the neighbourhood it was already in. At the very start there is no current neighbourhood, so the distance is undefined and scikit-learn records `inf` as the placeholder.

It is not an error, and it is not a point that is infinitely far from everything. It is simply the starting point, and every reachability plot begins with that spike. Ignore the first value whenever you read the y axis.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Cutting that one ordering at five radii

    The ordering above is already computed. `cluster_optics_dbscan` now reads it and applies a DBSCAN-style threshold at several values of `eps`, **without fitting anything again**.

    Watch two things as `eps` grows: the number of clusters, and the number of noise points. They move in opposite directions, for the reason the reachability plot makes visible: raising the cut line lets the algorithm reach over ridges it previously could not cross.

    ### `eps=0.5`, below most of the ridges
    """)
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
    return (wide_cuts,)


@app.cell
def _(wide_optics, wide_view):
    wide_view.plot_optics_dbscan(wide_optics, eps=0.5, height=230)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    At `eps=0.5` the cut line sits **below most of the ridges**, so it only clears the lowest valleys. Two clusters survive and **51 of the 100 points are noise**: more than half the data has been discarded, including the entire wide group, which never gets low enough to be reached.

    ### Raising the cut
    """)
    return


@app.cell
def _(wide_optics, wide_view):
    wide_view.plot_optics_dbscan(wide_optics, eps=1, height=230)
    return


@app.cell
def _(wide_optics, wide_view):
    wide_view.plot_optics_dbscan(wide_optics, eps=3, height=230)
    return


@app.cell
def _(wide_optics, wide_view):
    wide_view.plot_optics_dbscan(wide_optics, eps=4, height=230)
    return


@app.cell
def _(wide_optics, wide_view):
    wide_view.plot_optics_dbscan(wide_optics, eps=5, height=230)
    return


@app.cell
def _(mo, wide_cuts):
    mo.Html(wide_cuts.to_html())
    return


@app.cell
def _(adjusted_rand_score, cluster_optics_dbscan, mo, wide_cuts, wide_optics, wide_truth):
    _scored = wide_cuts.copy()
    _scored["adjusted Rand"] = [
        adjusted_rand_score(
            wide_truth,
            cluster_optics_dbscan(
                reachability=wide_optics.reachability_,
                core_distances=wide_optics.core_distances_,
                ordering=wide_optics.ordering_,
                eps=_eps,
            ),
        )
        for _eps in wide_cuts.index
    ]
    mo.Html(_scored.round(3).to_html())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Read the scored table against the plots, and something uncomfortable appears.

    | `eps` | Clusters | Noise | Adjusted Rand | What actually happened |
    | ---: | ---: | ---: | ---: | :--- |
    | 0.5 | 2 | 51 | 0.614 | Two groups found, the wide one discarded along with half the data |
    | **1** | **2** | **33** | **1.000** | **Two groups found, the entire third group labelled noise** |
    | 3 | 3 | 9 | 0.897 | All three found, but the boundaries are slightly off |
    | 4 | 3 | 2 | 0.971 | All three found, nearly exact |
    | 5 | 3 | 0 | 1.000 | All three found, exact |

    **`eps=1` scores a perfect 1.000 while reporting only two clusters and discarding 33 points.** Those 33 points are the entire third group, relabelled as noise. The index cannot see the difference between "found two groups and one large noise set" and "found all three groups", because renaming a whole group to `-1` is a legal rename as far as it is concerned.

    This is the trap the background section warned about, and it is not hypothetical. It is sitting in the middle of this table.

    The genuinely good fit is **`eps=5`**: three clusters, zero noise, index 1.000. The combination is what makes it trustworthy. A perfect index with a high noise count proves nothing.

    ### Try it yourself

    At `eps=1` the adjusted Rand index is 1.000 but only two clusters are reported. Explain how both can be true. Work out your answer, then reveal it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
Because **the index compares groupings, and "noise" is a single group.**

At `eps=1` the fit put one real group in cluster 0, one real group in cluster 1, and the **entire third group** into the noise label `-1`. As a grouping, that is a perfect match to the truth: three groups in the data, three groups in the labels, just with the third one renamed from "group 2" to "noise".

The index is invariant to renaming, so it reports 1.000. It cannot know that `-1` is supposed to mean "unclassified" rather than "group 3".

Two things expose the problem, and neither is the index:

- the **cluster count** is 2, not 3;
- the **noise count** is 33, not 0.

Always read the three numbers together. A perfect score is only meaningful when it comes with the expected number of clusters and zero discarded points.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Replace `None` so that `student_best_eps` holds the smallest `eps` in the cuts table that produces **three clusters with zero noise**. Read it from `wide_cuts`.
    """)
    return


@app.cell
def _(wide_cuts):
    student_best_eps = None
    print(student_best_eps)
    return (student_best_eps,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
```python
student_best_eps = float(
    wide_cuts[(wide_cuts["clusters"] == 3) & (wide_cuts["noise"] == 0)].index.min()
)
```
The answer is **5.0**.

The boolean condition selects the rows meeting both requirements, `.index` gives their `eps` labels, and `.min()` takes the smallest. `float(...)` tidies the type for display.

Note that `eps=1` is excluded by this filter even though it has the same perfect index, because it reports two clusters and 33 noise points. That is exactly the discrimination the table is teaching.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Four densities: a harder spread

    Now four groups with standard deviations 0.3, 0.05, 1.2, and 0.7. One is extremely tight (0.05) and they sit close together, so the ridges between them are lower than in the previous cloud.

    OPTICS is fitted once with `min_cluster_size=0.2`, so a group must hold at least a fifth of the 200 points. Then five cuts are scored.
    """)
    return


@app.cell
def _(
    OPTICS,
    adjusted_rand_score,
    cluster_census,
    cluster_optics_dbscan,
    cv,
    make_blobs,
    pd,
):
    four, four_truth = make_blobs(
        n_samples=200,
        centers=[(-2, 3), (0, 2), (3, -2), (7, 3)],
        cluster_std=[0.3, 0.05, 1.2, 0.7],
        random_state=2,
    )
    four_view = cv.ClusterVisualizer(pd.DataFrame(four, columns=["x", "y"]))
    four_optics = OPTICS(min_cluster_size=0.2).fit(four)
    four_groups, four_noise = cluster_census(four_optics.labels_)

    four_rows = []
    _optics_ari = adjusted_rand_score(four_truth, four_optics.labels_)
    four_rows.append(
        {
            "eps": "OPTICS labels",
            "clusters": four_groups,
            "noise": four_noise,
            "adjusted Rand": _optics_ari,
        }
    )
    for _eps in (0.5, 1, 1.5, 2, 4):
        _labels = cluster_optics_dbscan(
            reachability=four_optics.reachability_,
            core_distances=four_optics.core_distances_,
            ordering=four_optics.ordering_,
            eps=_eps,
        )
        _groups, _noise = cluster_census(_labels)
        four_rows.append(
            {
                "eps": str(_eps),
                "clusters": _groups,
                "noise": _noise,
                "adjusted Rand": adjusted_rand_score(four_truth, _labels),
            }
        )
    four_cuts = pd.DataFrame(four_rows).set_index("eps")
    four_view.plot_data(title="Four groups, four densities")
    return four_cuts, four_groups, four_noise, four_optics, four_view


@app.cell
def _(four_optics, four_view):
    four_view.plot_density_based_clustering(
        four_optics.labels_, title="OPTICS labels on the four-density cloud"
    )
    return


@app.cell(hide_code=True)
def _(mo, four_groups, four_noise):
    mo.md(f"""
    OPTICS reports **{four_groups} clusters and {four_noise} noise point**, which is the right answer: four groups found, exactly one point rejected.

    The group with standard deviation 0.05 is at (0, 2) and is essentially a single dot at this scale. Zoom.

    Now the cuts. The interesting feature of this cloud is that the four groups sit **closely together**, so the ridges between them are low. That means a moderate `eps` can step over the boundary between two real groups and merge them.
    """)
    return


@app.cell
def _(four_optics, four_view):
    four_view.plot_optics_dbscan(four_optics, eps=0.5, height=230)
    return


@app.cell
def _(four_optics, four_view):
    four_view.plot_optics_dbscan(four_optics, eps=1, height=230)
    return


@app.cell
def _(four_optics, four_view):
    four_view.plot_optics_dbscan(four_optics, eps=1.5, height=230)
    return


@app.cell
def _(four_optics, four_view):
    four_view.plot_optics_dbscan(four_optics, eps=2, height=230)
    return


@app.cell
def _(four_optics, four_view):
    four_view.plot_optics_dbscan(four_optics, eps=4, height=230)
    return


@app.cell
def _(mo, four_cuts):
    mo.Html(four_cuts.round(3).to_html())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Four stages, and every one of them fails differently.

    - **`eps=0.5`, too small.** **7 clusters and 44 noise.** The cut sits inside the valleys and chops them into pieces. Note the symptom: the cluster count is *too high*, which is the opposite of what a too-large radius does.
    - **`eps=1`.** **4 clusters and 8 noise**, index 0.951. Close, but eight genuine points were discarded.
    - **`eps=1.5`.** **4 clusters and 1 noise**, index 0.993. This reproduces the OPTICS labels almost exactly, which makes sense: the cut is now high enough to clear the ridges without crossing them.
    - **`eps=2`.** **2 clusters and 0 noise**, index 0.330. The cut has climbed over the low ridges between neighbouring groups and merged four into two. There is no noise, and the answer is badly wrong.
    - **`eps=4`.** **1 cluster and 0 noise**, index 0.000. Everything merged into one group. A "clean" summary with nothing discarded, and no clustering performed at all.

    Compare the last two rows with the first. **A too-small radius creates too many clusters; a too-large radius creates too few.** The noise count tells you which failure you are looking at only in combination with the cluster count.

    `eps=4` is the clearest illustration of the point that zero noise means nothing. One cluster, no noise, index 0.000: the algorithm has thrown all the structure away and has nothing left to call an outlier.

    ### Try it yourself

    Which `eps` value in `four_cuts` reproduces the OPTICS labels most closely? Replace `None` with a function that returns the `eps` whose adjusted Rand index is highest, excluding the `"OPTICS labels"` row.
    """)
    return


@app.cell
def _(four_cuts):
    student_four_best = None
    print(student_four_best)
    return (student_four_best,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
```python
_cuts_only = four_cuts.drop(index="OPTICS labels")
student_four_best = _cuts_only["adjusted Rand"].idxmax()
```
The answer is **1.5**, with an index of about 0.993.

`drop(index=...)` removes the OPTICS row so it cannot win against itself. `idxmax()` returns the index label of the largest value, which here is the string `"1.5"` rather than a number.

That result is the point of the section: the best cut is the one that reproduces what OPTICS already told you, and it is not the largest or the smallest `eps`.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    At `eps=4` the fit reports one cluster and zero noise. Explain why that is worse than the `eps=0.5` fit, which reports seven clusters and 44 noise points. Work out your answer, then reveal it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
Because **`eps=0.5` still contains the structure, fragmented, while `eps=4` has destroyed it.**

At `eps=0.5` the cut is too low, so it slices the valleys into pieces and discards a lot of points. The result is wrong, but the groups are still separated: it has split real groups rather than merging unrelated ones, and the index is 0.804, far above chance. Raise the cut a little and the pieces rejoin.

At `eps=4` the cut is above every ridge, so there is nothing to separate. All 200 points become one cluster, no point is an outlier, and the index is 0.000: the fit has the same information content as assigning every point to group 0. Nothing can be recovered from it, because the distinctions were never recorded.

The asymmetry matters in practice. Over-segmenting is visible, because the cluster count is obviously too high. Merging everything is quiet, because the summary looks tidy: one cluster, no noise, no error.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Earthquakes

    Real data now, from `quakes.csv`. 1,000 earthquakes with latitude, longitude, depth, magnitude, and a count of reporting stations.

    ### Which columns are clustered, and which are drawn

    This distinction is the most common source of confusion in these lessons, so it is worth stating plainly.

    - The **clustering** uses **standardized `depth` and `mag`**. Those two numbers are the coordinates the distance is computed on.
    - The **map** uses **`lat` and `long`**, because `ClusterVisualizer` plots its first two columns. Latitude and longitude are never part of the distance.

    So the map shows *where* each earthquake is, while the clustering grouped them by *how deep and how strong* they were. A cluster appearing in one part of the map does not mean the algorithm used geography; it means deep, strong earthquakes happen to be concentrated there.

    Standardizing matters because depth runs from 40 to 680 kilometres while magnitude runs from 4.0 to 6.4. Without scaling, depth would dominate every distance and magnitude would contribute almost nothing.
    """)
    return


@app.cell
def _(Path, StandardScaler, cv, pd):
    # Quakes: 1,000 seismic events near Fiji since 1964, magnitude above 4.0.
    # Source: Harvard PRIM-H project, via Dr John Woodhouse (Dept. of Geophysics, Harvard).
    # Distributed with R; documentation: https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/quakes.html
    # This file is a 1,000-row subsample of a 5,000-observation dataset.
    # Longitudes above 180 wrap past the antimeridian rather than being negative.
    # Save quakes.csv beside this notebook; in browser Marimo, upload it to the notebook's files.
    quakes = pd.read_csv(Path(__file__).resolve().parent / "quakes.csv", index_col="id")
    quake_columns = ["depth", "mag"]
    quake_z = pd.DataFrame(
        StandardScaler().fit_transform(quakes[quake_columns]),
        columns=quake_columns,
        index=quakes.index,
    )
    quake_view = cv.ClusterVisualizer(quakes)
    quake_ranges = quakes[quake_columns].describe().loc[["min", "max", "mean", "std"]]
    quake_view.plot_data(title="1,000 earthquakes, plotted by latitude and longitude")
    return quakes, quake_ranges, quake_view, quake_z


@app.cell(hide_code=True)
def _(mo, quakes, quake_ranges):
    mo.md(f"""
    **{len(quakes)} earthquakes**, columns `{", ".join(quakes.columns)}`.

    The two clustered columns span very different ranges:

    | | min | max | mean | std |
    | :--- | ---: | ---: | ---: | ---: |
    | depth (km) | {quake_ranges.loc["min", "depth"]:.0f} | {quake_ranges.loc["max", "depth"]:.0f} | {quake_ranges.loc["mean", "depth"]:.1f} | {quake_ranges.loc["std", "depth"]:.1f} |
    | magnitude | {quake_ranges.loc["min", "mag"]:.1f} | {quake_ranges.loc["max", "mag"]:.1f} | {quake_ranges.loc["mean", "mag"]:.2f} | {quake_ranges.loc["std", "mag"]:.2f} |

    Depth varies over 640 kilometres; magnitude varies over 2.4 points. On the raw scale, a depth difference of 100 would swamp any magnitude difference, so the clustering would effectively use depth alone. Standardizing puts both on the same footing.
    """)
    return


@app.cell
def _(DBSCAN, OPTICS, cluster_census, quake_z):
    quake_db = DBSCAN(eps=0.5, min_samples=5).fit(quake_z)
    quake_op = OPTICS(min_samples=5, min_cluster_size=0.2).fit(quake_z)
    quake_db_groups, quake_db_noise = cluster_census(quake_db.labels_)
    quake_op_groups, quake_op_noise = cluster_census(quake_op.labels_)
    quake_summary = f"DBSCAN: {quake_db_groups} clusters, {quake_db_noise} noise | OPTICS: {quake_op_groups} clusters, {quake_op_noise} noise"
    return (
        quake_db,
        quake_db_groups,
        quake_db_noise,
        quake_op,
        quake_op_groups,
        quake_op_noise,
        quake_summary,
    )


@app.cell(hide_code=True)
def _(mo, quake_summary):
    mo.md(f"""
    Both methods are fitted on the same standardized pair. **{quake_summary}**

    Neither finds more than one cluster. That is the honest result, and it is informative rather than disappointing: on these two standardized measurements, the earthquakes form one broad population rather than several separated groups. Depth and magnitude do not split them into distinct crowds.
    """)
    return


@app.cell
def _(quake_db, quake_view):
    quake_view.plot_density_based_clustering(
        quake_db.labels_, title="Earthquakes clustered with DBSCAN (on the map)"
    )
    return


@app.cell
def _(quake_op, quake_view):
    quake_view.plot_density_based_clustering(
        quake_op.labels_, title="Earthquakes clustered with OPTICS (on the map)"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Both maps look almost uniform, which matches the single cluster. The coloured points are the one cluster; the grey ones are noise.

    What the maps **do** show is geography that the clustering never used: the earthquakes trace an arc, which is the Pacific ring where tectonic plates meet. That pattern is real and visible, but it comes from `lat` and `long`, not from `depth` and `mag`. If you wanted to cluster by location, you would standardize and cluster those two columns instead.

    ### What the noise points have in common

    The box plots below show the standardized depth and magnitude, with the noise points overlaid in grey. This is where the noise definition becomes concrete.
    """)
    return


@app.cell
def _(plot_box_with_noise, quake_db, quake_z):
    plot_box_with_noise(
        quake_z,
        quake_db.labels_,
        title="DBSCAN noise on standardized depth and magnitude",
    )
    return


@app.cell
def _(plot_box_with_noise, quake_op, quake_z):
    plot_box_with_noise(
        quake_z,
        quake_op.labels_,
        title="OPTICS noise on standardized depth and magnitude",
    )
    return


@app.cell(hide_code=True)
def _(mo, quakes, quake_db, quake_op):
    _db_noise = quake_db.labels_ == -1
    _op_noise = quake_op.labels_ == -1
    mo.md(f"""
    The grey dots are not spread evenly. They sit at the **top of the magnitude box**, with means of **{quakes.mag[_db_noise].mean():.2f}** for the DBSCAN noise and **{quakes.mag[_op_noise].mean():.2f}** for the OPTICS noise, against an overall mean of **{quakes.mag.mean():.2f}**.

    So the points these algorithms rejected are the **strongest earthquakes**. They are not errors in the data; they are genuinely unlike the rest, sitting above the dense band that holds almost every earthquake. Density methods flag exactly this: points far from any crowd.

    That has a practical consequence worth stating. A density method will reject your most extreme and often most interesting observations, because being unusual is what "noise" means. Whether you want that depends on the question: for finding typical earthquakes, discarding the seven strongest is fine, and for studying them it would be a serious loss.

    ### Try it yourself

    Replace `None` so that `student_quake_noise_mag` holds the mean magnitude of the points OPTICS labelled as noise.
    """)
    return


@app.cell
def _(quakes, quake_op):
    student_quake_noise_mag = None
    print(student_quake_noise_mag)
    return (student_quake_noise_mag,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
```python
_is_noise = quake_op.labels_ == -1
student_quake_noise_mag = quakes.loc[_is_noise, "mag"].mean()
```
The answer is about **5.57**, against an overall mean of **4.62**.

The boolean mask `_is_noise` selects the noise rows, and `.loc` uses it to pick those magnitudes before averaging. Using the same mask on `quakes` rather than on `quake_z` gives the value in the original magnitude units, which is easier to interpret than a standardized score.

The gap of nearly one full magnitude point is the evidence that the rejected rows are systematically the strongest earthquakes, not random leftovers.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Which two columns did the clustering use as coordinates, and which two are shown in the map? Work out your answer, then reveal it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
The clustering used **standardized `depth` and `mag`**. The map shows **`lat` and `long`**, because `ClusterVisualizer` plots the first two columns of the table it is given.

The two are unrelated. Latitude and longitude were never part of the distance calculation, so the geography you see in the map is a display choice, not a clustering result.

That is why a cluster can appear concentrated in one region of the map without the algorithm having used location at all: deep or strong earthquakes happen to occur there. If you wanted to cluster by geography, you would standardize `lat` and `long` and cluster those instead.
""")
        },
        expanded=False,
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

    1. **OPTICS produces an ordering, not a partition.** It visits points from dense regions outward and records a reachability distance at each visit. The first value is `inf` because nothing precedes it; that is a convention, not an error.
    2. **The reachability plot is a landscape.** Valleys are dense groups and ridges are empty space between them. On the three-density cloud the plot shows three clear valleys, and OPTICS labels them as {wide_groups} clusters with {wide_noise} noise points without being asked for three.
    3. **`cluster_optics_dbscan` cuts that ordering; it fits nothing new.** One OPTICS fit can be cut at many values of `eps`, because the function only reads the stored reachability, core distances, and ordering.
    4. **Raising `eps` lowers noise and eventually merges groups.** On the three-density cloud, the noise counts fell 51 → 33 → 9 → 2 → 0 as `eps` went 0.5 → 1 → 3 → 4 → 5. On the four-density cloud a large enough `eps` collapsed four groups to two and then to one.
    5. **A perfect adjusted Rand index does not prove the clustering worked.** At `eps=1` on the three-density cloud the index is 1.000 while only two clusters are reported and 33 points, the whole third group, are labelled noise. The index treats all noise as one group, so renaming a group to "noise" is invisible to it. Always read the score beside the cluster count and the noise count.
    6. **The best cut reproduced what OPTICS already said.** On the four-density cloud the highest cut score was `eps=1.5`, which gave {int(four_cuts.loc["1.5", "clusters"])} clusters and {int(four_cuts.loc["1.5", "noise"])} noise points, matching the OPTICS labels of {four_groups} clusters and {four_noise} noise. Cutting is a way of reading the reachability structure, not a way of improving it.
    7. **`min_cluster_size` filters, while `eps` changes the threshold.** A value below 1 is a fraction of the sample, so `0.2` means at least 20% of the rows.
    8. **On the earthquakes, one cluster is the honest answer.** Using standardized depth and magnitude, DBSCAN found {quake_db_groups} cluster and {quake_db_noise} noise points, and OPTICS found {quake_op_groups} cluster and {quake_op_noise} noise. These two measurements do not divide the earthquakes into separate groups.
    9. **The noise points are the strongest earthquakes, not errors.** Their mean magnitude is well above the overall mean, so the methods rejected the most extreme observations. That is what noise means, and it is a trade-off: useful for finding typical events, harmful if the extremes are what you study.
    10. **The map and the clustering use different columns.** Clustering used depth and magnitude; the map shows latitude and longitude. The visible geography is a display choice, not a result.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check your understanding

    1. What does a reachability plot record, and what does a valley in it mean?
    2. Why is the first reachability value in the ordering infinite?
    3. Does `cluster_optics_dbscan` fit a new model? What does it read instead?
    4. What happens to the cluster count and the noise count as `eps` increases, and why?
    5. Explain how an adjusted Rand index of 1.000 can accompany a fit that discarded an entire real group.
    6. What does `min_cluster_size=0.2` require, and how does raising it differ from raising `eps`?
    7. Which two quake columns were clustered, and why were they standardized?
    8. Why are the earthquakes labelled as noise systematically stronger than the rest?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
1. **The reachability distance at each visit, in visit order.** A valley means a stretch where the search stayed close, which is a dense group. Ridges between valleys are empty space between groups.
2. **Nothing precedes the first point.** Reachability is measured relative to the neighbourhood the search came from, and at the start there is none, so scikit-learn stores `inf` as a placeholder. It is a convention, not a distance.
3. **No.** It reads the `reachability_`, `core_distances_`, and `ordering_` already stored on the OPTICS fit and applies a DBSCAN-style threshold at the `eps` you supply. That is why one fit can be cut many times cheaply.
4. **The noise count falls and the cluster count eventually falls too.** A higher line clears more ridges, so points that were unreachable join a cluster, and once the line passes a ridge that separates two real groups, those groups merge into one.
5. **Because all noise shares the single label `-1`, so it counts as one group.** If one entire real group is relabelled as noise and the others are found correctly, the grouping still matches the truth up to renaming, and the index is invariant to renaming. The cluster count and noise count are what expose it.
6. **At least 20% of the rows.** Raising it **drops small groups from the output** rather than turning points into noise, because the density profile is already computed and the parameter filters what is reported. Raising `eps` changes the threshold itself, so it does convert points to noise.
7. **Depth and magnitude.** They span very different ranges, 40 to 680 km against 4.0 to 6.4, so on the raw scale depth would dominate every distance and magnitude would barely contribute.
8. **Because they are genuinely far from the dense band.** Noise means "not part of any crowd", and the strongest earthquakes sit above the band holding almost all the others. Their mean magnitude is about 5.6 against an overall 4.6.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - [OPTICS](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html)
    - [cluster_optics_dbscan](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.cluster_optics_dbscan.html)
    - [Density-based clustering, scikit-learn user guide](https://scikit-learn.org/stable/modules/clustering.html#optics)
    - Ankerst, M., Breunig, M. M., Kriegel, H.-P., and Sander, J. (1999). OPTICS: Ordering points to identify the clustering structure. *ACM SIGMOD Record*, 28(2), 49–60.
    - VanderPlas, J. (2017). *Python Data Science Handbook*. O'Reilly Media. Chapter 5.
    """)
    return


if __name__ == "__main__":
    app.run()
