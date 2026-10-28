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
    from sklearn.cluster import DBSCAN, OPTICS
    from sklearn.datasets import make_blobs, make_circles, make_moons
    from sklearn.metrics import adjusted_rand_score
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    pio.templates.default = "plotly_white"
    px.defaults.color_discrete_sequence = px.colors.qualitative.G10

    return (
        DBSCAN,
        OPTICS,
        NearestNeighbors,
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

    By the end of this lesson, you should be able to:

    1. Classify any point as **core**, **border**, or **noise** given `eps`, `min_samples`, and the neighbours around it.
    2. Explain what `eps` and `min_samples` each control, and predict whether raising or lowering one increases noise or merges groups.
    3. Explain why a single `eps` cannot suit a dataset whose groups have different densities, and how OPTICS addresses that.
    4. Explain how `min_samples`, `xi`, and `min_cluster_size` change what OPTICS reports, and why a fraction is accepted.
    5. Interpret the adjusted Rand index on moons and rings, including how noise labels interact with it and why a perfect score does not prove that every point was clustered.

    ## Background

    **Density-based clustering** groups points that sit close together and marks the isolated ones as noise. It is the third family of methods in this course, after k-means and the two hierarchical approaches, and it answers a different question. K-means assumes groups are roughly round, and hierarchical clustering asks how points merge. Density methods instead ask: **where are the crowded regions, and where are the empty ones?**

    That framing brings two things the earlier methods lacked. It can follow irregular shapes, because a crescent is a crowded curve rather than a round blob. And it can refuse to assign a point at all, which is what "noise" means.

    ### The three kinds of point

    DBSCAN sorts every point into one of three categories. The definitions depend on two parameters.

    - `eps` is a **radius**. Two points are neighbours if the distance between them is at most `eps`.
    - `min_samples` is a **neighbour count threshold**.

    With those defined:

    | Point type | Definition |
    | :--- | :--- |
    | **Core** | Has at least `min_samples` neighbours within `eps`, counting itself |
    | **Border** | Within `eps` of a core point, but not a core point itself |
    | **Noise** | Neither of the above |

    A cluster grows from its core points. Two core points within `eps` of each other belong to the same cluster, and border points join whichever core point they are near. Anything the cluster never reaches stays noise, and is labelled **-1**.

    The counts matter when you read the plots: `-1` is not a cluster. A run that reports "1 cluster and 200 noise points" has found one group and rejected the rest.

    ### Two parameters, two failure modes

    Neither parameter is a difficulty setting. They control a trade-off, and each one fails in its own direction.

    - **`eps` too small** — almost nothing has enough neighbours within the radius, so points that should form groups are labelled noise. The clusters shatter into fragments or disappear.
    - **`eps` too large** — the radius reaches across the gap between two groups, so they merge into one. Noise vanishes because everything is reachable.
    - **`min_samples` too small** — a handful of neighbours is enough to be "core", so small accidental clumps are promoted to clusters and noise is under-reported.
    - **`min_samples` too large** — few points can meet the threshold, so genuine groups lose their core and dissolve into noise.

    On the two-blob data below you can watch all four of these happen.

    ### The limitation that motivates OPTICS

    DBSCAN has one radius, so it has one idea of "dense". Real data frequently has a tight group and a spread-out group side by side. One `eps` tuned for the tight group is too small for the wide one, and one tuned for the wide group will swallow the tight one and its surroundings. **No single value can be right for both.**

    OPTICS was designed for exactly this situation. Instead of committing to a radius, it **orders** the points by density and records, for each point, how far the algorithm had to reach to get to it. That number is the **reachability distance**. Dense regions produce small reachability values, and sparse regions produce large ones.

    The outcome is a profile of the data's density rather than a single partition. You then extract clusters from that profile, and the three parameters below control how.

    | Parameter | What it does |
    | :--- | :--- |
    | `min_samples` | How many neighbours count as "dense", the same idea as in DBSCAN |
    | `xi` | How steep a drop in reachability must be before a new cluster starts. Higher `xi` means fewer, larger clusters |
    | `min_cluster_size` | The smallest group OPTICS will report. A value below 1 is read as a **fraction of the sample**, so `0.1` means "at least 10% of the points" |

    `min_cluster_size` accepting a fraction is worth remembering: in scikit-learn, an integer is a count of points, and a float is a proportion. `0.5` and `89` can therefore mean entirely different things depending on how many rows you have.

    ## Measuring the result

    The moons and rings come with known generating labels, so the clusters can be scored. This lesson uses the **adjusted Rand index**: 1.0 means the partition matches exactly, and about 0 means no better than chance. Cluster numbers are arbitrary, so the score compares groupings, not names.

    One detail matters when reading it here, and it is a trap. **Noise is a label like any other**, so all noise points form one group as far as the index is concerned. Two consequences follow.

    - If an entire real group is labelled noise while the other groups are recovered correctly, **the index can still be 1.0**. The discarded group becomes the "noise cluster", and the partition is still identical to the truth once you accept that relabelling.
    - If only *some* points of a group become noise, those points are now grouped with noise from other groups, and the index falls.

    So a score of 1.0 does **not** prove that every group was found. It proves the labels are consistent with the truth up to renaming, and renaming a whole group to "noise" is a legal rename. **Always read the cluster count alongside the index**, and check the noise count separately, as this lesson does in every table.

    ## Datasets used

    All four datasets in this lesson are **generated in the notebook**, so nothing is downloaded and the results are reproducible anywhere.

    ### Two tight blobs

    From `sklearn.datasets.make_blobs`, 50 points, `random_state=0`, `cluster_std=0.5`. Small and well separated, so the parameter sweeps change one thing at a time. [Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html).

    ### Three densities

    100 points from `make_blobs` at three centres, with standard deviations **0.5, 0.1, and 2.0**. The uneven spreads are deliberate: they create exactly the situation a single `eps` cannot handle, which is the central point of the lesson.

    ### Half moons

    500 points from `sklearn.datasets.make_moons`, noise 0.05, `random_state=0`. Two interlocking crescents, a standard test case for shapes that a straight boundary cannot separate. [Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html).

    ### Concentric circles

    1,000 points from `sklearn.datasets.make_circles`, noise 0.08, `factor=0.2`, `random_state=0`. A ring around a circle, with an empty band between them. Density methods find that band naturally. [Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_circles.html).

    A note on the synthetic data: because we generate it, we know the true groups and can score the fits. Real data never offers that guarantee, so the scores here measure the algorithm rather than a realistic task. The trade-off is deliberate, since it isolates how `eps` and `min_samples` behave.

    Every scatter is a `ClusterVisualizer` Plotly figure, so you can zoom. **Zooming is part of the lesson**: the tight group in the three-density cloud is invisible at full scale, and reading it as "missing" is the mistake the lesson is trying to prevent. The scikit-learn defaults are `eps=0.5` and `min_samples=5`.
    """)
    return


@app.cell
def _(np):
    def cluster_census(labels):
        """Count clusters and noise points.

        Parameters
        ----------
        labels : array-like
            Cluster labels as returned by DBSCAN or OPTICS. Noise is labelled -1.

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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Two blobs: the three point types

    Fifty points in two well-separated groups. This cloud is easy on purpose, so that every change you see comes from the parameters rather than from a difficult shape.

    The first run uses the scikit-learn defaults, `eps=0.5` and `min_samples=5`.
    """)
    return


@app.cell
def _(DBSCAN, cluster_census, cv, make_blobs, pd):
    blobs, _blob_truth = make_blobs(
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
    blob_view.plot_data(title="Fifty points in two tight groups")
    return blob_models, blob_view, eps_table


@app.cell
def _(blob_models, blob_view):
    blob_view.plot_density_based_clustering(
        blob_models[0.5].labels_, title="DBSCAN with the default eps=0.5"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Two groups come out, with a handful of grey points that DBSCAN refused to assign. Those grey points are the **noise**. They sit at the fringes of each blob, further than `eps` from enough neighbours.

    ### The same fit, showing point types

    The next plot uses the same labels but distinguishes **core** points (filled, coloured), **border** points (hollow, coloured), and **noise** (grey). This is the plot to study, because it makes the definitions visible.
    """)
    return


@app.cell
def _(blob_models, blob_view):
    blob_view.plot_DBSCAN_categories(
        blob_models[0.5].labels_,
        blob_models[0.5].core_sample_indices_,
        title="Point types at eps=0.5, min_samples=5",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Read it outwards from the middle of a blob. The **filled** markers are core points: each one has at least five neighbours within 0.5 units, so it could recruit others. The **hollow** markers are border points: they sit within 0.5 of a core point, so they join the cluster, but they do not have five neighbours of their own and cannot recruit anyone else. The **grey** points are noise: no core point is close enough to reach them.

    That hierarchy is the whole algorithm. Clusters are grown only from core points, and everything else is either attached to one or left out.

    ### Try it yourself

    Replace `None` so that `student_core_count` holds the number of **core** points in the default fit, and `student_noise_count` holds the number of **noise** points.
    """)
    return


@app.cell
def _(blob_models):
    student_core_count = None
    student_noise_count = None
    print("core points:", student_core_count)
    print("noise points:", student_noise_count)
    return student_core_count, student_noise_count


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
```python
student_core_count = len(blob_models[0.5].core_sample_indices_)
student_noise_count = int((blob_models[0.5].labels_ == -1).sum())
```
There are **32** core points and **9** noise points. `core_sample_indices_` gives the row positions of the core points, so its length is the count. Noise is found by comparing the labels to `-1`; the `int(...)` converts the count to a plain integer so it prints cleanly.

Out of 50 points, 32 core plus 9 noise leaves 9 border points. All three types are present in one small dataset.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Why can a border point never start a new cluster of its own? Work out your answer, then reveal it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
Because a border point **does not have enough neighbours to be core**, and only core points are used to grow a cluster.

A border point is defined by its relationship to a core point: it is within `eps` of one. That is the only reason it belongs to a cluster at all. If you removed the core points, the border points would have nothing to attach to and would be noise.

The practical consequence is that a cluster is held together by its core, not its edge. A thin fringe of border points can look like a real group in a plot, but it cannot survive on its own.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Sweeping `eps` on the blobs

    `eps` is the radius. The next three plots hold `min_samples` at its default of 5 and change only the radius. **Predict before you look:** what should a smaller radius do to the noise count, and what should a larger one do to the number of clusters?

    The table below records all four runs, including the default you have already seen.
    """)
    return


@app.cell
def _(blob_models, blob_view):
    blob_view.plot_DBSCAN_categories(
        blob_models[0.35].labels_,
        blob_models[0.35].core_sample_indices_,
        title="eps=0.35: too small for this spacing",
    )
    return


@app.cell
def _(blob_models, blob_view):
    blob_view.plot_DBSCAN_categories(
        blob_models[1].labels_,
        blob_models[1].core_sample_indices_,
        title="eps=1: large enough to reach every point",
    )
    return


@app.cell
def _(blob_models, blob_view):
    blob_view.plot_DBSCAN_categories(
        blob_models[2].labels_,
        blob_models[2].core_sample_indices_,
        title="eps=2: wide enough to bridge the two groups",
    )
    return


@app.cell
def _(mo, eps_table):
    mo.Html(eps_table.to_html())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Read the table down the columns. The `noise` column and the `clusters` column move in opposite directions as the radius grows.

    - **`eps=0.35`, too small.** Noise jumps to **27** of the 50 points and only **15** remain core. More than half the cloud is now grey, and the two groups are visibly eaten away from the edges inward. Nothing has been miscomputed; the radius is simply smaller than the gaps between neighbours.
    - **`eps=0.5`, the default.** **32** core, **9** noise, **2** clusters. A sensible middle.
    - **`eps=1`.** Noise falls to **0** and all 50 points are core, but there are still **2** clusters. The radius is now larger than the gaps inside a blob, so every point can reach its neighbours, yet the gap between the blobs is still too wide to cross. This is the narrow region where the parameters are genuinely right.
    - **`eps=2`.** **1** cluster, **0** noise. The radius now spans the gap between the two blobs, so they merge. Note that the fit looks "clean" in a summary, because there is no noise at all, and it is completely wrong.

    That last row is the trap. **Zero noise is not a sign of a good fit.** It is often the signature of a radius so large that everything has been swept into one group. Always check the cluster count alongside the noise count.

    ### Try it yourself

    At `eps=2` there is no noise at all and only one cluster. Explain why a run with zero noise points can be worse than one with nine. Work out your answer, then reveal it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
Because **noise is what the algorithm discards, not what it gets wrong.** A large `eps` reaches so far that every point is within range of some core point, so none is left over. That removes noise by eliminating the gaps that made the two groups distinct.

At `eps=2` the radius is wide enough to cross the space between the blobs, so the two groups merge into one cluster. The summary reads "1 cluster, 0 noise", which sounds tidy, but the structure that mattered has been destroyed.

At `eps=0.5` the fit correctly reports two clusters and discards nine fringe points. Discarding a few points is the price of resolving the groups properly. **Judge a density fit by whether the groups survived, not by whether noise disappeared.**
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Sweeping `min_samples`

    Now the other parameter. `eps` stays at 0.5, and only `min_samples` changes. This threshold says how many neighbours a point needs before it counts as core.

    Small values are permissive: a few neighbours are enough, so more points become core and fewer are discarded. Large values are strict: a point needs a real crowd around it, so points on the fringes fail and become noise.

    The three plots below show 3, 5, and 8. The default, 5, is repeated so you can compare directly.
    """)
    return


@app.cell
def _(DBSCAN, blobs, cluster_census, pd):
    sample_models = {
        _samples: DBSCAN(min_samples=_samples).fit(blobs)
        for _samples in (3, 5, 8)
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
    return sample_models, sample_table


@app.cell
def _(blob_view, sample_models):
    blob_view.plot_DBSCAN_categories(
        sample_models[3].labels_,
        sample_models[3].core_sample_indices_,
        title="min_samples=3: a permissive threshold",
    )
    return


@app.cell
def _(blob_view, sample_models):
    blob_view.plot_DBSCAN_categories(
        sample_models[5].labels_,
        sample_models[5].core_sample_indices_,
        title="min_samples=5: the default",
    )
    return


@app.cell
def _(blob_view, sample_models):
    blob_view.plot_DBSCAN_categories(
        sample_models[8].labels_,
        sample_models[8].core_sample_indices_,
        title="min_samples=8: a strict threshold",
    )
    return


@app.cell
def _(mo, sample_table):
    mo.Html(sample_table.to_html())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - **`min_samples=3`.** **39** core points and **7** noise. Permissive: nearly everything qualifies as core, so almost nothing is discarded. If there were small accidental clumps in this data, this setting would start promoting them to clusters.
    - **`min_samples=5`.** **32** core and **9** noise.
    - **`min_samples=8`.** **19** core and **16** noise. Strict: a third of the cloud can no longer find eight neighbours within 0.5, so it is labelled noise. Notice that the two clusters survive, but their edges have thinned considerably.

    Compare this with the `eps` sweep. Both parameters increase noise when tightened, but they do it differently: `min_samples` removes points from the *edge* of a group, while `eps` shrinks the reach from *every* point, so it erodes the group all over. That is why `eps=0.35` leaves only 15 core points while `min_samples=8` leaves 19, even though both push noise up.

    ### Try it yourself

    Replace `None` so that `student_sample_noise` holds the noise count when `min_samples=8` and `eps` is at its default.
    """)
    return


@app.cell
def _(sample_models):
    student_sample_noise = None
    print(student_sample_noise)
    return (student_sample_noise,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
```python
student_sample_noise = int((sample_models[8].labels_ == -1).sum())
```
The answer is **16**. `sample_models` is keyed by the `min_samples` value, so `[8]` selects the strict fit. Comparing the labels to `-1` and summing finds the noise.

For reference, the same expression gives 9 at `min_samples=5` and 7 at `min_samples=3`, so the noise count nearly doubles as the threshold goes from 3 to 8.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Choosing `eps` with a k-distance plot

    The sweeps so far were exploratory: try a value, look, try another. In practice you want a starting point, and one common method is to plot the distance from each point to its k-th nearest neighbour, sorted from smallest to largest.

    The curve is flat where points are inside a dense group, then rises sharply where the group ends and the distances jump to the nearest other point. The **knee** of that curve is where the last real neighbour is, so a good `eps` is roughly the height of the knee.
    """)
    return


@app.cell
def _(NearestNeighbors, blobs, np, pd):
    _neighbours = NearestNeighbors(n_neighbors=5).fit(blobs)
    _distances, _ = _neighbours.kneighbors(blobs)
    k_distances = pd.DataFrame(
        {
            "point (sorted)": np.arange(len(blobs)),
            "distance to 5th neighbour": np.sort(_distances[:, -1]),
        }
    ).set_index("point (sorted)")
    return (k_distances,)


@app.cell
def _(k_distances, px):
    _fig = px.line(
        k_distances,
        y="distance to 5th neighbour",
        title="k-distance plot for the blobs, sorted",
    )
    _fig.add_hline(y=0.5, line_dash="dash", annotation_text="eps=0.5")
    _fig.update_layout(width=650, height=380, showlegend=False)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The distances run from about **0.24** up to **0.96**, and most of the rise is concentrated at the right-hand end. The dashed line marks `eps=0.5`. Points below it have their 5th neighbour inside the radius, so they can be core; points above it cannot.

    That is exactly the split the fit reports: the points above the line are the ones that became noise, and the fact that the curve stays low for most of its length is what makes 0.5 a defensible choice rather than a guess.

    Reading the knee is a guideline, not a formula. It gives you a starting value to test, which is why the sweeps come first in this lesson.

    ## 4. Three densities: where a single `eps` fails

    This cloud is built to break DBSCAN. Three centres, with standard deviations **0.5, 0.1, and 2.0**. One group is extremely tight, one is medium, and one is very spread out.

    Think about what that means for a radius. To be dense enough for the tight group, `eps` can be small. To be dense enough for the wide group, `eps` must be large. Those two requirements point in opposite directions, and one number cannot satisfy both.

    **Zoom in before drawing conclusions.** The tight group has a standard deviation of 0.1, so at full scale it collapses to what looks like a single point. It is there.
    """)
    return


@app.cell
def _(DBSCAN, OPTICS, cluster_census, cv, make_blobs, np, pd):
    np.random.seed(0)
    varied, _varied_truth = make_blobs(
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
    varied_view.plot_data(title="Three groups with very different densities")
    return varied_dbscan, varied_optics, varied_table, varied_view


@app.cell
def _(varied_dbscan, varied_view):
    varied_view.plot_DBSCAN_categories(
        varied_dbscan.labels_,
        varied_dbscan.core_sample_indices_,
        title="DBSCAN, eps=0.5 and min_samples=10",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What DBSCAN did here

    The fit reports **2 clusters and 44 noise points**. Only 56 of the 100 points were assigned at all.

    This is not a tuning mistake; it is the structural problem. With `eps=0.5`:

    - the **tight** group (standard deviation 0.1) is far too dense for this radius to be a problem, and it is recovered;
    - the **medium** group (0.5) is at the edge of what this radius handles;
    - the **wide** group (2.0) has neighbours much further apart than 0.5, so almost all of it fails the neighbour test and becomes noise.

    Raising `eps` enough to capture the wide group would simultaneously make the gap between the tight group and its neighbours irrelevant, inviting merges. There is no single value that fixes both ends. That is the limitation the next algorithm addresses.

    ### The same cloud with OPTICS

    OPTICS does not commit to one radius. It orders the points by density and records the reachability distance for each, so groups of different densities show up as separate features of one profile.

    With `min_samples=10` and `xi=0.05`, read the result against the DBSCAN run above.
    """)
    return


@app.cell
def _(varied_optics, varied_view):
    varied_view.plot_density_based_clustering(
        varied_optics["min_samples=10, xi=0.05"].labels_,
        title="OPTICS, min_samples=10 and xi=0.05",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **3 clusters and 0 noise**, against DBSCAN's 2 clusters and 44 noise points. All three densities are recovered, including the wide group that DBSCAN discarded.

    The mechanism is the ordering. Because OPTICS measures how far it had to reach to connect each point, a sparse group simply produces large reachability values, which is a different signal from "not dense enough to exist". DBSCAN has one threshold and must reject the sparse group; OPTICS can report it as a real cluster with a different characteristic density.

    That does not make OPTICS universally better. It has three parameters of its own, and each changes what gets reported, which is the next topic.

    ## 5. The OPTICS parameters

    OPTICS reports clusters by cutting its reachability profile. Three parameters control where those cuts happen, and each has a distinct effect.

    ### `min_samples`

    The same idea as in DBSCAN: how many neighbours make a point "dense". Raising it makes the density requirement stricter.
    """)
    return


@app.cell
def _(varied_optics, varied_view):
    varied_view.plot_density_based_clustering(
        varied_optics["min_samples=50"].labels_,
        title="min_samples=50: stricter density requirement",
    )
    return


@app.cell
def _(varied_optics, varied_view):
    varied_view.plot_density_based_clustering(
        varied_optics["min_samples=5"].labels_,
        title="min_samples=5: stricter threshold relaxed",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The two extremes fail in opposite directions, and both are in the table.

    - **`min_samples=50`** demands 50 neighbours before a point counts as dense. In a 100-point cloud that is half the sample. The result is **1 cluster and 0 noise**: everything collapses into one group, because almost no point can act as a local centre and the distinctions dissolve.
    - **`min_samples=5`** is permissive. It reports **4 clusters and 65 noise points**, so the density profile has been chopped into small pieces and most points are discarded.

    Only a middle value finds three. `min_samples` is not a setting you can push in one direction and expect improvement.

    ### `xi`

    `xi` sets how steep a drop in reachability must be before OPTICS declares a new cluster. A large `xi` demands a sharp change, so only the most obvious boundaries produce clusters, giving you **fewer and larger** groups. A small `xi` accepts gentle changes, giving **more and smaller** groups.
    """)
    return


@app.cell
def _(varied_optics, varied_view):
    varied_view.plot_density_based_clustering(
        varied_optics["xi=0.9"].labels_, title="xi=0.9: only sharp boundaries count"
    )
    return


@app.cell
def _(varied_optics, varied_view):
    varied_view.plot_density_based_clustering(
        varied_optics["xi=0.3"].labels_, title="xi=0.3: gentler boundaries accepted"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The direction matches the definition. **`xi=0.9`** is strict about steepness and reports **1 cluster with 67 noise points**: hardly any boundary is sharp enough, so the profile is barely cut. **`xi=0.3`** accepts gentler changes and reports **2 clusters with 33 noise**.

    Compare both with the working default of `xi=0.05`, which found 3 clusters and no noise. The lesson is that `xi` close to 1 is not "high quality", it is "almost nothing qualifies", and the failure shows up as noise rather than as an error message.

    ### `min_cluster_size`

    This one sets the smallest group OPTICS will report, and it accepts either a count or a fraction of the sample. With 100 points, a fraction of 0.3 means "at least 30 points", and a value of 0.4 means "at least 40".
    """)
    return


@app.cell
def _(varied_optics, varied_view):
    varied_view.plot_density_based_clustering(
        varied_optics["min_cluster_size=0.4"].labels_,
        title="min_cluster_size=0.4: needs at least 40 of 100 points",
    )
    return


@app.cell
def _(varied_optics, varied_view):
    varied_view.plot_density_based_clustering(
        varied_optics["min_cluster_size=0.3"].labels_,
        title="min_cluster_size=0.3: needs at least 30 of 100 points",
    )
    return


@app.cell
def _(varied_optics, varied_view):
    varied_view.plot_density_based_clustering(
        varied_optics["min_cluster_size=0.1"].labels_,
        title="min_cluster_size=0.1: needs at least 10 of 100 points",
    )
    return


@app.cell
def _(mo, varied_table):
    mo.Html(varied_table.to_html())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Read the three `min_cluster_size` rows at the bottom of the table, and note that they move in the opposite direction to what you might expect from the DBSCAN sweeps.

    - **`0.4` requires at least 40 points.** The wide group cannot supply 40 points in one dense region, so the only group large enough to survive is reported: **1 cluster**, no noise. Raising the size threshold *discarded* clusters rather than creating them.
    - **`0.3` requires at least 30 points.** **3 clusters**, no noise. All three groups are large enough to qualify.
    - **`0.1` requires only 10 points.** **4 clusters and 11 noise**. Lowering the threshold lets OPTICS report a small extra grouping, so more, smaller clusters appear.

    So `min_cluster_size` is a filter on size, not on quality. Raising it removes small groups from the output; it does not make the remaining ones better.

    And this is the difference from the DBSCAN sweeps: there, raising a parameter increased noise. Here, raising `min_cluster_size` *reduces* the number of reported groups while leaving noise at zero, because the points are simply absorbed into a larger cluster rather than rejected.

    ### Try it yourself

    Replace `None` so that `student_optics_best` holds the number of clusters reported by the OPTICS fit named `"min_samples=10, xi=0.05"` in `varied_table`.
    """)
    return


@app.cell
def _(varied_table):
    student_optics_best = None
    print(student_optics_best)
    return (student_optics_best,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
```python
student_optics_best = int(
    varied_table.loc["min_samples=10, xi=0.05", "clusters"]
)
```
The answer is **3**, matching the three groups in the data, with zero noise.

`loc` takes the row by name and then the column. The row index is the `fit` column, which is why the name is the long string rather than a number. `int(...)` is used so the value prints as `3` rather than as a NumPy integer.

Compare it with the DBSCAN row at the top of the same table: 2 clusters and 44 noise points.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    In this cloud, raising `min_cluster_size` reduced the number of reported clusters rather than increasing the noise. Explain why that differs from raising `eps` in DBSCAN. Work out your answer, then reveal it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
Because the two parameters act at different stages.

`eps` in DBSCAN is part of the **clustering itself**. It decides which points are close enough to be neighbours, so changing it changes which points can be core, and therefore which points end up as noise. A smaller radius rejects points.

`min_cluster_size` in OPTICS is a **filter applied to the result**. The density profile has already been computed; this parameter only decides which of the found groups are large enough to report. A group that fails the size test is dropped, and its points join a neighbouring cluster or are simply not reported as a separate cluster. That does not make them noise.

In short: `eps` changes what the algorithm can see, while `min_cluster_size` decides what it bothers to tell you.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Moons and rings

    The last two datasets are the irregular shapes from earlier lessons. Both are **standardized before fitting**, and that step matters more than it looks.

    DBSCAN measures distances in the units of the data. If one column has a much wider range than the other, that column dominates every distance, and `eps` ends up calibrated to it alone. Standardizing puts both columns on the same scale so `eps` means the same thing in each direction.

    This is not a formality. On the moons, clustering the **raw** coordinates with `eps=0.5` and `min_samples=10` produces **1 cluster and 0 noise** with an adjusted Rand index of **0.000**, a complete failure. Standardizing first gives **2 clusters, 0 noise, and an adjusted Rand index of 1.000**. Same data, same parameters, opposite outcome.

    The plots below use the original coordinates so the shapes stay recognisable, while the fits use the standardized values.
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
    moon_view.plot_data(title="Half moons: two interlocking crescents")
    return (
        circle_db,
        circle_db_ari,
        circle_db_groups,
        circle_db_noise,
        circle_op,
        circle_op_ari,
        circle_op_groups,
        circle_op_noise,
        circle_view,
        moon_db,
        moon_db_ari,
        moon_db_groups,
        moon_db_noise,
        moon_op,
        moon_op_ari,
        moon_op_groups,
        moon_op_noise,
        moon_view,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Half moons with DBSCAN

    DBSCAN finds groups by growing from dense cores. A crescent is a dense curve, so the method should be able to follow it, and unlike k-means it has no reason to cut across the middle.

    First the plain clustering, then the same fit with core and border points distinguished.
    """)
    return


@app.cell
def _(moon_db, moon_view):
    moon_view.plot_density_based_clustering(
        moon_db.labels_, title="Half moons, DBSCAN"
    )
    return


@app.cell
def _(moon_db, moon_view):
    moon_view.plot_DBSCAN_categories(
        moon_db.labels_,
        moon_db.core_sample_indices_,
        title="Half moons, DBSCAN with point types",
    )
    return


@app.cell
def _(moon_db, moon_op, moon_view):
    moon_view.plot_density_based_clustering(
        moon_op, title="Half moons, OPTICS"
    )
    return


@app.cell(hide_code=True)
def _(mo, moon_db_ari, moon_db_groups, moon_db_noise, moon_op_ari, moon_op_groups, moon_op_noise):
    mo.md(f"""
    Both methods recover the two crescents completely. DBSCAN reports **{moon_db_groups} clusters and {moon_db_noise} noise points** with an adjusted Rand index of **{moon_db_ari:.3f}**; OPTICS reports **{moon_op_groups} clusters and {moon_op_noise} noise** with **{moon_op_ari:.3f}**.

    A perfect score here needs a caveat. OPTICS and DBSCAN both recovered the two crescents and discarded **no** points, so the partition is unambiguous. But as the background section warns, a score of 1.0 alone would not have proved that: if one entire crescent had been labelled noise and the other clustered correctly, the index would also have been 1.0. It is the combination of **score 1.0, two clusters, and zero noise** that establishes the result.

    The point-type plot is worth a second look. Each crescent has a dense core running along its length and a thin halo of border points at the edges. That is what a crescent looks like to a density algorithm: a curve of high density, not a blob.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Concentric circles

    The rings are the harder shape for methods that compare distances, because the inner circle is *surrounded* by the outer ring. There is no direction in which a straight line separates them.

    Density has no such difficulty. Both rings are crowded curves, and the empty annulus between them is exactly what a density method looks for: a gap in the data.
    """)
    return


@app.cell
def _(circle_view):
    circle_view.plot_data(title="Concentric circles: a ring around a circle")
    return


@app.cell
def _(circle_db, circle_view):
    circle_view.plot_density_based_clustering(
        circle_db.labels_, title="Concentric circles, DBSCAN"
    )
    return


@app.cell
def _(circle_db, circle_view):
    circle_view.plot_DBSCAN_categories(
        circle_db.labels_,
        circle_db.core_sample_indices_,
        title="Concentric circles, DBSCAN with point types",
    )
    return


@app.cell
def _(circle_op, circle_view):
    circle_view.plot_density_based_clustering(
        circle_op, title="Concentric circles, OPTICS"
    )
    return


@app.cell(hide_code=True)
def _(circle_db_ari, circle_db_groups, circle_db_noise, circle_op_ari, circle_op_groups, circle_op_noise, mo):
    mo.md(f"""
    Both methods separate the two rings exactly. DBSCAN reports **{circle_db_groups} clusters and {circle_db_noise} noise** with an adjusted Rand index of **{circle_db_ari:.3f}**; OPTICS reports **{circle_op_groups} clusters and {circle_op_noise} noise** with **{circle_op_ari:.3f}**.

    The reason is visible in the data itself. The two rings are separated by an empty band, and a density method is built to find exactly that: the band is where the reachability jumps, so no cluster can grow across it.

    This is the same dataset that k-means could not separate without splitting both rings with a straight line, and that affinity propagation could not separate with a single preference. Density methods handle it naturally because they ask a different question: not "how far is this from a centre", but "is this point part of a crowd".
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Replace `None` so that `student_moon_ari` holds the adjusted Rand index of the DBSCAN fit on the moons, and `student_moon_noise` holds its noise count.
    """)
    return


@app.cell
def _(adjusted_rand_score, moon_db, moon_arms):
    student_moon_ari = None
    student_moon_noise = None
    print("adjusted Rand:", student_moon_ari)
    print("noise:", student_moon_noise)
    return student_moon_ari, student_moon_noise


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
```python
student_moon_ari = adjusted_rand_score(moon_arms, moon_db.labels_)
student_moon_noise = int((moon_db.labels_ == -1).sum())
```
The index is **1.000** and the noise count is **0**. Both arguments of `adjusted_rand_score` are label vectors over the same 500 points; the order does not matter, because the index is symmetric.

The noise count is **0**, which matters for interpreting that score. A perfect index on its own would also have been possible with a group entirely labelled noise, because renaming a whole group to `-1` leaves the partition consistent. Here the fit found the two crescents and discarded nothing, so the score reflects the clustering you can see in the plot.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Which of these is the correct reason the two moons are separated so cleanly? Work out your answer, then reveal it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
Because **each crescent is a region of consistently high density, separated from the other by a region of low density.**

DBSCAN does not look at direction or shape. It asks, for each point, whether enough neighbours sit within `eps`. Along a crescent the answer is yes, because the points are packed along the curve. Between the crescents there is a gap, so points there have too few neighbours and the cluster cannot grow across it.

That is why the method succeeds where k-means failed. K-means had to compare every point with two centres, and a straight boundary through the middle put both arms in each cluster. Density never needs a straight boundary; it only needs the gap to be wider than `eps`.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The score trap, demonstrated

    The background section claimed that relabelling a whole group as noise can leave the adjusted Rand index at 1.0. That is a strong claim, so here it is as a check you can run yourself.

    Take the correct labels for the moons, then relabel **one entire crescent** as noise and leave the other untouched. Nothing about the clustering has been fixed; half the data has simply been thrown away.
    """)
    return


@app.cell
def _(adjusted_rand_score, moon_arms):
    discarded_demo = moon_arms.copy()
    discarded_demo[moon_arms == 0] = -1
    demo_table_ari = adjusted_rand_score(moon_arms, discarded_demo)
    demo_clusters = len(set(discarded_demo.tolist())) - 1
    demo_noise = int((discarded_demo == -1).sum())
    demo_sentence = (
        f"Discarding one whole crescent: {demo_clusters} cluster(s), "
        f"{demo_noise} noise points, adjusted Rand index {demo_table_ari:.3f}"
    )
    demo_sentence
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **One cluster, 250 noise points, and an index of 1.000.** No clustering was performed at all, and the score reports perfect agreement.

    That is why this lesson never reads the index on its own. Read three numbers together:

    | What to check | Why |
    | :--- | :--- |
    | The adjusted Rand index | Whether the labels agree with the truth up to renaming |
    | The number of clusters | Whether the groups you expected were actually found |
    | The noise count | Whether any group was quietly discarded |

    A run that reports a perfect index alongside a suspiciously high noise count has not solved the problem. It has renamed it.
    """)
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

    1. **Every point is core, border, or noise.** A core point has at least `min_samples` neighbours within `eps`, counting itself. A border point is within `eps` of a core point but is not one. Everything else is noise, labelled `-1`. Out of the 50 blob points, the default fit classified 32 as core, 9 as border, and 9 as noise.
    2. **`eps` and `min_samples` both control a trade-off, and each fails in both directions.** Shrinking `eps` to 0.35 pushed noise to 27 of 50 points and left only 15 core. Raising it to 1 gave two clean clusters with no noise, and raising it to 2 merged the two blobs into one cluster. Raising `min_samples` to 8 pushed noise to 16 and cut core points to 19.
    3. **Zero noise does not mean a good fit.** At `eps=2` the summary reads one cluster and no noise, and the grouping is wrong. Noise is what the algorithm discards, so removing it can mean the distinctions were destroyed rather than resolved.
    4. **A single `eps` cannot serve groups of different densities.** On the three-density cloud, DBSCAN with `eps=0.5` found only {int(varied_table.loc["DBSCAN eps=0.5, min_samples=10", "clusters"])} clusters and rejected {int(varied_table.loc["DBSCAN eps=0.5, min_samples=10", "noise"])} of the 100 points, because the group with standard deviation 2.0 is far sparser than the radius assumes. The tight group is easy to miss at full scale; zoom before concluding it is absent.
    5. **OPTICS handles the mixed densities by ordering rather than thresholding.** The same cloud yielded **3 clusters and 0 noise** at `min_samples=10` and `xi=0.05`. Its parameters do different jobs: `min_samples` sets the density requirement, `xi` how steep a boundary must be, and `min_cluster_size` the smallest reportable group, where a fraction means a proportion of the sample.
    6. **`min_cluster_size` filters results rather than changing the clustering.** Raising it from 0.1 to 0.4 reduced the reported clusters from 4 to 1 while keeping noise at zero, because small groups are dropped rather than rejected. This is the opposite direction from raising `eps`, which creates noise.
    7. **Standardizing is decisive on the moons.** On the raw coordinates the default DBSCAN produced one cluster and an adjusted Rand index of 0.000; on standardized coordinates, two clusters and 1.000.
    8. **Density methods solve the shapes the earlier methods struggled with.** Moons: {moon_db_groups} clusters, {moon_db_noise} noise, adjusted Rand {moon_db_ari:.3f} for DBSCAN and {moon_op_ari:.3f} for OPTICS. Rings: {circle_db_groups} clusters, {circle_db_noise} noise, adjusted Rand {circle_db_ari:.3f} for DBSCAN and {circle_op_ari:.3f} for OPTICS. The rings separate because of the empty band between them, which no cluster can cross.
    9. **A perfect score needs the noise count beside it.** Because every noise point shares the same label, a whole group can be relabelled as noise while the adjusted Rand index stays at 1.0. Moons and rings both reach 1.0 here with two clusters *and* zero noise, and the zero is what rules out a silently discarded group.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check your understanding

    1. What are the three types of point in DBSCAN, and how is each defined?
    2. Why can a border point never start a cluster of its own?
    3. What happens to the noise count as `eps` decreases, and what happens to the cluster count as `eps` becomes very large?
    4. Why is a run with zero noise points not necessarily better than one with several?
    5. Why can no single `eps` work on a dataset whose groups have different densities, and what does OPTICS do differently?
    6. What does a `min_cluster_size` below 1 mean, and why does raising it reduce the number of reported clusters instead of increasing noise?
    7. Why does standardizing the moons change the result from one cluster to two?
    8. The adjusted Rand index is 1.0 on both the moons and the rings. What must be true for it to reach that value in this lesson?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
1. **Core** has at least `min_samples` neighbours within `eps`, counting itself. **Border** is within `eps` of a core point but is not core itself. **Noise** is neither, and is labelled `-1`.
2. **Only core points have enough neighbours to grow a cluster.** A border point qualifies only because a core point is within `eps` of it. Remove the core and it becomes noise, so it cannot recruit others.
3. **Noise rises as `eps` falls**, because fewer neighbours fall inside the smaller radius. As `eps` becomes very large, the **cluster count falls**, because the radius crosses the gaps between groups and merges them, eventually into one.
4. **Noise is what the algorithm discards.** A large radius can leave nothing to discard simply by sweeping everything into one group, as at `eps=2` on the blobs. Judge the fit by whether the real groups survived.
5. **One radius is one definition of dense.** A value small enough for a tight group is too small for a sparse one, and a value large enough for the sparse group swallows the tight one. OPTICS avoids committing to a radius: it orders points by density and records reachability distances, so groups of different densities appear as separate features of one profile.
6. **A fraction of the sample**, so `0.1` means at least 10% of the points. Raising it removes small groups from the output, and their points are absorbed or simply not reported, which does not make them noise. The profile was already computed; the parameter filters what is reported.
7. **Because DBSCAN measures distance in the units of the data.** On the raw moons, one column has a wider range, so `eps=0.5` is calibrated to that column and both crescents merge into a single cluster with an index of 0.000. Standardizing puts both columns on the same scale, and the same parameters then find two clusters with an index of 1.000.
8. **A perfect score requires the groups to be recovered, but not necessarily every point.** Because all noise points share one label, an entire group relabelled as noise still scores 1.0. What proves the result here is the combination: score 1.0, the expected number of clusters, and zero noise.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - [DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
    - [OPTICS](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html)
    - [Density-based clustering, scikit-learn user guide](https://scikit-learn.org/stable/modules/clustering.html#dbscan)
    - Ester, M., Kriegel, H.-P., Sander, J., and Xu, X. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. *KDD-96*.
    - Ankerst, M., Breunig, M. M., Kriegel, H.-P., and Sander, J. (1999). OPTICS: Ordering points to identify the clustering structure. *SIGMOD '99*.
    - VanderPlas, J. (2017). *Python Data Science Handbook*. O'Reilly Media. Chapter 5.
    """)
    return


if __name__ == "__main__":
    app.run()
