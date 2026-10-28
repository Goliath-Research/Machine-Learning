# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "plotly",
#     "scikit-learn>=1.2",
#     "scipy",
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
        "matplotlib": "matplotlib",
        "sklearn": "scikit-learn>=1.2",
        "scipy": "scipy",
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
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.io as pio
    from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.datasets import load_iris, make_blobs
    from sklearn.manifold import TSNE
    from sklearn.metrics import (
        adjusted_mutual_info_score,
        adjusted_rand_score,
        fowlkes_mallows_score,
        normalized_mutual_info_score,
    )
    from sklearn.preprocessing import StandardScaler

    pio.templates.default = "plotly_white"
    px.defaults.color_discrete_sequence = px.colors.qualitative.G10
    matplotlib.rcParams["figure.dpi"] = 110

    return (
        AgglomerativeClustering,
        StandardScaler,
        TSNE,
        adjusted_mutual_info_score,
        adjusted_rand_score,
        cv,
        dendrogram,
        fcluster,
        fowlkes_mallows_score,
        linkage,
        load_iris,
        make_blobs,
        mo,
        normalized_mutual_info_score,
        np,
        pd,
        plt,
        px,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Hierarchical Clustering

    ## Objectives

    By the end of this lesson, you should be able to:

    1. Explain what the height of a join means and what a horizontal cut does.
    2. Read a dendrogram from the first join at the bottom to the last merge at the top, and predict how many clusters a given cut height produces.
    3. Distinguish **single**, **complete**, **average**, and **ward** linkage by how each one measures the distance between two groups.
    4. Explain which external scores are corrected for chance and which is not, and why that changes how you read them.
    5. Describe which iris species separate cleanly under hierarchical clustering and which two do not.

    ## Background

    **Hierarchical clustering** builds a tree instead of a flat set of groups. It starts with every point in its own group and repeatedly merges the two closest groups, recording the height at which each merge happened. The result is a **dendrogram**: leaves at the bottom are the observations, and each join is a merge.

    - The **height** of a join is the distance between the two groups that merged.
    - Merges near the bottom are close together; merges near the top happened last, across large distances.
    - A **horizontal cut** at some height keeps every group that had not yet merged at that height. The number of vertical lines the cut crosses is the number of clusters.

    Nothing is "chosen" until you cut the tree. One dendrogram therefore contains every possible number of clusters at once, which is what makes this method different from k-means.

    ### Linkage: how two groups are measured

    The one decision that changes the whole tree is how you measure the distance between two *groups* rather than two points.

    | Linkage | Distance between two groups |
    | :--- | :--- |
    | **single** | the closest pair of points, one from each group |
    | **complete** | the farthest pair of points, one from each group |
    | **average** | the mean of all pairwise distances |
    | **ward** | the increase in within-group variance that the merge would cause |

    Single linkage can follow a chain of nearby points and join two groups through one thin bridge; this is called **chaining**. Complete linkage resists chaining, because a single close pair is not enough. Ward merges whatever keeps the groups most compact, which is why it is the default for most work.

    The dendrogram is drawn with matplotlib and SciPy, because SciPy computes the tree (`linkage`) and matplotlib draws it, and the helper below shows the tree beside the resulting scatter in one figure. `ClusterVisualizer` has no dendrogram method. Every later scatter and every score bar goes through `ClusterVisualizer`.

    ## Datasets used

    ### Ten blobs, and four hundred blobs

    Both come from `sklearn.datasets.make_blobs`, so they are generated in the notebook rather than downloaded. The ten-point cloud (`random_state=2`) is small enough to number every point and read the tree by hand. The 400-point cloud uses four centres with `cluster_std=1.8`, wide enough that the groups overlap, and it keeps the generator labels so the clusterings can be scored.

    ### Iris

    **Source.** The iris dataset, collected by the botanist **Edgar Anderson** and published by **R. A. Fisher** in 1936 in *The use of multiple measurements in taxonomic problems* (*Annals of Eugenics*, 7(2), 179–188). Loaded here with `sklearn.datasets.load_iris`. Also distributed by the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/53/iris).

    **What it holds.** 150 flowers, 50 from each of three species, with four measurements in centimetres:

    | Column | Meaning |
    | :--- | :--- |
    | sepal length (cm) | Length of the sepal |
    | sepal width (cm) | Width of the sepal |
    | petal length (cm) | Length of the petal |
    | petal width (cm) | Width of the petal |

    The three species are *Iris setosa*, *Iris versicolor*, and *Iris virginica*.

    **Why it suits this lesson.** One species, setosa, is clearly separable from the other two on these measurements. Versicolor and virginica overlap, and no clustering of four petal and sepal measurements can fully separate them, because some flowers genuinely have intermediate measurements. That asymmetry is the point: it shows what clustering can and cannot recover, and it is why the scores in this lesson land around 0.6 rather than 1.0.

    The dataset is a historical benchmark, used in statistics for nearly ninety years. That makes it easy to compare results with published work, and it also means it is unusually clean: no missing values, balanced classes, and measurements of a single flower type. Real botanical data is rarely this tidy.
    """)
    return


@app.cell
def _(dendrogram, fcluster, linkage, np, plt):
    def plot_clusters_from_dendrogram(
        X, method="ward", threshold=1, annotate=True, title=None
    ):
        """Draw a dendrogram beside the scatter plot produced by cutting it.

        Parameters
        ----------
        X : array-like of shape (n_samples, 2)
            Points to cluster. The first two columns are used as the x and y
            coordinates of the scatter plot.
        method : str, default "ward"
            Linkage passed to `scipy.cluster.hierarchy.linkage`. One of "ward",
            "single", "complete", "average", "centroid", or "median".
        threshold : float, default 1
            Height at which to cut the tree. The cut line is drawn on the
            dendrogram and `fcluster` uses the same value.
        annotate : bool, default True
            Number points with their row index. Useful on small clouds, where it
            lets you follow individual observations through the tree.
        title : str or None, default None
            Optional heading placed above both panels.

        Returns
        -------
        matplotlib.figure.Figure
            The figure, with the dendrogram on the left and the clusters on the
            right. Return it so marimo displays it, instead of relying on the
            global pyplot state.
        """
        data_labels = np.arange(X.shape[0])
        Z = linkage(X, method=method)
        labels = fcluster(Z, threshold, criterion="distance")
        num_clusters = len(np.unique(labels))

        figure, (ax_tree, ax_points) = plt.subplots(
            1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.35, 1]}
        )

        dendrogram(Z, ax=ax_tree)
        ax_tree.set_title(f"Dendrogram: {method} linkage")
        ax_tree.set_xlabel("Data points, in linkage order")
        ax_tree.set_ylabel("Distance between merged groups")
        ax_tree.grid(True, color="lightgrey", linestyle="-", linewidth=0.5)
        ax_tree.axhline(y=threshold, color="black", linestyle="--", linewidth=1)
        ax_tree.annotate(
            f"cut at {threshold}",
            xy=(0.02, threshold),
            xycoords=("axes fraction", "data"),
            va="bottom",
            fontsize=9,
        )

        ax_points.scatter(X[:, 0], X[:, 1], c=labels, cmap="gist_rainbow", s=45)
        ax_points.set_title(f"Cut at height {threshold}: {num_clusters} clusters")
        if annotate:
            for index, text in enumerate(data_labels):
                ax_points.annotate(
                    text=text,
                    xy=(X[index, 0], X[index, 1]),
                    textcoords="offset points",
                    xytext=(0, 6),
                    ha="center",
                    color="black",
                    size=9,
                )
        ax_points.grid(True, color="lightgrey", linestyle="-", linewidth=0.5)

        if title:
            figure.suptitle(title)
        figure.tight_layout()
        return figure

    return (plot_clusters_from_dendrogram,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. A small cloud you can read by hand

    Ten points, numbered so you can follow each one through the tree. A tree is easiest to read **from the bottom up**: start at the lowest join and work upward, because that is the order in which the merges actually happened.
    """)
    return


@app.cell
def _(fcluster, linkage, make_blobs, np, pd):
    tiny, _tiny_labels = make_blobs(random_state=2, n_samples=10)
    tiny_frame = pd.DataFrame(tiny, columns=["x", "y"]).round(2)
    _ward_tree = linkage(tiny, method="ward")
    tiny_merges = pd.DataFrame(
        [
            {
                "Merge": _step,
                "Left": str(int(_row[0])),
                "Right": str(int(_row[1])),
                "Height": _row[2],
                "New size": int(_row[3]),
            }
            for _step, _row in enumerate(_ward_tree)
        ]
    ).set_index("Merge")
    tiny_groups = fcluster(linkage(tiny, method="ward"), t=2, criterion="distance")
    tiny_cut = len(np.unique(tiny_groups))
    tiny_frame
    return tiny, tiny_cut, tiny_frame, tiny_merges


@app.cell
def _(tiny_merges):
    mo.Html(tiny_merges.round(3).to_html())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reading the tree from the bottom up

    The table lists the ten merges in the order ward performed them. Read it alongside the dendrogram below, starting at the lowest height.

    - **Merge 0, height 0.479:** points **1** and **5** join first. They are the closest pair in the cloud.
    - **Merge 1, height 0.545:** points **3** and **8** join, a second pair almost as close.
    - **Merge 2, height 1.355:** point **7** joins the 3-8 group, making a group of three.
    - **Merges 3 to 5, heights 1.375 to 1.737:** three more small groups form: 0 with 6, 9 with the 1-5 pair, and 2 with the 3-8-7 group.
    - **Merge 6, height 4.237:** two of those groups combine into four points.
    - **Merge 7, height 6.842:** two groups combine into six points.
    - **Merge 8, height 15.976:** the last merge joins everything.

    The final merge tells you the top-level split. Cutting just below 15.976 leaves two branches: **{1, 4, 5, 9}** and **{0, 2, 3, 6, 7, 8}**. Those are the two largest groups the tree offers, and they are the ones a cut at height 2 through 8 recovers.
    """)
    return


@app.cell
def _(plot_clusters_from_dendrogram, tiny):
    plot_clusters_from_dendrogram(
        tiny, threshold=2, title="Ten points, ward linkage, cut at height 2"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A cut at height **2** crosses four of the vertical lines, so it produces **four** clusters. Notice that the cut sits above the small joins near 0.5 and 1.4, so those are already inside a group, and below the joins at 4.2 and above, so those have not happened yet. A cut does not create groups; it decides which merges to keep.

    ### The same height under four linkages

    The next four figures all cut at height **4**. They do not have to agree on the number of clusters, because each linkage places its joins at different heights.
    """)
    return


@app.cell
def _(plot_clusters_from_dendrogram, tiny):
    plot_clusters_from_dendrogram(
        tiny, method="ward", threshold=4, title="Ward linkage"
    )
    return


@app.cell
def _(plot_clusters_from_dendrogram, tiny):
    plot_clusters_from_dendrogram(
        tiny, method="single", threshold=4, title="Single linkage"
    )
    return


@app.cell
def _(plot_clusters_from_dendrogram, tiny):
    plot_clusters_from_dendrogram(
        tiny, method="complete", threshold=4, title="Complete linkage"
    )
    return


@app.cell
def _(plot_clusters_from_dendrogram, tiny):
    plot_clusters_from_dendrogram(
        tiny, method="average", threshold=4, title="Average linkage"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    At the same height of 4, the four linkages give different answers: **ward gives 4 clusters, single gives 1, complete gives 3, and average gives 3**. The height is the same; the trees are not.

    Two lessons follow. First, a cut height is not a portable setting: you cannot reuse "4" across linkages and expect the same structure. Second, the shapes of the failure differ. In **single linkage** the whole cloud has already merged into one group by height 4, because a chain of near neighbors connects everything; that is chaining. In **ward** the groups stay compact and separate, so the same height still leaves four of them.

    ### Try it yourself

    What does the **height** of a join in a dendrogram represent? Work out your answer, then reveal it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
The height is the **distance between the two groups that were merged** at that join. A taller join means the groups were farther apart when they finally combined.

Check it against the merge table above: the first join sits at 0.479 and the last at 15.976. The bottom of the tree is where points are close, and the top is the single merge that covers the whole cloud.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Use `fcluster` to cut the ward tree of `tiny` at height **5** and count the clusters. Replace `None` so that `student_tiny_cut` holds that count.
    """)
    return


@app.cell
def _(fcluster, linkage, np, tiny):
    student_tiny_cut = None
    print(student_tiny_cut)
    return (student_tiny_cut,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
```python
student_tiny_cut = len(
    np.unique(fcluster(linkage(tiny, method="ward"), t=5, criterion="distance"))
)
```
The answer is **3**. A cut at 5 sits above the joins at 4.237 and below the join at 6.842, so it crosses three branches. Compare this with the cut at height 2, which gave four: raising the cut keeps fewer, larger groups.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Four overlapping blobs, four linkages

    This cloud has 400 points generated around **four** centers with `cluster_std=1.8`, which is wide enough that the groups overlap. Because we generated the data we also have the true group of every point, which lets us score the clusterings later.

    Each linkage is asked for four clusters and for three. Asking for the right number is not the same as finding the right groups, and asking for one too few shows how each linkage copes with being forced to merge.
    """)
    return


@app.cell
def _(AgglomerativeClustering, cv, make_blobs, pd):
    wide, wide_labels = make_blobs(
        n_samples=400, centers=4, cluster_std=1.8, random_state=10
    )
    wide_view = cv.ClusterVisualizer(
        pd.DataFrame(wide, columns=["x", "y"]), wide_labels
    )
    agg_labels = pd.DataFrame()
    for _method, _k in (
        ("ward", 4),
        ("ward", 3),
        ("single", 4),
        ("single", 3),
        ("complete", 4),
        ("complete", 3),
        ("average", 4),
        ("average", 3),
    ):
        agg_labels[f"{_method.title()}-{_k}"] = AgglomerativeClustering(
            n_clusters=_k, linkage=_method
        ).fit_predict(wide)
    wide_view.plot_data(title="Four hundred points around four overlapping centers")
    return agg_labels, wide_labels, wide_view


@app.cell
def _(wide_view):
    wide_view.plot_data_with_true_labels()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The plot above colours the points by the group that **generated** them. That is the answer key, and it is the reference every fit below is measured against. Four groups of exactly 100 points each, visibly overlapping in the middle of the cloud.

    ### Ward linkage, asking for four clusters

    Ward merges whatever increases the within-group variance least, so it prefers compact groups. Compare this plot with the answer key: the four colours should land in roughly the same four places, though the boundary between the two central groups will not be identical because the groups genuinely overlap there.

    The cluster sizes confirm it: **105, 103, 97, and 95** against the true 100 each. Nothing chained and nothing was stranded.
    """)
    return


@app.cell
def _(agg_labels, wide_view):
    wide_view.plot_clusters(
        agg_labels["Ward-4"], title="Ward linkage, 4 clusters (the correct number)"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now the same data and the same linkage, but we ask for **three** clusters. Ward has no choice but to merge two of the four real groups, so one cluster doubles in size to about **200** while the others stay near 100.

    Look for the oversized cluster in the plot: it is the one that spans two of the answer-key colours. Asking for one group too few does not produce "a cleaner version" of the answer; it forces a real distinction to be discarded.
    """)
    return


@app.cell
def _(agg_labels, wide_view):
    wide_view.plot_clusters(
        agg_labels["Ward-3"], title="Ward linkage, 3 clusters (one group too few)"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Single linkage, the same two questions

    Single linkage measures the distance between two groups as their **closest pair** of points. Before looking at the plot, predict what that should do on overlapping data: if any single pair of points from two groups happens to be close, those groups become mergeable, even when the groups as a whole are far apart.

    That is chaining, and it is what you should look for here.
    """)
    return


@app.cell
def _(agg_labels, wide_view):
    wide_view.plot_clusters(
        agg_labels["Single-4"],
        title="Single linkage, 4 clusters (chaining)",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This is the failure, and it is worth reading carefully. Asked for four clusters, single linkage returns **199, 199, 1, and 1**: two enormous halves and two isolated points.

    The two single-point clusters are the giveaway. A thin trail of points across the overlapping middle was enough to bridge two real groups into one, and the bridge cannot be undone further up the tree. The two points left on their own are simply the ones farthest from everything else, absorbed last.

    Note what the plot does *not* show: a slightly-wrong four-way split. It shows a qualitatively different structure, one where the requested number of clusters was met by producing two singletons.
    """)
    return


@app.cell
def _(agg_labels, wide_view):
    wide_view.plot_clusters(
        agg_labels["Single-3"],
        title="Single linkage, 3 clusters (the chaining persists)",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Asking for three does not repair it. The sizes become **200, 199, and 1**: the same two halves, still merged across the middle, with one point stranded. Chaining is a property of how the tree was built, not of how many clusters you request.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Complete linkage, the opposite rule

    Complete linkage measures the distance between two groups as their **farthest pair**. It is the strictest of the four rules: two groups cannot merge until *every* pair of points between them is reasonably close.

    Predict what that does to the chaining you just saw. A single close pair is no longer enough to force a merge, so the thin bridge across the middle should not be able to pull two groups together.
    """)
    return


@app.cell
def _(agg_labels, wide_view):
    wide_view.plot_clusters(
        agg_labels["Complete-4"],
        title="Complete linkage, 4 clusters (no chaining)",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The prediction holds. The sizes are **102, 100, 100, and 98**, the closest any linkage comes to the true 100 each, and there are no singletons. The plot shows four groups where the ward plot did, with the same unavoidable fuzziness in the middle.

    This is the best fit in the whole comparison, and the next two plots are its k = 3 counterpart.
    """)
    return


@app.cell
def _(agg_labels, wide_view):
    wide_view.plot_clusters(
        agg_labels["Complete-3"], title="Complete linkage, 3 clusters"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Under complete linkage, forcing three clusters produces **200, 102, and 98**: one genuine merge of two real groups, and the other two left intact. That is the honest failure mode when you ask for too few groups, and it is exactly what ward did as well.

    Contrast this with single linkage at three clusters, where the sizes were 200, 199, and 1. Both answers have a cluster of about 200, but only complete linkage's other clusters are real groups rather than leftovers.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Average linkage, the middle ground

    Average linkage measures the **mean** distance across all pairs between two groups. It sits between the two extremes: sensitive to overall separation like complete, but not as easily blocked by one far-flung point.

    Expect it to look a lot like complete linkage here, and to stay well clear of the chaining you saw under single linkage.
    """)
    return


@app.cell
def _(agg_labels, wide_view):
    wide_view.plot_clusters(
        agg_labels["Average-4"], title="Average linkage, 4 clusters"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Sizes are **105, 102, 98, and 95**, essentially the same four groups as complete linkage, with no singletons and no chaining.

    Now the same linkage with one group too few.
    """)
    return


@app.cell
def _(agg_labels, wide_view):
    wide_view.plot_clusters(
        agg_labels["Average-3"], title="Average linkage, 3 clusters"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Sizes are **105, 102, 98, and 95** at four clusters and **200, 102, and 98** at three, which are almost the same as complete linkage's. The plot is barely distinguishable from it, and that is the point: average and complete agree closely on well-behaved data, while single is the odd one out.

    ### All eight fits in one table

    Eight plots are hard to compare by eye. This table reduces each one to the two numbers that reveal its structure: the cluster sizes, and how well those clusters match the answer key.
    """)
    return


@app.cell
def _(agg_labels, adjusted_rand_score, np, pd, wide_labels):
    _rows = []
    for _name in agg_labels.columns:
        _sizes = sorted(np.bincount(agg_labels[_name]), reverse=True)
        _rows.append(
            {
                "Fit": _name,
                "Sizes": " / ".join(str(int(_size)) for _size in _sizes),
                "Smallest cluster": int(min(_sizes)),
                "Adjusted Rand": adjusted_rand_score(wide_labels, agg_labels[_name]),
            }
        )
    fit_summary = pd.DataFrame(_rows).set_index("Fit")
    mo.Html(fit_summary.round(3).to_html())
    return (fit_summary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Read the table one linkage at a time, comparing the 4-row with the 3-row above or below it.

    - **The `Smallest cluster` column is the fastest diagnostic.** For ward, complete, and average it never falls below 95. For single linkage it drops to **1** on both rows: that single number is the chaining, and you cannot see it in a bar chart of ARI.
    - **At four clusters**, complete (0.974), average (0.954), and ward (0.947) all recover the answer closely. Single linkage scores 0.496 because its 199/199 halves cut across two real groups.
    - **At three clusters**, the three sound linkages all lose about 0.25, because each must discard one real distinction. Single linkage barely moves (0.496 to 0.497), which is not resilience: it had already merged everything, so one fewer group changes nothing.
    - **The 3-row is never better than its 4-row for ward, complete, and average.** When you ask for the wrong number of groups, the fit degrades; it does not improvise a better answer.

    The four bar charts that follow score exactly these eight fits. Now that you know what each fit looks like, the bars should read as a summary rather than as the whole story. The highlighted bar in each chart is the best score among the eight.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Adjusted Rand index

    This is the score to read first, because it is corrected for chance and handles fits with different numbers of clusters fairly.
    """)
    return


@app.cell
def _(agg_labels, wide_view):
    wide_view.plot_adjusted_rand(agg_labels, title="Adjusted Rand index, all eight fits")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Complete-4** is highlighted at **0.974**, with Average-4 at 0.954 and Ward-4 at 0.947 close behind. Those three are the fits that recovered the four real groups.

    The two lowest bars are **Single-4** and **Single-3**, at 0.496 and 0.497. Note that they are almost identical despite asking for different numbers of clusters, which is the chaining signature again: the tree was already wrong before the cut, so the cut makes no difference.

    Also notice that the three k = 3 bars sit in a middle band around 0.69 to 0.70. Asking for one group too few is penalised, but far less than chaining is.

    ### Adjusted mutual information

    The same eight fits, scored by shared information rather than by pairwise agreement. It should tell the same story in a different currency.
    """)
    return


@app.cell
def _(agg_labels, wide_view):
    wide_view.plot_adjusted_mutual_info(
        agg_labels, title="Adjusted mutual information, all eight fits"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The ordering is nearly identical: **Complete-4** leads at **0.965**, ward and average are just behind at 0.939 and 0.946, and the single-linkage bars sink to 0.657 and 0.661.

    One difference is worth noticing. The gap between the good fits and the bad ones is small for this score in absolute terms, because mutual information is measured on a different scale from pairwise agreement. Compare *rankings* across scores, not the size of the gaps.

    ### Normalized mutual information

    Now an **unadjusted** score, kept here for comparison. Watch how the single-linkage bars look more respectable than they did under the adjusted scores.
    """)
    return


@app.cell
def _(agg_labels, wide_view):
    wide_view.plot_normalized_mutual_info(
        agg_labels, title="Normalized mutual information, all eight fits"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Normalized mutual information gives the single-linkage fits **0.660** and **0.663**, which is noticeably closer to the good fits than the adjusted scores suggested. That is the unadjusted score being generous: it has no chance correction, so a structurally broken partition still shares some information with the truth by accident.

    This is the practical reason the lesson distinguishes the two groups of scores.

    ### Fowlkes–Mallows

    The last chart, and the one to read most carefully. It is also unadjusted, and its floor is the highest of the four.
    """)
    return


@app.cell
def _(agg_labels, wide_view):
    wide_view.plot_fowlkes_mallows(
        agg_labels, title="Fowlkes–Mallows, all eight fits"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Every bar here is high. **Complete-4** leads at **0.980**, but **Single-4** still scores **0.702**, which looks like a respectable result for a fit that produced two singletons and cut across two real groups.

    Remember the shuffled-label baseline of about **0.25**: a meaningless clustering already scores a quarter of the maximum. So the usable range of this score on this data is roughly 0.25 to 1.00, not 0 to 1.00, and a value of 0.70 is far less impressive than it appears.

    Use this chart to rank fits that asked for the **same** number of clusters, where the comparison is fair. For anything else, go back to the adjusted Rand chart.
    """)
    return


@app.cell
def _(
    adjusted_mutual_info_score,
    adjusted_rand_score,
    agg_labels,
    fowlkes_mallows_score,
    normalized_mutual_info_score,
    np,
    pd,
    wide_labels,
):
    link_scores = pd.DataFrame(
        {
            "adjusted_rand": {
                _name: adjusted_rand_score(wide_labels, agg_labels[_name])
                for _name in agg_labels.columns
            },
            "adjusted_mutual_info": {
                _name: adjusted_mutual_info_score(wide_labels, agg_labels[_name])
                for _name in agg_labels.columns
            },
            "normalized_mutual_info": {
                _name: normalized_mutual_info_score(wide_labels, agg_labels[_name])
                for _name in agg_labels.columns
            },
            "fowlkes_mallows": {
                _name: fowlkes_mallows_score(wide_labels, agg_labels[_name])
                for _name in agg_labels.columns
            },
        }
    )
    best_link = link_scores["adjusted_rand"].idxmax()

    # Shuffle the labels to show each score's floor. This keeps the cluster sizes
    # and destroys only the correspondence with the true groups.
    _rng = np.random.RandomState(0)
    _shuffled = _rng.permutation(agg_labels["Ward-4"].to_numpy())
    chance_scores = pd.DataFrame(
        {
            "score": [
                "Adjusted Rand",
                "Adjusted Mutual Info",
                "Normalized Mutual Info",
                "Fowlkes-Mallows",
            ],
            "real fit": [
                adjusted_rand_score(wide_labels, agg_labels["Ward-4"]),
                adjusted_mutual_info_score(wide_labels, agg_labels["Ward-4"]),
                normalized_mutual_info_score(wide_labels, agg_labels["Ward-4"]),
                fowlkes_mallows_score(wide_labels, agg_labels["Ward-4"]),
            ],
            "same clusters, labels shuffled": [
                adjusted_rand_score(wide_labels, _shuffled),
                adjusted_mutual_info_score(wide_labels, _shuffled),
                normalized_mutual_info_score(wide_labels, _shuffled),
                fowlkes_mallows_score(wide_labels, _shuffled),
            ],
        }
    ).set_index("score")
    return best_link, chance_scores, link_scores


@app.cell
def _(link_scores):
    mo.Html(link_scores.round(3).to_html())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Every column is an **external** score: it compares the clustering with the labels that generated the data. Higher means closer agreement. Read them in two groups, because they do not share a baseline.

    | Score | Chance corrected? | What it measures |
    | :--- | :--- | :--- |
    | Adjusted Rand | yes | Agreement over pairs of points. Random labels score about 0. |
    | Adjusted Mutual Info | yes | Shared information, corrected for what random labels would share. About 0 at chance. |
    | Normalized Mutual Info | **no** | Shared information, scaled to 0 to 1. Its chance value is small but **not** 0. |
    | Fowlkes–Mallows | **no** | Geometric mean of pairwise precision and recall. Its chance value is **well above 0**. |

    Only the two adjusted scores are built so that random labels score about zero. For Fowlkes–Mallows, a score near zero is almost unreachable: the geometric mean of precision and recall stays moderate even when the labels are meaningless. The next cell demonstrates this rather than asking you to take it on trust, by shuffling one real clustering and rescoring it. The floor also depends on how many clusters you request, which is why a Fowlkes–Mallows score cannot be compared across different k.
    """)
    return


@app.cell
def _(chance_scores):
    mo.Html(chance_scores.round(3).to_html())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The shuffled column keeps the same cluster sizes and destroys only the correspondence with the truth. Read the drop:

    - **Adjusted Rand** falls to about **0.00**. A small negative value is normal.
    - **Adjusted Mutual Info** falls to about **0.00** as well.
    - **Normalized Mutual Info** falls to about 0.005, small but above zero.
    - **Fowlkes–Mallows** stays near **0.25** at four clusters.

    So a Fowlkes–Mallows score of 0.25 tells you almost nothing. Treat it as a tie-breaker between fits that used the **same** number of clusters, and never compare it across different k. For comparing clusterings with different numbers of groups, use the adjusted scores.

    ### Try it yourself

    Which external score is **not** corrected for chance, and therefore keeps a value well above zero even when the clustering is meaningless? Work out your answer, then reveal it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
The **Fowlkes–Mallows** score is not adjusted for chance, so shuffled labels still score about 0.25 at four clusters. Adjusted Rand and adjusted mutual information both fall to about 0.00.

Normalized mutual information is also unadjusted, but its chance value is small enough that it is far less misleading than Fowlkes–Mallows. That is why the lesson treats the two adjusted scores as the ones to rely on when comparing different numbers of clusters.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Which linkage agrees with the generator

    The table's last column answers the practical question. Complete linkage is best at **0.974**, ward and average follow at **0.947** and **0.954**, and single linkage trails at **0.496**.

    That ranking is worth stating carefully. It does not mean complete linkage is the right choice in general; on well-separated data single linkage would do fine, and its speed is an advantage. What the ranking shows is that on *this* data, where groups overlap and a thin bridge exists between them, the closest-pair rule is the risky one. The lesson is to know which failure each rule invites, not to memorise a winner.

    ### Try it yourself

    Which linkage measures the distance between two groups as the distance between their **closest pair** of points, and which uses their **farthest pair**? Reveal the answer when you have decided.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
**Single** linkage uses the closest pair; **complete** linkage uses the farthest pair.

Think about what that difference does. Single linkage asks only whether *any* two points are close, so one thin bridge of points is enough to join two groups. Complete linkage asks whether *every* pair is reasonably close, so a single close pair cannot force a merge. That is why single linkage chained the blobs into 199/199/1/1 while complete linkage kept them at 102/100/100/98.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Replace `None` so that `student_single_sizes` holds the four cluster sizes of the `Single-4` fit, using `np.bincount`. Then set `student_single_ari` to the adjusted Rand index of that same fit against `wide_labels`.
    """)
    return


@app.cell
def _(adjusted_rand_score, agg_labels, np, wide_labels):
    student_single_sizes = None
    student_single_ari = None
    print("Single-4 cluster sizes:", student_single_sizes)
    print("Single-4 ARI:", student_single_ari)
    return student_single_ari, student_single_sizes


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
```python
student_single_sizes = np.bincount(agg_labels["Single-4"])
student_single_ari = adjusted_rand_score(wide_labels, agg_labels["Single-4"])
```
The sizes are **199, 199, 1, and 1**, and the adjusted Rand index is about **0.496**, against **0.974** for complete linkage. `np.bincount` counts how many times each label appears, which is the quickest way to spot a chained result: two enormous groups and two singletons is the signature.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Iris

    The four measurements are standardized together. There is no holdout, so every flower contributes to the means and the standard deviations. The clustering uses those four columns.

    ### Why standardize here

    The four measurements are all in centimetres, so the units already agree; scale is not the problem in the way it was for the arrest rates in the k-means lesson. The spreads are the problem, and they differ a lot:

    | Measurement | Mean | Std | Min | Max |
    | :--- | ---: | ---: | ---: | ---: |
    | sepal length | 5.84 | 0.83 | 4.30 | 7.90 |
    | sepal width | 3.06 | 0.44 | 2.00 | 4.40 |
    | petal length | 3.76 | 1.77 | 1.00 | 6.90 |
    | petal width | 1.20 | 0.76 | 0.10 | 2.50 |

    Petal length varies four times as much as sepal width. Because distance sums squared differences, a measurement with a large spread contributes far more to every distance, so it effectively decides the clustering on its own. Standardizing puts all four on the same footing, one standard deviation, so each contributes comparably.

    This is a choice about what you want the distance to mean, not a correction of an error. Standardizing changes the answer, and on this dataset it changes the score as well.

    ### The measurements, before and after

    Box plots make the difference visible. The first shows the original centimetres, with the four measurements sitting at very different levels and spreads; the second shows the same columns standardized, all centred near 0 with comparable spread.
    """)
    return


@app.cell
def _(StandardScaler, load_iris, pd):
    # Iris: 150 flowers, 4 measurements in cm, 3 species.
    # Collected by Edgar Anderson; published by R. A. Fisher (1936), Annals of Eugenics 7(2), 179-188.
    # Bundled with scikit-learn, so no download is required: load_iris() reads a local copy.
    # Original distribution: https://archive.ics.uci.edu/dataset/53/iris
    iris = load_iris()
    iris_frame = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_frame["species"] = iris.target_names[iris.target]
    iris_z = pd.DataFrame(
        StandardScaler().fit_transform(iris_frame[iris.feature_names]),
        columns=iris.feature_names,
    )
    iris_ranges = iris_frame[iris.feature_names].describe().loc[
        ["mean", "std", "min", "max"]
    ]
    mo.Html(iris_ranges.round(3).to_html())
    return iris, iris_frame, iris_ranges, iris_z


@app.cell
def _(iris_frame, px):
    _original = iris_frame.melt(
        id_vars=["species"],
        var_name="measurement",
        value_name="centimetres",
        value_vars=iris_frame.columns[:-1],
    )
    _fig_original = px.box(
        _original,
        x="measurement",
        y="centimetres",
        color="measurement",
        title="Iris measurements in original centimetres",
    )
    _fig_original.update_layout(width=650, height=400, showlegend=False)
    _fig_original
    return


@app.cell
def _(iris_z, px):
    _standardized = iris_z.melt(
        var_name="measurement", value_name="standard deviations"
    )
    _fig_standardized = px.box(
        _standardized,
        x="measurement",
        y="standard deviations",
        color="measurement",
        title="The same measurements after standardizing",
    )
    _fig_standardized.update_layout(width=650, height=400, showlegend=False)
    _fig_standardized
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In the first figure the four boxes sit at different heights and have visibly different heights: petal length spans the widest range, sepal width the narrowest. In the second every box is centred near 0 with a comparable spread, so no single measurement dominates the distance calculation.

    ### Clustering the flowers

    t-SNE is **only a picture**. It is fit once, with `random_state=0`, and it is not the input to the clustering. Agglomerative clustering with ward linkage asks for three clusters using the four standardized measurements. The scores below compare those labels with the species; they do not score the t-SNE coordinates.
    """)
    return


@app.cell
def _(
    AgglomerativeClustering,
    TSNE,
    adjusted_mutual_info_score,
    adjusted_rand_score,
    cv,
    fowlkes_mallows_score,
    iris_frame,
    iris_z,
    normalized_mutual_info_score,
    pd,
):
    iris_plane = pd.DataFrame(
        TSNE(n_components=2, random_state=0, init="pca").fit_transform(iris_z),
        columns=["t-SNE 1", "t-SNE 2"],
    )
    iris_labels = AgglomerativeClustering(
        n_clusters=3, linkage="ward"
    ).fit_predict(iris_z)
    iris_view = cv.ClusterVisualizer(iris_plane, iris_frame["species"].to_numpy())
    iris_rand = adjusted_rand_score(iris_frame["species"], iris_labels)
    iris_ami = adjusted_mutual_info_score(iris_frame["species"], iris_labels)
    iris_nmi = normalized_mutual_info_score(iris_frame["species"], iris_labels)
    iris_fm = fowlkes_mallows_score(iris_frame["species"], iris_labels)
    iris_table = pd.crosstab(
        iris_frame["species"],
        pd.Series(iris_labels, name="cluster").map(lambda label: f"cluster {label}"),
    )
    iris_view.plot_data(title="Iris in the t-SNE view")
    return iris_ami, iris_fm, iris_labels, iris_nmi, iris_rand, iris_table, iris_view


@app.cell
def _(iris_view):
    iris_view.plot_data_with_true_labels(title="Species, t-SNE view")
    return


@app.cell
def _(iris_labels, iris_view):
    iris_view.plot_clusters(iris_labels, title="Ward, 3 clusters")
    return


@app.cell
def _(iris_table):
    mo.Html(iris_table.to_html())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reading the table

    The crosstab shows how the three fitted clusters line up with the three species. Two things stand out, and both match what the Jupyter reference observed.

    - **Setosa is almost perfectly isolated.** Cluster 1 holds 49 of the 50 setosa flowers and nothing else except a single virginica. The species is separable on these measurements, and ward finds it without being told the species.
    - **Versicolor and virginica are mixed.** The other two clusters split those two species rather than separating them: cluster 0 holds 23 versicolor and 48 virginica, while cluster 2 holds 27 versicolor and 2 virginica. No cluster is "the versicolor group", and none is "the virginica group".

    That mixing is the reason the scores are moderate rather than high. With three clusters, two of them are largely a split of the two overlapping species, so most versicolor and virginica flowers share a cluster with the other species. Only setosa is recovered cleanly.

    ### Try it yourself

    Which species separates cleanly into its own cluster, and which two are mixed together? Read `iris_table` above, then reveal the answer.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
**Setosa** separates cleanly: 49 of its 50 flowers sit alone in one cluster. **Versicolor and virginica** are mixed across the other two clusters.

Look at the largest entry in each row of the table. Setosa's row is concentrated in a single cluster, while versicolor's and virginica's rows are spread over two. The measurements separate one species well and the other two only partly, which no amount of clustering can fix, because the flowers themselves overlap.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Which linkage chooses the merge that increases the **within-group variance** the least? Work out your answer, then reveal it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
**Ward.** Single, complete, and average linkage all measure a distance between pairs of points. Ward measures the increase in within-group variance that the merge would cause, and merges whichever adds least.

That is why ward produced even groups of 102, 100, 100, and 98 on the blobs, while single linkage produced 199, 199, 1, and 1.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(
    best_link,
    chance_scores,
    iris_ami,
    iris_fm,
    iris_nmi,
    iris_rand,
    iris_table,
    link_scores,
    mo,
    tiny_cut,
):
    _main = iris_table.idxmax(axis=1)

    def _describe_species(species, counts):
        """Summarise where one species lands, including how pure that cluster is.

        Parameters
        ----------
        species : str
            Species name, used only for the message.
        counts : pandas.Series
            Counts of this species across the fitted clusters.

        Returns
        -------
        str
            A short description naming the cluster holding most of this species
            and reporting how many flowers in that cluster belong to other
            species, which is what makes a cluster mixed.
        """
        largest_cluster = counts.idxmax()
        own = int(counts.max())
        others = int(iris_table[largest_cluster].sum() - own)
        if others == 0:
            return f"{own} of {int(counts.sum())} in {largest_cluster}, a pure cluster"
        return (
            f"{own} of {int(counts.sum())} in {largest_cluster}, which also holds "
            f"{others} flowers of other species"
        )

    _lines = "\n".join(
        f"  - **{_species}**: {_describe_species(_species, _counts)}."
        for _species, _counts in iris_table.iterrows()
    )
    _single_ari = link_scores.loc["Single-4", "adjusted_rand"]
    _complete_ari = link_scores.loc["Complete-4", "adjusted_rand"]
    _fmi_chance = chance_scores.loc[
        "Fowlkes-Mallows", "same clusters, labels shuffled"
    ]
    mo.md(f"""
    ## Conclusions

    1. **A dendrogram shows every clustering at once.** Each join is a merge, and its height is the distance between the two groups that merged. Nothing is a cluster until you cut the tree; the cut decides which merges to keep.
    2. **A cut height is not portable between linkages.** At height 4 the tiny cloud gives 4 clusters with ward, 1 with single, 3 with complete, and 3 with average. The joins sit at different heights because the linkage rules differ, so one height crosses a different number of branches in each tree. A ward cut at height 2 gives {tiny_cut} clusters.
    3. **The linkage rule decides the failure mode.** Single linkage merges by the closest pair, so a chain of nearby points bridges groups. On the 400-blob cloud it produced groups of 199, 199, 1, and 1 and scored {_single_ari:.3f}; complete linkage, which uses the farthest pair, produced 102, 100, 100, and 98 and scored {_complete_ari:.3f}. Ward (0.947) and average (0.954) stayed close behind, so the outlier is single linkage, not a single "correct" method. The best adjusted Rand index here is **{best_link}**.
    4. **The four external scores do not share a baseline.** Adjusted Rand and adjusted mutual information are corrected for chance, so random labels score about 0. Normalized mutual information and Fowlkes–Mallows are **not** adjusted: shuffling the labels of a real fit still scores about **{_fmi_chance:.3f}** on Fowlkes–Mallows. Use the adjusted scores to compare different numbers of clusters, and compare Fowlkes–Mallows only between fits with the same k.
    5. **Asking for one cluster too few costs three of the four linkages about 0.25.** Ward falls from 0.947 to 0.685, complete from 0.974 to 0.690, and average from 0.954 to 0.701. Single linkage barely moves, from {_single_ari:.3f} to 0.497, because it had already chained everything: removing a cluster changes nothing when the groups were never separated.
    6. **On iris, one species separates and two do not.** Ward on the standardized measurements scores adjusted Rand index {iris_rand:.3f}, adjusted mutual information {iris_ami:.3f}, normalized mutual information {iris_nmi:.3f}, and Fowlkes–Mallows {iris_fm:.3f}. Setosa is nearly isolated, while versicolor and virginica share clusters:
    {_lines}
       That overlap belongs to the flowers, not to a mistake in the code.
    7. **t-SNE draws; it does not cluster.** The picture is a two-dimensional rendering for the eye. The tree was computed on the four standardized measurements.
    8. **Standardizing is a modelling decision, not a correction.** The four iris measurements share a unit but not a spread, so without standardizing the widest-ranging measurement dominates every distance. Standardizing changes the distances, and here it changes the score too. Which one you want depends on whether you believe all four measurements should count equally.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check your understanding

    1. What does the height of a join in a dendrogram represent?
    2. What does a horizontal cut do, and how do you count the clusters it produces?
    3. How does each of the four linkages measure the distance between two groups?
    4. Why can the same cut height produce different numbers of clusters under different linkages?
    5. Which of the four external scores are corrected for chance, and why does that matter when comparing fits with different numbers of clusters?
    6. Why did single linkage produce clusters of 199, 199, 1, and 1 on the blobs?
    7. Why is the t-SNE picture not the data that was clustered?
    8. What does standardizing the iris measurements change, and why is it a choice rather than a correction?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
1. **The distance between the two groups that merged.** A taller join means the groups were farther apart. In the tiny cloud the first join is at 0.479 and the last at 15.976.
2. **It keeps the merges below that height and discards the ones above.** The number of vertical lines the cut crosses is the number of clusters. A cut at height 2 on the tiny cloud crosses four.
3. **Single: the closest pair.** **Complete: the farthest pair.** **Average: the mean of all pairwise distances.** **Ward: the increase in within-group variance the merge would cause.**
4. **Each linkage places its joins at different heights.** The rules differ, so the trees differ, and one height therefore crosses a different number of branches. At height 4 the tiny cloud gives 4 clusters under ward but 1 under single.
5. **Adjusted Rand and adjusted mutual information are chance-corrected; normalized mutual information and Fowlkes–Mallows are not.** Random labels score about 0 on the adjusted ones, but Fowlkes–Mallows still scores roughly 0.25 to 0.35. That makes Fowlkes–Mallows unsafe for comparing different k, because its floor itself depends on k.
6. **Chaining.** Single linkage merges by the closest pair, so a thin trail of points between two blobs is enough to bridge them. Once bridged they cannot be separated higher in the tree, and the two most isolated points remain as clusters of one.
7. **t-SNE is a two-dimensional rendering.** It was fit once for the drawing, and the tree was computed on the four standardized measurements. The plot shows the data in a reduced space that the clustering never saw.
8. **It puts all four measurements on the same footing.** They share centimetres but not spread, and petal length varies four times as much as sepal width, so without standardizing it dominates every distance. Standardizing changes the distances and therefore the result; it is not a correction of an error, just a decision that each measurement should count equally.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - [AgglomerativeClustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html)
    - [Hierarchical clustering](https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering)
    - [SciPy linkage](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html)
    - [Iris dataset, UCI](https://archive.ics.uci.edu/dataset/53/iris)
    - Müller, A. C., & Guido, S. (2017). *Introduction to Machine Learning with Python*. O'Reilly Media. Chapter 3.
    - VanderPlas, J. (2017). *Python Data Science Handbook*. O'Reilly Media. Chapter 5.
    """)
    return


if __name__ == "__main__":
    app.run()
