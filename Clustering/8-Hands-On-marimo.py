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
    from sklearn.cluster import (
        AffinityPropagation,
        AgglomerativeClustering,
        KMeans,
        MeanShift,
        SpectralClustering,
        estimate_bandwidth,
    )
    from sklearn.metrics import (
        calinski_harabasz_score,
        davies_bouldin_score,
        silhouette_score,
    )
    from sklearn.preprocessing import StandardScaler

    pio.templates.default = "plotly_white"
    px.defaults.color_discrete_sequence = px.colors.qualitative.G10

    return (
        AffinityPropagation,
        AgglomerativeClustering,
        KMeans,
        MeanShift,
        Path,
        SpectralClustering,
        StandardScaler,
        calinski_harabasz_score,
        cv,
        davies_bouldin_score,
        estimate_bandwidth,
        mo,
        np,
        pd,
        px,
        silhouette_score,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Clustering a Restaurant Bill

    ## Objectives

    By the end of this lesson, you should be able to:

    1. Apply five clustering algorithms to the same two columns and explain how they differ.
    2. Explain why those two columns must be standardized before any distance is computed.
    3. Read three internal validation scores and name which direction each prefers.
    4. Explain how a validation score can be won by a cluster containing only two points.
    5. Remove a small group, re-fit, and explain why the z-scores must **not** be recomputed.
    6. Compare cluster membership against attributes that were never used in the distance.

    ## Background

    This is a hands-on lesson rather than a new algorithm. Every method here has had its own lesson earlier in the course, so the work now is to run them side by side on one dataset and see where they disagree.

    ### The data

    The `tips` table records one restaurant party per row: the bill, the tip, and some attributes of the visit. Only **`total_bill`** and **`tip`** enter the distance. The other five columns describe the party and are used later, only to characterise the groups.

    That distinction matters. A clustering that used `sex` or `day` would be grouping by those attributes; a clustering that ignores them can still be **compared** against them afterwards. The lesson does the second.

    ### Why standardize

    `total_bill` ranges from about 3 to 51 dollars, while `tip` ranges from 1 to 10. On the raw scale, a difference of 20 dollars in the bill outweighs the largest possible difference in the tip. Euclidean distance would be, in practice, almost entirely a bill difference.

    Standardizing subtracts each column's mean and divides by its standard deviation, so both columns end up on the same footing. Every row is used, because there is no holdout here: this is exploratory, not predictive.

    ### The algorithms

    Five methods, each with a different idea of what a cluster is:

    | Method | How it decides groups | You supply |
    | :--- | :--- | :--- |
    | **k-means** | nearest centre, minimizing within-cluster variance | the number of clusters |
    | **spectral** | connectivity in a neighbour graph | the number of clusters |
    | **agglomerative** | successive merges of the closest groups | the number of clusters |
    | **affinity propagation** | points elect exemplars by passing messages | a `preference` |
    | **mean shift** | windows climb to density peaks | a `bandwidth` |

    k-means, spectral, and agglomerative need `k` up front, so two values are tried for each: 2 and 3. Affinity propagation and mean shift decide the count themselves.

    **Mean shift uses an estimated bandwidth**, computed with quantile 0.3 and `random_state=0`, rather than a guessed value. That is deliberate: in an earlier lesson we saw that "the default bandwidth is 1" is wrong, and that a hand-picked radius can silently fragment or merge groups.

    ### The three scores

    All three are **internal**: they judge a clustering from the feature values alone, without needing true labels. This dataset has no true groups, so internal scores are the only option.

    | Score | Prefer | What it measures |
    | :--- | :--- | :--- |
    | Silhouette | **higher** | how much closer each point is to its own cluster than to the nearest other one |
    | Davies–Bouldin | **lower** | how similar each cluster is to its most similar neighbour |
    | Calinski–Harabasz | **higher** | between-cluster spread relative to within-cluster spread |

    They are not three versions of the same measurement, and they frequently name different winners. That disagreement is not a bug in the scores; it is the main lesson of this notebook.

    One more caution, learned the hard way earlier in the course: **a good score can be produced by a group of two points.** If two parties sit far from everything else, separating them makes the remaining groups look tight and well separated, and silhouette rewards that. The lesson below walks into exactly that case on purpose.

    ## Datasets used

    ### Tips

    **Source.** The `tips` dataset, collected by **Bryant and Smith** and distributed in the R package `reshape2`. It records 244 restaurant parties: the total bill, the tip, and attributes of the visit. Documentation is at the [R datasets reference](https://vincentarelbundock.github.io/Rdatasets/doc/reshape2/tips.html).

    **What it holds.** 244 rows and 7 columns:

    | Column | Meaning |
    | :--- | :--- |
    | `total_bill` | bill in dollars, including tax |
    | `tip` | tip in dollars |
    | `sex` | sex of the person paying |
    | `smoker` | whether the party included smokers |
    | `day` | day of the week |
    | `time` | Lunch or Dinner |
    | `size` | number of people in the party |

    There are **no missing values**. A copy, `tips.csv`, sits beside this notebook.

    **Two cautions.** First, this is observational data from one restaurant, so nothing here generalises to other places or times. Second, `tip` and `total_bill` are strongly related by construction: people tip roughly in proportion to what they spend. Clustering them together mostly recovers "cheap meals versus expensive meals", which is worth knowing before interpreting any group as a behavioural type.

    Every scatter below is a `ClusterVisualizer` plot, so you can hover and zoom.
    """)
    return


@app.cell
def _(Path, pd):
    # Tips: 244 restaurant parties. Collected by Bryant & Smith, distributed in
    # the R package reshape2. Documentation:
    # https://vincentarelbundock.github.io/Rdatasets/doc/reshape2/tips.html
    # Columns: total_bill, tip, sex, smoker, day, time, size. No missing values.
    # Save tips.csv beside this notebook; in browser Marimo, upload it to the notebook's files.
    tips = pd.read_csv(Path(__file__).resolve().parent / "tips.csv", index_col="id")
    return (tips,)


@app.cell
def _(mo, tips):
    mo.Html(tips.head().to_html())
    return


@app.cell
def _(np, pd, calinski_harabasz_score, davies_bouldin_score, silhouette_score):
    def score_labels(frame, label_map):
        """Score several clusterings on the same feature frame.

        Parameters
        ----------
        frame : pandas.DataFrame
            The columns the models were fitted on. Scores are computed in this
            space, so pass the same representation the model saw.
        label_map : dict
            Maps a fit name to its label array. Insertion order is preserved, so
            the bars appear in the order the fits were listed.

        Returns
        -------
        pandas.DataFrame
            One row per fit, indexed by name, with columns `silhouette`,
            `davies_bouldin`, and `calinski_harabasz`. A fit with fewer than two
            distinct labels has all three left as NaN, because none of the scores
            is defined for a single cluster.
        """
        rows = []
        for name, labels in label_map.items():
            labels = np.asarray(labels)
            if len(set(labels.tolist())) < 2:
                rows.append(
                    {
                        "fit": name,
                        "silhouette": np.nan,
                        "davies_bouldin": np.nan,
                        "calinski_harabasz": np.nan,
                    }
                )
                continue
            rows.append(
                {
                    "fit": name,
                    "silhouette": silhouette_score(frame, labels),
                    "davies_bouldin": davies_bouldin_score(frame, labels),
                    "calinski_harabasz": calinski_harabasz_score(frame, labels),
                }
            )
        return pd.DataFrame(rows).set_index("fit")

    return (score_labels,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Standardizing the two money columns

    Before any distance is computed, look at the two columns on their original scale. The box plots below show why standardization is not optional here.
    """)
    return


@app.cell
def _(pd, px, tips):
    _money = tips[["total_bill", "tip"]]
    _long = _money.melt(var_name="column", value_name="dollars")
    _fig = px.box(
        _long,
        x="column",
        y="dollars",
        color="column",
        title="Bill and tip in original dollars",
    )
    _fig.update_layout(width=600, height=400, showlegend=False)
    _fig
    return


@app.cell
def _(StandardScaler, pd, px, tips):
    money = tips[["total_bill", "tip"]]
    scaled = pd.DataFrame(
        StandardScaler().fit_transform(money),
        columns=money.columns,
        index=money.index,
    )
    _long = scaled.melt(var_name="column", value_name="standard deviations")
    _fig = px.box(
        _long,
        x="column",
        y="standard deviations",
        color="column",
        title="The same two columns, standardized",
    )
    _fig.update_layout(width=600, height=400, showlegend=False)
    _fig
    return money, scaled


@app.cell(hide_code=True)
def _(mo, money):
    mo.md(f"""
    Read the two plots together.

    - **Original scale.** The bill box sits far above the tip box and is much taller: `total_bill` runs from ${money.total_bill.min():.2f} to ${money.total_bill.max():.2f} with a standard deviation of ${money.total_bill.std():.2f}, while `tip` runs from ${money.tip.min():.2f} to ${money.tip.max():.2f} with a standard deviation of ${money.tip.std():.2f}. On this scale, one bill difference can be twenty times the entire range of tips.
    - **Standardized.** Both columns are now centred near 0 with a standard deviation of 1, so neither dominates the distance.

    Without this step, every algorithm in the lesson would effectively cluster on the bill alone and the tip would contribute almost nothing.
    """)
    return


@app.cell(hide_code=True)
def _(mo, scaled, tips):
    mo.md(f"""
    The standardized table has **{len(scaled)} rows** and two columns, one per money column. Every party is used: there is no train/test split, because this is exploratory analysis rather than prediction.

    The `ClusterVisualizer` below is given the **standardized** table. That matters for the scores that follow: an internal score measures distances in whatever space it is handed, so scoring a fit on the raw dollars would give a different, and misleading, answer.
    """)
    return


@app.cell
def _(cv, scaled):
    party_view = cv.ClusterVisualizer(scaled)
    party_view.plot_data(title="Bill and tip, standardized")
    return (party_view,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Replace `None` so that `student_ranges` holds the range of each money column in the original dollars, as a `pd.Series` or dict. Then set `student_ratio` to how many times larger the bill range is than the tip range.
    """)
    return


@app.cell
def _(money):
    student_ranges = None
    student_ratio = None
    print("ranges:\n", student_ranges)
    print(f"\nbill range / tip range = {student_ratio:.2f}")
    return student_ranges, student_ratio


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
```python
student_ranges = money.max() - money.min()
student_ratio = student_ranges["total_bill"] / student_ranges["tip"]
```
The bill spans about **48 dollars** and the tip about **9**, so the bill range is roughly **5.3 times** the tip range.

`.max() - .min()` gives the range of each column at once, because pandas applies the operation column-wise. The ratio is the concrete reason standardization is needed: the bill is not a little wider, it is several times wider, so it would dominate any Euclidean distance.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Eight fits on the same two columns

    Three methods that need `k` are run at 2 and 3; the two that decide the count themselves are run once each. Every fit sees the identical standardized table.

    The bandwidth for mean shift is estimated rather than guessed, with quantile 0.3 and a fixed `random_state`.
    """)
    return


@app.cell
def _(
    AffinityPropagation,
    AgglomerativeClustering,
    KMeans,
    MeanShift,
    SpectralClustering,
    estimate_bandwidth,
    scaled,
    score_labels,
):
    first_bandwidth = float(
        estimate_bandwidth(scaled, quantile=0.3, n_samples=len(scaled), random_state=0)
    )
    labels = {
        "k-means 2": KMeans(
            n_clusters=2, n_init="auto", random_state=123
        ).fit_predict(scaled),
        "k-means 3": KMeans(
            n_clusters=3, n_init="auto", random_state=123
        ).fit_predict(scaled),
        "spectral 2": SpectralClustering(
            n_clusters=2, assign_labels="kmeans", random_state=123, n_jobs=1
        ).fit_predict(scaled),
        "spectral 3": SpectralClustering(
            n_clusters=3, assign_labels="kmeans", random_state=123, n_jobs=1
        ).fit_predict(scaled),
        "agglomerative 2": AgglomerativeClustering(n_clusters=2).fit_predict(scaled),
        "agglomerative 3": AgglomerativeClustering(n_clusters=3).fit_predict(scaled),
        "affinity": AffinityPropagation(
            random_state=42, preference=-45, max_iter=500
        ).fit_predict(scaled),
        "mean shift": MeanShift(bandwidth=first_bandwidth).fit_predict(scaled),
    }
    first_scores = score_labels(scaled, labels)
    first_silhouette = first_scores["silhouette"].idxmax()
    first_davies = first_scores["davies_bouldin"].idxmin()
    first_calinski = first_scores["calinski_harabasz"].idxmax()
    first_counts = {_name: len(set(_l.tolist())) for _name, _l in labels.items()}
    return (
        first_bandwidth,
        first_calinski,
        first_counts,
        first_davies,
        first_scores,
        first_silhouette,
        labels,
    )


@app.cell(hide_code=True)
def _(first_bandwidth, first_counts, mo):
    mo.md(f"""
    The estimated bandwidth is **{first_bandwidth:.3f}**. Mean shift produced **{first_counts['mean shift']} clusters**, and affinity propagation produced **{first_counts['affinity']}**.

    Note those two numbers before looking at the plots. Neither method was told how many groups to find, and they did not agree with each other.

    ### k-means, two and three clusters
    """)
    return


@app.cell
def _(labels, party_view):
    party_view.plot_clusters(labels["k-means 2"], title="k-means, k = 2")
    return


@app.cell
def _(labels, party_view):
    party_view.plot_clusters(labels["k-means 3"], title="k-means, k = 3")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    With two clusters, k-means draws one boundary across the cloud. With three, it adds a second boundary and subdivides one side. Both partitions are valid answers to the question *"split this cloud into k compact groups"*; neither is more correct than the other, because there are no true groups to match.

    ### Spectral clustering
    """)
    return


@app.cell
def _(labels, party_view):
    party_view.plot_clusters(labels["spectral 2"], title="spectral, k = 2")
    return


@app.cell
def _(labels, party_view):
    party_view.plot_clusters(labels["spectral 3"], title="spectral, k = 3")
    return


@app.cell(hide_code=True)
def _(labels, mo, np):
    _sizes = np.bincount(labels["spectral 2"])
    mo.md(f"""
    **Look closely at the two-cluster spectral plot.** The cluster sizes are **{_sizes[0]} and {_sizes[1]}**: one group holds almost every party and the other holds **two points**.

    That is not a plotting artefact. Spectral clustering builds a neighbour graph and cuts it, and if two points are isolated enough to form their own connected piece, cutting the graph separates them cleanly. The algorithm did exactly what it was asked.

    Whether that is a *useful* answer is a different question, and it is the one the scores will answer badly. Hold that thought.

    ### Agglomerative clustering
    """)
    return


@app.cell
def _(labels, party_view):
    party_view.plot_clusters(labels["agglomerative 2"], title="agglomerative, k = 2")
    return


@app.cell
def _(labels, party_view):
    party_view.plot_clusters(labels["agglomerative 3"], title="agglomerative, k = 3")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Agglomerative clustering merges upward from individual points, so its boundaries look slightly different from k-means even at the same `k`. Both satisfy the same request, and both are defensible.

    ### Affinity propagation and mean shift
    """)
    return


@app.cell
def _(labels, party_view):
    party_view.plot_clusters(labels["affinity"], title="affinity propagation")
    return


@app.cell
def _(labels, party_view):
    party_view.plot_clusters(labels["mean shift"], title="mean shift")
    return


@app.cell(hide_code=True)
def _(first_counts, mo):
    mo.md(f"""
    These two chose their own counts: **{first_counts['affinity']}** groups for affinity propagation and **{first_counts['mean shift']}** for mean shift.

    Neither number is wrong. Both methods found the number of exemplars or density peaks that their criterion supports on this cloud. The fact that they differ from each other, and from the `k` we imposed on the other three methods, is the honest state of unsupervised learning on data with no true groups.

    ## 3. Scoring the eight fits

    The scores are computed on the standardized table, which is what every model saw.
    """)
    return


@app.cell
def _(mo, first_scores):
    mo.Html(first_scores.round(3).to_html())
    return


@app.cell(hide_code=True)
def _(first_davies, first_scores, first_silhouette, mo):
    mo.md(f"""
    Read the winners, then read the table again:

    - **Silhouette prefers `{first_silhouette}`** at {first_scores.loc[first_silhouette, 'silhouette']:.3f}.
    - **Davies–Bouldin prefers `{first_davies}`** at {first_scores.loc[first_davies, 'davies_bouldin']:.3f}.
    - **Calinski–Harabasz prefers a different fit again** — see the bar charts below.

    Two of the three scores agree, which is unusual and, in this case, suspicious. The fit they agree on is the one that isolated **two points**. Before trusting that agreement, look at what the winning fit actually contains.

    ### The same scores as bars
    """)
    return


@app.cell
def _(labels, party_view, pd):
    party_view.plot_silhouette(pd.DataFrame(labels), title="Silhouette, all 244 parties")
    return


@app.cell
def _(labels, party_view, pd):
    party_view.plot_davies_bouldin(
        pd.DataFrame(labels), title="Davies–Bouldin, all 244 parties"
    )
    return


@app.cell
def _(labels, party_view, pd):
    party_view.plot_calinski_harabasz(
        pd.DataFrame(labels), title="Calinski–Harabasz, all 244 parties"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    The scores disagree. Explain in one sentence why three internal scores can rank the same set of clusterings differently. Work out your answer, then reveal it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
Because **each score measures a different property of the partition**.

- Silhouette compares, for every point, its distance to its own cluster against its distance to the nearest other cluster. It rewards points being much closer to their own group.
- Davies–Bouldin compares each cluster's internal spread against its distance from its most similar neighbour, and penalises clusters that look alike.
- Calinski–Harabasz is a ratio of between-cluster to within-cluster dispersion, adjusted for the numbers of clusters and points, so it is sensitive to the overall spread rather than to individual points.

A partition can be excellent on one of those properties and mediocre on another. Nothing is broken; the scores answer different questions, and picking one is a judgement about what you care about.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. What is inside the winning group?

    The two-cluster spectral fit has a group of two points, and two of the three scores preferred that fit. So look at those two parties, in the original dollars, with every attribute that the clustering never saw.
    """)
    return


@app.cell
def _(labels, pd, scaled):
    spectral_labels = labels["spectral 2"]
    small_label = pd.Series(spectral_labels).value_counts().idxmin()
    small_index = scaled.index[spectral_labels == small_label]
    small_count = len(small_index)
    scaled_kept = scaled.drop(index=small_index)
    return scaled_kept, small_count, small_index, small_label


@app.cell(hide_code=True)
def _(mo, small_count, small_label, small_index, tips):
    _parties = tips.loc[small_index]
    mo.md(f"""
    The smallest spectral group is **label {small_label}** with **{small_count} parties**, at rows {", ".join(str(_i) for _i in small_index)}.

    In the original dollars they are:
    """)
    return


@app.cell
def _(mo, small_index, tips):
    mo.Html(tips.loc[small_index].to_html())
    return


@app.cell(hide_code=True)
def _(mo, small_count, small_index, tips):
    _p = tips.loc[small_index]
    mo.md(f"""
    Both parties spent far more than everyone else:

    - Bills of **${_p.total_bill.min():.2f}** and **${_p.total_bill.max():.2f}**, against a median bill of ${tips.total_bill.median():.2f}.
    - Tips of **${_p.tip.min():.2f}** and **${_p.tip.max():.2f}**, against a median tip of ${tips.tip.median():.2f}.

    Both were male, on a Saturday, at dinner, with 3 and 4 people. Those attributes are printed for context; **they played no part in the distance**. The two parties were separated because of how much money they spent, not because of who they were.

    ### Why the score liked this

    This is the pattern worth remembering. When two points sit far from everything else, separating them out makes the remaining group tighter. Silhouette measures how much closer points are to their own cluster than to the nearest other one, so pulling two outliers away raises the average for everyone left behind.

    Nothing about that is a calculation error. It is a score doing what it measures: rewarding separation. The problem is that a **two-point "cluster" is not a group**, it is two observations that happen to be far away. A number cannot tell the difference, but you can, by looking at the sizes.

    The practical rule: **always read the cluster sizes beside the score.**

    ### Try it yourself

    Replace `None` so that `student_small_sizes` holds the size of every cluster in the two-cluster spectral fit, as a `pd.Series` of counts.
    """)
    return


@app.cell
def _(labels, pd):
    student_small_sizes = None
    print(student_small_sizes)
    return (student_small_sizes,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
```python
student_small_sizes = pd.Series(labels["spectral 2"]).value_counts()
```
The sizes are **242 and 2**.

`value_counts()` counts how many rows carry each label; by default the largest comes first. Wrapping the label array in `pd.Series` is what gives you that method, since a NumPy array does not have it.

Seeing a 2 next to a 242 is the fastest way to notice this kind of result. The silhouette score alone would not have told you.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Removing those parties and re-fitting

    Set the two extreme parties aside and run every algorithm again on the remaining rows.

    **One thing must not change: the scaler.** The second round reuses the mean and standard deviation computed from all 244 rows. Removing rows and then recomputing the z-scores would silently reshuffle every value, so the second round would not be comparable with the first.
    """)
    return


@app.cell
def _(
    AffinityPropagation,
    AgglomerativeClustering,
    KMeans,
    MeanShift,
    SpectralClustering,
    cv,
    estimate_bandwidth,
    scaled_kept,
    score_labels,
):
    second_bandwidth = float(
        estimate_bandwidth(
            scaled_kept, quantile=0.3, n_samples=len(scaled_kept), random_state=0
        )
    )
    second_labels = {
        "k-means 2": KMeans(
            n_clusters=2, n_init="auto", random_state=123
        ).fit_predict(scaled_kept),
        "k-means 3": KMeans(
            n_clusters=3, n_init="auto", random_state=123
        ).fit_predict(scaled_kept),
        "spectral 2": SpectralClustering(
            n_clusters=2, assign_labels="kmeans", random_state=123, n_jobs=1
        ).fit_predict(scaled_kept),
        "spectral 3": SpectralClustering(
            n_clusters=3, assign_labels="kmeans", random_state=123, n_jobs=1
        ).fit_predict(scaled_kept),
        "agglomerative 2": AgglomerativeClustering(n_clusters=2).fit_predict(
            scaled_kept
        ),
        "agglomerative 3": AgglomerativeClustering(n_clusters=3).fit_predict(
            scaled_kept
        ),
        "affinity": AffinityPropagation(
            random_state=42, preference=-45, max_iter=500
        ).fit_predict(scaled_kept),
        "mean shift": MeanShift(bandwidth=second_bandwidth).fit_predict(scaled_kept),
    }
    second_scores = score_labels(scaled_kept, second_labels)
    second_silhouette = second_scores["silhouette"].idxmax()
    second_davies = second_scores["davies_bouldin"].idxmin()
    second_calinski = second_scores["calinski_harabasz"].idxmax()
    second_counts = {_n: len(set(_l.tolist())) for _n, _l in second_labels.items()}
    second_view = cv.ClusterVisualizer(scaled_kept)
    return (
        second_bandwidth,
        second_calinski,
        second_counts,
        second_davies,
        second_labels,
        second_scores,
        second_silhouette,
        second_view,
    )


@app.cell
def _(second_view):
    second_view.plot_data(title="After setting aside the two extreme parties")
    return


@app.cell(hide_code=True)
def _(mo, scaled_kept):
    mo.md(f"""
    **{len(scaled_kept)} parties remain.** The two extreme bills are gone, so the cloud is now dominated by ordinary meals.

    ### The scores after removal
    """)
    return


@app.cell
def _(mo, second_scores):
    mo.Html(second_scores.round(3).to_html())
    return


@app.cell(hide_code=True)
def _(first_counts, mo, second_bandwidth, second_counts, second_scores):
    mo.md(f"""
    Compare this with the first table, and notice what changed and what did not.

    - **The estimated bandwidth moved** to {second_bandwidth:.3f}, because the two extreme points no longer influence the nearest-neighbour distances.
    - **Mean shift now finds {second_counts['mean shift']} clusters**, down from {first_counts['mean shift']} in the first round.
    - **Silhouette dropped** from around 0.71 to around 0.55. Removing the outliers made the fit *less* separable by that measure, which confirms the two points had been doing the heavy lifting.
    - **The winners are now three different fits.** Before removal, silhouette and Davies–Bouldin both chose the spectral two-cluster fit; afterwards they no longer agree.

    That last point is the real lesson. Before removal, two scores agreed on a result that was an artefact of two points. After removal, nothing agrees, because there is no genuine two-group or three-group structure in this cloud to find.

    ### The second-round plots
    """)
    return


@app.cell
def _(second_labels, second_view):
    second_view.plot_clusters(second_labels["k-means 2"], title="k-means, k = 2, second round")
    return


@app.cell
def _(second_labels, second_view):
    second_view.plot_clusters(second_labels["spectral 2"], title="spectral, k = 2, second round")
    return


@app.cell
def _(second_labels, second_view):
    second_view.plot_clusters(second_labels["mean shift"], title="mean shift, second round")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The three plots look much more alike now. With the two extreme parties gone, the remaining cloud is a single elongated blob, and every method slices it in roughly the same place.

    That is a useful diagnostic: **when different algorithms agree, the structure is probably real. When they disagree wildly, either the data has genuine complexity or you are looking at artefacts.**
    """)
    return


@app.cell
def _(second_labels, second_view, pd):
    second_view.plot_silhouette(
        pd.DataFrame(second_labels), title="Silhouette, second round"
    )
    return


@app.cell
def _(second_labels, second_view, pd):
    second_view.plot_davies_bouldin(
        pd.DataFrame(second_labels), title="Davies–Bouldin, second round"
    )
    return


@app.cell
def _(second_labels, second_view, pd):
    second_view.plot_calinski_harabasz(
        pd.DataFrame(second_labels), title="Calinski–Harabasz, second round"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    The second round keeps the first round's scaler instead of recomputing it on the remaining rows. Explain why. Work out your answer, then reveal it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
Because **the two rounds must be measured on the same scale to be comparable.**

The first round standardized all 244 rows, so every value is expressed as "standard deviations from the mean of the full dataset". If you removed two rows and recomputed the mean and standard deviation, every remaining value would shift slightly. The same party would then have a different z-score in the second round than in the first, and any change in the scores would be a mix of two effects: the removal, and the rescaling.

Keeping the original scaler isolates the effect you actually wanted to study. The removed parties still sit inside the original mean and standard deviation, so the scale is stable.

There is a second, subtler reason. Using the full-data statistics is what you would do in a real pipeline where a model is fitted once and then applied to new data. Recomputing the scaler whenever the input changes is how leakage creeps in.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Do the clusters differ in anything besides money?

    The clustering used only `total_bill` and `tip`. The other five columns were ignored. Now compare them against the mean-shift groups from the second round, which is the fit with two balanced clusters.

    Each table shows the **percentage within a cluster**, so a cluster of 26 and a cluster of 216 can be compared fairly. Raw counts would mostly reflect the size difference.
    """)
    return


@app.cell
def _(cluster_index, second_labels):
    cluster_index = second_labels["mean shift"]
    return (cluster_index,)@app.cell
def _(cluster_index, mo, pd, scaled_kept, tips):
    _kept = tips.loc[scaled_kept.index]
    _lab = pd.Series(cluster_index, index=scaled_kept.index, name="cluster")
    sex_table = pd.crosstab(_lab, _kept["sex"], normalize="index").round(3) * 100
    mo.Html(sex_table.to_html())
    return


@app.cell
def _(cluster_index, mo, pd, scaled_kept, tips):
    _kept = tips.loc[scaled_kept.index]
    _lab = pd.Series(cluster_index, index=scaled_kept.index, name="cluster")
    smoker_table = pd.crosstab(_lab, _kept["smoker"], normalize="index").round(3) * 100
    mo.Html(smoker_table.to_html())
    return


@app.cell
def _(cluster_index, mo, pd, scaled_kept, tips):
    _kept = tips.loc[scaled_kept.index]
    _lab = pd.Series(cluster_index, index=scaled_kept.index, name="cluster")
    day_table = pd.crosstab(_lab, _kept["day"], normalize="index").round(3) * 100
    mo.Html(day_table.to_html())
    return


@app.cell
def _(cluster_index, mo, pd, scaled_kept, tips):
    _kept = tips.loc[scaled_kept.index]
    _lab = pd.Series(cluster_index, index=scaled_kept.index, name="cluster")
    time_table = pd.crosstab(_lab, _kept["time"], normalize="index").round(3) * 100
    mo.Html(time_table.to_html())
    return


@app.cell(hide_code=True)
def _(cluster_index, mo, np, scaled_kept, tips):
    _kept = tips.loc[scaled_kept.index]
    _sizes = np.bincount(cluster_index)
    _means = _kept.assign(cluster=cluster_index).groupby("cluster")[
        ["total_bill", "tip", "size"]
    ].mean()
    _lines = "\n".join(
        f"    - **Cluster {_c}** ({_sizes[_c]} parties): mean bill ${_r.total_bill:.2f}, "
        f"mean tip ${_r.tip:.2f}, mean party size {_r['size']:.1f}"
        for _c, _r in _means.iterrows()
    )
    mo.md(f"""
    Read the four tables for what separates the clusters, and what does not.

    - **Sex is nearly identical** in both groups: roughly 64% male and 36% female in each. The clustering did not find anything about sex.
    - **Smoking differs somewhat**: one cluster is 39% smokers, the other 27%. A real difference, though modest.
    - **Day and time differ only slightly.** Both clusters are mostly Saturday and Sunday, mostly dinner.

    So the honest conclusion is that the two groups are **not** separated by who the customers were. They are separated by what they spent, which is precisely the two columns the algorithms were given:

    {_lines}

    The second cluster spends roughly twice as much per party, which is what a clustering of `total_bill` and `tip` was always going to find. Its parties are also slightly larger, which explains part of the gap: more people means a bigger bill.

    **Caveat.** These are descriptive percentages from one restaurant. Turning "39% versus 27% smokers" into a claim about tipping behaviour would need a statistical test, and would still be confounded by the fact that larger parties spend more. This lesson stops at description.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Which of the three internal scores is the one you would prefer to be **small**? Replace `None` with its name as scikit-learn spells it.
    """)
    return


@app.cell
def _():
    student_small_score = None
    print(student_small_score)
    return (student_small_score,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
```python
student_small_score = "davies_bouldin"
```
**Davies–Bouldin** is the one to minimise, because it measures how similar each cluster is to its most similar neighbour. A small value means the clusters look different from each other.

Silhouette and Calinski–Harabasz are the other way round: larger is better. In the tables above, that is why the winners are found with `idxmax` for those two and `idxmin` for Davies–Bouldin.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(
    first_bandwidth,
    first_counts,
    first_davies,
    first_scores,
    first_silhouette,
    mo,
    scaled,
    second_bandwidth,
    second_calinski,
    second_counts,
    second_davies,
    second_scores,
    second_silhouette,
    small_count,
    tips,
):
    mo.md(f"""
    ## Conclusions

    1. **Two columns, standardized.** Only `total_bill` and `tip` entered the distance, and both were standardized first. On the raw scale the bill spans about 48 dollars against the tip's 9, so without scaling the bill would have dominated every algorithm.
    2. **Five methods, eight fits.** k-means, spectral, and agglomerative were each run at `k = 2` and `k = 3`; affinity propagation found {first_counts['affinity']} groups and mean shift found {first_counts['mean shift']} on their own. The disagreement in cluster count is normal on data with no true groups.
    3. **Mean shift used a computed bandwidth.** The estimate was {first_bandwidth:.3f}, not a guessed 1. Guessing would have been indefensible in a lesson that shows how sensitive the result is to that radius.
    4. **A score can be won by a two-point group.** The two-cluster spectral fit isolated **{small_count} parties** with the largest bills in the dataset. Silhouette preferred that fit at {first_scores.loc[first_silhouette, 'silhouette']:.3f} and Davies–Bouldin also preferred it at {first_scores.loc[first_davies, 'davies_bouldin']:.3f}, because pulling two outliers away makes the remaining group tighter. Neither score knows that two points are not a group. **Always read the cluster sizes beside the score.**
    5. **Removing the extremes changed the conclusion.** On the remaining {len(scaled) - small_count} parties the estimated bandwidth became {second_bandwidth:.3f} and silhouette fell to {second_scores.loc[second_silhouette, 'silhouette']:.3f}. The high first-round score had been mostly an artefact of two rows.
    6. **The scaler must not be recomputed.** The second round kept the first round's mean and standard deviation, so the two rounds are measured on one scale and the removal is the only thing that changed.
    7. **After removal, the three scores named three different winners**: silhouette `{second_silhouette}`, Davies–Bouldin `{second_davies}`, Calinski–Harabasz `{second_calinski}`. No fit is best on all three, and no fit is obviously right, because the cloud is one elongated blob rather than several groups.
    8. **The clusters differ in spending, not in who the customers were.** Sex proportions are nearly equal across clusters, and day and time differ only slightly. A modest difference appears for smoking. The clusters differ plainly in bill and tip, which are the columns the algorithms were given.
    9. **Interpretation stops at description.** One restaurant, 244 parties, no true labels, and internal scores only. Nothing here supports a claim about tipping behaviour in general.

    The single most transferable habit from this lesson: when a validation score looks excellent, check the group sizes before believing it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check your understanding

    1. Which two columns enter the clustering, and what happens to the other five?
    2. Why must those two columns be standardized, and what would happen if they were not?
    3. Which of the three scores is better when it is **small**?
    4. Why can three internal scores name three different winners on the same fits?
    5. The two-cluster spectral fit scored well. What was wrong with it as a clustering of this data?
    6. Why does the second round keep the first round's scaler instead of recomputing it?
    7. After the two extreme parties were removed, did the scores agree more or less than before? What does that tell you?
    8. The clusters differ in bill and tip. Do they also differ in sex, day, or time?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
1. **`total_bill` and `tip`.** The other five (`sex`, `smoker`, `day`, `time`, `size`) are kept in the table and used afterwards only to describe the groups; they never enter the distance.
2. **Because they are on very different scales.** The bill spans about 48 dollars and the tip about 9, so Euclidean distance would almost entirely reflect bill differences. Standardizing puts both on the footing of one standard deviation, so each contributes comparably.
3. **Davies–Bouldin**, which measures how similar each cluster is to its most similar neighbour. Silhouette and Calinski–Harabasz are the other way round.
4. **Because they measure different properties.** Silhouette compares each point's own cluster against the nearest other cluster; Davies–Bouldin compares cluster spread against neighbouring clusters; Calinski–Harabasz is a ratio of between- to within-cluster dispersion. A partition can be strong on one and weak on another.
5. **One of its two clusters contained only two points**, the two largest bills in the dataset. Separating two outliers makes the remaining group tighter, which raises silhouette. A two-point cluster is not a group, and the score cannot tell the difference.
6. **So the two rounds are comparable.** Recomputing the mean and standard deviation after removing rows would shift every remaining z-score, mixing the effect of removal with the effect of rescaling. Keeping the original scaler isolates the removal.
7. **Less.** Before removal two scores agreed on the spectral fit; afterwards all three named different fits. That is evidence the first-round agreement was an artefact of the two extreme rows rather than genuine structure.
8. **No, not meaningfully.** Sex proportions are nearly identical across clusters, and day and time differ only slightly; smoking shows a modest difference. The clear difference is in bill and tip, which are the columns the algorithms saw.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - [Clustering, scikit-learn user guide](https://scikit-learn.org/stable/modules/clustering.html)
    - [Silhouette analysis](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html)
    - [Tips dataset documentation](https://vincentarelbundock.github.io/Rdatasets/doc/reshape2/tips.html)
    - Bryant, P. G., and Smith, M. A. (1995). *Practical Data Analysis: Case Studies in Business Statistics*. Irwin.
    - Müller, A. C., and Guido, S. (2017). *Introduction to Machine Learning with Python*. O'Reilly Media. Chapter 3.
    - VanderPlas, J. (2017). *Python Data Science Handbook*. O'Reilly Media. Chapter 5.
    """)
    return


if __name__ == "__main__":
    app.run()
