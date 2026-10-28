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
    import warnings
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
    from sklearn.cluster import KMeans, SpectralClustering
    from sklearn.datasets import make_circles, make_moons
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import adjusted_rand_score

    warnings.filterwarnings(
        "ignore",
        message=r"JsProxy\.as_object_map\(\) is deprecated.*",
        category=RuntimeWarning,
        module=r"threadpoolctl",
    )

    return (
        KMeans,
        SpectralClustering,
        TfidfVectorizer,
        adjusted_rand_score,
        cv,
        make_circles,
        make_moons,
        mo,
        np,
        pd,
        warnings,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Spectral Clustering

    ## Objectives

    By the end of this lesson, you should be able to:

    1. Explain how a nearest-neighbor graph turns distances between points into connectivity.
    2. Explain why **k-means** cannot follow a moon, a ring, or a spiral, and describe what it does to them instead.
    3. Compare **k-means** and **spectral clustering** using the adjusted Rand index, and explain what that index does and does not tell you.
    4. Explain what `n_neighbors` controls, and what goes wrong when it is too small or too large.
    5. Describe how spectral clustering groups short documents that share vocabulary.

    ## Background

    **k-means** assigns a point to the nearest center and minimizes the squared distance to that center. That works for compact, roughly round groups. A moon, a ring, and a spiral are not round. A straight line between two centers cannot follow a curve, so a point on one arm of a curve can sit closer to the center of the other arm.

    **Spectral clustering** changes the question. Instead of asking *which center is nearest*, it asks *which points are connected to which*. It builds a graph whose vertices are the observations and whose edges join each point to its closest neighbors. Two points that are near each other along a curve are connected, even when a straight line between them would leave the curve.

    `affinity="nearest_neighbors"` selects that graph. Each point is joined to its `n_neighbors` closest neighbors. From the graph, the algorithm computes the graph Laplacian and its eigenvectors. Those eigenvectors are new coordinates in which points that belong to the same connected piece of the graph land near each other, so a simple method such as k-means can then separate them. This is the same idea as the kernel trick used with support vector machines: transform the data into a space where a linear boundary becomes possible. The transformation here follows connectivity rather than a polynomial or radial kernel.

    `n_neighbors` is therefore the most important setting. It controls how much of the data each point can "see".

    - **Too few neighbors** leaves the graph in disconnected fragments. The algorithm can only separate the fragments it finds, so a partition may follow pieces of a curve rather than the whole curve. scikit-learn warns that the spectral embedding may not work as expected.
    - **Too many neighbors** joins points across the shapes. The graph blurs the boundary between curves, and the separation degrades. This failure is often silent: no warning is raised.

    Neither mistake is detected for you. The number has to suit the dataset, so this lesson compares several values.

    `random_state=0` and `n_jobs=1` keep the neighbor-graph fits stable and reproducible. **Cluster labels are names, not ranks**: label 0 in one fit may be label 1 in the next.

    ## Measuring the result: adjusted Rand index

    The adjusted Rand index (ARI) compares a clustering with known groups. Here we know how the synthetic shapes were generated, so we can compare against the true arms and rings. The index counts pairs of points that are grouped together, and corrects for the agreement two random partitions would reach by chance.

    - **1.0** means the two partitions agree exactly.
    - **About 0.0** means the agreement is no better than chance.
    - Negative values are possible and mean worse than chance.

    The index **ignores the names of the clusters**. If the algorithm labels the two moons 0 and 1 and the truth labels them 1 and 0, the score is still 1.0. That is deliberate: clustering does not know, and cannot know, which group should be called 0.

    A high ARI does not prove that a grouping is useful. The true labels used here exist only because we generated the data.

    ## Datasets used

    Every dataset in this lesson is **generated in the notebook** rather than downloaded. Two of them come from scikit-learn's built-in generators, so the code that creates them is part of the lesson.

    ### Half moons

    From `sklearn.datasets.make_moons`, 300 points, noise 0.05, `random_state=10`. It places two interlocking crescents, each a half-circle offset from the other. Originally introduced in the scikit-learn documentation as a case where linear and centroid-based methods fail. [Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html).

    ### Concentric circles

    From `sklearn.datasets.make_circles`, 400 points, noise 0.1, `factor=0.2`, `random_state=123`. A small circle inside a larger ring. The `factor` parameter sets the ratio of the inner radius to the outer one, so 0.2 puts the inner circle well away from the ring. [Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_circles.html).

    ### Two spirals

    Generated by `create_spiral_data`, defined in this notebook. There is no library function for this shape, so the code builds it: points are placed along a spiral angle, then a second arm is created by negating both coordinates. The noise parameter thickens the arms so the two are not trivially separable. Each arm has 300 points.

    ### Short sentences

    **Source.** 26 sentences written for this lesson, about the sky, weather, animals, royalty, water, and Python. They are not from an external corpus, so there are no citations and no licensing questions. Each group of sentences deliberately shares distinctive vocabulary, which is what allows a word-weight representation to separate them. There are no true labels; we read the groups and interpret them.

    Because all four datasets are synthetic or purpose-written, the lesson is reproducible anywhere and free of privacy or licensing concerns. The cost is that none of them shows the messiness of real data. The iris and wine lessons supply that contrast.

    Each scatter below is a `ClusterVisualizer` plot, so you can hover and zoom to inspect individual points.
    """)
    return


@app.cell
def _(np):
    def create_spiral_data(n_points, noise=0.8, random_state=12345):
        """Generate two interleaved spiral arms.

        Parameters
        ----------
        n_points : int
            Number of points on each arm. The returned array has 2 * n_points rows.
        noise : float, default 0.8
            Spread added to each coordinate, which thickens the arms.
        random_state : int, default 12345
            Seed for a local random generator, so repeated calls return the same data
            without changing NumPy's global random state.

        Returns
        -------
        numpy.ndarray of shape (2 * n_points, 2)
            The first n_points rows are one arm and the last n_points rows are the
            other arm. Those two blocks are the labels used by the adjusted Rand index.
        """
        rng = np.random.RandomState(random_state)
        angle = np.sqrt(rng.rand(n_points, 1)) * 2 * np.pi
        arm_x = -np.cos(angle) * angle + rng.rand(n_points, 1) * noise
        arm_y = np.sin(angle) * angle + rng.rand(n_points, 1) * noise
        first_arm = np.hstack((arm_x, arm_y))
        second_arm = np.hstack((-arm_x, -arm_y))
        return np.vstack((first_arm, second_arm))

    return (create_spiral_data,)


@app.cell
def _(KMeans, SpectralClustering, adjusted_rand_score, cv, make_moons, pd):
    moons, moon_arms = make_moons(300, noise=0.05, random_state=10)
    moon_view = cv.ClusterVisualizer(pd.DataFrame(moons, columns=["x", "y"]))
    moon_labels = {}
    moon_rows = []
    for _k in (2, 3, 6):
        moon_labels[f"k-means {_k}"] = KMeans(
            n_clusters=_k, random_state=10, n_init="auto"
        ).fit_predict(moons)
        moon_labels[f"spectral {_k}"] = SpectralClustering(
            n_clusters=_k,
            affinity="nearest_neighbors",
            n_neighbors=25,
            assign_labels="kmeans",
            random_state=0,
            n_jobs=1,
        ).fit_predict(moons)
        moon_rows.append(
            {
                "k": _k,
                "k-means ARI": adjusted_rand_score(
                    moon_arms, moon_labels[f"k-means {_k}"]
                ),
                "spectral ARI": adjusted_rand_score(
                    moon_arms, moon_labels[f"spectral {_k}"]
                ),
            }
        )
    moon_scores = pd.DataFrame(moon_rows).set_index("k")
    moon_view.plot_data(title="Half Moons: two interlocking crescents")
    return moon_labels, moon_scores, moon_view


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Half moons

    The two crescents interlock. No straight line separates them, so any method that compares a point with a center will cut across both arms at once.

    ### k-means

    In the three plots below, look for a cluster that contains points from **both** crescents. With two centers, k-means draws one boundary through the middle of the picture, and each cluster holds part of the upper crescent and part of the lower one. Adding centers does not help: with six centers the extra groups are still cut out of the middle, because every center competes for points across both arms.
    """)
    return


@app.cell
def _(moon_labels, moon_view):
    moon_view.plot_clusters(
        moon_labels["k-means 2"], title="Half Moons: k-Means Clustering (k=2)"
    )
    return


@app.cell
def _(moon_labels, moon_view):
    moon_view.plot_clusters(
        moon_labels["k-means 3"], title="Half Moons: k-Means Clustering (k=3)"
    )
    return


@app.cell
def _(moon_labels, moon_view):
    moon_view.plot_clusters(
        moon_labels["k-means 6"], title="Half Moons: k-Means Clustering (k=6)"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Spectral clustering

    Now the same data with a neighbor graph. Each point keeps its 25 nearest neighbors, which are almost always points on its own crescent, because the crescents are thin and close to their own neighbors.

    With **k = 2**, the two arms come out entire and separate. The adjusted Rand index is 1.0, which means the fitted partition matches the generating arms exactly.
    """)
    return


@app.cell
def _(moon_labels, moon_view):
    moon_view.plot_clusters(
        moon_labels["spectral 2"],
        title="Half Moons: Spectral Clustering (k=2, 25 neighbors)",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What happens when we ask for more clusters

    With **k = 3** and **k = 6**, spectral clustering does not go back to cutting across the arms. Instead it subdivides them: one arm is split into pieces, and the other arm is split into other pieces.

    Read the plots carefully and check two things:

    1. Does any single cluster contain points from both crescents? It should not.
    2. Does the score fall as k rises? It does, because the true answer is two groups, so any subdivision disagrees with the labels even when each piece is pure.

    This is the difference between a method that cannot see the shape and a method that can. A lower score at k = 6 does not mean the six groups are wrong in an absolute sense; it means they are not the two groups we generated.
    """)
    return


@app.cell
def _(moon_labels, moon_view):
    moon_view.plot_clusters(
        moon_labels["spectral 3"],
        title="Half Moons: Spectral Clustering (k=3, 25 neighbors)",
    )
    return


@app.cell
def _(moon_labels, moon_view):
    moon_view.plot_clusters(
        moon_labels["spectral 6"],
        title="Half Moons: Spectral Clustering (k=6, 25 neighbors)",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Comparing the scores

    The table holds the adjusted Rand index for each fit. Both columns use the same generated arms, so the numbers are directly comparable.
    """)
    return


@app.cell
def _(moon_scores):
    moon_scores.round(3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Read the table as three separate questions.

    - **At k = 2**, where the answer is unambiguous, k-means scores about **0.25** and spectral scores **1.00**. k-means agrees with the truth on only about a quarter of the paired comparisons, which is slightly better than chance but far from the two arms.
    - **At k = 3 and k = 6**, k-means stays near **0.26** and **0.28**. Its answer has not improved with more centers, because the problem is not the number of groups.
    - **Spectral falls to 0.75 and then 0.34**, but for a different reason: the partition is still clean, it simply splits each crescent into smaller pieces. A score below 1 does not always mean a bad clustering. It can mean a finer clustering of the same shape.

    ### Try it yourself

    On the half moons, which method recovers both crescents when we ask for **k = 2**? Work out your answer, then reveal it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
**Spectral clustering.** At k = 2, k-means scores about 0.25 because one boundary cuts through both crescents. Spectral clustering with 25 neighbors scores 1.00, because each cluster holds exactly one crescent.

The cluster numbers are arbitrary, so the index compares groupings, not names.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    The table is stored in `moon_scores`. Replace `None` so that `student_moon_gap` holds the difference between the spectral and the k-means score at **k = 2**, that is spectral minus k-means.
    """)
    return


@app.cell
def _(moon_scores):
    student_moon_gap = None
    student_moon_gap
    return (student_moon_gap,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
```python
student_moon_gap = (
    moon_scores.loc[2, "spectral ARI"] - moon_scores.loc[2, "k-means ARI"]
)
```
The value is about **0.746**, since 1.000 − 0.254 = 0.746. `loc[2, ...]` selects the row labelled 2 and then one column by name. The gap is large because the neighbor graph follows each crescent, while k-means can only compare distances to a center.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Concentric circles

    A small circle sits inside a larger ring. The two groups are not close in any direction a straight line would use: the inner circle is *surrounded*, so a cut between the two centers runs through both groups. Distance to a center cannot express "inside", but connectivity can, because the inner points are connected to each other and not to the outer ring.

    ### k-means first

    k-means splits the picture with one straight boundary and mixes both rings.
    """)
    return


@app.cell
def _(KMeans, SpectralClustering, adjusted_rand_score, cv, make_circles, pd, warnings):
    circle_points, circle_rings = make_circles(
        n_samples=400, random_state=123, noise=0.1, factor=0.2
    )
    circle_view = cv.ClusterVisualizer(pd.DataFrame(circle_points, columns=["x", "y"]))
    circle_kmeans = KMeans(n_clusters=2, random_state=10, n_init="auto").fit_predict(
        circle_points
    )

    def _fit_circles(n_neighbors):
        """Fit spectral clustering on the circles and capture any warning.

        Parameters
        ----------
        n_neighbors : int
            Number of neighbors each point keeps in the affinity graph.

        Returns
        -------
        tuple
            The labels and a list of warning messages raised during the fit.
        """
        with warnings.catch_warnings(record=True) as _caught:
            warnings.simplefilter("always")
            _labels = SpectralClustering(
                n_clusters=2,
                affinity="nearest_neighbors",
                n_neighbors=n_neighbors,
                assign_labels="kmeans",
                random_state=0,
                n_jobs=1,
            ).fit_predict(circle_points)
        _messages = [str(_warning.message) for _warning in _caught]
        return _labels, _messages

    circle_spectral_sparse, circle_sparse_warnings = _fit_circles(4)
    circle_spectral_sound, circle_sound_warnings = _fit_circles(25)

    circle_scores = pd.DataFrame(
        [
            {
                "Fit": "k-means, k=2",
                "n_neighbors": "not used",
                "ARI": adjusted_rand_score(circle_rings, circle_kmeans),
                "Cluster sizes": str(sorted(pd.Series(circle_kmeans).value_counts().tolist(), reverse=True)),
                "Warning": "none",
            },
            {
                "Fit": "spectral, too few neighbors",
                "n_neighbors": "4",
                "ARI": adjusted_rand_score(circle_rings, circle_spectral_sparse),
                "Cluster sizes": str(sorted(pd.Series(circle_spectral_sparse).value_counts().tolist(), reverse=True)),
                "Warning": "; ".join(circle_sparse_warnings) if circle_sparse_warnings else "none",
            },
            {
                "Fit": "spectral, suitable value",
                "n_neighbors": "25",
                "ARI": adjusted_rand_score(circle_rings, circle_spectral_sound),
                "Cluster sizes": str(sorted(pd.Series(circle_spectral_sound).value_counts().tolist(), reverse=True)),
                "Warning": "; ".join(circle_sound_warnings) if circle_sound_warnings else "none",
            },
        ]
    ).set_index("Fit")
    circle_view.plot_data(title="Concentric Circles: a ring around a circle")
    return (
        circle_kmeans,
        circle_points,
        circle_rings,
        circle_scores,
        circle_spectral_sound,
        circle_view,
        circle_sparse_warnings,
    )


@app.cell
def _(circle_kmeans, circle_view):
    circle_view.plot_clusters(
        circle_kmeans, title="Concentric Circles: k-Means Clustering (k=2)"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Too few neighbors: `n_neighbors = 4`

    With only four neighbors per point, the graph breaks into many small fragments. scikit-learn detects that the graph is not fully connected and warns you.
    """)
    return


@app.cell
def _(circle_sparse_warnings, mo):
    if circle_sparse_warnings:
        mo.callout(
            mo.md(
                "scikit-learn warned: "
                + " ".join(f"`{message}`" for message in circle_sparse_warnings)
            ),
            kind="warn",
        )
    else:
        mo.md("No warning was raised.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The warning is accurate about the graph, but it is **not** a verdict on the clustering. A disconnected graph can still happen to produce a good partition, and a connected graph can produce a bad one. Read the warning as "the representation you built is fragmented", then check the result.

    In this fit the fragmentation does show: 400 points land in two very unequal groups of 351 and 49, and the adjusted Rand index is about **0.06**. The boundary follows the fragments, not the two rings.
    """)
    return


@app.cell
def _(circle_spectral_sparse, circle_view):
    circle_view.plot_clusters(
        circle_spectral_sparse,
        title="Concentric Circles: Spectral, 4 neighbors (graph fragmented)",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A suitable value: `n_neighbors = 25`

    With 25 neighbors each point is connected to enough of its own ring for the graph to hold together, and still not to the other ring. The fit is then exact: two groups of 200 points, and the adjusted Rand index is **1.00**.
    """)
    return


@app.cell
def _(circle_spectral_sound, circle_view):
    circle_view.plot_clusters(
        circle_spectral_sound,
        title="Concentric Circles: Spectral, 25 neighbors (k=2)",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Comparing the three fits

    The table collects the fits. The cluster sizes and the warning are the evidence, and the adjusted Rand index summarises the agreement with the generating rings.
    """)
    return


@app.cell
def _(circle_scores):
    mo.Html(circle_scores.round(3).to_html())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Three points to take away from this section.

    1. **k-means is not close.** Its index is about **0.05**, so it agrees with the true rings only slightly more than chance does.
    2. **A warning is a symptom, not a score.** At 4 neighbors the graph is fragmented and the result is poor. Between 10 and 20 neighbors the same warning still appears, yet the partition is already perfect. The warning tells you the graph is disconnected; it cannot tell you whether the answer is good.
    3. **`n_neighbors = 25` is a choice, not a law.** It suits these datasets because they are dense and noiseless enough for 25 to be both sufficient and safe. A different dataset can need a different value, and the next section shows what happens when the value is too large.

    ### Try it yourself

    The scores are in `circle_scores`, which is indexed by the `Fit` name. Replace `None` so that `student_rings` holds the adjusted Rand index of the fit that used **25 neighbors**.
    """)
    return


@app.cell
def _(circle_scores):
    student_rings = None
    student_rings
    return (student_rings,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
```python
student_rings = circle_scores.loc["spectral, suitable value", "ARI"]
```
The value is **1.0**. `loc` takes the row first and then the column, both by name. The same row shows two groups of 200 points and no warning, which is why this value suits the data: the graph stays connected without reaching across the gap between the rings.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    The fits above are done for you. Now fit one yourself. Replace each `None` so that `student_circle_model` is a `SpectralClustering` model for **2 clusters** on `circle_points`, using `affinity="nearest_neighbors"`, `n_neighbors=25`, `assign_labels="kmeans"`, `random_state=0`, and `n_jobs=1`. Then set `student_circle_score` to its adjusted Rand index against `circle_rings`.
    """)
    return


@app.cell
def _(SpectralClustering, adjusted_rand_score, circle_points, circle_rings):
    student_circle_model = None
    student_circle_score = None
    print("Model: ", student_circle_model)
    print("ARI:   ", student_circle_score)    
    return student_circle_model, student_circle_score


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
```python
student_circle_model = SpectralClustering(
    n_clusters=2,
    affinity="nearest_neighbors",
    n_neighbors=25,
    assign_labels="kmeans",
    random_state=0,
    n_jobs=1,
)
student_circle_score = adjusted_rand_score(
    circle_rings, student_circle_model.fit_predict(circle_points)
)
```
The score is **1.0**, matching the fit in the table. The five settings each do a job: `n_clusters` is how many groups you request, `affinity` chooses the neighbor graph, `n_neighbors` decides how far each point can "see", `assign_labels` selects the final step on the eigenvectors, and `random_state` keeps that step reproducible.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Spirals

    Two arms wind around each other. Points at the same radius on different arms are close in space, while points far apart along the same arm are distant. Distance therefore tells you almost nothing about which arm a point belongs to.

    `create_spiral_data` stacks one arm and then the other, and returns 600 rows. The first 300 rows are one arm and the next 300 are the other, so `np.repeat([0, 1], 300)` recovers the true arms. Those blocks are the labels used by the adjusted Rand index. The algorithm itself sees only the coordinates.

    ### k-means

    k-means divides the plane with a straight boundary. Because each arm wraps around the centre, both sides of that boundary contain parts of both arms. Each cluster mixes the two spirals.
    """)
    return


@app.cell
def _(
    KMeans,
    SpectralClustering,
    adjusted_rand_score,
    create_spiral_data,
    cv,
    np,
    pd,
):
    spirals = create_spiral_data(300)
    spiral_arms = np.repeat([0, 1], 300)
    spiral_view = cv.ClusterVisualizer(pd.DataFrame(spirals, columns=["x", "y"]))
    spiral_labels = {}
    spiral_rows = []
    for _k in (2, 3):
        spiral_labels[f"k-means {_k}"] = KMeans(
            n_clusters=_k, random_state=0, n_init="auto"
        ).fit_predict(spirals)
        spiral_labels[f"spectral {_k}"] = SpectralClustering(
            n_clusters=_k,
            affinity="nearest_neighbors",
            n_neighbors=25,
            assign_labels="kmeans",
            random_state=0,
            n_jobs=1,
        ).fit_predict(spirals)
        spiral_rows.append(
            {
                "k": _k,
                "k-means ARI": adjusted_rand_score(
                    spiral_arms, spiral_labels[f"k-means {_k}"]
                ),
                "spectral ARI": adjusted_rand_score(
                    spiral_arms, spiral_labels[f"spectral {_k}"]
                ),
            }
        )
    spiral_scores = pd.DataFrame(spiral_rows).set_index("k")
    spiral_view.plot_data(title="Spiral-Shaped Data: two interleaved arms")
    return spiral_labels, spiral_scores, spiral_view


@app.cell
def _(spiral_labels, spiral_view):
    spiral_view.plot_clusters(
        spiral_labels["k-means 2"],
        title="Spiral-Shaped Data: k-Means Clustering (k=2)",
    )
    return


@app.cell
def _(spiral_labels, spiral_view):
    spiral_view.plot_clusters(
        spiral_labels["k-means 3"],
        title="Spiral-Shaped Data: k-Means Clustering (k=3)",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Each cluster holds pieces of both arms. That is not a tuning problem, and adding a third centre does not fix it: the arms are defined by **continuity**, and a centre cannot represent a curve that bends back on itself.

    ### Spectral clustering

    With a neighbor graph of 25 neighbors, each point connects to nearby points along its own arm. The eigenvectors of that graph behave like a coordinate that "unrolls" the spiral: points that were close along an arm end up near each other, and the two arms separate. A straight cut in the new coordinates then corresponds to a curved boundary in the original picture.
    """)
    return


@app.cell
def _(spiral_labels, spiral_view):
    spiral_view.plot_clusters(
        spiral_labels["spectral 2"],
        title="Spiral-Shaped Data: Spectral Clustering (k=2, 25 neighbors)",
    )
    return


@app.cell
def _(spiral_labels, spiral_view):
    spiral_view.plot_clusters(
        spiral_labels["spectral 3"],
        title="Spiral-Shaped Data: Spectral Clustering (k=3, 25 neighbors)",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    At **k = 2** the two arms come out whole and the adjusted Rand index is **1.00**. At **k = 3** the extra cluster splits one arm, so the index drops to about **0.34** even though no cluster mixes the two arms.

    ### Comparing the scores
    """)
    return


@app.cell
def _(spiral_scores):
    mo.Html(spiral_scores.round(3).to_html())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    k-means stays near **0.02** and **0.07**: essentially no agreement with the true arms. Spectral reaches **1.00** at k = 2 and **0.34** at k = 3.

    ### The other edge: too many neighbors

    This section also shows why `n_neighbors` cannot simply be set as high as possible. Raising it from 25 to 40 makes the graph reach across the gap between the arms. The two spirals then merge, and the adjusted Rand index falls to about **0.09** with **no warning at all**. Too few neighbors warns you; too many does not. That silence is why the value has to be checked against the result rather than trusted.

    ### Try it yourself

    On the spirals, which method follows the arms at **k = 2** instead of cutting across them? Work out your answer, then reveal it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
**Spectral clustering.** k-means can only compare a point with a centre, so its boundary is a straight line and each cluster holds parts of both arms; its index is about 0.02. Spectral clustering with 25 neighbors follows connectivity along each arm and scores 1.00.

Asking for three clusters splits an arm rather than separating the arms, so the index falls to about 0.34.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Why can a **larger** `n_neighbors` make the spiral result worse, even though more neighbors mean more connections? Work out your answer, then reveal it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
The graph is the model. With 40 neighbors, some points on one arm are joined to points on the other, so the graph blurs the gap between the arms. The eigenvectors then no longer separate them, and the adjusted Rand index falls from 1.00 to about 0.09.

Unlike the too-small case, this failure raises no warning, because a densely connected graph is a perfectly valid graph. The representation is wrong, not broken.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Sentences

    So far every point had coordinates. Documents do not. `TfidfVectorizer` turns each sentence into numbers: one weight per word, where a word scores higher when it is frequent in a sentence but rare across the collection. Common English words are dropped, because they would make every sentence look similar.

    The result is a 26-row matrix with one column per remaining word (47 here). A row is a point in a high-dimensional space, and spectral clustering can build a neighbor graph from it exactly as it did for the shapes. Two sentences are neighbors when they use similar words.

    The cluster numbers are arbitrary here too. There are no true labels, so there is no adjusted Rand index to compute. We read the groups and interpret them.

    Because 26 sentences are far fewer than 300 points, they need **fewer** neighbors: `n_neighbors=10`. A large value would connect almost every sentence to every other and blur the topics.
    """)
    return


@app.cell
def _(SpectralClustering, TfidfVectorizer, np, pd):
    def spectral_clustering_docs(X, n_clusters=2, n_neighbors=10):
        """Cluster a document matrix with spectral clustering.

        Parameters
        ----------
        X : array-like of shape (n_documents, n_terms)
            Numeric representation of the documents, for example a TF-IDF matrix.
        n_clusters : int, default 2
            Number of clusters to request.
        n_neighbors : int, default 10
            Number of neighbors each document keeps in the affinity graph. Use a
            smaller value than you would for a few thousand points, because a small
            collection otherwise becomes one dense, connected graph.

        Returns
        -------
        numpy.ndarray of shape (n_documents,)
            Cluster label for each document. The labels are names, not ranks.
        """
        return SpectralClustering(
            n_clusters=n_clusters,
            affinity="nearest_neighbors",
            n_neighbors=n_neighbors,
            assign_labels="kmeans",
            random_state=0,
            n_jobs=1,
        ).fit_predict(X)


    def document_subject(sentence):
        """Return the topic of a sentence, used only to describe the clusters.

        Parameters
        ----------
        sentence : str
            One of the sentences in `documents`.

        Returns
        -------
        str
            One of "water", "sky", "code", "animals", "royalty", or "other".
        """
        text = sentence.lower()
        if "water" in text or "hydrat" in text:
            return "water"
        if "sky" in text or "sun" in text:
            return "sky"
        if "python" in text or "machine" in text or "intelligence" in text:
            return "code"
        if "fox" in text or "dog" in text:
            return "animals"
        if "crown" in text or "king" in text or "gold" in text or "diadem" in text:
            return "royalty"
        return "other"


    documents = [
        "The sky is blue and beautiful",
        "Love this blue and beautiful sky!",
        "The quick brown fox jumps over the lazy dog.",
        "A king's crown is made of gold",
        "The crown of a king is called a diadem",
        "The bright sun in the sky is shining",
        "The sun is bright and beautiful today",
        "The sun in the sky is very bright",
        "We can see the shining sun, the bright sun",
        "The fox jumped over the dog",
        "The fox is quicker than the lazy dog",
        "The dog is lazy but the brown fox is quick!",
        "Gold crowns are worn by kings",
        "The king's diadem is made of gold",
        "The water is clear and the sun is bright",
        "Drink water to stay hydrated",
        "Stay hydrated and drink water",
        "Python is a programming language",
        "Python language is used for machine learning",
        "Machine learning can be done with Python",
        "Artificial Intelligence is a branch of machine learning",
        "Learning Python is fun",
        "Python is easy to learn and powerful",
        "Machine learning is a fascinating field",
        "You need clear water to stay hydrated",
        "Stay healthy by drinking clear water",
    ]

    document_matrix = TfidfVectorizer(stop_words="english").fit_transform(documents).toarray()

    document_summary = pd.DataFrame(
        {
            "Items": [
                "documents",
                "TF-IDF word columns",
                "TF-IDF matrix shape",
                "subjects in the collection",
            ],
            "Value": [
                str(len(documents)),
                str(document_matrix.shape[1]),
                f"{document_matrix.shape[0]} rows x {document_matrix.shape[1]} columns",
                "water, sky, code, animals, royalty",
            ],
        }
    )
    return (
        document_matrix,
        document_subject,
        document_summary,
        documents,
        spectral_clustering_docs,
    )


@app.cell
def _(document_summary):
    document_summary
    return


@app.cell
def _(document_matrix, document_subject, documents, pd, spectral_clustering_docs):
    def summarise_clusters(n_clusters):
        """Describe each cluster by size and the topics it contains.

        Parameters
        ----------
        n_clusters : int
            Number of clusters requested from `spectral_clustering_docs`.

        Returns
        -------
        tuple
            The cluster labels, and a DataFrame with one row per cluster showing its
            size and the topics present.
        """
        labels = spectral_clustering_docs(document_matrix, n_clusters=n_clusters)
        rows = []
        for cluster in range(n_clusters):
            members = [i for i, label in enumerate(labels) if label == cluster]
            subjects = sorted({document_subject(documents[i]) for i in members})
            rows.append(
                {
                    "Cluster": cluster,
                    "Documents": len(members),
                    "Topics": ", ".join(subjects),
                    "Mixed": "yes" if len(subjects) > 1 else "no",
                }
            )
        return labels, pd.DataFrame(rows).set_index("Cluster")

    document_labels_3, document_clusters_3 = summarise_clusters(3)
    document_labels_4, document_clusters_4 = summarise_clusters(4)
    document_labels_5, document_clusters_5 = summarise_clusters(5)
    return (
        document_clusters_3,
        document_clusters_4,
        document_clusters_5,
        document_labels_3,
        document_labels_4,
        document_labels_5,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Asking for three clusters

    Three clusters is too few for five topics. Watch for a group that holds several subjects at once.
    """)
    return


@app.cell
def _(document_clusters_3):
    mo.Html(document_clusters_3.to_html())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The largest group holds 15 sentences and mixes **sky**, **water**, and **royalty**. With only three clusters, topics that share vocabulary get merged. Sky and water share words such as "sun" and "bright", and royalty separates easily because its words are distinctive.

    ### Asking for four clusters
    """)
    return


@app.cell
def _(document_clusters_4):
    mo.Html(document_clusters_4.to_html())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now royalty and animals have their own groups, and code has its own. The remaining group of 11 still mixes **sky** with **water**, because one sentence combines both: "The water is clear and the sun is bright."

    ### Asking for five clusters
    """)
    return


@app.cell
def _(document_clusters_5):
    mo.Html(document_clusters_5.to_html())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    At five clusters, sky and water finally separate, and the collection is close to its five subjects. Two details keep this honest:

    - One group of 7 mixes **royalty** with **sky**. A single sentence that borrows another topic's words can bridge two groups.
    - **Cluster numbers are arbitrary.** "Cluster 2" is not "the second best" group, and the numbering can change between runs.

    More clusters is not automatically better. Each extra group has to be interpreted, and small groups can be artifacts of one unusual sentence. Choosing a value here depends on what the reader finds useful, which is why the adjusted Rand index cannot be applied: there are no true labels, only our reading of the sentences.

    ### Try it yourself

    Replace `None` so that `student_doc_counts` holds the number of sentences in each cluster of the **four-cluster** fit, ordered by cluster number. `document_labels_4` holds those labels.
    """)
    return


@app.cell
def _(document_labels_4, pd):
    student_doc_counts = None
    student_doc_counts
    return (student_doc_counts,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
```python
student_doc_counts = pd.Series(document_labels_4).value_counts().sort_index()
```
The counts are **11, 4, 7, and 4**. `value_counts()` counts how many times each label appears, and `sort_index()` puts them in cluster order instead of size order. The four counts add to 26.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    The sentence collection is small: only 26 documents. Why does a **smaller** `n_neighbors` suit it, compared with the 300-point shape datasets? Work out your answer, then reveal it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
With 26 documents, 25 neighbors would connect almost every sentence to almost every other, and the topics would merge into one dense graph. Ten neighbors keeps each sentence tied to the few sentences that share its vocabulary.

The setting follows the size and density of the data. On the 300-point shapes the same reasoning gives a larger number, which is why the shape sections use 25 and this section uses 10.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(
    circle_scores,
    document_clusters_5,
    mo,
    moon_scores,
    spiral_scores,
):
    mo.md(f"""
    ## Conclusions

    1. **A neighbor graph turns distances into connectivity.** Spectral clustering does not compare a point with a center. It joins each point to its `n_neighbors` closest neighbors and separates the pieces of that graph. That is why it can follow a curve: points along an arm are connected even when a straight line between them leaves the arm.
    2. **k-means cannot follow these shapes.** Its index stays near **{moon_scores.loc[2, "k-means ARI"]:.3f}** on the moons, **{circle_scores.loc["k-means, k=2", "ARI"]:.3f}** on the rings, and **{spiral_scores.loc[2, "k-means ARI"]:.3f}** on the spirals. Adding centers does not rescue it, because the problem is the shape of the boundary, not the number of groups.
    3. **Spectral clustering recovers all three at 25 neighbors**, with an index of **{moon_scores.loc[2, "spectral ARI"]:.3f}** on the moons, **{circle_scores.loc["spectral, suitable value", "ARI"]:.3f}** on the rings, and **{spiral_scores.loc[2, "spectral ARI"]:.3f}** on the spirals.
    4. **It is not universally better, and `n_neighbors` runs both ways.** At 4 neighbors the rings fragment and the index drops to **{circle_scores.loc["spectral, too few neighbors", "ARI"]:.3f}**, with a warning. Too many neighbors is worse in a different way: the spirals fall to about 0.09 at 40 neighbors, with **no warning at all**. The setting must suit the data, and a good result is not evidence that the graph was sound.
    5. **Raising k subdivides rather than mixes.** On the moons, spectral scores {moon_scores.loc[3, "spectral ARI"]:.3f} at k = 3 and {moon_scores.loc[6, "spectral ARI"]:.3f} at k = 6 while never putting two crescents in one cluster. A lower index can mean a finer partition of the same shape, not a worse one.
    6. **The index ignores names.** It compares pairs of points that share a cluster, so it is unaffected by which group is called 0.
    7. **Documents with no coordinates.** `TfidfVectorizer` supplies the numbers, and the graph then joins sentences that share vocabulary. At five clusters the sky and water topics separate; one group still mixes royalty with sky. Cluster sizes at five clusters: {", ".join(str(int(value)) for value in document_clusters_5["Documents"])}. There are no true labels, so the index does not apply, and choosing the number of groups depends on what the reader finds useful.

    A last caution: the true labels used throughout exist only because we generated the data. In a real analysis, a high score would not prove that the groups correspond to anything meaningful.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check your understanding

    1. Why does k-means mix the two moons, even when we ask for more clusters?
    2. What does `affinity="nearest_neighbors"` build, and what does `n_neighbors` control?
    3. What does the "graph is not fully connected" warning tell you, and what does it not tell you?
    4. Why can the adjusted Rand index be 1 when the cluster numbers differ from the true arm numbers?
    5. Why does a larger `n_neighbors` make the spiral result worse, and why is that failure silent?
    6. On the moons, spectral clustering at k = 6 scores lower than at k = 2. Does that mean the six-cluster solution is bad?
    7. Why does the sentence collection need a smaller `n_neighbors` than the shape datasets?
    8. Can the adjusted Rand index be used to judge the sentence clustering?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
1. **It only compares distance to a center.** The crescents interlock, so a straight boundary through the middle puts points from both arms in one cluster. Adding centers subdivides the middle of the picture rather than following the arms.
2. **A graph of nearest neighbors.** Each observation becomes a vertex joined to its `n_neighbors` closest neighbors. `n_neighbors` sets how many connections each point keeps, so it decides whether the graph follows the shape or blurs it.
3. **The graph is fragmented.** It means the affinity graph has more than one connected component, so the embedding may not behave as expected. It does **not** tell you whether the resulting partition is good: between 10 and 20 neighbors the warning still fires while the rings are recovered perfectly. Always check the result.
4. **It compares pairs, not names.** A pair of points on the same arm should share a cluster regardless of whether that cluster is called 0 or 1. Relabelling does not change the score.
5. **The graph reaches across the gap.** With 40 neighbors, points on one arm connect to points on the other, so the eigenvectors no longer separate the arms and the index falls to about 0.09. No warning appears, because a densely connected graph is a valid graph, not an error.
6. **No.** The six clusters are still pure; each arm is simply split into pieces. The index compares the result with the two true arms, so a finer partition of the same shape scores lower. A low index can reflect a different question, not a worse answer.
7. **There are only 26 documents.** A large `n_neighbors` would connect almost every pair of sentences and merge the topics. Ten neighbors keeps each sentence tied to the few that share its vocabulary.
8. **No.** The index needs known true labels, and the sentences have none. The clusters are assessed by reading them and judging whether the topics are coherent.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - [SpectralClustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html)
    - [Spectral clustering, scikit-learn user guide](https://scikit-learn.org/stable/modules/clustering.html#spectral-clustering)
    - Müller, A. C., & Guido, S. (2017). *Introduction to Machine Learning with Python*. O'Reilly Media. Chapter 3.
    - VanderPlas, J. (2017). *Python Data Science Handbook*. O'Reilly Media. Chapter 5.
    """)
    return


if __name__ == "__main__":
    app.run()
