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
    from sklearn.cluster import AffinityPropagation
    from sklearn.datasets import load_wine, make_blobs, make_moons
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.manifold import TSNE
    from sklearn.metrics import (
        adjusted_mutual_info_score,
        adjusted_rand_score,
        fowlkes_mallows_score,
        normalized_mutual_info_score,
    )
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import StandardScaler

    pio.templates.default = "plotly_white"
    px.defaults.color_discrete_sequence = px.colors.qualitative.G10

    return (
        AffinityPropagation,
        StandardScaler,
        TSNE,
        TfidfVectorizer,
        adjusted_mutual_info_score,
        adjusted_rand_score,
        cosine_similarity,
        cv,
        fowlkes_mallows_score,
        load_wine,
        make_blobs,
        make_moons,
        mo,
        normalized_mutual_info_score,
        np,
        pd,
        px,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Affinity Propagation

    ## Objectives

    By the end of this lesson, you should be able to:

    1. Explain what an **exemplar** is and why it is always one of the observed points.
    2. Explain why affinity propagation does not take an `n_clusters` argument, and how `preference` controls the number of groups instead.
    3. Predict what happens to the number of exemplars as `preference` becomes more negative, and verify it on two datasets.
    4. Explain why affinity propagation can miss the two moons even after the number of clusters looks plausible.
    5. Score the wine clusters against the three cultivars, and explain which of the four external scores are corrected for chance.

    ## Background

    **Affinity propagation** is a clustering method that decides the number of groups for you. Most clustering methods, including k-means and hierarchical clustering, need you to say how many groups you want. Affinity propagation instead asks the data to elect representatives.

    ### Exemplars

    An **exemplar** is an actual observation that is chosen to represent a group. Every other point is assigned to one exemplar, and the exemplar is the "centre" of its group.

    This is the first important difference from k-means. A k-means centre is the **mean** of its points, which is usually a location where no observation sits. An exemplar is always a real row of your data: one of the flowers, one of the wines, or one of the sentences. You can point at it and name it. That property is why the method suits problems where a representative example is more useful than an average.

    ### Message passing

    The algorithm works by passing two kinds of messages between points, over and over, until they settle.

    - A **responsibility** message travels from a point to a candidate exemplar. It says how strongly this point prefers that candidate over all the others.
    - An **availability** message travels the other way. It says how suitable the candidate is as an exemplar, given how many points are already competing for it.

    The two messages are combined into **affinities**. A point that is a good exemplar for many others accumulates high affinity, and it elects itself. Points that are never the best choice for anyone end up assigned to somebody else. When the messages stop changing, the points that elected themselves are the exemplars, and everybody else joins the exemplar they prefer.

    Nothing in that process counts clusters. The number of groups emerges from which points win the election.

    ### Preference

    If the number of groups is not supplied, something has to control how many points are willing to stand for election. That role belongs to **`preference`**.

    - `preference` is the willingness of each point to become an exemplar. In scikit-learn it is also the initial value of the availability message.
    - A **higher** (less negative, or positive) preference makes more points willing, so more exemplars survive, so you get **more** clusters.
    - A **more negative** preference makes fewer points willing, so **fewer** clusters survive. Push it far enough and everything collapses into two groups, then one.

    The default in scikit-learn is the median of the input similarities. That default is a reasonable starting point, but on most data it produces far more clusters than you want, which is why the examples below compare it with explicit values.

    Reading the effect of `preference` is the main practical skill in this lesson. It is a dial for the number of groups, not a measure of quality.

    ### When affinity propagation struggles

    The messages are built from **distances or similarities**. That has two consequences worth knowing before you look at the plots.

    - **It can miss shapes that bend.** Just like k-means, the method prefers data where a group is a compact blob. If two crescents curve toward each other, their points can be close in distance even though they belong to different arms, and the messages have no way to know the difference. Lowering the preference reduces the number of exemplars; it does not teach the algorithm about shape.
    - **The preference sweep is brittle.** The number of clusters can change abruptly as `preference` moves, and there is no automatic rule for choosing it. You compare a few values and read the result.

    ### Similarity for text

    Documents have no coordinates, so a distance cannot be computed directly. This lesson uses `TfidfVectorizer` to turn each sentence into a vector of word weights, then **cosine similarity** to compare them. Cosine similarity measures the angle between two vectors, which ignores their length: a short sentence and a long sentence that use the same vocabulary score as similar.

    Because the similarity matrix is computed first, the fit is called with `affinity="precomputed"`. That tells affinity propagation to treat the square matrix you pass in as the similarity between every pair of points, rather than computing similarities from coordinates itself.

    ## Measuring the result

    The wine data come with three known cultivar labels, so the clusters can be compared against them. This lesson uses four **external** scores. They do not share a baseline.

    | Score | Chance corrected? | What it measures |
    | :--- | :--- | :--- |
    | Adjusted Rand | yes | Agreement over pairs of points. Random labels score about 0. |
    | Adjusted Mutual Info | yes | Shared information, corrected for what random labels would share. About 0 at chance. |
    | Normalized Mutual Info | **no** | Shared information, scaled to 0 to 1. Its chance value is small but not 0. |
    | Fowlkes–Mallows | **no** | Geometric mean of pairwise precision and recall. Its chance value is **well above 0**. |

    Only the two adjusted scores are built so that meaningless labels score about zero. **Fowlkes–Mallows is the one to be careful with**: a meaningless clustering still scores around 0.3, so a Fowlkes–Mallows value of 0.75 is less impressive than it looks. Compare it only between fits that asked for the same number of clusters, or read it alongside Adjusted Rand.

    Affinity propagation does not accept `n_clusters`, but you can still compare runs that happened to produce different numbers of clusters. The adjusted scores make that comparison fair; Fowlkes–Mallows does not.

    **Cluster numbers are names, not ranks.** Cluster 0 is not "the first" or "the best" group, and the numbering can change between runs.

    ## Datasets used

    - **Blobs.** 50 points around three centres, `cluster_std=1.5`, `random_state=10`. Small and well separated, so the method should succeed easily.
    - **Half moons.** 100 points, noise 0.08, `random_state=10`. Two interlocking crescents, which is the shape that distance-based methods struggle with.
    - **Seven short sentences** about nature and artificial intelligence. No coordinates, so a similarity matrix has to be built.
    - **Wine.** Thirteen chemical measurements of 178 wines from three Italian cultivars, loaded from scikit-learn. Real data, with correlated columns and overlapping groups.

    Every scatter below is a `ClusterVisualizer` plot, so you can hover and zoom. `random_state=0` keeps the fits reproducible.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Blobs: the method working as intended

    Fifty points, three well-separated blobs. This is the easy case: the groups are compact, so distance is a fair description of similarity.

    Notice what we do **not** supply. There is no `n_clusters=3`. The call below passes only `random_state` and `max_iter`.
    """)
    return


@app.cell
def _(AffinityPropagation, cv, make_blobs, pd):
    blobs, blob_truth = make_blobs(
        n_samples=50, centers=3, cluster_std=1.5, random_state=10
    )
    blob_frame = pd.DataFrame(blobs, columns=["x", "y"])
    blob_view = cv.ClusterVisualizer(blob_frame, blob_truth)
    blob_model = AffinityPropagation(random_state=0, max_iter=500).fit(blobs)
    blob_count = len(blob_model.cluster_centers_indices_)
    blob_view.plot_data(title="Fifty points around three centres")
    return blob_count, blob_frame, blob_model, blob_view


@app.cell
def _(blob_model, blob_view, blob_count):
    blob_view.plot_clusters(
        blob_model.labels_,
        blob_model.cluster_centers_,
        title=f"Affinity propagation found {blob_count} exemplars on its own",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reading the result

    The black markers are the exemplars. Three things are worth checking here.

    - **The count came out at three**, exactly the number of blobs that were generated, and nobody asked for three.
    - **The black markers sit on top of actual points**, not between the blobs. Compare this with the k-means lesson, where the centres floated in open space. You can see this because each marker lands on a plotted dot.
    - **The adjusted Rand index is 1.0**, so the recovered partition matches the generated groups exactly.

    The next cell confirms the third point, and shows which points were elected.
    """)
    return


@app.cell
def _(adjusted_rand_score, blob_count, blob_model, blob_truth, np):
    blob_sizes = np.bincount(blob_model.labels_)
    blob_summary = (
        f"{blob_count} exemplars, sizes {', '.join(str(int(s)) for s in blob_sizes)}, "
        f"adjusted Rand {adjusted_rand_score(blob_truth, blob_model.labels_):.3f}"
    )
    blob_summary
    return (blob_summary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    The fitted model is `blob_model`. Replace `None` so that `student_exemplars` holds the **row indices** of the points that were elected as exemplars.
    """)
    return


@app.cell
def _(blob_model):
    student_exemplars = None
    print(student_exemplars)
    return (student_exemplars,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
```python
student_exemplars = blob_model.cluster_centers_indices_
```
`cluster_centers_indices_` gives the positions in the original data of the points that became exemplars. For this fit it holds three row numbers, and the cluster centres are exactly those rows, which is why the black markers sit on top of dots.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Why is an exemplar always one of the observed points, while a k-means centre usually is not? Work out your answer, then reveal it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
Because affinity propagation **elects one of the existing points** to represent each group. The messages only ever travel between real observations, and a point wins the election for itself. No new location is ever computed.

K-means instead computes the **mean** of the points in each cluster and moves its centre there. The mean of a set of points is usually a position where no observation sits, which is why k-means centres often appear in empty space between two groups.

The practical consequence is that an exemplar can be named and inspected. A k-means centre is a synthetic average.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Half moons: the number of clusters is not the whole story

    The moons are the shape from the k-means and spectral lessons: two crescents that interlock. Distance is a poor guide here, because points on opposite arms can be neighbours in space.

    With the **default preference** the method finds eight exemplars. That is more groups than the two crescents, so the next plots ask two separate questions:

    1. Does lowering the preference reduce the count to two?
    2. If it does, does it then find the two crescents?
    """)
    return


@app.cell
def _(AffinityPropagation, cv, make_moons, pd):
    moons, moon_truth = make_moons(100, noise=0.08, random_state=10)
    moon_frame = pd.DataFrame(moons, columns=["x", "y"])
    moon_view = cv.ClusterVisualizer(moon_frame, moon_truth)
    moon_default = AffinityPropagation(random_state=0, max_iter=500).fit(moons)
    moon_low = AffinityPropagation(
        preference=-50, random_state=0, max_iter=500
    ).fit(moons)
    moon_default_count = len(moon_default.cluster_centers_indices_)
    moon_low_count = len(moon_low.cluster_centers_indices_)
    moon_view.plot_data(title="Two interlocking crescents")
    return (
        moon_default,
        moon_default_count,
        moon_low,
        moon_low_count,
        moon_view,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The default preference

    Eight exemplars for two visible crescents. Look for the same crescent being represented by several different colours: that is subdivision, not separation.
    """)
    return


@app.cell
def _(moon_default, moon_default_count, moon_view):
    moon_view.plot_clusters(
        moon_default.labels_,
        moon_default.cluster_centers_,
        title=f"Default preference: {moon_default_count} exemplars",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Lowering the preference to -50

    Two exemplars survive. The count now matches the number of crescents, which is encouraging, but look at **which points share a colour** before drawing any conclusion.
    """)
    return


@app.cell
def _(moon_low, moon_low_count, moon_view):
    moon_view.plot_clusters(
        moon_low.labels_,
        moon_low.cluster_centers_,
        title=f"preference = -50: {moon_low_count} exemplars",
    )
    return


@app.cell
def _(adjusted_rand_score, moon_default, moon_low, moon_truth):
    moon_summary = (
        f"default: adjusted Rand {adjusted_rand_score(moon_truth, moon_default.labels_):.3f}, "
        f"-50: adjusted Rand {adjusted_rand_score(moon_truth, moon_low.labels_):.3f}"
    )
    moon_summary
    return (moon_summary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The count was right and the answer was still wrong

    The adjusted Rand index is about **0.24** with the default preference and about **0.26** at `preference=-50`. Both are close to chance, even though the second fit produced exactly two clusters.

    That pairing is the most important observation in this section. **Getting the number of clusters right is not the same as finding the right clusters.** At `preference=-50` each colour contains points from both crescents, so the two groups cut across the shapes rather than following them.

    Lowering the preference achieved the thing it controls: it reduced the number of exemplars. It could not achieve the thing it does not control: understanding that a crescent is a curved shape. The messages are built from distances, and on interlocking crescents distance is actively misleading.

    Compare this with the spectral clustering lesson, where a neighbour graph *could* follow the crescents. Affinity propagation could too, if you gave it a similarity that respects connectivity rather than raw distance. That is exactly what `affinity="precomputed"` allows, and it is what the next section does for text.

    ### Try it yourself

    Which argument controls how willing a point is to become an exemplar, and in which direction does it reduce the number of clusters? Work out your answer, then reveal it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
**`preference`**, and making it **more negative** reduces the number of clusters.

The default run found 8 exemplars on the moons; `preference=-50` found 2. On the wine data the same dial walks the count down as it becomes more negative.

Two cautions. First, the effect is not gradual and there is no formula: you have to try values and look. Second, fewer clusters is not automatically better, as the moons section shows.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Replace `None` so that `student_moon_gap` holds the difference in adjusted Rand index between the `preference=-50` fit and the default fit, that is `-50` minus default. Negative values are fine.
    """)
    return


@app.cell
def _(adjusted_rand_score, moon_default, moon_low, moon_truth):
    student_moon_gap = None
    print(student_moon_gap)
    return (student_moon_gap,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
```python
student_moon_gap = (
    adjusted_rand_score(moon_truth, moon_low.labels_)
    - adjusted_rand_score(moon_truth, moon_default.labels_)
)
```
The value is about **+0.025**, so lowering the preference changed the score by almost nothing even though it cut the number of clusters from 8 to 2. That is the point of the exercise: the cluster count moved a long way, and the agreement with the true crescents barely moved at all.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Sentences: a similarity that suits the data

    The seven sentences have no coordinates, so the first job is to produce numbers. `TfidfVectorizer` builds one weight per word, giving higher weight to words that are frequent in a sentence but rare in the collection. Common English words are dropped.

    Then **cosine similarity** compares the sentences. It measures the angle between two word-weight vectors, so a short sentence and a long sentence that use the same vocabulary count as similar. That is the right notion of similarity for text, and it is deliberately different from the raw distance used on the moons.

    Because the similarities are computed first, the fit is told `affinity="precomputed"`.
    """)
    return


@app.cell
def _(TfidfVectorizer, cosine_similarity):
    sentences = [
        "Nature is beautiful.",
        "I love nature.",
        "Machine learning is a facet of AI.",
        "Nature is my best friend.",
        "General AI is coming soon.",
        "AI is creating a new era.",
        "Cluster Analysis is a branch of AI.",
    ]
    sentence_weights = TfidfVectorizer(stop_words="english").fit_transform(sentences)
    sentence_similarity = cosine_similarity(sentence_weights)
    sentence_facts = (
        f"{len(sentences)} sentences, {sentence_weights.shape[1]} word columns, "
        f"similarity matrix {sentence_similarity.shape[0]} x {sentence_similarity.shape[1]}"
    )
    sentence_facts
    return sentence_similarity, sentence_facts, sentences


@app.cell
def _(AffinityPropagation, sentence_similarity):
    sentence_model = AffinityPropagation(
        affinity="precomputed", random_state=0, max_iter=500
    ).fit(sentence_similarity)
    sentence_count = len(sentence_model.cluster_centers_indices_)
    return sentence_count, sentence_model


@app.cell
def _(mo, pd, sentence_model, sentences):
    sentence_table = pd.DataFrame(
        {
            "Sentence": sentences,
            "Cluster": sentence_model.labels_,
            "Exemplar": [
                "yes" if _i in set(sentence_model.cluster_centers_indices_) else ""
                for _i in range(len(sentences))
            ],
        }
    ).sort_values(["Cluster", "Sentence"])
    mo.Html(sentence_table.to_html())
    return


@app.cell(hide_code=True)
def _(mo, sentence_count):
    mo.md(f"""
    ### Reading the result

    The method found **{sentence_count} clusters**, and the table shows which sentences share one. The `Exemplar` column marks the sentence that represents each group.

    Two things follow from the exemplar rule being applied to text.

    - **Each exemplar is a real sentence**, not an average of several sentences. You can read the representative of each group and judge whether it stands for the rest.
    - **The groups are about vocabulary, not meaning.** The nature sentences share the word "nature"; the AI sentences share "AI". That works here because the vocabulary is distinctive. Two sentences that mean the same thing using different words would not be grouped, because they share no words to give them a similar angle.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Replace `None` so that `student_sentence_exemplars` holds the **sentences themselves** that were elected as exemplars, as a list. `sentence_model` holds the fit.
    """)
    return


@app.cell
def _(sentence_model, sentences):
    student_sentence_exemplars = None
    print(student_sentence_exemplars)
    return (student_sentence_exemplars,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
```python
student_sentence_exemplars = [
    sentences[i] for i in sentence_model.cluster_centers_indices_
]
```
The list holds one representative sentence for each cluster: a nature sentence and an AI sentence. `cluster_centers_indices_` gives positions in the original data, so you use them to look the sentences back up.

This is the property that makes affinity propagation attractive for text: the output includes a real example per group rather than a list of word weights that nobody can read.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    What does `affinity="precomputed"` tell scikit-learn to expect, and why is it needed here? Work out your answer, then reveal it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
It tells scikit-learn that you are passing **a square matrix of pairwise similarities** between observations, rather than the observations themselves.

It is needed because the sentences have no coordinates, so there is no distance to compute from raw features. `cosine_similarity` builds the matrix first, and the fit treats your numbers as the similarities directly.

If you passed the TF-IDF matrix instead and left the default affinity, the method would compute Euclidean distances between word weights, which is not the comparison you want for text.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Wine: choosing a preference with evidence

    Thirteen chemical measurements of 178 wines from three cultivars. Unlike the synthetic data, these measurements are real, correlated, and on different scales, so they are standardized first.

    There is no holdout: every wine contributes to the means and standard deviations used for standardization.

    ### Why the preference has to be searched

    Three cultivars are not something the algorithm knows. We want about three clusters, so `preference` has to be chosen to produce them. Rather than guessing once, this section sweeps four values and compares the results. Each value is a separate fit.
    """)
    return


@app.cell
def _(AffinityPropagation, StandardScaler, load_wine, np, pd):
    wine = load_wine()
    wine_frame = pd.DataFrame(wine.data, columns=wine.feature_names)
    wine_frame["cultivar"] = wine.target_names[wine.target]
    wine_z = pd.DataFrame(
        StandardScaler().fit_transform(wine.data), columns=wine.feature_names
    )

    wine_labels = {}
    wine_rows = []
    for _preference in (-50, -100, -200, -500):
        _model = AffinityPropagation(
            random_state=0, preference=_preference, max_iter=500
        ).fit(wine_z)
        _name = f"preference={_preference}"
        wine_labels[_name] = _model.labels_
        wine_rows.append(
            {
                "Preference": _preference,
                "Clusters": len(_model.cluster_centers_indices_),
            }
        )
    wine_sweep = pd.DataFrame(wine_rows).set_index("Preference")
    wine_labels = pd.DataFrame(wine_labels)

    # The fit used for the detailed analysis below.
    wine_model = AffinityPropagation(
        random_state=0, preference=-200, max_iter=500
    ).fit(wine_z)
    wine_count = len(wine_model.cluster_centers_indices_)
    wine_shape = f"{wine_frame.shape[0]} wines, {wine_frame.shape[1] - 1} measurements"
    return (
        wine,
        wine_count,
        wine_frame,
        wine_labels,
        wine_model,
        wine_shape,
        wine_sweep,
        wine_z,
    )


@app.cell
def _(wine_shape):
    wine_shape
    return


@app.cell
def _(mo, wine_frame):
    mo.Html(wine_frame.head().round(2).to_html())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The preference dial in numbers

    The table records how many clusters each value produced. Read it as a dial, not a score: it says nothing about quality.
    """)
    return


@app.cell
def _(wine_sweep):
    mo.Html(wine_sweep.to_html())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The count walks down as the preference becomes more negative: **8, 5, 3, 2**. Each step makes fewer points willing to stand as exemplars, so fewer groups survive.

    This is the behaviour to remember from the moons section, now seen as a controlled sweep. On the moons, one step from the default to -50 moved 8 clusters to 2. Here four deliberate steps move 8 clusters to 2. The mapping from `preference` to cluster count is **specific to the dataset**, which is why it has to be searched rather than calculated.

    At `preference=-200` the count is three, matching the number of cultivars. The rest of the section examines that fit.

    ### The t-SNE picture

    t-SNE reduces the thirteen measurements to two dimensions **for display only**. It is fitted once, with `random_state=0`, and it is not the input to the clustering. The clustering used all thirteen standardized columns.
    """)
    return


@app.cell
def _(TSNE, cv, pd, wine, wine_count, wine_model, wine_z):
    wine_plane = pd.DataFrame(
        TSNE(n_components=2, random_state=0, init="pca").fit_transform(wine_z),
        columns=["t-SNE 1", "t-SNE 2"],
    )
    wine_view = cv.ClusterVisualizer(wine_plane, wine.target)
    wine_view.plot_data(title="Wine in the t-SNE view (the cultivars are not shown)")
    wine_plane
    return wine_plane, wine_view


@app.cell
def _(wine_view):
    wine_view.plot_data_with_true_labels(title="The three cultivars in the t-SNE view")
    return


@app.cell
def _(wine_count, wine_model, wine_view):
    wine_view.plot_clusters(
        wine_model.labels_,
        title=f"Affinity propagation, preference=-200: {wine_count} clusters",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice that no exemplar markers are drawn on this plot. The exemplars are real wines, described by thirteen chemical measurements, so they have no position in a two-dimensional t-SNE picture. That is a limitation of the display, not of the method: the exemplar wines exist, they simply cannot be plotted here.

    Comparing the two plots above is a fair check, because t-SNE was fitted on the same standardized data the clustering used. The clusters should line up roughly with the cultivar colours, and they do, with visible overlap where the cultivars themselves overlap.

    ### Scoring against the cultivars

    The scores compare the three clusters with the three cultivar labels. Cluster numbers are arbitrary, so a perfect fit could label the cultivars in any order.
    """)
    return


@app.cell
def _(
    adjusted_mutual_info_score,
    adjusted_rand_score,
    fowlkes_mallows_score,
    normalized_mutual_info_score,
    pd,
    wine,
    wine_model,
):
    wine_scores = pd.DataFrame(
        {
            "Score": [
                "Adjusted Rand",
                "Adjusted Mutual Info",
                "Normalized Mutual Info",
                "Fowlkes-Mallows",
            ],
            "Value": [
                adjusted_rand_score(wine.target, wine_model.labels_),
                adjusted_mutual_info_score(wine.target, wine_model.labels_),
                normalized_mutual_info_score(wine.target, wine_model.labels_),
                fowlkes_mallows_score(wine.target, wine_model.labels_),
            ],
            "Chance corrected": ["yes", "yes", "no", "no"],
        }
    ).set_index("Score")
    return (wine_scores,)


@app.cell
def _(mo, wine_scores):
    mo.Html(wine_scores.round(4).to_html())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Read the four values with the baseline in mind, not as four confirmations of the same thing.

    - **Adjusted Rand is about 0.62.** Roughly two thirds of the pairwise agreement you could hope for. This is the number to quote.
    - **Adjusted Mutual Info is about 0.61**, telling the same story in a different currency.
    - **Fowlkes–Mallows is about 0.75**, which looks impressive and is the least trustworthy of the four. Shuffled labels on this data still score about **0.33**, so the usable range is roughly 0.33 to 1.00 rather than 0 to 1.00.
    - **Normalized Mutual Info is about 0.61** and is unadjusted, though its chance value is small enough that it is far less misleading than Fowlkes–Mallows.

    So the honest summary is "good but not close to perfect agreement", which the next table explains.

    ### Where the clusters and the cultivars disagree

    A crosstab shows exactly which wines land in which cluster.
    """)
    return


@app.cell
def _(np, pd, wine, wine_model):
    wine_table = pd.crosstab(
        pd.Series(wine.target_names[wine.target], name="cultivar"),
        pd.Series(wine_model.labels_, name="cluster"),
    )
    wine_sizes = np.bincount(wine_model.labels_)
    return wine_sizes, wine_table


@app.cell
def _(mo, wine_table):
    mo.Html(wine_table.to_html())
    return


@app.cell(hide_code=True)
def _(mo, wine_sizes):
    mo.md(f"""
    Cluster sizes are **{", ".join(str(int(_size)) for _size in wine_sizes)}**, so the three clusters are of similar size. That is not evidence of quality; it is only worth knowing when reading the table.

    The pattern is more encouraging than the scores alone suggested, but it is not a recovery of the three labels.

    - **Each cluster is dominated by one cultivar.** Cluster 0 holds 49 of the 59 cultivar-0 wines and only 5 wines from elsewhere, so it is close to pure. Cluster 1 holds 58 of the 71 cultivar-1 wines, and cluster 2 holds 46 of the 48 cultivar-2 wines.
    - **But each cluster also absorbs wines from a neighbour.** Cluster 0 takes 10 cultivar-1 wines, cluster 1 takes 10 cultivar-2 wines, and cluster 2 takes 10 cultivar-1 wines. Those 20 misassigned wines are what hold the adjusted Rand index down to 0.62.

    In other words, the grouping is largely right, and the errors are concentrated where cultivars 1 and 2 meet chemically. No cluster is "the cultivar 1 group" in a pure sense, and the chemical measurements that separate those two cultivars overlap enough that a clustering of these thirteen columns cannot fully undo it.

    ### Do the scores agree on which preference is best?

    The sweep produced four fits with different numbers of clusters. Scoring all four shows whether the choice of `preference` is obvious.
    """)
    return


@app.cell
def _(
    adjusted_mutual_info_score,
    adjusted_rand_score,
    fowlkes_mallows_score,
    normalized_mutual_info_score,
    pd,
    wine,
    wine_labels,
):
    sweep_scores = pd.DataFrame(
        {
            "Clusters": {
                _name: len(pd.unique(wine_labels[_name])) for _name in wine_labels
            },
            "Adjusted Rand": {
                _name: adjusted_rand_score(wine.target, wine_labels[_name])
                for _name in wine_labels
            },
            "Adjusted Mutual Info": {
                _name: adjusted_mutual_info_score(wine.target, wine_labels[_name])
                for _name in wine_labels
            },
            "Fowlkes-Mallows": {
                _name: fowlkes_mallows_score(wine.target, wine_labels[_name])
                for _name in wine_labels
            },
        }
    )
    return (sweep_scores,)


@app.cell
def _(mo, sweep_scores):
    mo.Html(sweep_scores.round(3).to_html())
    return


@app.cell(hide_code=True)
def _(adjusted_rand_score, mo, np, sweep_scores, wine, wine_labels):
    _shuffled = np.random.RandomState(0).permutation(wine_labels["preference=-200"])
    mo.md(f"""
    The three scores do **not** agree on a single winner, and that is the honest result.

    - **Adjusted Rand is highest at preference=-200**, with about {sweep_scores.loc["preference=-200", "Adjusted Rand"]:.3f}. This is the fit that produced three clusters.
    - **Adjusted Mutual Info is highest at preference=-100**, with about {sweep_scores.loc["preference=-100", "Adjusted Mutual Info"]:.3f}, even though that fit produced five clusters. Splitting a group can share slightly more information without agreeing more on pairs.
    - **Fowlkes-Mallows is also highest at preference=-200**, at about {sweep_scores.loc["preference=-200", "Fowlkes-Mallows"]:.3f}. Remember that its chance floor on this data is about 0.34, so treat the ranking as a tie-breaker only.

    To see that floor, this fit's labels shuffled at random score {adjusted_rand_score(wine.target, _shuffled):.3f} on Adjusted Rand but well above zero on Fowlkes-Mallows. Comparing the two columns side by side is the safest habit.

    There is no automatic rule that picks `preference=-200` for you. The justification is a combination of the adjusted Rand index and the fact that three clusters matches the three known cultivars, which is information from outside the algorithm.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Replace `None` so that `student_wine_count` holds the number of clusters in the `preference=-200` fit, and `student_wine_ari` holds its adjusted Rand index against the true cultivars.
    """)
    return


@app.cell
def _(adjusted_rand_score, wine, wine_model):
    student_wine_count = None
    student_wine_ari = None
    print("clusters:", student_wine_count)
    print("adjusted Rand:", student_wine_ari)
    return student_wine_ari, student_wine_count


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
```python
student_wine_count = len(wine_model.cluster_centers_indices_)
student_wine_ari = adjusted_rand_score(wine.target, wine_model.labels_)
```
The count is **3** and the adjusted Rand index is about **0.622**. `cluster_centers_indices_` has one entry per exemplar, so its length is the number of clusters; this is the same expression used for the blobs and the moons.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Which of the four external scores is **not** corrected for chance, and why does that make its value of about 0.75 look better than it is? Work out your answer, then reveal it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
**Fowlkes–Mallows** is not corrected for chance. Shuffled labels on this data still score about **0.34**, so the usable range is roughly 0.34 to 1.00 and a value of 0.75 is far less impressive than it appears.

Adjusted Rand and Adjusted Mutual Info both fall to about 0 when the labels are shuffled, which is why they are the ones to quote. Normalized Mutual Info is also unadjusted, but its chance value is small.

The practical rule: use the adjusted scores when comparing fits with different numbers of clusters, and compare Fowlkes–Mallows only between fits that asked for the same number.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(
    blob_count,
    blob_summary,
    moon_default_count,
    moon_low_count,
    moon_summary,
    mo,
    sentence_count,
    sweep_scores,
    wine_count,
    wine_scores,
    wine_sizes,
):
    mo.md(f"""
    ## Conclusions

    1. **An exemplar is a real observation.** Affinity propagation elects points to represent groups, so the representatives can be named and inspected. A k-means centre is a computed mean and usually sits where no point exists.
    2. **There is no `n_clusters` argument, and `preference` takes its place.** A more negative preference makes fewer points willing to stand, so fewer clusters survive. On the moons the default produced {moon_default_count} exemplars and `preference=-50` produced {moon_low_count}; on the wine data a sweep from -50 to -500 walked the count down through 8, 5, 3, and 2.
    3. **The blobs work cleanly.** {blob_count} exemplars were found without being asked for, sitting on top of real points. {blob_summary}.
    4. **The right number of clusters is not the right answer.** On the moons, `preference=-50` produced exactly two clusters, but each one held points from both crescents. {moon_summary.capitalize()}. The dial controls the count, not the geometry, and distance cannot describe a crescent.
    5. **The similarity you supply decides what the method can see.** For the sentences, cosine similarity of TF-IDF weights grouped {sentence_count} clusters by shared vocabulary, and each exemplar was a readable sentence. That works for text because the vocabulary is distinctive; sentences that mean the same thing in different words would not be grouped.
    6. **On the wine data, three clusters score well but not perfectly.** {wine_count} clusters at `preference=-200`, with adjusted Rand {wine_scores.loc["Adjusted Rand", "Value"]:.3f} and adjusted mutual information {wine_scores.loc["Adjusted Mutual Info", "Value"]:.3f}. Cluster sizes are {", ".join(str(int(_size)) for _size in wine_sizes)}.
    7. **The scores disagree about the best preference.** Adjusted Rand peaks at -200, but adjusted mutual information peaks at -100 with five clusters. The choice of `preference` rests on the adjusted Rand index plus the outside knowledge that there are three cultivars. No formula makes that choice for you.
    8. **Fowlkes–Mallows needs care.** Its value of {wine_scores.loc["Fowlkes-Mallows", "Value"]:.3f} sits against a chance floor of about 0.34 on this data. Quote the adjusted scores, and treat Fowlkes–Mallows as a tie-breaker between fits with the same number of clusters.
    9. **t-SNE drew the picture, it did not cluster.** The fits used the standardized measurements, and on the wine data the exemplars cannot even be drawn, because they live in thirteen dimensions.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check your understanding

    1. What is an exemplar, and how does it differ from a k-means centre?
    2. Why does affinity propagation have no `n_clusters` argument, and what does `preference` control instead?
    3. What happens to the number of clusters as `preference` becomes more negative, and why?
    4. On the moons, `preference=-50` produced two clusters but an adjusted Rand index near 0.26. Explain how both can be true.
    5. What does `affinity="precomputed"` expect, and why is it used for the sentences?
    6. Why is cosine similarity preferred over Euclidean distance for text?
    7. Which external scores are corrected for chance, and which one has a chance floor near 0.34 on the wine data?
    8. Which data did the wine clustering actually use, and why can the exemplars not be drawn on the t-SNE plot?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
1. **A real observation elected to represent a group.** Every other point is assigned to one exemplar. A k-means centre is instead the mean of its points, so it is usually a location where no observation sits.
2. **The exemplars are elected by the data, so the count is an output rather than an input.** `preference` sets how willing each point is to stand as an exemplar, which is what determines how many groups survive.
3. **It falls.** A more negative preference makes fewer points willing to become exemplars. On the wine data the counts were 8, 5, 3, and 2 as the preference went from -50 to -500.
4. **The dial worked and the geometry did not.** Lowering the preference reduced the count to two, which it controls. It cannot teach the method that a crescent is a curve, so each cluster still contains points from both arms, and the adjusted Rand index stays near chance.
5. **A square matrix of pairwise similarities** between observations, rather than the observations themselves. It is used for the sentences because text has no coordinates, so `cosine_similarity` computes the similarities first.
6. **It compares the angle between word-weight vectors and ignores their length.** That suits text, where a short and a long sentence on the same subject should count as similar. Euclidean distance would be dominated by document length and by high-weight rare words.
7. **Adjusted Rand and adjusted mutual information are corrected.** Normalized mutual information and Fowlkes–Mallows are not. Fowlkes–Mallows has a chance floor near 0.34 on the wine data, so its 0.75 is less impressive than it looks.
8. **The clustering used all thirteen standardized chemical measurements.** t-SNE is only a two-dimensional picture for the eye. The exemplars are real wines with thirteen coordinates, so they have no position in that picture.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - [AffinityPropagation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AffinityPropagation.html)
    - [Affinity propagation, scikit-learn user guide](https://scikit-learn.org/stable/modules/clustering.html#affinity-propagation)
    - [Wine dataset, UCI](https://archive.ics.uci.edu/dataset/109/wine)
    - Frey, B. J., and Dueck, D. (2007). Clustering by passing messages between data points. *Science*, 315(5814), 972–976.
    - VanderPlas, J. (2017). *Python Data Science Handbook*. O'Reilly Media. Chapter 5.
    """)
    return


if __name__ == "__main__":
    app.run()
