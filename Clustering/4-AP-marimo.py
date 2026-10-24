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
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Affinity Propagation

    ## Objectives

    - Let the points choose their own exemplars instead of asking for a number of clusters.
    - Lower the preference and watch the number of exemplars fall.
    - Compare the wine clusters with the three cultivars.

    ## Background

    Affinity propagation passes messages between points until a set of exemplars remains. An exemplar is a point that represents a cluster. The other points are assigned to the exemplar they exchange messages with.

    `preference` is how willing a point is to become an exemplar. A more negative preference produces fewer exemplars. The fit does not take `n_clusters`. These fits use `random_state=0` and `max_iter=500`.

    The scatters come from `ClusterVisualizer`. Black markers are exemplars when the centers live in the same two columns as the picture.

    ## Datasets Used

    - **Blobs.** 50 points, three centers, `cluster_std=1.5`, `random_state=10`.
    - **Half moons.** 100 points, noise 0.08, `random_state=10`.
    - **Seven short sentences** about nature and artificial intelligence.
    - **Wine.** Thirteen chemical measurements of wines from three Italian cultivars, loaded from scikit-learn.
    """)
    return


@app.cell
def _(AffinityPropagation, cv, make_blobs, pd):
    blobs, _blob_labels = make_blobs(
        n_samples=50, centers=3, cluster_std=1.5, random_state=10
    )
    blob_view = cv.ClusterVisualizer(pd.DataFrame(blobs, columns=["x", "y"]))
    blob_model = AffinityPropagation(random_state=0, max_iter=500).fit(blobs)
    blob_clusters = len(blob_model.cluster_centers_indices_)
    print(f"Blobs: {blob_clusters} exemplars")
    blob_view.plot_data(title="Blobs")
    return blob_clusters, blob_model, blob_view


@app.cell
def _(blob_model, blob_view):
    blob_view.plot_clusters(
        blob_model.labels_,
        blob_model.cluster_centers_,
        title=f"Affinity propagation, {len(blob_model.cluster_centers_indices_)} exemplars",
    )
    return


@app.cell
def _(AffinityPropagation, cv, make_moons, pd):
    moons, _moon_arms = make_moons(100, noise=0.08, random_state=10)
    moon_view = cv.ClusterVisualizer(pd.DataFrame(moons, columns=["x", "y"]))
    moon_default = AffinityPropagation(random_state=0, max_iter=500).fit(moons)
    moon_low = AffinityPropagation(preference=-50, random_state=0, max_iter=500).fit(
        moons
    )
    moon_default_clusters = len(moon_default.cluster_centers_indices_)
    moon_low_clusters = len(moon_low.cluster_centers_indices_)
    print(f"Default preference: {moon_default_clusters} exemplars")
    print(f"preference = -50: {moon_low_clusters} exemplars")
    moon_view.plot_data(title="Half Moons")
    return moon_default, moon_default_clusters, moon_low, moon_low_clusters, moon_view


@app.cell
def _(moon_default, moon_default_clusters, moon_view):
    moon_view.plot_clusters(
        moon_default.labels_,
        moon_default.cluster_centers_,
        title=f"Half Moons - Affinity Propagation ({moon_default_clusters} exemplars)",
    )
    return


@app.cell
def _(moon_low, moon_low_clusters, moon_view):
    moon_view.plot_clusters(
        moon_low.labels_,
        moon_low.cluster_centers_,
        title=f"Half Moons - Affinity Propagation (preference = -50, {moon_low_clusters} exemplars)",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A lower preference cuts the number of exemplars. It does not by itself put one exemplar on each moon. Points from both moons can still share an exemplar, because the messages use distance and the moons bend toward each other.

    For the sentences, cosine similarity compares the direction of the word-weight vectors. The affinity matrix is passed in already computed, so `affinity="precomputed"`.
    """)
    return


@app.cell
def _(AffinityPropagation, TfidfVectorizer, cosine_similarity):
    sentences = [
        "Nature is beautiful.",
        "I love nature.",
        "Machine learning is a facet of AI.",
        "Nature is my best friend.",
        "General AI is coming soon.",
        "AI is creating a new era.",
        "Cluster Analysis is a branch of AI.",
    ]
    weights = TfidfVectorizer(stop_words="english").fit_transform(sentences)
    sentence_model = AffinityPropagation(
        affinity="precomputed", random_state=0, max_iter=500
    ).fit(cosine_similarity(weights))
    for _cluster in sorted(set(sentence_model.labels_)):
        print(f"Cluster {_cluster}")
        for _sentence in (
            sentences[_i]
            for _i, _label in enumerate(sentence_model.labels_)
            if _label == _cluster
        ):
            print(f"  {_sentence}")
    sentence_clusters = len(sentence_model.cluster_centers_indices_)
    print(f"{sentence_clusters} clusters")
    return (sentence_clusters,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Wine

    The thirteen measurements are standardized together. There is no holdout. t-SNE, with `random_state=0`, is only the picture. Affinity propagation sees the standardized measurements, with `preference=-200`. The exemplars live in those thirteen columns, so they are not drawn on the t-SNE plane.

    The scores compare the clusters with the three cultivar labels. The cluster numbers do not have to match the cultivar numbers.
    """)
    return


@app.cell
def _(
    AffinityPropagation,
    StandardScaler,
    TSNE,
    adjusted_mutual_info_score,
    adjusted_rand_score,
    cv,
    fowlkes_mallows_score,
    load_wine,
    normalized_mutual_info_score,
    pd,
):
    wine = load_wine()
    wine_z = StandardScaler().fit_transform(wine.data)
    wine_plane = pd.DataFrame(
        TSNE(n_components=2, random_state=0, init="pca").fit_transform(wine_z),
        columns=["t-SNE 1", "t-SNE 2"],
    )
    wine_view = cv.ClusterVisualizer(wine_plane, wine.target)
    wine_model = AffinityPropagation(
        random_state=0, preference=-200, max_iter=500
    ).fit(wine_z)
    wine_clusters = len(wine_model.cluster_centers_indices_)
    wine_rand = adjusted_rand_score(wine.target, wine_model.labels_)
    wine_ami = adjusted_mutual_info_score(wine.target, wine_model.labels_)
    wine_nmi = normalized_mutual_info_score(wine.target, wine_model.labels_)
    wine_fm = fowlkes_mallows_score(wine.target, wine_model.labels_)
    wine_table = pd.crosstab(
        pd.Series(wine.target_names[wine.target], name="cultivar"),
        pd.Series(wine_model.labels_, name="cluster"),
    )
    print(f"{wine_clusters} clusters")
    print(
        f"ARI {wine_rand:.3f}   AMI {wine_ami:.3f}   NMI {wine_nmi:.3f}   FMI {wine_fm:.3f}"
    )
    print(wine_table.to_string())
    wine_view.plot_data(title="Wine in the t-SNE view")
    return (
        wine_ami,
        wine_clusters,
        wine_fm,
        wine_model,
        wine_nmi,
        wine_rand,
        wine_table,
        wine_view,
    )


@app.cell
def _(wine_view):
    wine_view.plot_data_with_true_labels(title="Cultivars")
    return


@app.cell
def _(wine_clusters, wine_model, wine_view):
    wine_view.plot_clusters(
        wine_model.labels_,
        title=f"Affinity Propagation Results in the t-SNE Space ({wine_clusters} clusters)",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Which argument makes a point less willing to become an exemplar when it is set more negative? Replace `None` with its name.
    """)
    return


@app.cell
def _():
    student_preference = None
    print(student_preference)
    return (student_preference,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r'''
```python
student_preference = "preference"
```
**preference.** On the moons, the default value and `preference=-50` produce different numbers of exemplars. Fewer exemplars are not the same thing as one exemplar per moon.
''')
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(
    blob_clusters,
    mo,
    moon_default_clusters,
    moon_low_clusters,
    sentence_clusters,
    wine_ami,
    wine_clusters,
    wine_fm,
    wine_nmi,
    wine_rand,
    wine_table,
):
    _main = wine_table.idxmax()
    _lines = "\n".join(
        f"- Cluster {_cluster} draws most of its wines from **{_cultivar}**."
        for _cluster, _cultivar in _main.items()
    )
    mo.md(f"""
    ## Conclusions

    **Key takeaways:**

    - On the blobs, affinity propagation finds {blob_clusters} exemplars without being told to look for three.
    - On the moons, the default preference finds {moon_default_clusters} exemplars. `preference=-50` finds {moon_low_clusters}. Fewer exemplars are not the same thing as one exemplar per moon.
    - The seven sentences form {sentence_clusters} clusters. The printed lists show which sentences share an exemplar.
    - On the wines, `preference=-200` finds {wine_clusters} clusters. Adjusted Rand index is {wine_rand:.3f}, adjusted mutual information is {wine_ami:.3f}, normalized mutual information is {wine_nmi:.3f}, and the Fowlkes–Mallows score is {wine_fm:.3f}.
    {_lines}
    - The wine picture is a t-SNE view. The messages were passed on the thirteen standardized measurements.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check your understanding

    1. What is an exemplar?
    2. Why is there no `n_clusters` argument in these fits?
    3. What happens to the number of clusters when preference becomes more negative?
    4. Why can two moons still share an exemplar after the preference is lowered?
    5. What does `affinity="precomputed"` expect?
    6. Which matrix was affinity propagation fit on for the wines?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
1. **A real data point chosen to represent a cluster:** The other points are assigned to an exemplar. It is not a mean computed in empty space.
2. **The algorithm chooses the exemplars:** Preference changes how willing points are to take that role. The count is a result.
3. **It falls:** A large negative preference leaves fewer points willing to be exemplars.
4. **The moons bend toward each other:** Messages follow distance. A lower preference reduces the count. It does not teach the algorithm the shape.
5. **A similarity matrix you already computed:** For the sentences that matrix is cosine similarity of the word weights.
6. **The thirteen standardized measurements:** t-SNE only places the bottles for the drawing.
""")
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - [AffinityPropagation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AffinityPropagation.html)
    - Frey, B. J., and Dueck, D. (2007). Clustering by passing messages between data points. *Science*, 315(5814), 972–976.
    """)
    return


if __name__ == "__main__":
    app.run()
