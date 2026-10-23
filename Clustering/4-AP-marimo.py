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
        fowlkes_mallows_score,
        load_wine,
        make_blobs,
        make_moons,
        normalized_mutual_info_score,
    )


@app.cell
def _(np, pd, sns):
    def draw_clusters(axes, x, y, labels, title, centers=None):
        frame = pd.DataFrame(
            {"x": np.asarray(x), "y": np.asarray(y), "cluster": np.asarray(labels).astype(str)}
        )
        sns.scatterplot(data=frame, x="x", y="y", hue="cluster", s=28, ax=axes, legend=False)
        if centers is not None:
            axes.scatter(centers[:, 0], centers[:, 1], c="black", marker="x", s=70)
        axes.set_title(title)

    return (draw_clusters,)


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

    `preference` is how willing a point is to become an exemplar. A more negative preference produces fewer exemplars. The fit does not take `n_clusters`.

    ## Datasets Used

    - **Blobs.** 50 points, three centers, `cluster_std=1.5`, `random_state=10`.
    - **Half moons.** 100 points, noise 0.08, `random_state=10`.
    - **Seven short sentences** about nature and artificial intelligence.
    - **Wine.** Thirteen chemical measurements of wines from three Italian cultivars, loaded from scikit-learn.
    """)
    return


@app.cell
def _(AffinityPropagation, draw_clusters, make_blobs, plt):
    blobs, _blob_labels = make_blobs(
        n_samples=50, centers=3, cluster_std=1.5, random_state=10
    )
    blob_model = AffinityPropagation(random_state=0, max_iter=500).fit(blobs)
    blob_clusters = len(blob_model.cluster_centers_indices_)
    _figure, _axes = plt.subplots(figsize=(5, 4))
    draw_clusters(
        _axes,
        blobs[:, 0],
        blobs[:, 1],
        blob_model.labels_,
        f"{blob_clusters} exemplars",
        centers=blob_model.cluster_centers_,
    )
    print(f"Blobs: {blob_clusters} clusters")
    return _figure


@app.cell
def _(AffinityPropagation, draw_clusters, make_moons, plt):
    moons, _moon_arms = make_moons(100, noise=0.08, random_state=10)
    moon_default = AffinityPropagation(random_state=0, max_iter=500).fit(moons)
    moon_low = AffinityPropagation(preference=-50, random_state=0, max_iter=500).fit(moons)
    moon_default_clusters = len(moon_default.cluster_centers_indices_)
    moon_low_clusters = len(moon_low.cluster_centers_indices_)
    _figure, _axes = plt.subplots(1, 2, figsize=(9, 4))
    draw_clusters(
        _axes[0],
        moons[:, 0],
        moons[:, 1],
        moon_default.labels_,
        f"Default preference, {moon_default_clusters} exemplars",
        centers=moon_default.cluster_centers_,
    )
    draw_clusters(
        _axes[1],
        moons[:, 0],
        moons[:, 1],
        moon_low.labels_,
        f"preference = -50, {moon_low_clusters} exemplars",
        centers=moon_low.cluster_centers_,
    )
    _axes[0].set_aspect("equal", adjustable="datalim")
    _axes[1].set_aspect("equal", adjustable="datalim")
    _figure.tight_layout()
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A lower preference cuts the number of exemplars. It does not by itself teach the algorithm that the two moons are the two groups. Points from both moons can still share an exemplar, because the messages use distance and the moons bend toward each other.

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
            sentences[i] for i, label in enumerate(sentence_model.labels_) if label == _cluster
        ):
            print(f"  {_sentence}")
    sentence_clusters = len(sentence_model.cluster_centers_indices_)
    print(f"{sentence_clusters} clusters")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Wine

    The thirteen measurements are standardized together. t-SNE, with `random_state=0`, is only the picture. Affinity propagation sees the standardized measurements, with `preference=-200`.

    The scores compare the exemplars with the three cultivar labels. The cluster numbers do not have to match the cultivar numbers.
    """)
    return


@app.cell
def _(
    AffinityPropagation,
    StandardScaler,
    TSNE,
    adjusted_mutual_info_score,
    adjusted_rand_score,
    draw_clusters,
    fowlkes_mallows_score,
    load_wine,
    normalized_mutual_info_score,
    pd,
    plt,
):
    wine = load_wine()
    wine_z = StandardScaler().fit_transform(wine.data)
    wine_view = TSNE(n_components=2, random_state=0, init="pca").fit_transform(wine_z)
    wine_model = AffinityPropagation(random_state=0, preference=-200, max_iter=500).fit(wine_z)
    wine_clusters = len(wine_model.cluster_centers_indices_)
    wine_rand = adjusted_rand_score(wine.target, wine_model.labels_)
    wine_ami = adjusted_mutual_info_score(wine.target, wine_model.labels_)
    wine_nmi = normalized_mutual_info_score(wine.target, wine_model.labels_)
    wine_fm = fowlkes_mallows_score(wine.target, wine_model.labels_)
    wine_table = pd.crosstab(
        pd.Series(wine.target_names[wine.target], name="cultivar"),
        pd.Series(wine_model.labels_, name="cluster"),
    )
    _figure, _axes = plt.subplots(1, 2, figsize=(10, 4))
    draw_clusters(_axes[0], wine_view[:, 0], wine_view[:, 1], wine.target, "Cultivars")
    draw_clusters(
        _axes[1],
        wine_view[:, 0],
        wine_view[:, 1],
        wine_model.labels_,
        f"Affinity propagation, {wine_clusters} clusters",
    )
    _figure.tight_layout()
    print(
        f"ARI {wine_rand:.3f}   AMI {wine_ami:.3f}   NMI {wine_nmi:.3f}   FMI {wine_fm:.3f}"
    )
    print(wine_table.to_string())
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Which argument makes a point less willing to become an exemplar when it is more negative? Replace `None` with its name.
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
**preference.** A large negative value leaves fewer exemplars. The algorithm still chooses how many.
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
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check your understanding

    1. What is an exemplar?
    2. Why is there no `n_clusters` argument in these fits?
    3. What happens to the number of clusters when preference becomes more negative?
    4. Why use cosine similarity for the sentences?
    5. What do the external scores compare on the wines?
    6. Why can cluster 0 fail to mean cultivar 0?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
1. **A point that stands for its cluster:** Other points are assigned to an exemplar. The exemplar is one of the original rows.
2. **The messages choose the number:** Preference influences that choice. It is not a request for a fixed k.
3. **It usually falls:** Points are less willing to become exemplars, so more of them join an existing one.
4. **It compares direction, not length:** Two sentences can use the same words with different totals. Cosine similarity ignores that difference in length.
5. **The exemplars and the three cultivars:** A high score means wines of the same cultivar tend to share an exemplar.
6. **The labels are assigned by the algorithm:** The crosstab is the translation between those labels and the cultivar names.
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
    - [Clustering](https://scikit-learn.org/stable/modules/clustering.html#affinity-propagation)
    - VanderPlas, J. (2017). *Python Data Science Handbook*. O'Reilly Media. Chapter 5.
    """)
    return


if __name__ == "__main__":
    app.run()
