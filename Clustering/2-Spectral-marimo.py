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
    from sklearn.cluster import KMeans, SpectralClustering
    from sklearn.datasets import make_circles, make_moons
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import adjusted_rand_score

    return (
        KMeans,
        SpectralClustering,
        TfidfVectorizer,
        adjusted_rand_score,
        make_circles,
        make_moons,
    )


@app.cell
def _(np, pd, sns):
    def draw_clusters(axes, x, y, labels, title):
        frame = pd.DataFrame(
            {"x": np.asarray(x), "y": np.asarray(y), "cluster": np.asarray(labels).astype(str)}
        )
        sns.scatterplot(data=frame, x="x", y="y", hue="cluster", s=22, ax=axes, legend=False)
        axes.set_aspect("equal", adjustable="datalim")
        axes.set_title(title)

    def spiral_points(n_points, noise=0.8):
        rng = np.random.default_rng(12345)
        angle = np.sqrt(rng.random((n_points, 1))) * 2 * np.pi
        arm_x = -np.cos(angle) * angle + rng.random((n_points, 1)) * noise
        arm_y = np.sin(angle) * angle + rng.random((n_points, 1)) * noise
        points = np.vstack((np.hstack((arm_x, arm_y)), np.hstack((-arm_x, -arm_y))))
        arms = np.repeat([0, 1], n_points)
        return points, arms

    return draw_clusters, spiral_points


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Spectral Clustering

    ## Objectives

    - See where k-means cuts a moon, a ring, or a spiral.
    - Separate those shapes with a neighbor graph instead of a straight cut.
    - Group short documents by the words they share.

    ## Background

    k-means assigns a point to the nearest center. That works for round blobs. A moon, a ring, and a spiral are not round, so the nearest center can mix two shapes.

    Spectral clustering builds a graph of nearest neighbors and clusters in that graph. `affinity="nearest_neighbors"` is that graph. `n_neighbors` is how many neighbors each point keeps. Too few neighbors can leave the graph in separate pieces, and scikit-learn warns that the embedding may not behave as expected.

    ## Datasets Used

    - **Half moons.** 300 points from `make_moons`, noise 0.05, `random_state=10`.
    - **Concentric circles.** 400 points from `make_circles`, noise 0.1, factor 0.2, `random_state=123`.
    - **Two spirals.** 300 points on each arm.
    - **Twenty-five short sentences** about weather, animals, royalty, water, and Python.
    """)
    return


@app.cell
def _(
    KMeans,
    SpectralClustering,
    adjusted_rand_score,
    draw_clusters,
    make_moons,
    pd,
    plt,
):
    moons, moon_arms = make_moons(300, noise=0.05, random_state=10)
    moon_rows = []
    _figure, _axes = plt.subplots(2, 3, figsize=(11, 7))
    for _column, _k in enumerate((2, 3, 6)):
        _kmeans = KMeans(n_clusters=_k, random_state=10, n_init="auto").fit_predict(moons)
        _spectral = SpectralClustering(
            n_clusters=_k,
            affinity="nearest_neighbors",
            n_neighbors=20,
            assign_labels="kmeans",
            random_state=0,
            n_jobs=1,
        ).fit_predict(moons)
        draw_clusters(_axes[0, _column], moons[:, 0], moons[:, 1], _kmeans, f"k-means, k = {_k}")
        draw_clusters(_axes[1, _column], moons[:, 0], moons[:, 1], _spectral, f"Spectral, k = {_k}")
        moon_rows.append(
            {
                "k": _k,
                "kmeans_ari": adjusted_rand_score(moon_arms, _kmeans),
                "spectral_ari": adjusted_rand_score(moon_arms, _spectral),
            }
        )
    moon_scores = pd.DataFrame(moon_rows).set_index("k")
    _figure.tight_layout()
    print(moon_scores.round(3).to_string())
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The adjusted Rand index compares a clustering with the two true arms. It is 1 when the groups match, and near 0 when the match is no better than chance. The index does not care which group is called 0.

    For the rings, 20 neighbors leaves the graph disconnected. The fit below uses 40 neighbors.
    """)
    return


@app.cell
def _(
    KMeans,
    SpectralClustering,
    adjusted_rand_score,
    draw_clusters,
    make_circles,
    plt,
):
    circles, circle_rings = make_circles(
        n_samples=400, random_state=123, noise=0.1, factor=0.2
    )
    circle_kmeans = KMeans(n_clusters=2, random_state=10, n_init="auto").fit_predict(circles)
    circle_spectral = SpectralClustering(
        n_clusters=2,
        affinity="nearest_neighbors",
        n_neighbors=40,
        assign_labels="kmeans",
        random_state=0,
        n_jobs=1,
    ).fit_predict(circles)
    circle_kmeans_ari = adjusted_rand_score(circle_rings, circle_kmeans)
    circle_spectral_ari = adjusted_rand_score(circle_rings, circle_spectral)
    _figure, _axes = plt.subplots(1, 2, figsize=(9, 4))
    draw_clusters(_axes[0], circles[:, 0], circles[:, 1], circle_kmeans, f"k-means, ARI {circle_kmeans_ari:.3f}")
    draw_clusters(
        _axes[1], circles[:, 0], circles[:, 1], circle_spectral, f"Spectral, ARI {circle_spectral_ari:.3f}"
    )
    _figure.tight_layout()
    return _figure


@app.cell
def _(KMeans, SpectralClustering, adjusted_rand_score, draw_clusters, pd, plt, spiral_points):
    spirals, spiral_arms = spiral_points(300)
    spiral_rows = []
    _figure, _axes = plt.subplots(2, 2, figsize=(9, 8))
    for _column, _k in enumerate((2, 3)):
        _kmeans = KMeans(n_clusters=_k, random_state=0, n_init="auto").fit_predict(spirals)
        _spectral = SpectralClustering(
            n_clusters=_k,
            affinity="nearest_neighbors",
            n_neighbors=10,
            assign_labels="kmeans",
            random_state=0,
            n_jobs=1,
        ).fit_predict(spirals)
        draw_clusters(_axes[0, _column], spirals[:, 0], spirals[:, 1], _kmeans, f"k-means, k = {_k}")
        draw_clusters(_axes[1, _column], spirals[:, 0], spirals[:, 1], _spectral, f"Spectral, k = {_k}")
        spiral_rows.append(
            {
                "k": _k,
                "kmeans_ari": adjusted_rand_score(spiral_arms, _kmeans),
                "spectral_ari": adjusted_rand_score(spiral_arms, _spectral),
            }
        )
    spiral_scores = pd.DataFrame(spiral_rows).set_index("k")
    _figure.tight_layout()
    print(spiral_scores.round(3).to_string())
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Twenty-five sentences

    `TfidfVectorizer` turns each sentence into word weights. Common English words are dropped. Spectral clustering then uses the neighbor graph of those weights. The cluster numbers are arbitrary, so the list below is the thing to read: sentences about the same subject should land together.
    """)
    return


@app.cell
def _(SpectralClustering, TfidfVectorizer):
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
    document_labels = SpectralClustering(
        n_clusters=4,
        affinity="nearest_neighbors",
        n_neighbors=8,
        assign_labels="kmeans",
        random_state=0,
        n_jobs=1,
    ).fit_predict(document_matrix)
    for _cluster in sorted(set(document_labels)):
        print(f"Cluster {_cluster}")
        for _sentence in (documents[i] for i, label in enumerate(document_labels) if label == _cluster):
            print(f"  {_sentence}")
    document_sizes = {int(_cluster): int((document_labels == _cluster).sum()) for _cluster in set(document_labels)}
    print(f"{len(documents)} sentences in {len(document_sizes)} clusters: {document_sizes}")
    return document_sizes


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Which value of `affinity` builds the neighbor graph? Replace `None` with that string.
    """)
    return


@app.cell
def _():
    student_affinity = None
    print(student_affinity)
    return (student_affinity,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r'''
```python
student_affinity = "nearest_neighbors"
```
**nearest_neighbors.** Each point is joined to `n_neighbors` nearby points. The clustering then works on that graph.
''')
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(circle_kmeans_ari, circle_spectral_ari, document_sizes, mo, moon_scores, spiral_scores):
    mo.md(f"""
    ## Conclusions

    **Key takeaways:**

    - On the moons, k-means at k = 2 has adjusted Rand index {moon_scores.loc[2, "kmeans_ari"]:.3f}. Spectral clustering at the same k has {moon_scores.loc[2, "spectral_ari"]:.3f}.
    - On the rings, k-means has adjusted Rand index {circle_kmeans_ari:.3f} and spectral clustering has {circle_spectral_ari:.3f}. The spectral fit uses 40 neighbors so the graph stays connected.
    - On the spirals, k-means at k = 2 has adjusted Rand index {spiral_scores.loc[2, "kmeans_ari"]:.3f}. Spectral clustering has {spiral_scores.loc[2, "spectral_ari"]:.3f}.
    - The {sum(document_sizes.values())} sentences are split into groups of sizes {", ".join(str(size) for size in document_sizes.values())}. Read the printed lists for the shared subject. The cluster numbers themselves are arbitrary.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check your understanding

    1. Why does a nearest center mix the two moons?
    2. What does `n_neighbors` control?
    3. What does a disconnected neighbor graph warn you about?
    4. Why can the adjusted Rand index ignore the names of the clusters?
    5. What does the TF–IDF step produce?
    6. Why is a sentence that mentions both water and the sun a hard case?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
1. **The moons are not round blobs:** Points on opposite moons can be closer to the same center than to the rest of their own moon.
2. **The size of the local graph:** Each point keeps that many neighbors. A larger value connects more of the shape.
3. **The embedding can split the data into pieces the graph never joins:** Raising `n_neighbors` is the usual response when the shapes are far apart.
4. **It compares pairs of points:** Two points that belong together in the truth should belong together in the clustering, whatever the labels are called.
5. **One row of word weights per sentence:** Words that appear in many sentences get less weight.
6. **It shares words with two themes:** The neighbor graph has to put it with one group.
""")
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - [SpectralClustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html)
    - [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
    - VanderPlas, J. (2017). *Python Data Science Handbook*. O'Reilly Media. Chapter 5.
    """)
    return


if __name__ == "__main__":
    app.run()
