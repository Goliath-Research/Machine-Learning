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
    from sklearn.cluster import KMeans, SpectralClustering
    from sklearn.datasets import make_circles, make_moons
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import adjusted_rand_score

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
    )


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

    Every scatter below is a `ClusterVisualizer` plot, so you can zoom. `random_state=0` and `n_jobs=1` keep the neighbor-graph fits stable on this machine. The adjusted Rand index compares a clustering with the known arms or rings. It is 1 when the groups match, and near 0 when the match is no better than chance. The index does not care which group is called 0.

    ## Datasets Used

    - **Half moons.** 300 points from `make_moons`, noise 0.05, `random_state=10`.
    - **Concentric circles.** 400 points from `make_circles`, noise 0.1, factor 0.2, `random_state=123`.
    - **Two spirals.** 300 points on each arm.
    - **Short sentences** about weather, animals, royalty, water, and Python.
    """)
    return


@app.cell
def _(np):
    def create_spiral_data(n_points, noise=0.8):
        """Generate two spiral arms of n_points each."""
        np.random.seed(12345)
        n = np.sqrt(np.random.rand(n_points, 1)) * 360 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.rand(n_points, 1) * noise
        d1y = np.sin(n) * n + np.random.rand(n_points, 1) * noise
        return np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y))))

    return (create_spiral_data,)


@app.cell
def _(
    KMeans,
    SpectralClustering,
    adjusted_rand_score,
    cv,
    make_moons,
    pd,
):
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
            n_neighbors=20,
            assign_labels="kmeans",
            random_state=0,
            n_jobs=1,
        ).fit_predict(moons)
        moon_rows.append(
            {
                "k": _k,
                "kmeans_ari": adjusted_rand_score(moon_arms, moon_labels[f"k-means {_k}"]),
                "spectral_ari": adjusted_rand_score(
                    moon_arms, moon_labels[f"spectral {_k}"]
                ),
            }
        )
    moon_scores = pd.DataFrame(moon_rows).set_index("k")
    print(moon_scores.round(3).to_string())
    moon_view.plot_data(title="Half Moons")
    return moon_labels, moon_scores, moon_view


@app.cell
def _(moon_labels, moon_view):
    moon_view.plot_clusters(
        moon_labels["k-means 2"], title="Half Moons: k-Means Clustering (k=2)"
    )
    return


@app.cell
def _(moon_labels, moon_view):
    moon_view.plot_clusters(
        moon_labels["spectral 2"], title="Half Moons: Spectral Clustering (k=2)"
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
        moon_labels["spectral 3"], title="Half Moons: Spectral Clustering (k=3)"
    )
    return


@app.cell
def _(moon_labels, moon_view):
    moon_view.plot_clusters(
        moon_labels["k-means 6"], title="Half Moons: k-Means Clustering (k=6)"
    )
    return


@app.cell
def _(moon_labels, moon_view):
    moon_view.plot_clusters(
        moon_labels["spectral 6"], title="Half Moons: Spectral Clustering (k=6)"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Concentric circles

    Twenty neighbors leaves this graph disconnected. The warning is the lesson. The next fit raises `n_neighbors` to 40, which is the connected graph drawn in the notebook.
    """)
    return


@app.cell
def _(
    KMeans,
    SpectralClustering,
    adjusted_rand_score,
    cv,
    make_circles,
    pd,
):
    circles, circle_rings = make_circles(
        n_samples=400, random_state=123, noise=0.1, factor=0.2
    )
    circle_view = cv.ClusterVisualizer(pd.DataFrame(circles, columns=["x", "y"]))
    circle_kmeans = KMeans(n_clusters=2, random_state=10, n_init="auto").fit_predict(
        circles
    )
    circle_spectral_20 = SpectralClustering(
        n_clusters=2,
        affinity="nearest_neighbors",
        n_neighbors=20,
        assign_labels="kmeans",
        random_state=0,
        n_jobs=1,
    ).fit_predict(circles)
    circle_spectral_40 = SpectralClustering(
        n_clusters=2,
        affinity="nearest_neighbors",
        n_neighbors=40,
        assign_labels="kmeans",
        random_state=0,
        n_jobs=1,
    ).fit_predict(circles)
    circle_kmeans_ari = adjusted_rand_score(circle_rings, circle_kmeans)
    circle_spectral_20_ari = adjusted_rand_score(circle_rings, circle_spectral_20)
    circle_spectral_ari = adjusted_rand_score(circle_rings, circle_spectral_40)
    print(f"k-means ARI {circle_kmeans_ari:.3f}")
    print(f"spectral, 20 neighbors, ARI {circle_spectral_20_ari:.3f}")
    print(f"spectral, 40 neighbors, ARI {circle_spectral_ari:.3f}")
    circle_view.plot_data(title="Concentric Circles")
    return (
        circle_kmeans,
        circle_kmeans_ari,
        circle_spectral_20,
        circle_spectral_20_ari,
        circle_spectral_40,
        circle_spectral_ari,
        circle_view,
    )


@app.cell
def _(circle_kmeans, circle_view):
    circle_view.plot_clusters(
        circle_kmeans, title="Concentric Circles: k-Means Clustering (k=2)"
    )
    return


@app.cell
def _(circle_spectral_20, circle_view):
    circle_view.plot_clusters(
        circle_spectral_20,
        title="Concentric Circles: Spectral Clustering (k=2, 20 neighbors)",
    )
    return


@app.cell
def _(circle_spectral_40, circle_view):
    circle_view.plot_clusters(
        circle_spectral_40, title="Concentric Circles: Spectral Clustering (k=2)"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Spirals

    `create_spiral_data` stacks one arm, then the other. The first 300 rows are one arm and the next 300 are the other. Those two blocks are the labels used for the adjusted Rand index. k-means still sees only coordinates.
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
            n_neighbors=10,
            assign_labels="kmeans",
            random_state=0,
            n_jobs=1,
        ).fit_predict(spirals)
        spiral_rows.append(
            {
                "k": _k,
                "kmeans_ari": adjusted_rand_score(
                    spiral_arms, spiral_labels[f"k-means {_k}"]
                ),
                "spectral_ari": adjusted_rand_score(
                    spiral_arms, spiral_labels[f"spectral {_k}"]
                ),
            }
        )
    spiral_scores = pd.DataFrame(spiral_rows).set_index("k")
    print(spiral_scores.round(3).to_string())
    spiral_view.plot_data(title="Spiral-Shaped Data")
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
        spiral_labels["spectral 2"],
        title="Spiral-Shaped Data: Spectral Clustering (k=2)",
    )
    return


@app.cell
def _(spiral_labels, spiral_view):
    spiral_view.plot_clusters(
        spiral_labels["k-means 3"],
        title="Spiral-Shaped Data: k-Means Clustering (k=3)",
    )
    return


@app.cell
def _(spiral_labels, spiral_view):
    spiral_view.plot_clusters(
        spiral_labels["spectral 3"],
        title="Spiral-Shaped Data: Spectral Clustering (k=3)",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sentences

    `TfidfVectorizer` turns each sentence into word weights and drops common English words. Spectral clustering then uses the neighbor graph of those weights. The cluster numbers are arbitrary. Read the lists.
    """)
    return


@app.cell
def _(SpectralClustering, TfidfVectorizer, np):
    def spectral_clustering_docs(X, documents, n_clusters=2, n_neighbors=10):
        """Perform spectral clustering on the document matrix and print each cluster."""
        clustering = SpectralClustering(
            n_clusters=n_clusters,
            affinity="nearest_neighbors",
            n_neighbors=n_neighbors,
            assign_labels="kmeans",
            random_state=0,
            n_jobs=1,
        )
        labels = clustering.fit_predict(X)
        for i in range(n_clusters):
            print(f"Cluster {i}:")
            for doc_index in np.where(labels == i)[0]:
                print(f" - {documents[doc_index]}")
        return labels

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
    print(f"{len(documents)} sentences, {document_matrix.shape[1]} word weights")
    print("\nAsking for 4 clusters")
    labels4 = spectral_clustering_docs(document_matrix, documents, n_clusters=4)
    document_sizes = {
        int(_cluster): int((labels4 == _cluster).sum()) for _cluster in set(labels4)
    }

    def _subject(sentence):
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

    mixed = []
    for _cluster in sorted(set(labels4)):
        _subjects = sorted(
            {
                _subject(documents[_i])
                for _i, _label in enumerate(labels4)
                if _label == _cluster
            }
        )
        if len(_subjects) > 1:
            mixed.append(
                f"Cluster {_cluster} mixes {', '.join(_subjects)}"
            )
    document_mix = (
        "; ".join(mixed) if mixed else "Each printed cluster stays on one subject"
    )
    print(document_sizes)
    print(document_mix)
    return document_mix, document_sizes, documents


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Which value of `affinity` builds the neighbor graph used on the moons, rings, and spirals? Replace `None` with that string.
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
**nearest_neighbors.** Each point is joined to `n_neighbors` nearby points. The clustering then works on that graph. On the rings, 20 neighbors is not enough to keep the graph in one piece, and 40 is.
''')
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(
    circle_kmeans_ari,
    circle_spectral_20_ari,
    circle_spectral_ari,
    document_mix,
    document_sizes,
    documents,
    mo,
    moon_scores,
    spiral_scores,
):
    mo.md(f"""
    ## Conclusions

    **Key takeaways:**

    - On the moons, k-means at k = 2 has adjusted Rand index {moon_scores.loc[2, "kmeans_ari"]:.3f}. Spectral clustering with 20 neighbors has {moon_scores.loc[2, "spectral_ari"]:.3f}.
    - On the rings, k-means has adjusted Rand index {circle_kmeans_ari:.3f}. Spectral clustering with 20 neighbors has {circle_spectral_20_ari:.3f}. With 40 neighbors it has {circle_spectral_ari:.3f}.
    - On the spirals, k-means at k = 2 has adjusted Rand index {spiral_scores.loc[2, "kmeans_ari"]:.3f}. Spectral clustering with 10 neighbors has {spiral_scores.loc[2, "spectral_ari"]:.3f}.
    - The {len(documents)} sentences, asked for 4 clusters, land in groups of sizes {", ".join(str(size) for size in document_sizes.values())}. {document_mix}. The cluster numbers themselves are arbitrary.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check your understanding

    1. Why does k-means mix the two moons?
    2. What does `affinity="nearest_neighbors"` build?
    3. What does the disconnected-graph warning mean for the rings?
    4. Why can the adjusted Rand index be 1 when the cluster numbers differ from the true arm numbers?
    5. What does the sentence clustering actually compare?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
1. **It cuts by nearest center:** The moons bend around each other, so a straight cut through the middle puts points from both arms in one cluster.
2. **A graph:** Each point keeps `n_neighbors` neighbors. Spectral clustering then separates pieces of that graph.
3. **The rings are not one connected graph at 20 neighbors:** The embedding can fail to follow the rings. Raising the neighbor count to 40 joins the graph.
4. **It compares pairs, not names:** Two points on the same arm should share a cluster, whatever that cluster is called.
5. **Word weights:** `TfidfVectorizer` turns the sentences into numbers. Neighbors are sentences with similar weights, after common English words are dropped.
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
    - VanderPlas, J. (2017). *Python Data Science Handbook*. O'Reilly Media. Chapter 5.
    """)
    return


if __name__ == "__main__":
    app.run()
