# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "plotly",
#     "scikit-learn",
#     "scipy",
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
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
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
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Hierarchical Clustering

    ## Objectives

    - Read a dendrogram from the joins at the bottom to the last merge at the top.
    - Compare ward, single, complete, and average linkage on the same cloud.
    - Score those clusterings against known labels, then repeat the comparison on the iris measurements.

    ## Background

    Hierarchical clustering joins points into larger groups. The height of a join is the distance between the groups being merged. A horizontal cut keeps every join below that height and produces one clustering.

    Linkage is the distance between two groups. Single linkage uses the closest pair. Complete linkage uses the farthest pair. Average linkage uses the mean of the pairs. Ward linkage chooses the merge that increases the within-group variance the least.

    The dendrogram is drawn with matplotlib and SciPy. Plotly did not draw dendrograms when these lessons were written, and `ClusterVisualizer` has no dendrogram method. Every later scatter, and every external score bar, goes through `ClusterVisualizer`.

    ## Datasets Used

    - **Ten blobs.** A tiny cloud from `make_blobs`, `random_state=2`, so every point can be numbered.
    - **Four hundred blobs.** Four centers, `cluster_std=1.8`, `random_state=10`, with the labels that generated them.
    - **Iris.** Four measurements of 150 flowers from three species, loaded from scikit-learn.
    """)
    return


@app.cell
def _(dendrogram, fcluster, linkage, np, plt):
    def plot_clusters_from_dendrogram(X, method="ward", threshold=1):
        """Plot the dendrogram with a threshold, and the associate scatter plot with the clusters."""
        data_labels = np.array([i for i in range(X.shape[0])])
        Z = linkage(X, method=method)
        plt.figure(figsize=(10, 4))
        plt.subplot(121)
        dendrogram(Z)
        plt.title(f"Dendrogram - ({method} method)")
        plt.xlabel("Data Points")
        plt.ylabel("Euclidean Distance")
        plt.grid(True, color="lightgrey", linestyle="-", linewidth=0.5)
        plt.axhline(y=threshold, color="grey", linestyle="--")
        plt.subplot(122)
        labels = fcluster(Z, threshold, criterion="distance")
        num_clusters = len(np.unique(labels))
        plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="gist_rainbow")
        plt.title("Scatter Plot - (%d clusters)" % num_clusters)
        for i, txt in enumerate(data_labels):
            plt.annotate(
                text=txt,
                xy=(X[i, 0], X[i, 1]),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
                color="k",
                size=9,
            )
        plt.grid(True, color="lightgrey", linestyle="-", linewidth=0.5)
        figure = plt.gcf()
        figure.tight_layout()
        return figure

    return (plot_clusters_from_dendrogram,)


@app.cell
def _(fcluster, linkage, make_blobs, np):
    tiny, _tiny_labels = make_blobs(random_state=2, n_samples=10)
    tiny_groups = fcluster(linkage(tiny, method="ward"), t=2, criterion="distance")
    tiny_cut = len(np.unique(tiny_groups))
    print(f"A cut at height 2 produces {tiny_cut} clusters")
    return tiny, tiny_cut


@app.cell
def _(plot_clusters_from_dendrogram, tiny):
    plot_clusters_from_dendrogram(tiny, threshold=2)
    return


@app.cell
def _(plot_clusters_from_dendrogram, tiny):
    plot_clusters_from_dendrogram(tiny, method="ward", threshold=4)
    return


@app.cell
def _(plot_clusters_from_dendrogram, tiny):
    plot_clusters_from_dendrogram(tiny, method="single", threshold=4)
    return


@app.cell
def _(plot_clusters_from_dendrogram, tiny):
    plot_clusters_from_dendrogram(tiny, method="complete", threshold=4)
    return


@app.cell
def _(plot_clusters_from_dendrogram, tiny):
    plot_clusters_from_dendrogram(tiny, method="average", threshold=4)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The same height does not mean the same number of clusters once the linkage changes. Single linkage can chain nearby points into one long group. Ward tends to keep the groups more even.

    The next cloud has labels from the generator. Each linkage is asked for four clusters and for three. The titles name the linkage that was actually fit. The adjusted Rand index, adjusted mutual information, normalized mutual information, and Fowlkes–Mallows score compare those labels with the generator labels. Higher is closer agreement. A score near 0 is no better than chance.
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
    wide_view.plot_data()
    return agg_labels, wide_labels, wide_view


@app.cell
def _(wide_view):
    wide_view.plot_data_with_true_labels()
    return


@app.cell
def _(agg_labels, wide_view):
    wide_view.plot_clusters(
        agg_labels["Ward-4"], title="Agglomerative Clustering - Ward Linkage, 4 clusters"
    )
    return


@app.cell
def _(agg_labels, wide_view):
    wide_view.plot_clusters(
        agg_labels["Ward-3"], title="Agglomerative Clustering - Ward Linkage, 3 clusters"
    )
    return


@app.cell
def _(agg_labels, wide_view):
    wide_view.plot_clusters(
        agg_labels["Single-4"],
        title="Agglomerative Clustering - Single Linkage, 4 clusters",
    )
    return


@app.cell
def _(agg_labels, wide_view):
    wide_view.plot_clusters(
        agg_labels["Single-3"],
        title="Agglomerative Clustering - Single Linkage, 3 clusters",
    )
    return


@app.cell
def _(agg_labels, wide_view):
    wide_view.plot_clusters(
        agg_labels["Complete-4"],
        title="Agglomerative Clustering - Complete Linkage, 4 clusters",
    )
    return


@app.cell
def _(agg_labels, wide_view):
    wide_view.plot_clusters(
        agg_labels["Complete-3"],
        title="Agglomerative Clustering - Complete Linkage, 3 clusters",
    )
    return


@app.cell
def _(agg_labels, wide_view):
    wide_view.plot_clusters(
        agg_labels["Average-4"],
        title="Agglomerative Clustering - Average Linkage, 4 clusters",
    )
    return


@app.cell
def _(agg_labels, wide_view):
    wide_view.plot_clusters(
        agg_labels["Average-3"],
        title="Agglomerative Clustering - Average Linkage, 3 clusters",
    )
    return


@app.cell
def _(agg_labels, wide_view):
    wide_view.plot_adjusted_rand(agg_labels, title="Agglomerative Clustering")
    return


@app.cell
def _(agg_labels, wide_view):
    wide_view.plot_adjusted_mutual_info(agg_labels, title="Agglomerative Clustering")
    return


@app.cell
def _(agg_labels, wide_view):
    wide_view.plot_normalized_mutual_info(agg_labels, title="Agglomerative Clustering")
    return


@app.cell
def _(agg_labels, wide_view):
    wide_view.plot_fowlkes_mallows(agg_labels, title="Agglomerative Clustering")
    return


@app.cell
def _(
    adjusted_mutual_info_score,
    adjusted_rand_score,
    agg_labels,
    fowlkes_mallows_score,
    normalized_mutual_info_score,
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
    print(link_scores.round(3).to_string())
    print(f"Highest adjusted Rand index: {best_link}")
    return best_link, link_scores


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Iris

    The four measurements are standardized together. There is no holdout, so every flower contributes to the means and the standard deviations. The clustering uses those four columns.

    t-SNE is only a picture. It is fit once, with `random_state=0`, and it is not the input to the clustering. Agglomerative clustering with ward linkage asks for three clusters. The bars below compare those labels with the species. They do not score the t-SNE coordinates.
    """)
    return


@app.cell
def _(
    AgglomerativeClustering,
    StandardScaler,
    TSNE,
    adjusted_mutual_info_score,
    adjusted_rand_score,
    cv,
    fowlkes_mallows_score,
    load_iris,
    normalized_mutual_info_score,
    pd,
):
    iris = load_iris()
    iris_frame = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_frame["species"] = iris.target_names[iris.target]
    iris_z = pd.DataFrame(
        StandardScaler().fit_transform(iris_frame[iris.feature_names]),
        columns=iris.feature_names,
    )
    iris_plane = pd.DataFrame(
        TSNE(n_components=2, random_state=0, init="pca").fit_transform(iris_z),
        columns=["t-SNE 1", "t-SNE 2"],
    )
    iris_labels = AgglomerativeClustering(n_clusters=3).fit_predict(iris_z)
    iris_view = cv.ClusterVisualizer(iris_plane, iris_frame["species"].to_numpy())
    iris_rand = adjusted_rand_score(iris_frame["species"], iris_labels)
    iris_ami = adjusted_mutual_info_score(iris_frame["species"], iris_labels)
    iris_nmi = normalized_mutual_info_score(iris_frame["species"], iris_labels)
    iris_fm = fowlkes_mallows_score(iris_frame["species"], iris_labels)
    iris_table = pd.crosstab(
        iris_frame["species"], pd.Series(iris_labels, name="cluster")
    )
    print(
        f"ARI {iris_rand:.3f}   AMI {iris_ami:.3f}   NMI {iris_nmi:.3f}   FMI {iris_fm:.3f}"
    )
    print(iris_table.to_string())
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Which linkage chooses the merge that increases the within-group variance the least? Replace `None` with its name.
    """)
    return


@app.cell
def _():
    student_linkage = None
    print(student_linkage)
    return (student_linkage,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r'''
```python
student_linkage = "ward"
```
**ward.** Single, complete, and average linkage use a distance between pairs. Ward uses the change in variance. On the labeled blobs, the score bars show which linkage agrees with the generator.
''')
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(
    best_link,
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
    _lines = "\n".join(
        f"- **{_species}** has most of its flowers in cluster {_cluster}."
        for _species, _cluster in _main.items()
    )
    mo.md(f"""
    ## Conclusions

    **Key takeaways:**

    - On the ten-point cloud, a ward cut at height 2 produces {tiny_cut} clusters. The four linkage pictures at height 4 do not have to produce the same count, because the joins sit at different heights.
    - On the labeled blobs, the highest adjusted Rand index is **{best_link}**, at {link_scores.loc[best_link, "adjusted_rand"]:.3f}. Its Fowlkes–Mallows score is {link_scores.loc[best_link, "fowlkes_mallows"]:.3f}. The bars are those four scores, drawn by `ClusterVisualizer` against the generator labels.
    - Asking for three clusters when four were generated lowers the agreement. The table shows the size of that drop for each linkage.
    - On the iris flowers, ward linkage with three clusters has adjusted Rand index {iris_rand:.3f}, adjusted mutual information {iris_ami:.3f}, normalized mutual information {iris_nmi:.3f}, and Fowlkes–Mallows score {iris_fm:.3f}.
    {_lines}
    - t-SNE only places the flowers for the drawing. The joins were computed on the four standardized measurements.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check your understanding

    1. What does the height of a join in the dendrogram mean?
    2. What does a horizontal cut do?
    3. How does single linkage measure the distance between two groups?
    4. Why can the same cut height produce different numbers of clusters?
    5. What are the four external scores comparing?
    6. Why is the t-SNE picture not the data that was clustered?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
1. **The distance at which two groups merged:** Taller joins happen later.
2. **It keeps the groups that had not yet merged above that height:** The number of vertical lines the cut crosses is the number of clusters.
3. **By the closest pair of points:** One bridge of nearby points can pull two groups together.
4. **The joins sit at different heights:** Linkage changes which groups merge first, so a fixed height can cross a different number of branches.
5. **The clustering and the known labels:** They do not look at the coordinates again. They ask whether points that share a true label share a cluster.
6. **t-SNE is a two-dimensional view:** The agglomerative fit used the standardized sepal and petal measurements.
""")
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - [AgglomerativeClustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html)
    - [Hierarchical clustering](https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering)
    - VanderPlas, J. (2017). *Python Data Science Handbook*. O'Reilly Media. Chapter 5.
    """)
    return


if __name__ == "__main__":
    app.run()
