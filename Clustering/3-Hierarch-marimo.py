# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "scikit-learn",
#     "seaborn",
#     "scipy",
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
        dendrogram,
        fcluster,
        fowlkes_mallows_score,
        linkage,
        load_iris,
        make_blobs,
        normalized_mutual_info_score,
    )


@app.cell
def _(np, pd, sns):
    def draw_clusters(axes, x, y, labels, title):
        frame = pd.DataFrame(
            {"x": np.asarray(x), "y": np.asarray(y), "cluster": np.asarray(labels).astype(str)}
        )
        sns.scatterplot(data=frame, x="x", y="y", hue="cluster", s=28, ax=axes, legend=False)
        axes.set_title(title)

    return (draw_clusters,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Hierarchical Clustering

    ## Objectives

    - Read a dendrogram from the joins at the bottom to the last merge at the top.
    - Compare ward, single, complete, and average linkage.
    - Score those clusterings against known labels, then repeat the idea on the iris measurements.

    ## Background

    Hierarchical clustering joins points into larger groups. The height of a join is the distance between the groups being merged. A horizontal cut keeps every join below that height and produces one clustering.

    Linkage is the distance between two groups. Single linkage uses the closest pair. Complete linkage uses the farthest pair. Average linkage uses the mean of the pairs. Ward linkage chooses the merge that increases the within-group variance the least.

    ## Datasets Used

    - **Ten blobs.** A tiny cloud from `make_blobs`, `random_state=2`, so every point can be numbered.
    - **Four hundred blobs.** Four centers, `cluster_std=1.8`, `random_state=10`, with the labels that generated them.
    - **Iris.** Four measurements of 150 flowers from three species, loaded from scikit-learn.
    """)
    return


@app.cell
def _(dendrogram, fcluster, linkage, make_blobs, np, plt):
    tiny, _tiny_labels = make_blobs(random_state=2, n_samples=10)
    tiny_link = linkage(tiny, method="ward")
    tiny_groups = fcluster(tiny_link, t=2, criterion="distance")
    print(f"A cut at height 2 produces {len(np.unique(tiny_groups))} clusters")
    _figure, _axes = plt.subplots(1, 2, figsize=(10, 4))
    dendrogram(tiny_link, ax=_axes[0])
    _axes[0].axhline(2, color="gray", linestyle="--")
    _axes[0].set_title("Ward dendrogram")
    _axes[0].set_ylabel("Distance")
    _axes[1].scatter(tiny[:, 0], tiny[:, 1], c=tiny_groups, cmap="tab10", s=40)
    for _i, _xy in enumerate(tiny):
        _axes[1].annotate(str(_i), _xy, textcoords="offset points", xytext=(0, 4), ha="center")
    _axes[1].set_title("Cut at height 2")
    _figure.tight_layout()
    return _figure


@app.cell
def _(dendrogram, fcluster, linkage, np, plt, tiny):
    _figure, _axes = plt.subplots(2, 2, figsize=(10, 8))
    for _ax, _method in zip(_axes.ravel(), ("ward", "single", "complete", "average")):
        _link = linkage(tiny, method=_method)
        _groups = fcluster(_link, t=4, criterion="distance")
        dendrogram(_link, ax=_ax)
        _ax.axhline(4, color="gray", linestyle="--")
        _ax.set_title(f"{_method}: {len(np.unique(_groups))} clusters below height 4")
    _figure.tight_layout()
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The same height does not mean the same number of clusters once the linkage changes. Single linkage can chain nearby points into one long group. Ward tends to keep the groups more even.

    The next cloud has labels from the generator. Each linkage is asked for four clusters and for three. The adjusted Rand index, adjusted mutual information, normalized mutual information, and Fowlkes–Mallows score all compare those labels with the generator labels. Higher is closer agreement. A score near 0 is no better than chance.
    """)
    return


@app.cell
def _(
    AgglomerativeClustering,
    adjusted_mutual_info_score,
    adjusted_rand_score,
    draw_clusters,
    fowlkes_mallows_score,
    make_blobs,
    normalized_mutual_info_score,
    pd,
    plt,
):
    wide, wide_labels = make_blobs(
        n_samples=400, centers=4, cluster_std=1.8, random_state=10
    )
    link_rows = []
    _figure, _axes = plt.subplots(2, 2, figsize=(9, 8))
    for _ax, _method in zip(_axes.ravel(), ("ward", "single", "complete", "average")):
        _labels = AgglomerativeClustering(n_clusters=4, linkage=_method).fit_predict(wide)
        draw_clusters(_ax, wide[:, 0], wide[:, 1], _labels, _method)
        link_rows.append({"linkage": _method, "clusters": 4, "labels": _labels})
    for _method in ("ward", "single", "complete", "average"):
        link_rows.append(
            {
                "linkage": _method,
                "clusters": 3,
                "labels": AgglomerativeClustering(n_clusters=3, linkage=_method).fit_predict(wide),
            }
        )
    score_rows = []
    for _row in link_rows:
        _labels = _row["labels"]
        score_rows.append(
            {
                "fit": f"{_row['linkage']}-{_row['clusters']}",
                "adjusted_rand": adjusted_rand_score(wide_labels, _labels),
                "adjusted_mutual_info": adjusted_mutual_info_score(wide_labels, _labels),
                "normalized_mutual_info": normalized_mutual_info_score(wide_labels, _labels),
                "fowlkes_mallows": fowlkes_mallows_score(wide_labels, _labels),
            }
        )
    link_scores = pd.DataFrame(score_rows).set_index("fit")
    best_link = link_scores["adjusted_rand"].idxmax()
    _figure.tight_layout()
    print(link_scores.round(3).to_string())
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Iris

    The four measurements are standardized together. There is no holdout, so every flower contributes to the means and the standard deviations. The clustering uses those four columns.

    t-SNE is only a picture. It is fit once, with `random_state=0`, and it is not the input to the clustering. Agglomerative clustering with ward linkage asks for three clusters, one for each species.
    """)
    return


@app.cell
def _(
    AgglomerativeClustering,
    StandardScaler,
    TSNE,
    adjusted_mutual_info_score,
    adjusted_rand_score,
    draw_clusters,
    fowlkes_mallows_score,
    load_iris,
    normalized_mutual_info_score,
    pd,
    plt,
):
    iris = load_iris()
    iris_frame = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_frame["species"] = iris.target_names[iris.target]
    iris_z = pd.DataFrame(
        StandardScaler().fit_transform(iris_frame[iris.feature_names]),
        columns=iris.feature_names,
    )
    iris_view = TSNE(n_components=2, random_state=0, init="pca").fit_transform(iris_z)
    iris_labels = AgglomerativeClustering(n_clusters=3).fit_predict(iris_z)
    iris_rand = adjusted_rand_score(iris_frame["species"], iris_labels)
    iris_ami = adjusted_mutual_info_score(iris_frame["species"], iris_labels)
    iris_nmi = normalized_mutual_info_score(iris_frame["species"], iris_labels)
    iris_fm = fowlkes_mallows_score(iris_frame["species"], iris_labels)
    iris_table = pd.crosstab(
        iris_frame["species"], pd.Series(iris_labels, name="cluster")
    )
    _figure, _axes = plt.subplots(1, 2, figsize=(10, 4))
    draw_clusters(
        _axes[0], iris_view[:, 0], iris_view[:, 1], iris_frame["species"], "Species, t-SNE view"
    )
    draw_clusters(
        _axes[1], iris_view[:, 0], iris_view[:, 1], iris_labels, "Ward, 3 clusters"
    )
    _figure.tight_layout()
    print(
        f"ARI {iris_rand:.3f}   AMI {iris_ami:.3f}   NMI {iris_nmi:.3f}   FMI {iris_fm:.3f}"
    )
    print(iris_table.to_string())
    return _figure


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
**ward.** Single, complete, and average linkage use a distance between pairs. Ward uses the change in variance.
''')
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(best_link, iris_ami, iris_fm, iris_nmi, iris_rand, iris_table, link_scores, mo):
    _main = iris_table.idxmax(axis=1)
    _lines = "\n".join(
        f"- **{_species}** has most of its flowers in cluster {_cluster}."
        for _species, _cluster in _main.items()
    )
    mo.md(f"""
    ## Conclusions

    **Key takeaways:**

    - On the labeled blobs, the highest adjusted Rand index is **{best_link}**, at {link_scores.loc[best_link, "adjusted_rand"]:.3f}. Its Fowlkes–Mallows score is {link_scores.loc[best_link, "fowlkes_mallows"]:.3f}.
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
