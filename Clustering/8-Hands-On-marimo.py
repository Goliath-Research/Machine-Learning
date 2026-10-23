# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "pydataset",
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
    from pydataset import data
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

    return (
        AffinityPropagation,
        AgglomerativeClustering,
        KMeans,
        MeanShift,
        SpectralClustering,
        StandardScaler,
        estimate_bandwidth,
        calinski_harabasz_score,
        data,
        davies_bouldin_score,
        silhouette_score,
    )


@app.cell
def _(np, pd, sns):
    def draw_clusters(axes, x, y, labels, title):
        frame = pd.DataFrame(
            {"x": np.asarray(x), "y": np.asarray(y), "cluster": np.asarray(labels).astype(str)}
        )
        sns.scatterplot(data=frame, x="x", y="y", hue="cluster", s=28, ax=axes, legend=False)
        axes.set_title(title)
        axes.set_xlabel("total bill, z")
        axes.set_ylabel("tip, z")

    return (draw_clusters,)


@app.cell
def _(
    calinski_harabasz_score,
    davies_bouldin_score,
    np,
    pd,
    silhouette_score,
):
    def score_labels(matrix, label_map):
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
                    "silhouette": silhouette_score(matrix, labels),
                    "davies_bouldin": davies_bouldin_score(matrix, labels),
                    "calinski_harabasz": calinski_harabasz_score(matrix, labels),
                }
            )
        return pd.DataFrame(rows).set_index("fit")

    return (score_labels,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Clustering a Restaurant Bill

    ## Objectives

    - Cluster bills by the amount of the bill and the tip.
    - Compare k-means, spectral clustering, agglomerative clustering, affinity propagation, and mean shift on the same two columns.
    - Look at the smallest spectral group, set those rows aside, and score the fits again.

    ## Background

    The tips table records one restaurant party per row. This lesson uses only `total_bill` and `tip`. Sex, smoker, day, time, and party size stay in the table so the small group can be described, but they do not enter the distance.

    The two money columns are standardized on every row. There is no holdout in this clustering. Mean shift uses a bandwidth estimated at quantile 0.3, with `random_state=0` so the radius does not change between runs. A second round drops a few parties and keeps the z-scores from the first scaler, so those dropped bills are still inside the original mean and standard deviation.

    ## Datasets Used

    - **Tips.** 244 parties from `pydataset`: bill, tip, sex, smoker, day, time, and party size.
    """)
    return


@app.cell
def _(
    AffinityPropagation,
    AgglomerativeClustering,
    KMeans,
    MeanShift,
    SpectralClustering,
    StandardScaler,
    data,
    estimate_bandwidth,
    draw_clusters,
    pd,
    plt,
    score_labels,
):
    tips = data("tips")
    money = tips[["total_bill", "tip"]]
    scaled = pd.DataFrame(
        StandardScaler().fit_transform(money),
        columns=money.columns,
        index=money.index,
    )
    labels = {
        "k-means 2": KMeans(n_clusters=2, n_init="auto", random_state=123).fit_predict(scaled),
        "k-means 3": KMeans(n_clusters=3, n_init="auto", random_state=123).fit_predict(scaled),
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
        "mean shift": MeanShift(
            bandwidth=estimate_bandwidth(
                scaled, quantile=0.3, n_samples=len(scaled), random_state=0
            )
        ).fit_predict(scaled),
    }
    first_scores = score_labels(scaled, labels)
    silhouette_winner = first_scores["silhouette"].idxmax()
    davies_winner = first_scores["davies_bouldin"].idxmin()
    calinski_winner = first_scores["calinski_harabasz"].idxmax()
    _figure, _axes = plt.subplots(2, 4, figsize=(12, 6))
    for _ax, (_name, _lab) in zip(_axes.ravel(), labels.items()):
        draw_clusters(_ax, scaled["total_bill"], scaled["tip"], _lab, _name)
    _figure.tight_layout()
    print(f"{len(tips)} parties")
    print(first_scores.round(3).to_string())
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Silhouette and Calinski–Harabasz prefer a high score. Davies–Bouldin prefers a low score. The names above the bars in the conclusions are those winners, taken from this run.

    The notebook then looks at spectral clustering with two clusters, even when another fit wins a score. The smallest of those two groups is printed in the original units.
    """)
    return


@app.cell
def _(labels, pd, scaled, tips):
    spectral_labels = labels["spectral 2"]
    small_label = pd.Series(spectral_labels).value_counts().idxmin()
    small_parties = tips.loc[spectral_labels == small_label].copy()
    small_count = len(small_parties)
    print(f"Smallest spectral-2 cluster: label {small_label}, {small_count} parties")
    print(small_parties.to_string(index=False))
    kept = spectral_labels != small_label
    scaled_kept = scaled.loc[kept]


@app.cell
def _(
    AffinityPropagation,
    AgglomerativeClustering,
    KMeans,
    MeanShift,
    SpectralClustering,
    draw_clusters,
    estimate_bandwidth,
    plt,
    scaled_kept,
    score_labels,
):
    second_labels = {
        "k-means 2": KMeans(n_clusters=2, n_init="auto", random_state=123).fit_predict(scaled_kept),
        "k-means 3": KMeans(n_clusters=3, n_init="auto", random_state=123).fit_predict(scaled_kept),
        "spectral 2": SpectralClustering(
            n_clusters=2, assign_labels="kmeans", random_state=123, n_jobs=1
        ).fit_predict(scaled_kept),
        "spectral 3": SpectralClustering(
            n_clusters=3, assign_labels="kmeans", random_state=123, n_jobs=1
        ).fit_predict(scaled_kept),
        "agglomerative 2": AgglomerativeClustering(n_clusters=2).fit_predict(scaled_kept),
        "agglomerative 3": AgglomerativeClustering(n_clusters=3).fit_predict(scaled_kept),
        "affinity": AffinityPropagation(
            random_state=42, preference=-45, max_iter=500
        ).fit_predict(scaled_kept),
        "mean shift": MeanShift(
            bandwidth=estimate_bandwidth(
                scaled_kept, quantile=0.3, n_samples=len(scaled_kept), random_state=0
            )
        ).fit_predict(scaled_kept),
    }
    second_scores = score_labels(scaled_kept, second_labels)
    second_silhouette = second_scores["silhouette"].idxmax()
    second_davies = second_scores["davies_bouldin"].idxmin()
    second_calinski = second_scores["calinski_harabasz"].idxmax()
    _figure, _axes = plt.subplots(2, 4, figsize=(12, 6))
    for _ax, (_name, _lab) in zip(_axes.ravel(), second_labels.items()):
        draw_clusters(
            _ax, scaled_kept["total_bill"], scaled_kept["tip"], _lab, _name
        )
    _figure.tight_layout()
    print(second_scores.round(3).to_string())
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Which score should be small when the clusters are compact and well separated? Replace `None` with its name.
    """)
    return


@app.cell
def _():
    student_compact = None
    print(student_compact)
    return (student_compact,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r'''
```python
student_compact = "davies_bouldin"
```
**davies_bouldin.** Silhouette and Calinski–Harabasz are read in the other direction: a larger score is the better separation on those two.
''')
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(
    calinski_winner,
    davies_winner,
    first_scores,
    mo,
    second_calinski,
    second_davies,
    second_scores,
    second_silhouette,
    silhouette_winner,
    small_count,
):
    mo.md(f"""
    ## Conclusions

    **Key takeaways:**

    - On all {244} parties, silhouette prefers **{silhouette_winner}** ({first_scores.loc[silhouette_winner, "silhouette"]:.3f}). Davies–Bouldin prefers **{davies_winner}** ({first_scores.loc[davies_winner, "davies_bouldin"]:.3f}). Calinski–Harabasz prefers **{calinski_winner}** ({first_scores.loc[calinski_winner, "calinski_harabasz"]:.0f}).
    - The two-cluster spectral fit has a smallest group of {small_count} parties. Those rows are printed above, in dollars, with the party attributes that were not used in the distance.
    - After those parties are removed, silhouette prefers **{second_silhouette}** ({second_scores.loc[second_silhouette, "silhouette"]:.3f}). Davies–Bouldin prefers **{second_davies}**. Calinski–Harabasz prefers **{second_calinski}**.
    - The second-round z-scores still use the mean and standard deviation from the full table, including the parties that were removed.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check your understanding

    1. Which two columns enter the clustering?
    2. Why are those columns standardized?
    3. Which direction is a better Davies–Bouldin score?
    4. Why might the three scores name different fits?
    5. What is special about the parties printed from the spectral fit?
    6. What did the second-round scaler still include?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
1. **`total_bill` and `tip`:** Day, time, sex, smoker, and party size are used only to describe a group after it is found.
2. **The bill is a larger number than the tip:** Without standardization, distance is mostly a distance in the bill.
3. **Down:** A smaller Davies–Bouldin score means each cluster is less like its nearest neighbor.
4. **They reward different shapes:** One fit can have a higher silhouette and a worse Davies–Bouldin score. The table is the comparison.
5. **They are the smaller of the two spectral groups:** The fit was not told to find outliers. The small group is just the cluster with fewer parties.
6. **The removed parties:** Their bills and tips were part of the mean and the standard deviation. The second round did not compute those again.
""")
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - [Clustering](https://scikit-learn.org/stable/modules/clustering.html)
    - [Tips dataset](https://vincentarelbundock.github.io/Rdatasets/doc/reshape2/tips.html), from the R datasets used by `pydataset`
    - VanderPlas, J. (2017). *Python Data Science Handbook*. O'Reilly Media. Chapter 5.
    """)
    return


if __name__ == "__main__":
    app.run()
