# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
#     "pandas",
#     "plotly",
#     "pydataset",
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
        calinski_harabasz_score,
        cv,
        data,
        davies_bouldin_score,
        estimate_bandwidth,
        mo,
        np,
        pd,
        silhouette_score,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Clustering a Restaurant Bill

    ## Objectives

    - Cluster parties by the bill and the tip.
    - Compare k-means, spectral clustering, agglomerative clustering, affinity propagation, and mean shift on those two columns.
    - Look at the smallest spectral group, set those rows aside, and score the fits again.

    ## Background

    The tips table records one restaurant party per row. This lesson uses only `total_bill` and `tip`. Sex, smoker, day, time, and party size stay in the table so the small group can be described, but they do not enter the distance.

    The two money columns are standardized on every row. There is no holdout. Mean shift uses a bandwidth estimated at quantile 0.3, with `random_state=0`, so the radius is computed rather than left at a guessed value of 1. A second round drops a few parties and keeps the z-scores from the first scaler, so those dropped bills are still inside the original mean and standard deviation.

    `ClusterVisualizer` stores the scaled bill and tip, which is the same table the models see. Its bars therefore score the fit. Silhouette and Calinski–Harabasz prefer a high score. Davies–Bouldin prefers a low score.

    ## Datasets Used

    - **Tips.** 244 parties from `pydataset`: bill, tip, sex, smoker, day, time, and party size.
    """)
    return


@app.cell
def _(
    calinski_harabasz_score,
    davies_bouldin_score,
    np,
    pd,
    silhouette_score,
):
    def score_labels(frame, label_map):
        """Score each clustering on the columns the model was fit on.

        frame: the columns passed to fit.
        label_map: a name for each label array.
        A fit with fewer than two labels is left blank.
        """
        rows = []
        for _name, _labels in label_map.items():
            _labels = np.asarray(_labels)
            if len(set(_labels.tolist())) < 2:
                rows.append(
                    {
                        "fit": _name,
                        "silhouette": np.nan,
                        "davies_bouldin": np.nan,
                        "calinski_harabasz": np.nan,
                    }
                )
                continue
            rows.append(
                {
                    "fit": _name,
                    "silhouette": silhouette_score(frame, _labels),
                    "davies_bouldin": davies_bouldin_score(frame, _labels),
                    "calinski_harabasz": calinski_harabasz_score(frame, _labels),
                }
            )
        return pd.DataFrame(rows).set_index("fit")

    return (score_labels,)


@app.cell
def _(
    AffinityPropagation,
    AgglomerativeClustering,
    KMeans,
    MeanShift,
    SpectralClustering,
    StandardScaler,
    cv,
    data,
    estimate_bandwidth,
    pd,
    score_labels,
):
    tips = data("tips")
    money = tips[["total_bill", "tip"]]
    scaled = pd.DataFrame(
        StandardScaler().fit_transform(money),
        columns=money.columns,
        index=money.index,
    )
    party_view = cv.ClusterVisualizer(scaled)
    labels = {
        "k-means 2": KMeans(n_clusters=2, n_init="auto", random_state=123).fit_predict(
            scaled
        ),
        "k-means 3": KMeans(n_clusters=3, n_init="auto", random_state=123).fit_predict(
            scaled
        ),
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
    print(f"{len(tips)} parties")
    print(first_scores.round(3).to_string())
    party_view.plot_data(title="Bill and tip, standardized")
    return (
        calinski_winner,
        davies_winner,
        first_scores,
        labels,
        party_view,
        scaled,
        silhouette_winner,
        tips,
    )


@app.cell
def _(labels, party_view):
    party_view.plot_clusters(labels["k-means 2"], title="k-means 2")
    return


@app.cell
def _(labels, party_view):
    party_view.plot_clusters(labels["k-means 3"], title="k-means 3")
    return


@app.cell
def _(labels, party_view):
    party_view.plot_clusters(labels["spectral 2"], title="spectral 2")
    return


@app.cell
def _(labels, party_view):
    party_view.plot_clusters(labels["spectral 3"], title="spectral 3")
    return


@app.cell
def _(labels, party_view):
    party_view.plot_clusters(labels["agglomerative 2"], title="agglomerative 2")
    return


@app.cell
def _(labels, party_view):
    party_view.plot_clusters(labels["agglomerative 3"], title="agglomerative 3")
    return


@app.cell
def _(labels, party_view):
    party_view.plot_clusters(labels["affinity"], title="affinity propagation")
    return


@app.cell
def _(labels, party_view):
    party_view.plot_clusters(labels["mean shift"], title="mean shift")
    return


@app.cell
def _(labels, party_view, pd):
    party_view.plot_silhouette(pd.DataFrame(labels), title="Silhouette, all parties")
    return


@app.cell
def _(labels, party_view, pd):
    party_view.plot_davies_bouldin(
        pd.DataFrame(labels), title="Davies-Bouldin, all parties"
    )
    return


@app.cell
def _(labels, party_view, pd):
    party_view.plot_calinski_harabasz(
        pd.DataFrame(labels), title="Calinski-Harabasz, all parties"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The two-cluster spectral fit is the one the lesson inspects, whether or not it wins every score. The smallest of those two groups is printed in the original dollars, with the party attributes that were not used in the distance.
    """)
    return


@app.cell
def _(labels, pd, scaled, tips):
    spectral_labels = labels["spectral 2"]
    small_label = pd.Series(spectral_labels).value_counts().idxmin()
    small_index = scaled.index[spectral_labels == small_label]
    small_parties = tips.loc[small_index]
    small_count = len(small_parties)
    print(f"Smallest spectral-2 cluster: label {small_label}, {small_count} parties")
    print(small_parties.to_string(index=False))
    scaled_kept = scaled.drop(index=small_index)
    return scaled_kept, small_count, small_parties


@app.cell
def _(
    AffinityPropagation,
    AgglomerativeClustering,
    KMeans,
    MeanShift,
    SpectralClustering,
    cv,
    estimate_bandwidth,
    pd,
    scaled_kept,
    score_labels,
):
    second_labels = {
        "k-means 2": KMeans(n_clusters=2, n_init="auto", random_state=123).fit_predict(
            scaled_kept
        ),
        "k-means 3": KMeans(n_clusters=3, n_init="auto", random_state=123).fit_predict(
            scaled_kept
        ),
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
        "mean shift": MeanShift(
            bandwidth=estimate_bandwidth(
                scaled_kept, quantile=0.3, n_samples=len(scaled_kept), random_state=0
            )
        ).fit_predict(scaled_kept),
    }
    second_view = cv.ClusterVisualizer(scaled_kept)
    second_scores = score_labels(scaled_kept, second_labels)
    second_silhouette = second_scores["silhouette"].idxmax()
    second_davies = second_scores["davies_bouldin"].idxmin()
    second_calinski = second_scores["calinski_harabasz"].idxmax()
    print(second_scores.round(3).to_string())
    second_view.plot_data(title="After dropping the small spectral group")
    return (
        second_calinski,
        second_davies,
        second_labels,
        second_scores,
        second_silhouette,
        second_view,
    )


@app.cell
def _(second_labels, second_view):
    second_view.plot_clusters(second_labels["k-means 2"], title="k-means 2, second round")
    return


@app.cell
def _(second_labels, second_view):
    second_view.plot_clusters(second_labels["spectral 2"], title="spectral 2, second round")
    return


@app.cell
def _(second_labels, second_view):
    second_view.plot_clusters(
        second_labels["mean shift"], title="mean shift, second round"
    )
    return


@app.cell
def _(second_labels, second_view, pd):
    second_view.plot_silhouette(
        pd.DataFrame(second_labels), title="Silhouette, after the small group is removed"
    )
    return


@app.cell
def _(second_labels, second_view, pd):
    second_view.plot_davies_bouldin(
        pd.DataFrame(second_labels),
        title="Davies-Bouldin, after the small group is removed",
    )
    return


@app.cell
def _(second_labels, second_view, pd):
    second_view.plot_calinski_harabasz(
        pd.DataFrame(second_labels),
        title="Calinski-Harabasz, after the small group is removed",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Which of the three scores should be small when the clusters are compact and well separated? Replace `None` with its name.
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
**davies_bouldin.** Silhouette and Calinski–Harabasz are read the other way: a larger score is the better separation. The winners in the conclusions are `idxmax` or `idxmin` on this run, not a fixed name.
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
    tips,
):
    mo.md(f"""
    ## Conclusions

    **Key takeaways:**

    - On all {len(tips)} parties, silhouette prefers **{silhouette_winner}** ({first_scores.loc[silhouette_winner, "silhouette"]:.3f}). Davies–Bouldin prefers **{davies_winner}** ({first_scores.loc[davies_winner, "davies_bouldin"]:.3f}). Calinski–Harabasz prefers **{calinski_winner}** ({first_scores.loc[calinski_winner, "calinski_harabasz"]:.0f}).
    - The two-cluster spectral fit has a smallest group of {small_count} parties. Those rows are printed above, in dollars, with the party attributes that were not used in the distance.
    - After those parties are removed, silhouette prefers **{second_silhouette}** ({second_scores.loc[second_silhouette, "silhouette"]:.3f}). Davies–Bouldin prefers **{second_davies}**. Calinski–Harabasz prefers **{second_calinski}**.
    - The second-round z-scores still use the mean and standard deviation from the full table, including the parties that were removed. The bars score that same scaled table.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check your understanding

    1. Which two columns enter the clustering?
    2. Why are those columns standardized?
    3. Which score is better when it is smaller?
    4. What is special about the smallest spectral group of two clusters?
    5. Do the second-round z-scores get a new mean and standard deviation?
    6. Why can the three scores name different winners?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
1. **total_bill and tip:** The other columns describe the parties. They are printed for the small group and they are not in the distance.
2. **A bill of 50 dollars would dwarf a tip of 5:** Standardization gives each column a standard deviation of 1. Every party is used, because there is no holdout.
3. **Davies–Bouldin:** Silhouette and Calinski–Harabasz prefer larger values.
4. **It is a handful of large bills:** The printed rows are the smallest spectral-2 group. A score can look strong because of that handful.
5. **No:** The second round keeps the first scaler. The dropped parties still sit inside the original mean and standard deviation.
6. **They punish different shapes:** One score can reward a tiny, very tight group that another score does not prefer. The names in the conclusions are whichever row wins on this run.
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
    - [Tips](https://vincentarelbundock.github.io/Rdatasets/doc/reshape2/tips.html)
    """)
    return


if __name__ == "__main__":
    app.run()
