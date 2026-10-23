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
    from sklearn.datasets import load_iris
    from sklearn.decomposition import PCA
    from sklearn.manifold import MDS, Isomap, LocallyLinearEmbedding, TSNE
    from sklearn.preprocessing import StandardScaler

    return Isomap, LocallyLinearEmbedding, MDS, PCA, StandardScaler, TSNE, load_iris


@app.cell
def _(sns):
    def draw_species(axes, frame, x, y, title):
        sns.scatterplot(data=frame, x=x, y=y, hue="species", s=28, ax=axes, linewidth=0)
        axes.set_title(title)
        axes.legend(fontsize=7, markerscale=0.8)

    def outside_whiskers(column, value):
        q1, q3 = column.quantile((0.25, 0.75))
        iqr = q3 - q1
        return bool(value < q1 - 1.5 * iqr or value > q3 + 1.5 * iqr)

    return draw_species, outside_whiskers


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Iris in Two Dimensions

    ## Objectives

    - Standardize the four iris measurements and draw them with PCA, MDS, locally linear embedding, Isomap, and t-SNE.
    - Read the share of variance kept by the first two principal components.
    - Find the virginica flower nearest the versicolor center in the t-SNE plane, and compare its raw measurements with the virginica whiskers.

    ## Background

    Each method sees the four standardized measurements and returns two coordinates for the picture. The species colors are added afterward. The scaler is fit on all 150 flowers. There is no holdout. These pictures describe the table.

    t-SNE is run with `init="pca"` and `random_state=20`. A second run, after one flower is removed, is a new fit. Every remaining point can move. The notebook marks a hand-chosen coordinate, `(1.5, 0.83)`. That coordinate belongs to one run of the algorithm. The flower below is chosen from this run: the virginica point closest to the mean of the versicolor points in the t-SNE plane.

    ## Datasets Used

    - **Iris.** 150 flowers from `load_iris`. The measurements are sepal length, sepal width, petal length, and petal width, in centimeters. The species are setosa, versicolor, and virginica.
    """)
    return


@app.cell
def _(load_iris, pd, plt, sns):
    iris = load_iris()
    iris_frame = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_frame["species"] = iris.target_names[iris.target]
    species_counts = iris_frame.species.value_counts()
    print(f"{len(iris_frame)} flowers")
    print(species_counts.to_string())
    print(f"Missing values: {int(iris_frame.isnull().sum().sum())}")
    _long = iris_frame.melt(id_vars="species", var_name="measure", value_name="cm")
    _figure, _axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.countplot(data=iris_frame, x="species", ax=_axes[0], hue="species", legend=False)
    _axes[0].set_title("Flowers per species")
    sns.boxplot(data=_long, x="measure", y="cm", hue="species", ax=_axes[1])
    _axes[1].set_title("Measurements in centimeters")
    _axes[1].tick_params(axis="x", rotation=25)
    _axes[1].legend(fontsize=7)
    _figure.tight_layout()
    return _figure


@app.cell
def _(StandardScaler, iris_frame, pd, plt, sns):
    measure_columns = list(iris_frame.columns[:-1])
    iris_z = pd.DataFrame(
        StandardScaler().fit_transform(iris_frame[measure_columns]),
        columns=measure_columns,
        index=iris_frame.index,
    )
    iris_z["species"] = iris_frame.species
    _long = iris_z.melt(id_vars="species", var_name="measure", value_name="z")
    _figure, _axes = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=_long, x="measure", y="z", ax=_axes, color="C0")
    _axes.set_title("Standardized measurements")
    _axes.tick_params(axis="x", rotation=25)
    return _figure


@app.cell
def _(PCA, draw_species, iris_z, measure_columns, pd, plt):
    iris_pca_model = PCA(n_components=2).fit(iris_z[measure_columns])
    iris_share = iris_pca_model.explained_variance_ratio_
    iris_pca = pd.DataFrame(iris_pca_model.transform(iris_z[measure_columns]), columns=["PCA 1", "PCA 2"])
    iris_pca["species"] = iris_z.species.to_numpy()
    print(f"Explained variance ratio: {iris_share.round(3)}")
    print(f"Together: {iris_share.sum():.3f}")
    _figure, _axes = plt.subplots(figsize=(5.5, 4))
    draw_species(_axes, iris_pca, "PCA 1", "PCA 2", "PCA")
    return _figure


@app.cell
def _(MDS, draw_species, iris_z, measure_columns, pd, plt):
    iris_mds = pd.DataFrame(
        MDS(
            n_components=2,
            metric_mds=True,
            init="random",
            n_init=1,
            random_state=25,
            normalized_stress="auto",
        ).fit_transform(iris_z[measure_columns]),
        columns=["MDS 1", "MDS 2"],
    )
    iris_mds["species"] = iris_z.species.to_numpy()
    _figure, _axes = plt.subplots(figsize=(5.5, 4))
    draw_species(_axes, iris_mds, "MDS 1", "MDS 2", "MDS")
    return _figure


@app.cell
def _(LocallyLinearEmbedding, draw_species, iris_z, measure_columns, pd, plt):
    iris_lle = pd.DataFrame(
        LocallyLinearEmbedding(
            n_neighbors=50,
            n_components=2,
            random_state=50,
            method="modified",
            eigen_solver="dense",
        ).fit_transform(iris_z[measure_columns]),
        columns=["LLE 1", "LLE 2"],
    )
    iris_lle["species"] = iris_z.species.to_numpy()
    _figure, _axes = plt.subplots(figsize=(5.5, 4))
    draw_species(_axes, iris_lle, "LLE 1", "LLE 2", "Locally linear embedding")
    return _figure


@app.cell
def _(Isomap, draw_species, iris_z, measure_columns, pd, plt):
    iris_isomap = pd.DataFrame(
        Isomap(n_neighbors=50, n_components=2, eigen_solver="dense").fit_transform(iris_z[measure_columns]),
        columns=["Isomap 1", "Isomap 2"],
    )
    iris_isomap["species"] = iris_z.species.to_numpy()
    _figure, _axes = plt.subplots(figsize=(5.5, 4))
    draw_species(_axes, iris_isomap, "Isomap 1", "Isomap 2", "Isomap")
    return _figure


@app.cell
def _(TSNE, draw_species, iris_z, measure_columns, np, pd, plt):
    iris_tsne = pd.DataFrame(
        TSNE(random_state=20, init="pca").fit_transform(iris_z[measure_columns]),
        columns=["t-SNE 1", "t-SNE 2"],
        index=iris_z.index,
    )
    iris_tsne["species"] = iris_z.species
    _versicolor = iris_tsne.loc[iris_tsne.species == "versicolor", ["t-SNE 1", "t-SNE 2"]].to_numpy()
    _virginica = iris_tsne.loc[iris_tsne.species == "virginica", ["t-SNE 1", "t-SNE 2"]]
    _center = _versicolor.mean(axis=0)
    _distances = np.linalg.norm(_virginica.to_numpy() - _center, axis=1)
    nearest_virginica = int(_virginica.index[_distances.argmin()])
    nearest_coordinates = _virginica.loc[nearest_virginica].to_numpy()
    print(f"Nearest virginica row: {nearest_virginica}")
    print(f"t-SNE coordinates: {nearest_coordinates.round(3)}")
    _figure, _axes = plt.subplots(figsize=(5.5, 4))
    draw_species(_axes, iris_tsne, "t-SNE 1", "t-SNE 2", "t-SNE")
    _axes.scatter(
        nearest_coordinates[0],
        nearest_coordinates[1],
        s=160,
        facecolors="none",
        edgecolors="black",
        linewidths=1.2,
    )
    return _figure


@app.cell
def _(TSNE, iris_frame, iris_z, measure_columns, nearest_virginica, outside_whiskers, pd, plt, sns):
    iris_kept = iris_z.drop(index=nearest_virginica)
    iris_tsne_kept = pd.DataFrame(
        TSNE(random_state=20, init="pca").fit_transform(iris_kept[measure_columns]),
        columns=["t-SNE 1", "t-SNE 2"],
    )
    iris_tsne_kept["species"] = iris_kept.species.to_numpy()
    outlier_values = iris_frame.loc[nearest_virginica, measure_columns].astype(float)
    virginica = iris_frame.loc[iris_frame.species == "virginica", measure_columns]
    outside_features = [name for name in measure_columns if outside_whiskers(virginica[name], outlier_values[name])]
    print(outlier_values.to_string())
    print(f"Measurements outside the virginica whiskers: {outside_features or 'none'}")
    _long = virginica.melt(var_name="measure", value_name="cm")
    _figure, _axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.scatterplot(data=iris_tsne_kept, x="t-SNE 1", y="t-SNE 2", hue="species", s=28, ax=_axes[0], linewidth=0)
    _axes[0].set_title("t-SNE refit without that flower")
    _axes[0].legend(fontsize=7)
    sns.boxplot(data=_long, x="measure", y="cm", ax=_axes[1], color="C2")
    _axes[1].scatter(list(measure_columns), outlier_values.to_numpy(), s=80, facecolors="none", edgecolors="black")
    _axes[1].set_title("That flower on the virginica boxes")
    _axes[1].tick_params(axis="x", rotation=25)
    _figure.tight_layout()
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Which method keeps shortest-path distances along the neighbor graph? Replace `None` with its name.
    """)
    return


@app.cell
def _():
    student_method = None
    print(student_method)
    return (student_method,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r'''
```python
student_method = "Isomap"
```
**Isomap.** The neighbor graph supplies the paths. PCA keeps straight variance. MDS keeps the distances it was given. Locally linear embedding keeps local reconstruction weights. t-SNE keeps neighbor relationships.
''')
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(iris_share, mo, nearest_virginica, outside_features, species_counts):
    _outside = ", ".join(outside_features) if outside_features else "none of them"
    mo.md(f"""
    ## Conclusions

    **Key takeaways:**

    - The table has {int(species_counts["setosa"])} setosa, {int(species_counts["versicolor"])} versicolor, and {int(species_counts["virginica"])} virginica flowers.
    - The first two principal components keep {iris_share[0]:.1%} and {iris_share[1]:.1%} of the variance in the standardized measurements. Together that is {iris_share.sum():.1%}.
    - In this t-SNE run, virginica row {nearest_virginica} is the closest virginica flower to the versicolor center. Of its four raw measurements, the ones outside the virginica whiskers are: {_outside}.
    - Removing that row and fitting t-SNE again moves the remaining flowers. The second picture is not the first picture with one dot erased.
    - The scaler and every embedding were fit on the flowers in the picture. The species labels were used only to color the points and to name the nearest virginica flower.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check your understanding

    1. Why are the four measurements standardized?
    2. What do the two PCA percentages add up to?
    3. Why can a point that sits among another species in a t-SNE picture still be ordinary in the original centimeters?
    4. What changes when t-SNE is fit a second time after a row is removed?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
1. **The centimeters are not on one scale:** Petal length spreads more than sepal width. Standardization gives each column a standard deviation of 1 before a distance or a component is computed.
2. **The share those two axes keep:** Each percentage is that component's fraction of the total variance. Their sum is the fraction the picture retains. The rest is in the two components that were not drawn.
3. **t-SNE does not preserve the original distances:** A flower can land near another species in the picture because of the embedding, while its four measurements still sit inside the virginica boxes. The whisker count is the check.
4. **The whole map is recomputed:** t-SNE has no separate transform step. The second fit places the remaining 149 flowers from scratch, using the same seed.
""")
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - [Manifold learning](https://scikit-learn.org/stable/modules/manifold.html)
    - [Iris](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html)
    - VanderPlas, J. (2017). *Python Data Science Handbook*. O'Reilly Media. Chapter 5.
    """)
    return


if __name__ == "__main__":
    app.run()
