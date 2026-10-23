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
    from pathlib import Path

    from sklearn.datasets import load_digits
    from sklearn.decomposition import PCA
    from sklearn.manifold import Isomap, LocallyLinearEmbedding, TSNE
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsClassifier

    return (
        Isomap,
        KNeighborsClassifier,
        LocallyLinearEmbedding,
        PCA,
        Path,
        TSNE,
        cross_val_score,
        load_digits,
    )


@app.cell
def _(np, pd, plt, sns):
    def along_word(position, embedded):
        embedded = np.asarray(embedded)
        axis = embedded[:, int(np.argmax(np.var(embedded, axis=0)))]
        return abs(float(np.corrcoef(position, axis)[0, 1]))

    def draw_letters(axes, frame, title):
        sns.scatterplot(
            data=frame,
            x="x0",
            y="x1",
            hue="labels",
            hue_order=list("GOLIATH"),
            s=12,
            ax=axes,
            linewidth=0,
        )
        axes.set_title(title)
        axes.set_aspect("equal", adjustable="datalim")
        axes.legend(fontsize=7, markerscale=0.6)

    def plot_digits(images):
        figure, axes = plt.subplots(
            4, 10, figsize=(8, 3.4), gridspec_kw={"hspace": 0.15, "wspace": 0.05}
        )
        for ax, image in zip(np.ravel(axes), images[:40]):
            ax.imshow(np.asarray(image).reshape(8, 8), cmap="binary", interpolation="nearest", vmin=0, vmax=16)
            ax.set_xticks([])
            ax.set_yticks([])
        return figure

    return along_word, draw_letters, plot_digits


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Manifold Learning

    ## Objectives

    - Unroll the saved S-shaped GOLIATH cloud with locally linear embedding, Isomap, and t-SNE.
    - Compare t-SNE at perplexity 5, 30, 50, and 120.
    - Place the digits with two principal components and with t-SNE, and score each picture with a 5-neighbor classifier.

    ## Background

    Locally linear embedding rebuilds each point from its neighbors and keeps those local weights. Isomap keeps shortest-path distances on the neighbor graph, a stand-in for distance along the surface. t-SNE places points so that neighbors stay neighbors. Perplexity sets how many neighbors that effort considers. The default is 30.

    The along-word score is the absolute correlation between the original row order and the longer axis of the embedding. The file is stored in that row order. A high score means the word still runs along one axis. It does not by itself mean the letters are readable.

    ## Datasets Used

    - **S-shaped GOLIATH.** `GOLIATH_S_curve.csv`, the cloud the notebook saved. `Goliath.png`, which was sampled to build it, is not in this folder.
    - **Digits.** 1,797 images, 64 pixels each, from `load_digits`.
    """)
    return


@app.cell
def _(Path, np, pd, plt):
    curve_path = Path(__file__).with_name("GOLIATH_S_curve.csv")
    curve = pd.read_csv(curve_path)
    word_position = np.arange(len(curve))
    print(f"{len(curve)} points from {curve_path.name}")
    print(curve.labels.value_counts().reindex(list("GOLIATH")).to_string())
    _figure = plt.figure(figsize=(6, 4.5))
    _axes = _figure.add_subplot(111, projection="3d")
    for _letter in "GOLIATH":
        _rows = curve.labels.to_numpy() == _letter
        _axes.scatter(curve.loc[_rows, "x0"], curve.loc[_rows, "x1"], curve.loc[_rows, "x2"], s=8, label=_letter)
    _axes.set_title("S-shaped GOLIATH")
    _axes.legend(fontsize=7, markerscale=0.6)
    return _figure


@app.cell
def _(LocallyLinearEmbedding, along_word, curve, draw_letters, pd, plt, word_position):
    lle_agreement = {}
    _figure, _axes = plt.subplots(1, 3, figsize=(11, 3.5))
    for _ax, _neighbors in zip(_axes, (82, 97, 101)):
        _embedded = LocallyLinearEmbedding(
            n_neighbors=_neighbors,
            n_components=2,
            random_state=20,
            method="modified",
            eigen_solver="dense",
        ).fit_transform(curve[["x0", "x1", "x2"]])
        lle_agreement[_neighbors] = along_word(word_position, _embedded)
        _layout = pd.DataFrame({"x0": _embedded[:, 0], "x1": _embedded[:, 1], "labels": curve.labels})
        draw_letters(_ax, _layout, f"LLE, {_neighbors} neighbors")
        print(f"LLE {_neighbors} neighbors: along-word correlation {lle_agreement[_neighbors]:.3f}")
    _figure.tight_layout()
    return _figure


@app.cell
def _(Isomap, along_word, curve, draw_letters, pd, plt, word_position):
    isomap_agreement = {}
    _figure, _axes = plt.subplots(1, 2, figsize=(8, 3.6))
    for _ax, _neighbors, _solver in zip(_axes, (62, 100), ("dense", "auto")):
        _embedded = Isomap(n_neighbors=_neighbors, n_components=2, eigen_solver=_solver).fit_transform(
            curve[["x0", "x1", "x2"]]
        )
        isomap_agreement[_neighbors] = along_word(word_position, _embedded)
        _layout = pd.DataFrame({"x0": _embedded[:, 0], "x1": _embedded[:, 1], "labels": curve.labels})
        draw_letters(_ax, _layout, f"Isomap, {_neighbors} neighbors")
        print(f"Isomap {_neighbors} neighbors: along-word correlation {isomap_agreement[_neighbors]:.3f}")
    _figure.tight_layout()
    return _figure


@app.cell
def _(TSNE, along_word, curve, draw_letters, pd, plt, word_position):
    tsne_agreement = {}
    _settings = (
        (30, "auto"),
        (5, 50),
        (50, "auto"),
        (120, 100),
    )
    _figure, _axes = plt.subplots(2, 2, figsize=(8, 7))
    for _ax, (_perplexity, _learning) in zip(_axes.ravel(), _settings):
        _model = TSNE(
            random_state=20,
            init="pca",
            perplexity=_perplexity,
            learning_rate=_learning,
        )
        _embedded = _model.fit_transform(curve[["x0", "x1", "x2"]])
        tsne_agreement[_perplexity] = along_word(word_position, _embedded)
        _layout = pd.DataFrame({"x0": _embedded[:, 0], "x1": _embedded[:, 1], "labels": curve.labels})
        draw_letters(_ax, _layout, f"t-SNE, perplexity {_perplexity}")
        print(f"Perplexity {_perplexity}: along-word correlation {tsne_agreement[_perplexity]:.3f}")
    _figure.tight_layout()
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Digits

    The 5-neighbor score is a check on the picture, not part of either fit. It is the mean accuracy of 5-fold cross-validation on the two coordinates, using the digit labels only as the thing to predict. `n_jobs=1`.
    """)
    return


@app.cell
def _(
    KNeighborsClassifier,
    PCA,
    TSNE,
    cross_val_score,
    load_digits,
    pd,
    plot_digits,
    plt,
    sns,
):
    digits = load_digits()
    digit_pca = PCA(n_components=2).fit_transform(digits.data)
    digit_tsne = TSNE(random_state=42, init="pca").fit_transform(digits.data)
    _checker = KNeighborsClassifier(n_neighbors=5)
    pca_neighbor_score = float(cross_val_score(_checker, digit_pca, digits.target, cv=5, n_jobs=1).mean())
    tsne_neighbor_score = float(cross_val_score(_checker, digit_tsne, digits.target, cv=5, n_jobs=1).mean())
    print(f"5-neighbor accuracy on the PCA plane: {pca_neighbor_score:.3f}")
    print(f"5-neighbor accuracy on the t-SNE plane: {tsne_neighbor_score:.3f}")
    _grids = plot_digits(digits.data)
    _grids.suptitle("First 40 digits")
    _figure, _axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for _ax, _embedded, _title in (
        (_axes[0], digit_pca, "PCA"),
        (_axes[1], digit_tsne, "t-SNE"),
    ):
        _frame = pd.DataFrame(
            {"x": _embedded[:, 0], "y": _embedded[:, 1], "digit": digits.target.astype(str)}
        )
        sns.scatterplot(data=_frame, x="x", y="y", hue="digit", s=12, ax=_ax, linewidth=0)
        _ax.set_title(_title)
        _ax.legend(title="", fontsize=6, markerscale=0.6, ncol=2)
    return _grids, _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Which t-SNE argument sets the balance between local and wider neighborhoods? Replace `None` with that argument name.
    """)
    return


@app.cell
def _():
    student_perplexity = None
    print(student_perplexity)
    return (student_perplexity,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r'''
```python
student_perplexity = "perplexity"
```
**perplexity.** A small value keeps the fit focused on the nearest points. A larger value asks it to respect a wider neighborhood. The default is 30.
''')
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(isomap_agreement, lle_agreement, mo, pca_neighbor_score, tsne_agreement, tsne_neighbor_score):
    mo.md(f"""
    ## Conclusions

    **Key takeaways:**

    - Modified locally linear embedding has along-word correlations {lle_agreement[82]:.3f} at 82 neighbors, {lle_agreement[97]:.3f} at 97, and {lle_agreement[101]:.3f} at 101.
    - Isomap has correlations {isomap_agreement[62]:.3f} at 62 neighbors and {isomap_agreement[100]:.3f} at 100.
    - t-SNE along-word correlations are {tsne_agreement[5]:.3f} at perplexity 5, {tsne_agreement[30]:.3f} at 30, {tsne_agreement[50]:.3f} at 50, and {tsne_agreement[120]:.3f} at 120.
    - On the digits, 5-neighbor accuracy is {pca_neighbor_score:.3f} in the PCA plane and {tsne_neighbor_score:.3f} in the t-SNE plane. Neither embedding was shown the labels.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check your understanding

    1. What does locally linear embedding try to keep?
    2. What distance does Isomap keep?
    3. What does a larger perplexity ask t-SNE to pay attention to?
    4. Why is the 5-neighbor score not part of the t-SNE fit?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
1. **Local weights:** Each point is rebuilt from its neighbors. The embedding tries to keep those weights. It does not try to keep long distances.
2. **Shortest paths on the neighbor graph:** Those paths follow the surface when the surface is covered by the neighbors. A Euclidean shortcut would cut through the bend.
3. **A wider neighborhood:** Perplexity 5 stays very local. Perplexity 120 pulls in much more of the cloud. The along-word correlations are the comparison on this file.
4. **The labels are only a check:** t-SNE sees the pixel values. The classifier is fit afterward on the two coordinates to measure how separated the digits became.
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
    - [t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)
    - VanderPlas, J. (2017). *Python Data Science Handbook*. O'Reilly Media. Chapter 5.
    """)
    return


if __name__ == "__main__":
    app.run()
