# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
#     "pandas",
#     "plotly",
#     "scikit-learn>=1.2",
# ]
# ///

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
async def _():
    import importlib.util as _import_util
    import sys as _sys

    # Setup: run this cell to install any missing packages.
    # You do not need to modify this code.
    _packages = {
        "numpy": "numpy",
        "pandas": "pandas",
        "plotly": "plotly",
        "sklearn": "scikit-learn>=1.2",
    }
    _missing = [
        _package for _module, _package in _packages.items()
        if _import_util.find_spec(_module) is None
    ]
    if _missing:
        if _sys.platform == "emscripten":
            # Browser Python uses micropip; it cannot launch subprocesses.
            import micropip as _micropip
            await _micropip.install(_missing)
        else:
            import subprocess as _subprocess
            _subprocess.check_call([_sys.executable, "-m", "pip", "install", *_missing])

    # The imports below wait until package installation has completed.
    packages_ready = True
    return (packages_ready,)


@app.cell
def _(packages_ready):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))

    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.io as pio
    import PlotBuilder as pb
    from plotly.subplots import make_subplots
    from sklearn.datasets import load_digits
    from sklearn.decomposition import PCA
    from sklearn.manifold import Isomap, LocallyLinearEmbedding, TSNE
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsClassifier

    pio.templates.default = "plotly_white"
    px.defaults.color_discrete_sequence = px.colors.qualitative.G10
    builder = pb.PlotBuilder(width=780, height=380)

    return (
        Isomap,
        KNeighborsClassifier,
        LocallyLinearEmbedding,
        PCA,
        Path,
        TSNE,
        builder,
        cross_val_score,
        go,
        load_digits,
        make_subplots,
        mo,
        np,
        pd,
        px,
    )


@app.cell
def _(builder, go, make_subplots, np, pd):
    def along_word(position, embedded):
        """Absolute correlation between the word order and an embedding axis.

        Parameters
        ----------
        position : array-like
            The original row order, which is the word order.
        embedded : array-like
            Two coordinates per point.

        Returns
        -------
        float
            Absolute correlation with the longest axis, so a flipped axis is not
            penalised.
        """
        embedded = np.asarray(embedded)
        axis = embedded[:, int(np.argmax(np.var(embedded, axis=0)))]
        return abs(float(np.corrcoef(position, axis)[0, 1]))

    def letters_panel(frames, captions, title="", rows=1, cols=None, width=1150):
        """Place several letter embeddings side by side for comparison.

        Parameters
        ----------
        frames : list of pandas.DataFrame
            One layout per panel, each with `x0`, `x1`, and `labels`.
        captions : list of str
            Panel titles, in reading order.
        title : str, default ""
            Figure title.
        rows, cols : int
            Grid shape.
        width : int, default 1150
            Figure width in pixels.

        Returns
        -------
        plotly.graph_objects.Figure
            The assembled figure, sharing one legend.
        """
        cols = cols or len(frames)
        figure = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=captions,
            horizontal_spacing=0.04,
            vertical_spacing=0.14,
        )
        for index, frame in enumerate(frames):
            row = index // cols + 1
            col = index % cols + 1
            for letter in "GOLIATH":
                mask = frame["labels"].to_numpy() == letter
                figure.add_trace(
                    go.Scatter(
                        x=frame["x0"].to_numpy()[mask],
                        y=frame["x1"].to_numpy()[mask],
                        mode="markers",
                        name=letter,
                        legendgroup=letter,
                        showlegend=index == 0,
                        marker=dict(size=4, opacity=0.8),
                    ),
                    row=row,
                    col=col,
                )
        figure.update_layout(
            width=width,
            height=340 if rows == 1 else 720,
            title=title,
            margin=dict(t=70, b=35, l=45, r=20),
            legend=dict(orientation="h", itemsizing="constant"),
        )
        return figure

    def plot_digits(images):
        """Draw the first 40 digit images as a 4 by 10 Plotly grid.

        Parameters
        ----------
        images : array-like
            One row of 64 pixel values per image.

        Returns
        -------
        plotly.graph_objects.Figure
            The grid, with axes hidden.
        """
        return builder.digit_grid(images, 4, 10, "The first 40 digits")

    return along_word, letters_panel, plot_digits


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

    - **S-shaped GOLIATH.** `GOLIATH_S_curve.csv`, the saved cloud: 1,000 points sampled from the rendered word and wrapped onto an S-shaped surface. `Goliath.png`, the image those points were sampled from, is also in this folder. The cloud is read from the CSV rather than regenerated, so the row order that the word-order score depends on is exactly the saved order.
    - **Digits.** 1,797 images, 64 pixels each, from `load_digits`.
    """)
    return


@app.cell
def _(Path, np, pd, px):
    curve_path = Path(__file__).with_name("GOLIATH_S_curve.csv")
    curve = pd.read_csv(curve_path)
    word_position = np.arange(len(curve))
    letter_counts = curve.labels.value_counts().reindex(list("GOLIATH"))
    _figure = px.scatter_3d(
        curve,
        x="x0",
        y="x1",
        z="x2",
        color="labels",
        title="The S-shaped GOLIATH cloud, in three dimensions",
        category_orders={"labels": list("GOLIATH")},
    )
    _figure.update_traces(marker=dict(size=2, opacity=0.85))
    _figure.update_layout(
        width=760, height=520, margin=dict(t=55, b=20, l=20, r=20),
        legend=dict(itemsizing="constant"),
    )
    _figure
    return curve, letter_counts, word_position


@app.cell
def _(LocallyLinearEmbedding, along_word, curve, letters_panel, pd, word_position):
    lle_agreement = {}
    _layouts = []
    _captions = []
    for _neighbors in (82, 97, 101):
        _embedded = LocallyLinearEmbedding(
            n_neighbors=_neighbors,
            n_components=2,
            random_state=20,
            method="modified",
            eigen_solver="dense",
        ).fit_transform(curve[["x0", "x1", "x2"]])
        lle_agreement[_neighbors] = along_word(word_position, _embedded)
        _layouts.append(
            pd.DataFrame(
                {"x0": _embedded[:, 0], "x1": _embedded[:, 1], "labels": curve.labels}
            )
        )
        _captions.append(f"LLE, {_neighbors} neighbours")
    letters_panel(
        _layouts, _captions, title="Locally linear embedding at three neighbourhood sizes"
    )
    return (lle_agreement,)


@app.cell
def _(Isomap, along_word, curve, letters_panel, pd, word_position):
    isomap_agreement = {}
    _layouts = []
    _captions = []
    for _neighbors, _solver in ((62, "dense"), (100, "auto")):
        _embedded = Isomap(
            n_neighbors=_neighbors, n_components=2, eigen_solver=_solver
        ).fit_transform(curve[["x0", "x1", "x2"]])
        isomap_agreement[_neighbors] = along_word(word_position, _embedded)
        _layouts.append(
            pd.DataFrame(
                {"x0": _embedded[:, 0], "x1": _embedded[:, 1], "labels": curve.labels}
            )
        )
        _captions.append(f"Isomap, {_neighbors} neighbours")
    letters_panel(_layouts, _captions, title="Isomap at two neighbourhood sizes", width=900)
    return (isomap_agreement,)


@app.cell
def _(TSNE, along_word, curve, letters_panel, pd, word_position):
    tsne_agreement = {}
    _layouts = []
    _captions = []
    for _perplexity, _learning in ((30, "auto"), (5, 50), (50, "auto"), (120, 100)):
        _model = TSNE(
            random_state=20, init="pca", perplexity=_perplexity, learning_rate=_learning
        )
        _embedded = _model.fit_transform(curve[["x0", "x1", "x2"]])
        tsne_agreement[_perplexity] = along_word(word_position, _embedded)
        _layouts.append(
            pd.DataFrame(
                {"x0": _embedded[:, 0], "x1": _embedded[:, 1], "labels": curve.labels}
            )
        )
        _captions.append(f"t-SNE, perplexity {_perplexity}")
    letters_panel(
        _layouts,
        _captions,
        title="t-SNE at four perplexities",
        rows=2,
        cols=2,
        width=900,
    )
    return (tsne_agreement,)


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
    builder,
    cross_val_score,
    load_digits,
    pd,
    plot_digits,
):
    digits = load_digits()
    digit_pca = PCA(n_components=2).fit_transform(digits.data)
    digit_tsne = TSNE(random_state=42, init="pca").fit_transform(digits.data)
    _checker = KNeighborsClassifier(n_neighbors=5)
    pca_neighbor_score = float(
        cross_val_score(_checker, digit_pca, digits.target, cv=5, n_jobs=1).mean()
    )
    tsne_neighbor_score = float(
        cross_val_score(_checker, digit_tsne, digits.target, cv=5, n_jobs=1).mean()
    )
    plot_digits(digits.data)
    return digit_pca, digit_tsne, digits, pca_neighbor_score, tsne_neighbor_score


@app.cell
def _(builder, digit_pca, digit_tsne, digits, pd):
    _labels = digits.target.astype(str)
    _left = pd.DataFrame(
        {"c1": digit_pca[:, 0], "c2": digit_pca[:, 1], "digit": _labels}
    )
    _right = pd.DataFrame(
        {"c1": digit_tsne[:, 0], "c2": digit_tsne[:, 1], "digit": _labels}
    )
    builder.scatter_by_group(
        _left, "c1", "c2", "digit", "Digits in the plane of two principal components", size=4, equal=False
    )
    return


@app.cell
def _(builder, digit_tsne, digits, pd):
    _right = pd.DataFrame(
        {"c1": digit_tsne[:, 0], "c2": digit_tsne[:, 1], "digit": digits.target.astype(str)}
    )
    builder.scatter_by_group(
        _right, "c1", "c2", "digit", "The same digits placed by t-SNE", size=4, equal=False
    )
    return


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
