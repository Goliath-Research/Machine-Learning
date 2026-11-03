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
    from plotly.subplots import make_subplots
    from scipy.stats import spearmanr
    from sklearn.datasets import load_digits
    from sklearn.decomposition import PCA
    from sklearn.manifold import Isomap, LocallyLinearEmbedding, TSNE
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsClassifier

    pio.templates.default = "plotly_white"
    px.defaults.color_discrete_sequence = px.colors.qualitative.G10
    LETTER_COLORS = {
        letter: px.colors.qualitative.G10[index]
        for index, letter in enumerate("GOLIATH")
    }

    return (
        Isomap, KNeighborsClassifier, LETTER_COLORS, LocallyLinearEmbedding, PCA,
        Path, TSNE, cross_val_score, go, load_digits, make_subplots, mo, np, pd,
        px, spearmanr,
    )


@app.cell
def _(np, spearmanr):
    def along_word(position, embedded):
        """Best |Spearman| between word order and an embedding axis or PC1."""
        embedded = np.asarray(embedded, dtype=float)
        position = np.asarray(position)
        scores = [
            abs(float(spearmanr(position, embedded[:, column]).statistic))
            for column in range(embedded.shape[1])
        ]
        centered = embedded - embedded.mean(axis=0)
        _, _, _vt = np.linalg.svd(centered, full_matrices=False)
        _pc1 = centered @ _vt[0]
        scores.append(abs(float(spearmanr(position, _pc1).statistic)))
        return float(max(scores))

    return (along_word,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `along_word`

    Max |Spearman| between row index and each embedding axis or PC1 — rotation-safe order score, not letter readability.
    """)
    return


@app.cell
def _(LETTER_COLORS, go, make_subplots):
    def letters_panel(frames, captions, title="", rows=1, cols=None, width=880):
        """Letter scatters in a grid with shared colours and width at most 880."""
        cols = cols or len(frames)
        width = min(int(width), 880)
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
                        marker=dict(size=4, opacity=0.8, color=LETTER_COLORS[letter]),
                    ),
                    row=row,
                    col=col,
                )
            axis_suffix = "" if col == 1 and row == 1 else str(index + 1)
            figure.update_yaxes(
                scaleanchor=f"x{axis_suffix}", scaleratio=1, row=row, col=col
            )
        figure.update_layout(
            width=width,
            height=340 if rows == 1 else 660,
            title=title,
            margin=dict(t=70, b=35, l=45, r=20),
            legend=dict(orientation="h", itemsizing="constant", title="letter"),
        )
        return figure

    return (letters_panel,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `letters_panel`

    Shared legend, fixed `LETTER_COLORS`, 1:1 `scaleanchor` per panel, width capped at 880.
    """)
    return


@app.cell
def _(go, make_subplots, np):
    def plot_digits(images, width=880):
        """First 40 digit images upright in a 4×10 Plotly heatmap grid."""
        data = np.asarray(images)
        if data.ndim == 2:
            data = data.reshape(-1, 8, 8)
        rows, cols = 4, 10
        width = min(int(width), 880)
        figure = make_subplots(
            rows=rows,
            cols=cols,
            horizontal_spacing=0.01,
            vertical_spacing=0.02,
        )
        shown = 0
        for row in range(1, rows + 1):
            for col in range(1, cols + 1):
                if shown >= min(len(data), rows * cols):
                    break
                figure.add_trace(
                    go.Heatmap(
                        z=np.flipud(data[shown]),
                        showscale=False,
                        colorscale="Greys",
                        hoverinfo="skip",
                    ),
                    row=row,
                    col=col,
                )
                shown += 1
        _left, _right, _top, _bottom = 10, 10, 55, 10
        _cell = (width - _left - _right) * (1 - 0.01 * (cols - 1)) / cols
        _plot_h = rows * _cell / (1 - 0.02 * (rows - 1))
        figure.update_layout(
            width=width,
            height=int(_top + _bottom + _plot_h),
            title="The first 40 digits",
            margin=dict(t=_top, b=_bottom, l=_left, r=_right),
            plot_bgcolor="white",
        )
        figure.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
        figure.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
        return figure

    return (plot_digits,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `plot_digits`

    `go.Heatmap` with `np.flipud` for upright 8×8 glyphs and square cells at width ≤ 880.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Manifold Learning

    ## Objectives

    By the end of this lesson, you should be able to:

    1. Explain why distance *along* a bent sheet differs from straight-line distance *through* space, and why that breaks linear projection on the S-shaped GOLIATH cloud.
    2. State what each method preserves — local reconstruction weights (LLE), graph shortest paths (Isomap), neighbourhood probabilities (t-SNE) — and why preserving one property does not preserve the others.
    3. Interpret the along-word score as order along *some* embedding direction (Spearman on each axis and on PC1), not as letter readability.
    4. Tune neighbourhood size (`n_neighbors` or `perplexity`) and read when local detail or global order improves or collapses on this dataset.
    5. Contrast PCA as a reusable **`transform`** with t-SNE as a full refit that cannot extend to new points.

    ## Background

    On a folded sheet, neighbours *along* the surface can be far *through* the air — PCA only sees the latter. **LLE** keeps local reconstruction weights; **Isomap** keeps graph shortest paths; **t-SNE** keeps neighbourhood probabilities; **PCA** is the linear baseline. **`n_neighbors`** / **`perplexity`** trade local detail against global smoothing.

    ## Datasets Used

    - **S-shaped GOLIATH.** `GOLIATH_S_curve.csv`: 1,000 points on an S-shaped surface, row order matching the word sequence used by the along-word score.
    - **Digits.** 1,797 images of 64 pixels from `load_digits`.
    """)
    return


@app.cell
def _(Path, np, pd, px):
    curve_path = Path(__file__).with_name("GOLIATH_S_curve.csv")
    curve = pd.read_csv(curve_path)
    word_position = np.arange(len(curve))
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
    return curve, word_position


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Locally linear embedding

    For each point \(x_i\), LLE finds weights \(W_{ij}\) so \(x_i \approx \sum_j W_{ij} x_j\) over neighbours. The 2-D embedding minimises \(\Phi(Y)=\sum_i \| y_i - \sum_j W_{ij} y_j \|^2\). **What to look for:** ribbons should follow one surface; compare the slider across neighbourhood sizes.
    """)
    return


@app.cell
def _(LocallyLinearEmbedding, curve, pd):
    _coords = curve[["x0", "x1", "x2"]].to_numpy()
    lle_cache = {}
    for _n in range(55, 106):
        _emb = LocallyLinearEmbedding(
            n_neighbors=_n,
            n_components=2,
            random_state=20,
            method="modified",
            eigen_solver="dense",
        ).fit_transform(_coords)
        lle_cache[_n] = pd.DataFrame(
            {"x0": _emb[:, 0], "x1": _emb[:, 1], "labels": curve.labels}
        )
    return (lle_cache,)


@app.cell
def _(mo):
    lle_neighbor_slider = mo.ui.slider(
        start=55, stop=105, step=1, value=97, label="LLE n_neighbors"
    )
    lle_neighbor_slider
    return (lle_neighbor_slider,)


@app.cell
def _(along_word, letters_panel, lle_cache, lle_neighbor_slider, word_position):
    _n = int(lle_neighbor_slider.value)
    _frame = lle_cache[_n]
    _score = along_word(word_position, _frame[["x0", "x1"]].to_numpy())
    lle_slider_figure = letters_panel(
        [_frame],
        [f"LLE, {_n} neighbours (along-word {_score:.3f})"],
        title="Locally linear embedding at the chosen neighbourhood size",
    )
    lle_slider_figure
    return


@app.cell
def _(along_word, lle_cache, word_position):
    lle_agreement = {
        _n: along_word(
            word_position, lle_cache[_n][["x0", "x1"]].to_numpy()
        )
        for _n in (82, 97, 101)
    }
    return (lle_agreement,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Isomap

    Geodesic distance \(D^G_{ij}\) is the shortest path on the `n_neighbors` graph; Isomap embeds so Euclidean distances match \(D^G_{ij}\). **What to look for:** letter colours match the legend; broken graphs look like scattered islands.
    """)
    return


@app.cell
def _(Isomap, curve, pd):
    _coords = curve[["x0", "x1", "x2"]].to_numpy()
    isomap_cache = {}
    for _n in range(55, 106):
        _solver = "dense" if _n <= 70 else "auto"
        _emb = Isomap(
            n_neighbors=_n, n_components=2, eigen_solver=_solver
        ).fit_transform(_coords)
        isomap_cache[_n] = pd.DataFrame(
            {"x0": _emb[:, 0], "x1": _emb[:, 1], "labels": curve.labels}
        )
    return (isomap_cache,)


@app.cell
def _(mo):
    isomap_neighbor_slider = mo.ui.slider(
        start=55, stop=105, step=1, value=62, label="Isomap n_neighbors"
    )
    isomap_neighbor_slider
    return (isomap_neighbor_slider,)


@app.cell
def _(
    along_word,
    isomap_cache,
    isomap_neighbor_slider,
    letters_panel,
    word_position,
):
    _n = int(isomap_neighbor_slider.value)
    _frame = isomap_cache[_n]
    _score = along_word(word_position, _frame[["x0", "x1"]].to_numpy())
    isomap_slider_figure = letters_panel(
        [_frame],
        [f"Isomap, {_n} neighbours (along-word {_score:.3f})"],
        title="Isomap at the chosen neighbourhood size",
    )
    isomap_slider_figure
    return


@app.cell
def _(along_word, isomap_cache, word_position):
    isomap_agreement = {
        _n: along_word(
            word_position, isomap_cache[_n][["x0", "x1"]].to_numpy()
        )
        for _n in (62, 100)
    }
    return (isomap_agreement,)


@app.cell
def _(PCA, along_word, curve, letters_panel, pd, word_position):
    _coords = curve[["x0", "x1", "x2"]].to_numpy()
    pca_emb = PCA(n_components=2).fit_transform(_coords)
    pca_frame = pd.DataFrame(
        {"x0": pca_emb[:, 0], "x1": pca_emb[:, 1], "labels": curve.labels}
    )
    pca_along = along_word(word_position, pca_emb)
    pca_compare_figure = letters_panel(
        [pca_frame],
        [f"PCA (along-word {pca_along:.3f})"],
        title="Linear PCA on the same 3-D coordinates",
    )
    pca_compare_figure
    return (pca_along,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### t-SNE

    Minimise \(C=\sum_i \mathrm{KL}(P_i \,\|\, Q_i)\) between high-D neighbour probabilities \(P\) and 2-D Student-\(t\) neighbours \(Q\). **What to look for:** low perplexity fragments the word; near 120 the thread returns — axes still carry no metric meaning.
    """)
    return


@app.cell
def _(TSNE, curve, pd):
    _coords = curve[["x0", "x1", "x2"]].to_numpy()
    tsne_cache = {}
    for _p in range(5, 121, 5):
        _lr = 50 if _p == 5 else ("auto" if _p != 120 else 100)
        _emb = TSNE(
            random_state=20, init="pca", perplexity=_p, learning_rate=_lr
        ).fit_transform(_coords)
        tsne_cache[_p] = pd.DataFrame(
            {"x0": _emb[:, 0], "x1": _emb[:, 1], "labels": curve.labels}
        )
    return (tsne_cache,)


@app.cell
def _(mo):
    tsne_perplexity_slider = mo.ui.slider(
        start=5, stop=120, step=5, value=30, label="t-SNE perplexity"
    )
    tsne_perplexity_slider
    return (tsne_perplexity_slider,)


@app.cell
def _(
    along_word,
    letters_panel,
    tsne_cache,
    tsne_perplexity_slider,
    word_position,
):
    _p = int(tsne_perplexity_slider.value)
    _frame = tsne_cache[_p]
    _score = along_word(word_position, _frame[["x0", "x1"]].to_numpy())
    tsne_slider_figure = letters_panel(
        [_frame],
        [f"t-SNE, perplexity {_p} (along-word {_score:.3f})"],
        title="t-SNE at the chosen perplexity",
    )
    tsne_slider_figure
    return


@app.cell
def _(along_word, letters_panel, tsne_cache, word_position):
    tsne_agreement = {}
    _layouts = []
    _captions = []
    for _perplexity in (5, 30, 50, 120):
        _frame = tsne_cache[_perplexity]
        tsne_agreement[_perplexity] = along_word(
            word_position, _frame[["x0", "x1"]].to_numpy()
        )
        _layouts.append(_frame)
        _captions.append(
            f"Perplexity {_perplexity} (score {tsne_agreement[_perplexity]:.3f})"
        )
    tsne_quad_figure = letters_panel(
        _layouts,
        _captions,
        title="t-SNE at four perplexities",
        rows=2,
        cols=2,
        width=880,
    )
    tsne_quad_figure
    return (tsne_agreement,)


@app.cell
def _(
    isomap_agreement,
    lle_agreement,
    mo,
    pca_along,
    pd,
    tsne_agreement,
):
    _rows = [
        ("PCA (linear)", "—", pca_along),
        ("LLE", "82 neighbours", lle_agreement[82]),
        ("LLE", "97 neighbours", lle_agreement[97]),
        ("LLE", "101 neighbours", lle_agreement[101]),
        ("Isomap", "62 neighbours", isomap_agreement[62]),
        ("Isomap", "100 neighbours", isomap_agreement[100]),
        ("t-SNE", "perplexity 5", tsne_agreement[5]),
        ("t-SNE", "perplexity 30", tsne_agreement[30]),
        ("t-SNE", "perplexity 50", tsne_agreement[50]),
        ("t-SNE", "perplexity 120", tsne_agreement[120]),
    ]
    score_table = pd.DataFrame(
        _rows, columns=["Method", "Setting", "Along-word", "Notes"]
    )
    score_table["Notes"] = [
        "Global linear collapse",
        "Manifold",
        "Manifold",
        "Manifold",
        "Graph geodesics",
        "Graph geodesics",
        "Very local",
        "Default-scale",
        "Mid perplexity",
        "Wide neighbourhood",
    ]
    mo.Html(score_table.to_html(index=False, float_format=lambda x: f"{x:.3f}"))
    return


@app.cell(hide_code=True)
def _(mo, pca_along):
    mo.md(f"""
    PCA reaches only **{pca_along:.3f}** on along-word; several manifold settings cluster near **1.0**, so read layouts as well as the table.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Digits

    The 5-neighbour score is cross-validated k-NN accuracy on the 2-D coordinates — a label check, not part of the embedding objective.
    """)
    return


@app.cell
def _(
    KNeighborsClassifier,
    PCA,
    TSNE,
    cross_val_score,
    load_digits,
    plot_digits,
):
    digits = load_digits()
    digit_pca_model = PCA(n_components=2).fit(digits.data)
    digit_pca = digit_pca_model.transform(digits.data)
    digit_tsne = TSNE(random_state=42, init="pca").fit_transform(digits.data)
    _checker = KNeighborsClassifier(n_neighbors=5)
    pca_neighbor_score = float(
        cross_val_score(_checker, digit_pca, digits.target, cv=5, n_jobs=1).mean()
    )
    tsne_neighbor_score = float(
        cross_val_score(_checker, digit_tsne, digits.target, cv=5, n_jobs=1).mean()
    )
    plot_digits(digits.data[:40])
    return (
        digit_pca,
        digit_pca_model,
        digit_tsne,
        digits,
        pca_neighbor_score,
        tsne_neighbor_score,
    )


@app.cell
def _(digit_pca, digit_tsne, digits, go, make_subplots, np, pd):
    _labels = digits.target.astype(str)
    _left = pd.DataFrame(
        {"x0": digit_pca[:, 0], "x1": digit_pca[:, 1], "digit": _labels}
    )
    _right = pd.DataFrame(
        {"x0": digit_tsne[:, 0], "x1": digit_tsne[:, 1], "digit": _labels}
    )
    digits_figure = make_subplots(
        rows=1, cols=2, subplot_titles=("PCA (2 components)", "t-SNE")
    )
    for _panel, _frame, _index in ((_left, 0), (_right, 1)):
        for _digit in map(str, range(10)):
            _mask = _frame["digit"].to_numpy() == _digit
            digits_figure.add_trace(
                go.Scatter(
                    x=_frame["x0"].to_numpy()[_mask],
                    y=_frame["x1"].to_numpy()[_mask],
                    mode="markers",
                    name=_digit,
                    legendgroup=_digit,
                    showlegend=_index == 0,
                    marker=dict(size=4, opacity=0.75),
                ),
                row=1,
                col=_index + 1,
            )
    digits_figure.update_layout(
        width=880,
        height=380,
        title="All digits: PCA plane versus t-SNE",
        margin=dict(t=70, b=35, l=45, r=20),
        legend=dict(orientation="h", title="digit"),
    )
    digits_figure
    return


@app.cell(hide_code=True)
def _(mo, pca_neighbor_score, tsne_neighbor_score):
    mo.md(f"""
    t-SNE 5-NN accuracy **{tsne_neighbor_score:.3f}** beats PCA **{pca_neighbor_score:.3f}**; cluster gaps in t-SNE are not pixel distances.
    """)
    return


@app.cell
def _(PCA, TSNE, digits, go, mo, np, pd):
    _split = 1500
    _train = digits.data[:_split]
    _holdout = digits.data[_split:]
    _holdout_labels = digits.target[_split:]
    pca_reuse = PCA(n_components=2).fit(_train)
    _pca_new = pca_reuse.transform(_holdout)
    TSNE(random_state=42, init="pca").fit_transform(np.vstack([_train, _holdout]))
    _reuse_frame = pd.DataFrame(
        {"x0": _pca_new[:, 0], "x1": _pca_new[:, 1], "digit": _holdout_labels.astype(str)}
    )
    transform_figure = go.Figure()
    for _digit in map(str, range(10)):
        _mask = _reuse_frame["digit"].to_numpy() == _digit
        transform_figure.add_trace(
            go.Scatter(
                x=_reuse_frame["x0"].to_numpy()[_mask],
                y=_reuse_frame["x1"].to_numpy()[_mask],
                mode="markers",
                name=_digit,
                marker=dict(size=5, opacity=0.85),
            )
        )
    transform_figure.update_layout(
        width=760,
        height=380,
        title="PCA transform on 297 held-out digits (fit on the first 1,500)",
        margin=dict(t=55, b=40, l=45, r=20),
        legend=dict(title="digit"),
    )
    mo.vstack(
        [
            transform_figure,
            mo.md(
                "t-SNE has no **`transform`**: adding holdout points requires a **refit** on the combined matrix, and the layout will not align with an earlier t-SNE picture. Only PCA provides a fixed projection for new rows."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Which t-SNE argument sets the balance between local and wider neighbourhoods? Replace `None` with that argument name.
    """)
    return


@app.cell
def _():
    student_perplexity = None
    return (student_perplexity,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answer": mo.md(r"""
```python
student_perplexity = "perplexity"
```
**perplexity.** Smaller values weight nearest neighbours heavily; larger values widen the effective neighbourhood (default 30).
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    In one sentence, what object does LLE try to keep fixed when it moves points into 2-D?
    """)
    return


@app.cell
def _():
    student_lle_preserves = ""
    return (student_lle_preserves,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answer": mo.md(r"""
**Local reconstruction weights:** each point should still be approximated by the same linear combination of its neighbours after embedding.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Using `curve` with columns `x0`, `x1`, `x2`, fit **PCA** with two components and store the embedding in `student_pca_2d` (a NumPy array of shape `(n_points, 2)`).
    """)
    return


@app.cell
def _(PCA, curve):
    student_pca_2d = None
    return (student_pca_2d,)


@app.cell(hide_code=True)
def _(PCA, curve, mo):
    mo.accordion(
        {
            "Show answer": mo.md(r"""
```python
student_pca_2d = PCA(n_components=2).fit_transform(curve[["x0", "x1", "x2"]])
```
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    After fitting PCA on a training matrix `X_train`, what method places new rows `X_new` into the same component space?
    """)
    return


@app.cell
def _():
    student_transform_call = ""
    return (student_transform_call,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answer": mo.md(r"""
```python
X_new_pca = pca_model.transform(X_new)
```
Use the **same** fitted `PCA` instance; **`fit_transform`** on new data alone would learn different directions.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(
    isomap_agreement,
    lle_agreement,
    mo,
    pca_along,
    pca_neighbor_score,
    tsne_agreement,
    tsne_neighbor_score,
):
    mo.md(f"""
    ## Conclusions

    1. **Manifold geometry:** PCA on the S-curve reaches along-word **{pca_along:.3f}** because it cannot follow the sheet; nonlinear methods recover order near **1.0** when neighbourhoods are tuned.
    2. **No single winner on this file:** t-SNE at perplexity 120 (**{tsne_agreement[120]:.3f}**), LLE at 97/101 (**{lle_agreement[97]:.3f}** / **{lle_agreement[101]:.3f}**), and Isomap (**{isomap_agreement[62]:.3f}**–**{isomap_agreement[100]:.3f}**) tie within rounding — not a strict "global methods won" ranking.
    3. **Perplexity and neighbours matter:** t-SNE falls to **{tsne_agreement[5]:.3f}** at perplexity 5 while wide settings recover global thread; the sliders show the same trade-off for LLE and Isomap.
    4. **Scores ≠ readable letters:** high along-word means monotonic order along some direction; overlapping glyphs can still look wrong while correlating with row index.
    5. **Reuse vs refit:** PCA **`transform`** scores **{pca_neighbor_score:.3f}** on digit neighbours with a fixed map; t-SNE reaches **{tsne_neighbor_score:.3f}** locally but must be refit to add points.
    """)
    return


if __name__ == "__main__":
    app.run()
