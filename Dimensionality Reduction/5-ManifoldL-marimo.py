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
    LLE_NEIGHBOR_KEYS = sorted({*range(55, 106, 5), 80, 100, 105})
    ISO_NEIGHBOR_KEYS = sorted({*range(55, 106, 5), 62})
    TSNE_PERPLEXITY_KEYS = [5, 10, 20, 30, 40, 50, 60, 80, 100, 120]
    TSNE_LEARNING_RATE = 200

    return (
        ISO_NEIGHBOR_KEYS,
        KNeighborsClassifier,
        LETTER_COLORS,
        LLE_NEIGHBOR_KEYS,
        LocallyLinearEmbedding,
        PCA,
        Path,
        TSNE,
        TSNE_LEARNING_RATE,
        TSNE_PERPLEXITY_KEYS,
        cross_val_score,
        go,
        load_digits,
        make_subplots,
        mo,
        np,
        pd,
        px,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Manifold Learning

    ## Objectives

    By the end of this lesson, you should be able to:

    1. Explain why distance *along* a bent sheet differs from straight-line distance *through* space, and why that breaks linear projection on the S-shaped GOLIATH cloud.
    2. State what each method preserves — local reconstruction weights (LLE), graph shortest paths (Isomap), neighbourhood probabilities (t-SNE) — and why preserving one property does not preserve the others.
    3. Interpret the along-word score as how monotonically the saved row order follows *some* direction in the plane (not letter readability).
    4. Tune neighbourhood size (`n_neighbors` or `perplexity`) and read when local detail or global order improves or collapses on this dataset.
    5. Contrast PCA as a reusable **`transform`** with t-SNE as a full refit that cannot extend to new points.

    ## Background

    On a folded sheet, neighbours *along* the surface can be far *through* the air — PCA only sees the latter. **LLE** keeps local reconstruction weights; **Isomap** keeps graph shortest paths; **t-SNE** keeps neighbourhood probabilities; **PCA** is the linear baseline. **`n_neighbors`** / **`perplexity`** trade local detail against global smoothing.

    The **along-word score** measures monotonicity along the best direction in the plane, treating a **reversed** word order as equally good (`max(s, 1−s)` per axis). Low scores mean the row order folds back on itself; a mirrored but readable GOLIATH still scores high.

    ## Datasets Used

    - **S-shaped GOLIATH.** `GOLIATH_S_curve.csv`: 1,000 points on an S-shaped surface, row order matching the word sequence used by the along-word score.
    - **Digits.** 1,797 images of 64 pixels from `load_digits`.
    """)
    return


@app.cell
def _(np):
    def along_word(embedded):
        """Monotonicity of CSV row order along the best plane direction."""
        emb = np.asarray(embedded, dtype=float)
        _n = len(emb)
        _best = 0.0
        _directions = [emb[:, 0], emb[:, 1]]
        _centered = emb - emb.mean(axis=0)
        _, _, _vt = np.linalg.svd(_centered, full_matrices=False)
        _directions.append(_centered @ _vt[0])
        _pairs = _n * (_n - 1) // 2
        for _coord in _directions:
            _order = np.argsort(_coord)
            _rank = np.empty(_n, dtype=int)
            _rank[_order] = np.arange(_n)
            _inversions = sum(
                1 for _i in range(_n) for _j in range(_i + 1, _n) if _rank[_i] > _rank[_j]
            )
            _score = 1.0 - _inversions / _pairs
            _best = max(_best, _score, 1.0 - _score)
        return float(_best)

    return (along_word,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `along_word`

    The CSV rows trace the word. A honest unrolling lets you walk the rows monotonically along some direction; walking backwards counts the same. The score is the best such monotonicity over the two axes or the first principal direction — high means the global sequence survived (even mirrored), not that each letter is legible.
    """)
    return


@app.cell
def _(LETTER_COLORS, go, make_subplots):
    def letters_panel(frames, captions, title="", rows=1, cols=None, width=840):
        """Letter scatters in a grid; width capped at 840 px."""
        cols = cols or len(frames)
        width = min(int(width), 840)
        _single = rows == 1 and cols == 1 and len(frames) == 1
        figure = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=captions,
            horizontal_spacing=0.04,
            vertical_spacing=0.12,
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
            if not _single:
                axis_suffix = "" if col == 1 and row == 1 else str(index + 1)
                figure.update_yaxes(
                    scaleanchor=f"x{axis_suffix}", scaleratio=1, row=row, col=col
                )
        figure.update_layout(
            width=width,
            height=380 if rows == 1 else 500,
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

    Compare several letter layouts in one figure with a shared legend and consistent colours. Multi-panel views keep equal axis scaling so panels are comparable; single plots leave axes free so a thin ribbon is not squashed.
    """)
    return


@app.cell
def _(go, make_subplots, np):
    def plot_digits(images, width=840):
        """First 40 digit images upright in a 4×10 Plotly heatmap grid."""
        data = np.asarray(images)
        if data.ndim == 2:
            data = data.reshape(-1, 8, 8)
        rows, cols = 4, 10
        width = min(int(width), 840)
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
            height=min(500, int(_top + _bottom + _plot_h)),
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

    A small gallery of 8×8 handwritten digits, flipped upright and drawn with square cells so strokes are not stretched.
    """)
    return


@app.cell
def _(Path, np, pd, px):
    curve_path = Path(__file__).with_name("GOLIATH_S_curve.csv")
    curve = pd.read_csv(curve_path)
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
        width=760, height=500, margin=dict(t=55, b=20, l=20, r=20),
        legend=dict(itemsizing="constant"),
    )
    _figure
    return (curve,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Locally linear embedding

    For each point \(x_i\), LLE finds weights \(W_{ij}\) so \(x_i \approx \sum_j W_{ij} x_j\) over neighbours. The 2-D embedding minimises \(\Phi(Y)=\sum_i \| y_i - \sum_j W_{ij} y_j \|^2\). **What to look for:** smooth ribbons; at 80 neighbours the word can appear **mirrored** yet score high, at 105 as a clean **arc** — read the picture, not only the number.
    """)
    return


@app.cell
def _(LLE_NEIGHBOR_KEYS, LocallyLinearEmbedding, curve, pd):
    _coords = curve[["x0", "x1", "x2"]].to_numpy()
    lle_cache = {}
    for _n in LLE_NEIGHBOR_KEYS:
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
def _(LLE_NEIGHBOR_KEYS, mo):
    lle_neighbor_slider = mo.ui.slider(
        start=min(LLE_NEIGHBOR_KEYS),
        stop=max(LLE_NEIGHBOR_KEYS),
        step=5,
        value=100,
        label="LLE n_neighbors",
    )
    lle_neighbor_slider
    return (lle_neighbor_slider,)


@app.cell
def _(LLE_NEIGHBOR_KEYS, along_word, letters_panel, lle_cache, lle_neighbor_slider):
    _n = min(LLE_NEIGHBOR_KEYS, key=lambda k: abs(k - int(lle_neighbor_slider.value)))
    _frame = lle_cache[_n]
    _score = along_word(_frame[["x0", "x1"]].to_numpy())
    lle_slider_figure = letters_panel(
        [_frame],
        [f"LLE, {_n} neighbours (along-word {_score:.3f})"],
        title="Locally linear embedding at the chosen neighbourhood size",
    )
    lle_slider_figure
    return


@app.cell
def _(along_word, lle_cache):
    lle_agreement = {
        _n: along_word(lle_cache[_n][["x0", "x1"]].to_numpy())
        for _n in (80, 100, 105)
    }
    return (lle_agreement,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Isomap

    Geodesic distance \(D^G_{ij}\) is the shortest path on the `n_neighbors` graph; Isomap embeds so Euclidean distances match \(D^G_{ij}\). **What to look for:** connected graphs follow the sheet; broken graphs scatter letters and drop the along-word score.
    """)
    return


@app.cell
def _(ISO_NEIGHBOR_KEYS, Isomap, curve, pd):
    _coords = curve[["x0", "x1", "x2"]].to_numpy()
    isomap_cache = {}
    for _n in ISO_NEIGHBOR_KEYS:
        _solver = "dense" if _n <= 70 else "auto"
        _emb = Isomap(
            n_neighbors=_n, n_components=2, eigen_solver=_solver
        ).fit_transform(_coords)
        isomap_cache[_n] = pd.DataFrame(
            {"x0": _emb[:, 0], "x1": _emb[:, 1], "labels": curve.labels}
        )
    return (isomap_cache,)


@app.cell
def _(ISO_NEIGHBOR_KEYS, mo):
    isomap_neighbor_slider = mo.ui.slider(
        start=min(ISO_NEIGHBOR_KEYS),
        stop=max(ISO_NEIGHBOR_KEYS),
        step=5,
        value=62,
        label="Isomap n_neighbors",
    )
    isomap_neighbor_slider
    return (isomap_neighbor_slider,)


@app.cell
def _(
    ISO_NEIGHBOR_KEYS,
    along_word,
    isomap_cache,
    isomap_neighbor_slider,
    letters_panel,
):
    _n = min(ISO_NEIGHBOR_KEYS, key=lambda k: abs(k - int(isomap_neighbor_slider.value)))
    _frame = isomap_cache[_n]
    _score = along_word(_frame[["x0", "x1"]].to_numpy())
    isomap_slider_figure = letters_panel(
        [_frame],
        [f"Isomap, {_n} neighbours (along-word {_score:.3f})"],
        title="Isomap at the chosen neighbourhood size",
    )
    isomap_slider_figure
    return


@app.cell
def _(along_word, isomap_cache):
    isomap_agreement = {
        _n: along_word(isomap_cache[_n][["x0", "x1"]].to_numpy())
        for _n in (62, 100)
    }
    return (isomap_agreement,)


@app.cell
def _(PCA, along_word, curve, letters_panel, pd):
    _coords = curve[["x0", "x1", "x2"]].to_numpy()
    pca_emb = PCA(n_components=2).fit_transform(_coords)
    pca_frame = pd.DataFrame(
        {"x0": pca_emb[:, 0], "x1": pca_emb[:, 1], "labels": curve.labels}
    )
    pca_along = along_word(pca_emb)
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

    Minimise \(C=\sum_i \mathrm{KL}(P_i \,\|\, Q_i)\) between high-D neighbour probabilities \(P\) and 2-D Student-\(t\) neighbours \(Q\) (learning rate fixed in this lesson). **What to look for:** along-word moves unevenly with perplexity — high values help overall but not monotonically; cluster gaps are still not distances.
    """)
    return


@app.cell
def _(TSNE, TSNE_LEARNING_RATE, TSNE_PERPLEXITY_KEYS, curve, pd):
    _coords = curve[["x0", "x1", "x2"]].to_numpy()
    tsne_cache = {}
    for _p in TSNE_PERPLEXITY_KEYS:
        _emb = TSNE(
            random_state=20,
            init="pca",
            perplexity=_p,
            learning_rate=TSNE_LEARNING_RATE,
        ).fit_transform(_coords)
        tsne_cache[_p] = pd.DataFrame(
            {"x0": _emb[:, 0], "x1": _emb[:, 1], "labels": curve.labels}
        )
    return (tsne_cache,)


@app.cell
def _(TSNE_PERPLEXITY_KEYS, mo):
    _step = TSNE_PERPLEXITY_KEYS[1] - TSNE_PERPLEXITY_KEYS[0]
    tsne_perplexity_slider = mo.ui.slider(
        start=min(TSNE_PERPLEXITY_KEYS),
        stop=max(TSNE_PERPLEXITY_KEYS),
        step=_step,
        value=30,
        label="t-SNE perplexity",
    )
    tsne_perplexity_slider
    return (tsne_perplexity_slider,)


@app.cell
def _(
    TSNE_PERPLEXITY_KEYS,
    along_word,
    letters_panel,
    tsne_cache,
    tsne_perplexity_slider,
):
    _p = min(
        TSNE_PERPLEXITY_KEYS,
        key=lambda k: abs(k - int(tsne_perplexity_slider.value)),
    )
    _frame = tsne_cache[_p]
    _score = along_word(_frame[["x0", "x1"]].to_numpy())
    tsne_slider_figure = letters_panel(
        [_frame],
        [f"t-SNE, perplexity {_p} (along-word {_score:.3f})"],
        title="t-SNE at the chosen perplexity",
    )
    tsne_slider_figure
    return


@app.cell
def _(along_word, letters_panel, tsne_cache):
    tsne_agreement = {
        _p: along_word(_frame[["x0", "x1"]].to_numpy())
        for _p, _frame in tsne_cache.items()
    }
    _layouts = []
    _captions = []
    for _perplexity in (5, 30, 50, 120):
        _frame = tsne_cache[_perplexity]
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
        width=840,
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
        ("PCA (linear)", "—", pca_along, "Cannot unroll the bend"),
        ("LLE", "80 neighbours", lle_agreement[80], "Mirrored, readable word"),
        ("LLE", "100 neighbours", lle_agreement[100], "Clean unrolling"),
        ("LLE", "105 neighbours", lle_agreement[105], "Clean arc"),
        ("Isomap", "62 neighbours", isomap_agreement[62], "Graph geodesics"),
        ("Isomap", "100 neighbours", isomap_agreement[100], "Graph geodesics"),
        ("t-SNE", "perplexity 5", tsne_agreement[5], "Very local"),
        ("t-SNE", "perplexity 30", tsne_agreement[30], "Default-scale"),
        ("t-SNE", "perplexity 50", tsne_agreement[50], "Mid perplexity"),
        ("t-SNE", "perplexity 120", tsne_agreement[120], "Wide neighbourhood"),
    ]
    score_table = pd.DataFrame(
        _rows, columns=["Method", "Setting", "Along-word", "Notes"]
    )
    mo.Html(score_table.to_html(index=False, float_format=lambda x: f"{x:.3f}"))
    return


@app.cell(hide_code=True)
def _(isomap_agreement, lle_agreement, mo, pca_along, tsne_agreement):
    _tsne_vals = list(tsne_agreement.values())
    mo.md(f"""
    PCA **{pca_along:.3f}**; LLE/Isomap **{min(lle_agreement[80], lle_agreement[100], lle_agreement[105], isomap_agreement[62], isomap_agreement[100]):.3f}**–**{max(lle_agreement[80], lle_agreement[100], lle_agreement[105], isomap_agreement[62], isomap_agreement[100]):.3f}** (mirrored LLE 80 **{lle_agreement[80]:.3f}**). t-SNE is **uneven** (**{min(_tsne_vals):.3f}**–**{max(_tsne_vals):.3f}**): **{tsne_agreement[40]:.3f}** at 40, **{tsne_agreement[50]:.3f}** at 50, **{tsne_agreement[100]:.3f}** vs **{tsne_agreement[120]:.3f}** at 100 vs 120.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Digits

    The 5-neighbour score is cross-validated k-NN accuracy on the 2-D coordinates — a digit-label check on the picture, separate from the GOLIATH along-word score and from the embedding objective.
    """)
    return


@app.cell
def _(
    KNeighborsClassifier,
    PCA,
    TSNE,
    TSNE_LEARNING_RATE,
    cross_val_score,
    load_digits,
    plot_digits,
):
    digits = load_digits()
    digit_pca_model = PCA(n_components=2).fit(digits.data)
    digit_pca = digit_pca_model.transform(digits.data)
    digit_tsne = TSNE(
        random_state=42, init="pca", learning_rate=TSNE_LEARNING_RATE
    ).fit_transform(digits.data)
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
def _(digit_pca, digit_tsne, digits, go, make_subplots, pd):
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
    for _frame, _index in ((_left, 0), (_right, 1)):
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
        width=840,
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
    t-SNE separates digit classes more clearly in the plane (5-NN accuracy **{tsne_neighbor_score:.3f}** vs **{pca_neighbor_score:.3f}** for PCA). That measures local class structure in the picture, not whether you can **`transform`** new images.
    """)
    return


@app.cell
def _(PCA, digits, go, pd):
    _split = 1500
    _train = digits.data[:_split]
    _holdout = digits.data[_split:]
    _holdout_labels = digits.target[_split:]
    pca_reuse = PCA(n_components=2).fit(_train)
    _pca_new = pca_reuse.transform(_holdout)
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
    transform_figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    t-SNE has no **`transform`**: new points require refitting on the combined data, and the new layout will not align with an earlier t-SNE picture. PCA is the reusable map — the held-out scatter above uses the same projection learned from the training rows only.
    """)
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
def _():
    student_pca_2d = None
    return (student_pca_2d,)


@app.cell(hide_code=True)
def _(mo):
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

@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answer": mo.md(r"""
Call **`transform`** on the same fitted PCA object: `fitted_pca.transform(X_new)`.

Running **`fit_transform`** on `X_new` alone would learn new directions and break compatibility with the training projection.
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
    tsne_agreement,
):
    _tsne_vals = list(tsne_agreement.values())
    _manifold_high = max(
        lle_agreement[80],
        lle_agreement[100],
        lle_agreement[105],
        isomap_agreement[62],
        isomap_agreement[100],
    )
    mo.md(f"""
    ## Conclusions

    1. **Manifold geometry:** PCA on the S-curve reaches along-word **{pca_along:.3f}**; tuned LLE and Isomap reach up to **{_manifold_high:.3f}**, including a mirrored GOLIATH at 80 neighbours (**{lle_agreement[80]:.3f}**), well above PCA.
    2. **No single winner on this file:** Isomap **{isomap_agreement[62]:.3f}**–**{isomap_agreement[100]:.3f}**, LLE at 100 neighbours **{lle_agreement[100]:.3f}**, and t-SNE at 120 **{tsne_agreement[120]:.3f}** all recover order — compare layouts, not just decimals.
    3. **Perplexity and neighbours matter:** t-SNE along-word is uneven (**{min(_tsne_vals):.3f}**–**{max(_tsne_vals):.3f}** across the slider cache), e.g. **{tsne_agreement[40]:.3f}** at perplexity 40, **{tsne_agreement[50]:.3f}** at 50, with **{tsne_agreement[100]:.3f}** at 100 beating **{tsne_agreement[120]:.3f}** at 120; LLE at 105 is **{lle_agreement[105]:.3f}** on a clean arc.
    4. **Scores ≠ readable letters:** along-word tracks global monotonicity (either orientation); overlapping letters can still look wrong while scoring moderately (PCA **{pca_along:.3f}**).
    5. **Reuse vs refit:** PCA **`transform`** maps held-out digits in the scatter above; t-SNE has no equivalent — every new point set needs a new fit and a new layout.
    """)
    return


if __name__ == "__main__":
    app.run()
