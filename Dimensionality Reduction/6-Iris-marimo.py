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
    from scipy.spatial import procrustes as scipy_procrustes
    from sklearn.datasets import load_iris
    from sklearn.decomposition import PCA
    from sklearn.manifold import Isomap, LocallyLinearEmbedding, MDS, TSNE
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    pio.templates.default = "plotly_white"
    px.defaults.color_discrete_sequence = px.colors.qualitative.G10
    SPECIES_ORDER = ["setosa", "versicolor", "virginica"]

    return (
        Isomap,
        KNeighborsClassifier,
        LocallyLinearEmbedding,
        MDS,
        NearestNeighbors,
        PCA,
        SPECIES_ORDER,
        StandardScaler,
        TSNE,
        cross_val_score,
        go,
        load_iris,
        mo,
        np,
        pd,
        px,
        scipy_procrustes,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Iris in Two Dimensions

    ## Objectives

    By the end of this lesson, you should be able to:

    1. Explain what each method optimises, and why two methods given identical input can disagree about which flowers belong together.
    2. Read the share of variance kept by two principal components, and say what that share does *not* guarantee.
    3. Explain why the axes of an MDS, LLE, Isomap, or t-SNE plot carry no meaning, while the groupings can still be real.
    4. Explain why t-SNE cannot place a new point without refitting everything.
    5. Describe why a flower can appear inside another species in a t-SNE plot while remaining ordinary on every original measurement.

    ## Background

    Five methods, one standardized input (four columns), two coordinates out. None uses species labels; colour is added afterward.

    | Method | Preserves | Labels? |
    | :--- | :--- | :---: |
    | **PCA** | variance directions | no |
    | **MDS** | pairwise distances | no |
    | **LLE** | neighbour weights | no |
    | **Isomap** | graph shortest paths | no |
    | **t-SNE** | neighbour probabilities | no |

    Only **PCA** axes are interpretable loadings; other plots have arbitrary axes — read **groupings**, not axis meaning. High PCA variance share still may not separate species; t-SNE keeps neighbourhoods, not distances, and has no **`transform`**.

    ## Datasets used

    **Iris** (Fisher, 1936; [UCI](https://archive.ics.uci.edu/dataset/53/iris)): 150 flowers, four cm measurements, three species. Setosa is easy to split; versicolor and virginica overlap in 4-D.
    """)
    return


@app.cell
def _(SPECIES_ORDER, go, px):
    def species_scatter(frame, x, y, title="", equal=False, width=840, height=500):
        """2-D species scatter with go.Scatter traces (width and height capped)."""
        width = min(int(width), 840)
        height = min(int(height), 500)
        figure = go.Figure()
        for index, species in enumerate(SPECIES_ORDER):
            mask = frame["species"] == species
            figure.add_trace(
                go.Scatter(
                    x=frame.loc[mask, x],
                    y=frame.loc[mask, y],
                    mode="markers",
                    name=species,
                    marker=dict(size=7, opacity=0.85, color=px.colors.qualitative.G10[index]),
                )
            )
        figure.update_layout(
            width=width,
            height=height,
            title=title,
            margin=dict(t=55, b=45, l=55, r=20),
            legend_title_text="species",
        )
        if equal:
            figure.update_yaxes(scaleanchor="x", scaleratio=1)
        return figure

    return (species_scatter,)


@app.cell
def _(SPECIES_ORDER, go, np, px):
    def pca_biplot(scores, species, loadings, names, share, title="", width=840, height=500):
        """PCA scores with loading vectors drawn from the origin."""
        width = min(int(width), 840)
        height = min(int(height), 500)
        figure = go.Figure()
        for index, label in enumerate(SPECIES_ORDER):
            mask = np.asarray(species) == label
            figure.add_trace(
                go.Scatter(
                    x=scores[mask, 0],
                    y=scores[mask, 1],
                    mode="markers",
                    name=label,
                    marker=dict(size=7, opacity=0.85, color=px.colors.qualitative.G10[index]),
                )
            )
        _scale = 3.0 * np.sqrt(float(np.sum(share)))
        for index, name in enumerate(names):
            _xl = loadings[0, index] * _scale
            _yl = loadings[1, index] * _scale
            figure.add_trace(
                go.Scatter(
                    x=[0, _xl],
                    y=[0, _yl],
                    mode="lines+markers",
                    name=name,
                    line=dict(width=2, color="dimgray"),
                    marker=dict(size=6, color="dimgray"),
                    showlegend=True,
                )
            )
            figure.add_annotation(
                x=_xl,
                y=_yl,
                text=name,
                showarrow=False,
                font=dict(color="dimgray", size=10),
                xanchor="left" if _xl >= 0 else "right",
                yanchor="bottom" if _yl >= 0 else "top",
            )
        figure.update_layout(
            width=width,
            height=height,
            title=title,
            margin=dict(t=55, b=45, l=55, r=20),
            legend_title_text="species / loadings",
        )
        figure.update_yaxes(scaleanchor="x", scaleratio=1)
        return figure

    return (pca_biplot,)


@app.cell
def _(scipy_procrustes):
    def layout_shift(baseline, comparison):
        """Procrustes disparity after aligning two layouts (0 = identical shape)."""
        _, _, disparity = scipy_procrustes(baseline, comparison)
        return float(disparity)

    return (layout_shift,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "About `species_scatter`": mo.md(
                "One trace per species with shared colours across embeddings; plot size is capped for the notebook layout."
            ),
            "About `pca_biplot`": mo.md(
                "Scores plus grey loading arrows from the origin; length reflects variance share and labels sit at each arrow tip."
            ),
            "About `layout_shift`": mo.md(
                "Procrustes disparity after aligning two layouts — near zero when shapes match, larger when settings rearrange the map."
            ),
        },
        expanded=False,
    )
    return


@app.cell
def _(load_iris, pd):
    iris = load_iris()
    iris_frame = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_frame["species"] = iris.target_names[iris.target]
    measure_columns = list(iris.feature_names)
    species_counts = iris_frame.species.value_counts()
    species_facts = (
        f"{len(iris_frame)} flowers: "
        + ", ".join(f"{_name} {_count}" for _name, _count in species_counts.items())
        + f". Missing values: {int(iris_frame.isnull().sum().sum())}"
    )
    spreads = iris_frame[measure_columns].max() - iris_frame[measure_columns].min()
    spread_ratio = float(spreads.max() / spreads.min())
    widest_measurement = spreads.idxmax()
    petal_length_spread = float(spreads["petal length (cm)"])
    narrow_spread = float(spreads.min())
    return (
        iris,
        iris_frame,
        measure_columns,
        narrow_spread,
        petal_length_spread,
        species_counts,
        species_facts,
        spread_ratio,
        spreads,
        widest_measurement,
    )


@app.cell(hide_code=True)
def _(
    iris_frame,
    measure_columns,
    mo,
    narrow_spread,
    petal_length_spread,
    species_facts,
    spread_ratio,
    spreads,
    widest_measurement,
):
    _rows = "\n".join(
        f"| {_name} | {iris_frame[_name].min():.1f} | {iris_frame[_name].max():.1f} | {spreads[_name]:.1f} |"
        for _name in measure_columns
    )
    mo.md(f"""
    **{species_facts}.**

    Petal length spreads **{petal_length_spread:.1f} cm**; sepal width and petal width tie at **{narrow_spread:.1f} cm**. The ratio of widest to narrowest spread is **{spread_ratio:.2f}** ({widest_measurement} vs the {narrow_spread:.1f} cm columns).

    | measurement | min | max | spread |
    | :--- | ---: | ---: | ---: |
    {_rows}

    On the raw scale, distance methods would overweight petal length. Standardize with \\(z_{{ij}}=(x_{{ij}}-\\mu_j)/\\sigma_j\\). **Setosa** has short petals; **versicolor** and **virginica** overlap on all four.
    """)
    return


@app.cell
def _(SPECIES_ORDER, iris_frame, measure_columns, pd, px):
    _long = pd.melt(
        iris_frame,
        id_vars="species",
        value_vars=measure_columns,
        var_name="measurement",
        value_name="centimetres",
    )
    _figure = px.box(
        _long,
        x="measurement",
        y="centimetres",
        color="species",
        title="The four measurements by species, in original centimetres",
        category_orders={"species": SPECIES_ORDER},
    )
    _figure.update_layout(width=840, height=500, margin=dict(t=60, b=80, l=55, r=20))
    _figure.update_xaxes(tickangle=-20)
    _figure
    return


@app.cell
def _(StandardScaler, iris, iris_frame, measure_columns, np, pd):
    iris_z = pd.DataFrame(
        StandardScaler().fit_transform(iris_frame[measure_columns]),
        columns=measure_columns,
        index=iris_frame.index,
    )
    iris_z["species"] = iris_frame["species"]
    _X = iris_z[measure_columns].to_numpy()
    _virginica = np.flatnonzero(iris.target == 2)
    _centroid = _X[iris.target == 1].mean(axis=0)
    highlight_row = int(
        _virginica[np.argmin(np.linalg.norm(_X[_virginica] - _centroid, axis=1))]
    )
    return highlight_row, iris_z


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Replace `None` so that `student_widest` holds the measurement with the largest spread and `student_ratio` holds how many times larger that spread is than the smallest. Use `iris_frame` and `measure_columns`.
    """)
    return


@app.cell
def _():
    student_widest = None
    student_ratio = None
    return student_ratio, student_widest


@app.cell(hide_code=True)
def _(
    mo,
    narrow_spread,
    petal_length_spread,
    spread_ratio,
    widest_measurement,
):
    mo.accordion(
        {
            "Show answer": mo.md(f"""
```python
spreads = iris_frame[measure_columns].max() - iris_frame[measure_columns].min()
student_widest = spreads.idxmax()
student_ratio = spreads.max() / spreads.min()
```
**{widest_measurement}** spreads **{petal_length_spread:.1f} cm**; sepal width and petal width tie at **{narrow_spread:.1f} cm**. The ratio is **{spread_ratio:.2f}**, not three-to-one.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Principal components

    PCA finds orthogonal directions of maximal variance. With two components, the **explained variance share** is

    $$
    \frac{\lambda_1 + \lambda_2}{\sum_k \lambda_k}.
    $$

    The biplot shows scores and **loadings** — how each measurement contributes to PC1 and PC2.
    """)
    return


@app.cell
def _(PCA, iris_z, measure_columns, np, pd, pca_biplot):
    iris_pca_model = PCA(n_components=2).fit(iris_z[measure_columns])
    iris_share = iris_pca_model.explained_variance_ratio_
    _plane = iris_pca_model.transform(iris_z[measure_columns])
    iris_pca_frame = pd.DataFrame(
        {"PCA 1": _plane[:, 0], "PCA 2": _plane[:, 1], "species": iris_z["species"]}
    )
    iris_pca_figure = pca_biplot(
        _plane,
        iris_z["species"].to_numpy(),
        iris_pca_model.components_,
        measure_columns,
        iris_share,
        title="PCA biplot on standardized measurements",
    )
    iris_pca_figure
    return iris_pca_model, iris_share


@app.cell(hide_code=True)
def _(iris_pca_model, iris_share, measure_columns, mo):
    _loadings = iris_pca_model.components_
    mo.md(f"""
    Components keep **{iris_share[0]:.1%}** and **{iris_share[1]:.1%}** (**{iris_share.sum():.1%}** total).

    | measurement | PC1 | PC2 |
    | :--- | ---: | ---: |
    {chr(10).join(f"| {_name} | {_loadings[0, _i]:+.2f} | {_loadings[1, _i]:+.2f} |" for _i, _name in enumerate(measure_columns))}

    **PC1** mixes petal length and width with **sepal length** (same sign) — a general size axis. **PC2** is dominated by **sepal width**, and **both sepal loadings are positive**, so high PC2 means **wider and somewhat longer** sepals.

    Setosa separates on the left; versicolor and virginica form a **touching band** with no empty gap. Keeping **{iris_share.sum():.1%}** of the variance does not by itself separate those species.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Replace `None` so that `student_pca_share` holds the combined share of variance for the two PCA components (between 0 and 1).
    """)
    return


@app.cell
def _():
    student_pca_share = None
    return (student_pca_share,)


@app.cell(hide_code=True)
def _(iris_share, mo):
    mo.accordion(
        {
            "Show answer": mo.md(f"""
```python
student_pca_share = iris_share.sum()
```
The two components keep **{iris_share.sum():.1%}** of the variance — high spread capture, not guaranteed species separation.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Only PCA names its axes. In one sentence, why can MDS or t-SNE not expose loadings the same way?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answer": mo.md(r"""
They optimise **layout coordinates** to match distances or neighbourhood probabilities, not a fixed linear projection of the original columns — so there is no weight vector per measurement to read from an axis.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Four non-linear embeddings

    **MDS stress** (metric case) compares embedded distances \(\hat d_{ij}\) to input distances \(d_{ij}\):

    $$
    \sigma = \sum_{i<j}\bigl(d_{ij}-\hat d_{ij}\bigr)^2.
    $$

    **t-SNE** minimises KL divergence between neighbour probabilities \(P\) and \(Q\):

    $$
    C = \sum_i \mathrm{KL}(P_i \,\|\, Q_i).
    $$

    Same standardized input, four different criteria — axes arbitrary, read groupings only.
    """)
    return


@app.cell
def _(MDS, iris_z, measure_columns, pd, species_scatter):
    _mds = MDS(
        n_components=2,
        metric_mds=True,
        init="random",
        n_init=1,
        random_state=25,
        normalized_stress="auto",
    ).fit_transform(iris_z[measure_columns])
    species_scatter(
        pd.DataFrame({"MDS 1": _mds[:, 0], "MDS 2": _mds[:, 1], "species": iris_z["species"]}),
        "MDS 1",
        "MDS 2",
        "MDS: matching pairwise distances",
    )
    return


@app.cell
def _(Isomap, iris_z, measure_columns, pd, species_scatter):
    _isomap = Isomap(n_neighbors=30, n_components=2, eigen_solver="dense").fit_transform(
        iris_z[measure_columns]
    )
    species_scatter(
        pd.DataFrame(
            {"Isomap 1": _isomap[:, 0], "Isomap 2": _isomap[:, 1], "species": iris_z["species"]}
        ),
        "Isomap 1",
        "Isomap 2",
        "Isomap on a neighbour graph",
    )
    return


@app.cell
def _(TSNE, iris_z, measure_columns, np, pd):
    _baseline = TSNE(
        random_state=20, init="random", perplexity=30
    ).fit_transform(iris_z[measure_columns])
    tsne_baseline = pd.DataFrame(
        {
            "t-SNE 1": _baseline[:, 0],
            "t-SNE 2": _baseline[:, 1],
            "species": iris_z["species"],
        },
        index=iris_z.index,
    )
    return (tsne_baseline,)


@app.cell
def _(mo):
    tsne_perplexity_slider = mo.ui.slider(
        5, 50, value=30, step=5, show_value=True, label="t-SNE perplexity"
    )
    tsne_seed_slider = mo.ui.slider(
        0, 99, value=20, step=1, show_value=True, label="t-SNE random seed"
    )
    lle_neighbors_slider = mo.ui.slider(
        10, 50, value=30, step=5, show_value=True, label="LLE n_neighbors"
    )
    mo.hstack([tsne_perplexity_slider, tsne_seed_slider, lle_neighbors_slider], gap=2)
    return lle_neighbors_slider, tsne_perplexity_slider, tsne_seed_slider


@app.cell
def _(
    TSNE,
    iris_z,
    layout_shift,
    measure_columns,
    pd,
    species_scatter,
    tsne_baseline,
    tsne_perplexity_slider,
    tsne_seed_slider,
):
    _perp = int(tsne_perplexity_slider.value)
    _seed = int(tsne_seed_slider.value)
    _tsne = TSNE(
        random_state=_seed, init="random", perplexity=_perp
    ).fit_transform(iris_z[measure_columns])
    tsne_shift = layout_shift(
        tsne_baseline[["t-SNE 1", "t-SNE 2"]].to_numpy(), _tsne
    )
    tsne_live_frame = pd.DataFrame(
        {"t-SNE 1": _tsne[:, 0], "t-SNE 2": _tsne[:, 1], "species": iris_z["species"]}
    )
    tsne_live_figure = species_scatter(
        tsne_live_frame,
        "t-SNE 1",
        "t-SNE 2",
        f"t-SNE (perplexity {_perp}, seed {_seed}; Procrustes shift {tsne_shift:.3f})",
    )
    tsne_live_figure
    return (tsne_shift,)


@app.cell
def _(
    LocallyLinearEmbedding,
    iris_z,
    lle_neighbors_slider,
    measure_columns,
    pd,
    species_scatter,
):
    _n = int(lle_neighbors_slider.value)
    _lle = LocallyLinearEmbedding(
        n_neighbors=_n,
        n_components=2,
        random_state=50,
        method="modified",
        eigen_solver="dense",
    ).fit_transform(iris_z[measure_columns])
    lle_live_figure = species_scatter(
        pd.DataFrame({"LLE 1": _lle[:, 0], "LLE 2": _lle[:, 1], "species": iris_z["species"]}),
        "LLE 1",
        "LLE 2",
        f"LLE ({_n} neighbours)",
    )
    lle_live_figure
    return


@app.cell
def _(
    KNeighborsClassifier,
    PCA,
    TSNE,
    cross_val_score,
    iris,
    iris_z,
    measure_columns,
    pd,
):
    _X = iris_z[measure_columns].to_numpy()
    _pca = PCA(n_components=2).fit_transform(_X)
    _tsne = TSNE(random_state=20, init="random", perplexity=30).fit_transform(_X)
    _checker = KNeighborsClassifier(n_neighbors=5)
    knn_raw = float(
        cross_val_score(_checker, _X, iris.target, cv=5, n_jobs=1).mean()
    )
    knn_pca = float(
        cross_val_score(_checker, _pca, iris.target, cv=5, n_jobs=1).mean()
    )
    knn_tsne = float(
        cross_val_score(_checker, _tsne, iris.target, cv=5, n_jobs=1).mean()
    )
    knn_table = pd.DataFrame(
        [
            ("Raw standardized", knn_raw, "Original 4-D z-scores"),
            ("PCA (2D)", knn_pca, "Linear 2-D projection"),
            (
                "t-SNE (2D)",
                knn_tsne,
                f"Beats PCA ({knn_pca:.3f}) in 2-D, below raw ({knn_raw:.3f})",
            ),
        ],
        columns=["Embedding", "5-NN accuracy", "Notes"],
    )
    knn_table
    return knn_pca, knn_raw, knn_tsne


@app.cell(hide_code=True)
def _(knn_pca, knn_raw, knn_tsne, mo):
    mo.md(f"""
    **5-NN species accuracy (cross-validated):** raw standardized **{knn_raw:.3f}**, PCA **{knn_pca:.3f}**, t-SNE **{knn_tsne:.3f}**. t-SNE separates species better than PCA in two dimensions (**{knn_tsne:.3f}** vs **{knn_pca:.3f}**) but still scores below the raw four-dimensional input (**{knn_raw:.3f}**). Versicolor and virginica still **touch**; there is no empty channel between them.
    """)
    return


@app.cell(hide_code=True)
def _(highlight_row, mo):
    mo.md(f"""
    ## 3. Row {highlight_row} in t-SNE space

    We highlight the **virginica** flower closest to the **versicolor** centroid in standardized 4-D (row **{highlight_row}** on this table). It sits among versicolor points in the baseline t-SNE map — check its neighbours and raw measurements.
    """)
    return


@app.cell
def _(NearestNeighbors, highlight_row, iris, iris_z, measure_columns, np):
    _X = iris_z[measure_columns].to_numpy()
    _nn = NearestNeighbors(n_neighbors=6).fit(_X)
    _indices = _nn.kneighbors(_X[highlight_row : highlight_row + 1], return_distance=False)[0, 1:]
    highlight_neighbor_species = iris.target_names[iris.target[_indices]]
    highlight_versicolor_neighbors = int(np.sum(iris.target[_indices] == 1))
    return highlight_neighbor_species, highlight_versicolor_neighbors


@app.cell
def _(go, highlight_row, species_scatter, tsne_baseline):
    _highlight = tsne_baseline.copy()
    _row = _highlight.loc[highlight_row]
    _figure = species_scatter(
        _highlight,
        "t-SNE 1",
        "t-SNE 2",
        f"Baseline t-SNE with row {highlight_row} circled",
    )
    _figure.add_trace(
        go.Scatter(
            x=[_row["t-SNE 1"]],
            y=[_row["t-SNE 2"]],
            mode="markers",
            marker=dict(size=14, color="rgba(0,0,0,0)", line=dict(color="black", width=2)),
            name=f"row {highlight_row}",
            showlegend=True,
        )
    )
    _figure
    return


@app.cell(hide_code=True)
def _(highlight_row, highlight_neighbor_species, highlight_versicolor_neighbors, mo):
    _list = ", ".join(highlight_neighbor_species)
    mo.md(f"""
    In standardized 4-D space, **{highlight_versicolor_neighbors} of 5** nearest neighbours of row **{highlight_row}** are **versicolor** ({_list}). t-SNE is honouring that overlap — not claiming the flower is atypical in centimetres.
    """)
    return


@app.cell
def _(highlight_row, iris_frame, measure_columns, np):
    _virginica_raw = iris_frame.loc[iris_frame.species == "virginica", measure_columns]
    _values = iris_frame.loc[highlight_row, measure_columns].astype(float)
    _rows = []
    for _name in measure_columns:
        _q1, _q3 = _virginica_raw[_name].quantile([0.25, 0.75])
        _iqr = _q3 - _q1
        _low, _high = _q1 - 1.5 * _iqr, _q3 + 1.5 * _iqr
        _value = float(_values[_name])
        _rows.append(
            {
                "measurement": _name,
                "value": _value,
                "virginica low": _low,
                "virginica high": _high,
                "outside whiskers": bool(_value < _low or _value > _high),
            }
        )
    outlier_check = np.array([_r["outside whiskers"] for _r in _rows])
    outlier_table = _rows
    return outlier_check, outlier_table


@app.cell
def _(mo, outlier_table, pd):
    mo.Html(pd.DataFrame(outlier_table).set_index("measurement").round(2).to_html())
    return


@app.cell(hide_code=True)
def _(mo, outlier_check):
    mo.md(f"""
    **None** of the four measurements fall outside virginica whiskers (**{int(outlier_check.sum())}** flags). Embedding position reports **neighbours**; the table reports **typicality**.
    """)
    return


@app.cell(hide_code=True)
def _(highlight_row, mo):
    mo.md(f"""
    ## 4. Refitting without row {highlight_row}

    t-SNE has no **`transform`**. The next cell refits on 149 flowers (same seed and perplexity as the baseline) and reports how much the layout moves for the flowers that remain.
    """)
    return


@app.cell
def _(TSNE, highlight_row, iris_z, layout_shift, measure_columns, np, tsne_baseline):
    _X = iris_z[measure_columns].to_numpy()
    _ref = tsne_baseline[["t-SNE 1", "t-SNE 2"]].to_numpy()
    _kept = np.delete(_X, highlight_row, axis=0)
    _refit = TSNE(random_state=20, init="random", perplexity=30).fit_transform(_kept)
    _common = tsne_baseline.drop(index=highlight_row)[["t-SNE 1", "t-SNE 2"]].to_numpy()
    refit_shift = layout_shift(_common, _refit)
    tsne_shift_p5 = layout_shift(
        _ref, TSNE(random_state=20, init="random", perplexity=5).fit_transform(_X)
    )
    tsne_shift_p25 = layout_shift(
        _ref, TSNE(random_state=20, init="random", perplexity=25).fit_transform(_X)
    )
    tsne_shift_p50 = layout_shift(
        _ref, TSNE(random_state=20, init="random", perplexity=50).fit_transform(_X)
    )
    tsne_shift_seed0 = layout_shift(
        _ref, TSNE(random_state=0, init="random", perplexity=30).fit_transform(_X)
    )
    return refit_shift, tsne_shift_p5, tsne_shift_p25, tsne_shift_p50, tsne_shift_seed0


@app.cell(hide_code=True)
def _(
    highlight_row,
    mo,
    refit_shift,
    tsne_shift_p5,
    tsne_shift_p25,
    tsne_shift_p50,
    tsne_shift_seed0,
):
    mo.md(f"""
    Dropping row **{highlight_row}** moves the remaining layout only **{refit_shift:.3f}** (Procrustes). **Perplexity** matters mainly at low values versus the baseline map (**{tsne_shift_p5:.3f}** at 5, **{tsne_shift_p25:.3f}** at 25, **{tsne_shift_p50:.3f}** at 50); **seed** nudges it too (**{tsne_shift_seed0:.3f}** at seed 0). A new flower still needs a full refit — there is no **`transform`**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    `iris_z` is already standardized. Fit **PCA** with two components on `measure_columns` and store the coordinates in `student_iris_pca` with shape `(150, 2)`.
    """)
    return


@app.cell
def _():
    student_iris_pca = None
    return (student_iris_pca,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answer": mo.md(r"""
```python
student_iris_pca = PCA(n_components=2).fit_transform(iris_z[measure_columns])
```
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(highlight_row, highlight_versicolor_neighbors, mo):
    mo.md(f"""
    ### Try it yourself

    Row **{highlight_row}** has **{highlight_versicolor_neighbors}** versicolor flowers among its five nearest neighbours in standardized 4-D. Does that force it to look mislabelled in t-SNE? Answer in one sentence.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answer": mo.md(r"""
No — t-SNE keeps **local neighbourhoods**, so a virginica surrounded by versicolor in measurement space can sit among them in the plane while still having ordinary virginica measurements.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(highlight_row, mo):
    mo.md(f"""
    ### Try it yourself

    Using `NearestNeighbors` on `iris_z[measure_columns]`, store the count of versicolor flowers among the five nearest neighbours of row **{highlight_row}** in `student_highlight_versicolor` (an integer).
    """)
    return


@app.cell
def _():
    student_highlight_versicolor = None
    return (student_highlight_versicolor,)


@app.cell(hide_code=True)
def _(highlight_row, highlight_versicolor_neighbors, iris, iris_z, measure_columns, mo, np):
    mo.accordion(
        {
            "Show answer": mo.md(f"""
```python
_X = iris_z[measure_columns].to_numpy()
_nn = NearestNeighbors(n_neighbors=6).fit(_X)
_indices = _nn.kneighbors(_X[{highlight_row}:{highlight_row + 1}], return_distance=False)[0, 1:]
student_highlight_versicolor = int(np.sum(iris.target[_indices] == 1))
```
The count is **{highlight_versicolor_neighbors}** versicolor neighbours (target code 1).
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(
    highlight_row,
    highlight_versicolor_neighbors,
    iris_share,
    mo,
    refit_shift,
    tsne_shift_p5,
    tsne_shift_p25,
    tsne_shift_p50,
    tsne_shift_seed0,
):
    mo.md(f"""
    ## Conclusions

    1. **Different optima, different pictures:** PCA, MDS, LLE, Isomap, and t-SNE share the same four standardized columns yet preserve variance, distances, weights, graph paths, or neighbourhoods — so their 2-D groupings disagree even when setosa always separates.
    2. **Variance share ≠ class separation:** two PCA components keep **{iris_share.sum():.1%}** of the variance, but versicolor and virginica still overlap in the biplot because they overlap in measurement space.
    3. **Meaningful axes only for PCA:** MDS, LLE, Isomap, and t-SNE axes are arbitrary; read **which species cluster**, not what the horizontal axis "means".
    4. **t-SNE is a fit, not a map:** there is no **`transform`** for a new flower; dropping row **{highlight_row}** shifts the layout only **{refit_shift:.3f}**. Procrustes shifts vs the baseline map are **{tsne_shift_p5:.3f}** at perplexity 5, **{tsne_shift_p25:.3f}** at 25, **{tsne_shift_p50:.3f}** at 50, and **{tsne_shift_seed0:.3f}** when the seed is 0 — perplexity matters mainly at low values.
    5. **Neighbourhood ≠ typicality:** row **{highlight_row}** sits among versicolor in t-SNE with **{highlight_versicolor_neighbors} of 5** versicolor neighbours in 4-D, yet its centimetre values stay inside virginica whiskers — check measurements, not plot position alone.
    """)
    return


if __name__ == "__main__":
    app.run()
