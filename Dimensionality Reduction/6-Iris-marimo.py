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

    This lesson is the synthesis for the folder. Every method here has had its own lesson, and now they are applied to one dataset so their pictures can be compared.

    All five take the **same input**: four standardized iris measurements. All five return **two coordinates per flower**. What differs is the criterion each one optimises, and therefore which relationships survive the collapse to two dimensions.

    | Method | What it tries to preserve | Uses labels? |
    | :--- | :--- | :--- |
    | **PCA** | the directions of greatest variance | no |
    | **MDS** | the pairwise distances between flowers | no |
    | **Locally linear embedding** | each flower's reconstruction weights from its neighbours | no |
    | **Isomap** | shortest-path distances along a neighbour graph | no |
    | **t-SNE** | the probability that two flowers are neighbours | no |

    **None of them uses the species labels.** The colours are added afterwards, which is what makes these plots a fair test: if named species land in separate regions, the method found structure it was never told about.

    ### Why the axes have no meaning

    Only PCA produces axes with an interpretation: its first axis is the direction of greatest variance, and its components are weighted mixtures of the four measurements. The other four are **non-linear embeddings** whose axes are arbitrary — a rotated or flipped version of the same picture is equally valid.

    So when reading any of the non-PCA plots, ask *which groups are separate*, not *what the x axis means*. There is no answer to the second question.

    ### Two cautions that this lesson is built around

    **A high variance share is not the same as a good picture.** PCA will report that its two components keep a large fraction of the variance, and it is tempting to read that as "these two dimensions are enough". Whether they separate the species is a separate question, and the plot answers it, not the percentage.

    **t-SNE distances are not measurements.** t-SNE preserves *neighbourhoods*, not distances. Two flowers close together in its picture are neighbours in the original space; wide visual gaps are not measured distances. And because it has no `transform` method, placing a new flower requires refitting the whole map.

    That last property is the subject of the final section, and it teaches something that generalises: **a picture is a fit, not a projection.**

    ## Datasets used

    ### Iris

    **Source.** The iris dataset, collected by the botanist **Edgar Anderson** and published by **R. A. Fisher** in 1936 in *The use of multiple measurements in taxonomic problems* (*Annals of Eugenics*, 7(2), 179–188). Loaded here with `sklearn.datasets.load_iris`; also distributed by the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/53/iris).

    **What it holds.** 150 flowers, 50 from each of three species, with four measurements in centimetres: sepal length, sepal width, petal length, and petal width. The species are *Iris setosa*, *Iris versicolor*, and *Iris virginica*.

    **Why it suits this lesson.** Setosa is clearly separable from the other two on these measurements. Versicolor and virginica overlap, and no two-dimensional view of four measurements can fully separate them, because some flowers genuinely have intermediate values.
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `species_scatter`

    One trace per species using **`go.Scatter`**. Embeddings use free axis scaling so clusters are not squashed; PCA biplots may request equal scaling when comparing score and loading geometry.
    """)
    return


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
            figure.add_trace(
                go.Scatter(
                    x=[0, loadings[0, index] * _scale],
                    y=[0, loadings[1, index] * _scale],
                    mode="lines+markers",
                    name=name,
                    line=dict(width=2),
                    marker=dict(size=6),
                    showlegend=True,
                )
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `pca_biplot`

    Flower scores plus **loading vectors** for each measurement, scaled by explained variance so arrow length reflects how much that feature contributes to the plane. This replaces a separate arrow overlay on a second figure.
    """)
    return


@app.cell
def _(scipy_procrustes):
    def layout_shift(baseline, comparison):
        """Procrustes disparity after aligning two layouts (0 = identical shape)."""
        _, _, disparity = scipy_procrustes(baseline, comparison)
        return float(disparity)

    return (layout_shift,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `layout_shift`

    Compares two embeddings of the same flowers after removing rotation, reflection, and scale. A small value means the **neighbourhood structure** barely moved; a large value means seed or perplexity reshaped the map.
    """)
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
    _spreads = iris_frame[measure_columns].max() - iris_frame[measure_columns].min()
    spread_ratio = float(_spreads.max() / _spreads.min())
    widest_measurement = _spreads.idxmax()
    return (
        iris,
        iris_frame,
        measure_columns,
        species_counts,
        species_facts,
        spread_ratio,
        widest_measurement,
        _spreads,
    )


@app.cell(hide_code=True)
def _(iris_frame, measure_columns, mo, species_facts, spread_ratio, widest_measurement, _spreads):
    _rows = "\n".join(
        f"| {_name} | {iris_frame[_name].min():.1f} | {iris_frame[_name].max():.1f} | {_spreads[_name]:.1f} |"
        for _name in measure_columns
    )
    mo.md(f"""
    **{species_facts}.**

    Petal length spreads **{_spreads['petal length (cm)']:.1f} cm**; sepal width and petal width tie at **{_spreads['sepal width (cm)']:.1f} cm**. The ratio of widest to narrowest spread is **{spread_ratio:.2f}** ({widest_measurement} vs the 2.4 cm columns).

    | measurement | min | max | spread |
    | :--- | ---: | ---: | ---: |
    {_rows}

    On the raw scale, distance methods would overweight petal length. Standardization fixes that.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Standardization

    Each column is centred and scaled to unit variance:

    $$
    z_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}.
    $$

    The boxes below use **raw** centimetres; every embedding in this lesson uses **`z`**.
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
def _(StandardScaler, iris_frame, measure_columns, pd):
    iris_z = pd.DataFrame(
        StandardScaler().fit_transform(iris_frame[measure_columns]),
        columns=measure_columns,
        index=iris_frame.index,
    )
    iris_z["species"] = iris_frame["species"]
    return (iris_z,)


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
def _(mo, spread_ratio, widest_measurement):
    mo.accordion(
        {
            "Show answer": mo.md(f"""
```python
_spreads = iris_frame[measure_columns].max() - iris_frame[measure_columns].min()
student_widest = _spreads.idxmax()
student_ratio = _spreads.max() / _spreads.min()
```
**{widest_measurement}** spreads **5.9 cm**; sepal width and petal width tie at **2.4 cm**. The ratio is **{spread_ratio:.2f}**, not three-to-one.
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
def _(PCA, iris_z, measure_columns, np, pca_biplot):
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
    return iris_pca_frame, iris_pca_model, iris_share


@app.cell(hide_code=True)
def _(iris_pca_model, iris_share, measure_columns, mo):
    _loadings = iris_pca_model.components_
    mo.md(f"""
    Components keep **{iris_share[0]:.1%}** and **{iris_share[1]:.1%}** (**{iris_share.sum():.1%}** total).

    | measurement | PC1 | PC2 |
    | :--- | ---: | ---: |
    {chr(10).join(f"| {_name} | {_loadings[0, _i]:+.2f} | {_loadings[1, _i]:+.2f} |" for _i, _name in enumerate(measure_columns))}

    **PC1** mixes petal length and width with **sepal length** (same sign) — a general size axis. **PC2** is dominated by **sepal width**, with **sepal length** loading the same way (both positive here): wide, relatively short sepals versus the opposite.

    Setosa separates on the left; versicolor and virginica form a **touching band** with no empty gap.
    """)
    return


@app.cell(hide_code=True)
def _(iris_share, mo):
    mo.md(f"""
    **What the variance share does not promise.** Two components keep **{iris_share.sum():.1%}** of the variance — a faithful summary of spread — yet versicolor and virginica still overlap because they overlap in four dimensions.
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
def _(LocallyLinearEmbedding, iris_z, measure_columns, pd, species_scatter):
    _lle = LocallyLinearEmbedding(
        n_neighbors=30,
        n_components=2,
        random_state=50,
        method="modified",
        eigen_solver="dense",
    ).fit_transform(iris_z[measure_columns])
    species_scatter(
        pd.DataFrame({"LLE 1": _lle[:, 0], "LLE 2": _lle[:, 1], "species": iris_z["species"]}),
        "LLE 1",
        "LLE 2",
        "Locally linear embedding",
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
    _baseline = TSNE(random_state=20, init="pca", perplexity=30).fit_transform(
        iris_z[measure_columns]
    )
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
        5, 50, value=30, step=5, show_value=True, label="LLE n_neighbors"
    )
    mo.hstack([tsne_perplexity_slider, tsne_seed_slider, lle_neighbors_slider], gap=2)
    return lle_neighbors_slider, tsne_perplexity_slider, tsne_seed_slider


@app.cell
def _(
    LocallyLinearEmbedding,
    TSNE,
    iris_z,
    layout_shift,
    lle_neighbors_slider,
    measure_columns,
    pd,
    species_scatter,
    tsne_baseline,
    tsne_perplexity_slider,
    tsne_seed_slider,
):
    _perp = int(tsne_perplexity_slider.value)
    _seed = int(tsne_seed_slider.value)
    _n = int(lle_neighbors_slider.value)
    _tsne = TSNE(
        random_state=_seed, init="pca", perplexity=_perp
    ).fit_transform(iris_z[measure_columns])
    _lle = LocallyLinearEmbedding(
        n_neighbors=_n,
        n_components=2,
        random_state=50,
        method="modified",
        eigen_solver="dense",
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
    lle_live_figure = species_scatter(
        pd.DataFrame({"LLE 1": _lle[:, 0], "LLE 2": _lle[:, 1], "species": iris_z["species"]}),
        "LLE 1",
        "LLE 2",
        f"LLE ({_n} neighbours)",
    )
    mo.vstack([tsne_live_figure, lle_live_figure])
    return tsne_shift,


@app.cell
def _(
    KNeighborsClassifier,
    PCA,
    TSNE,
    cross_val_score,
    iris,
    iris_z,
    measure_columns,
    mo,
    pd,
):
    _X = iris_z[measure_columns].to_numpy()
    _pca = PCA(n_components=2).fit_transform(_X)
    _tsne = TSNE(random_state=20, init="pca", perplexity=30).fit_transform(_X)
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
    _rows = [
        ("Raw standardized", knn_raw, "Original 4-D z-scores", "—"),
        ("PCA (2D)", knn_pca, "Linear projection", "—"),
        ("t-SNE (2D)", knn_tsne, "Strongest 2-D separation", "—"),
    ]
    knn_table = pd.DataFrame(
        _rows, columns=["Embedding", "5-NN accuracy", "Notes", "Extra"]
    )
    mo.Html(knn_table.to_html(index=False, float_format=lambda x: f"{x:.3f}"))
    return knn_pca, knn_raw, knn_tsne


@app.cell(hide_code=True)
def _(knn_pca, knn_raw, knn_tsne, mo):
    mo.md(f"""
    **5-NN species accuracy (cross-validated):** raw standardized **{knn_raw:.3f}**, PCA **{knn_pca:.3f}**, t-SNE **{knn_tsne:.3f}** — t-SNE separates classes most in the plane, but versicolor and virginica still **touch**; there is no empty channel between them.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Row 133 in t-SNE space

    The virginica flower **row 133** sits among versicolor points in the baseline t-SNE map. Check its **four-dimensional neighbours** and its raw measurements.
    """)
    return


@app.cell
def _(NearestNeighbors, iris, iris_z, measure_columns, np):
    _X = iris_z[measure_columns].to_numpy()
    _nn = NearestNeighbors(n_neighbors=6).fit(_X)
    _indices = _nn.kneighbors(_X[133:134], return_distance=False)[0, 1:]
    row133_neighbor_species = iris.target_names[iris.target[_indices]]
    row133_versicolor_neighbors = int(np.sum(iris.target[_indices] == 1))
    return row133_neighbor_species, row133_versicolor_neighbors


@app.cell
def _(go, species_scatter, tsne_baseline):
    _highlight = tsne_baseline.copy()
    _row = _highlight.loc[133]
    _figure = species_scatter(
        _highlight,
        "t-SNE 1",
        "t-SNE 2",
        "Baseline t-SNE with row 133 circled",
    )
    _figure.add_trace(
        go.Scatter(
            x=[_row["t-SNE 1"]],
            y=[_row["t-SNE 2"]],
            mode="markers",
            marker=dict(size=14, color="rgba(0,0,0,0)", line=dict(color="black", width=2)),
            name="row 133",
            showlegend=True,
        )
    )
    _figure
    return


@app.cell(hide_code=True)
def _(mo, row133_neighbor_species, row133_versicolor_neighbors):
    _list = ", ".join(row133_neighbor_species)
    mo.md(f"""
    In standardized 4-D space, **{row133_versicolor_neighbors} of 5** nearest neighbours of row 133 are **versicolor** ({_list}). t-SNE is honouring that overlap — not claiming the flower is atypical in centimetres.
    """)
    return


@app.cell
def _(iris_frame, measure_columns, np):
    _virginica_raw = iris_frame.loc[iris_frame.species == "virginica", measure_columns]
    _values = iris_frame.loc[133, measure_columns].astype(float)
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
def _(mo):
    mo.md(r"""
    ## 4. Refitting without row 133

    t-SNE has no **`transform`**. The next cell refits on 149 flowers (same seed and perplexity as the baseline) and reports how much the layout moves for the flowers that remain.
    """)
    return


@app.cell
def _(TSNE, iris_z, layout_shift, measure_columns, np, tsne_baseline):
    _X = iris_z[measure_columns].to_numpy()
    _kept = np.delete(_X, 133, axis=0)
    _refit = TSNE(random_state=20, init="pca", perplexity=30).fit_transform(_kept)
    _common = tsne_baseline.drop(index=133)[["t-SNE 1", "t-SNE 2"]].to_numpy()
    refit_shift = layout_shift(_common, _refit)
    return refit_shift,


@app.cell(hide_code=True)
def _(mo, refit_shift):
    mo.md(f"""
    Procrustes shift after dropping row 133 is only **{refit_shift:.3f}** — the map barely moves for the remaining flowers. **Seed** and **perplexity** (see sliders) change the picture far more; the lesson is that t-SNE cannot place a new flower without a full refit.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Using `iris_z` and `measure_columns`, fit `StandardScaler` is already applied — store the 2-D PCA coordinates of all flowers in `student_iris_pca` with shape `(150, 2)`.
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
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Row 133 has three versicolor flowers among its five nearest neighbours in 4-D. Does that force it to look like a mislabelled versicolor in t-SNE? Answer in one sentence.
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
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Row 133 lands among versicolor flowers in the t-SNE plot but has ordinary virginica measurements. Explain how both can be true. Work out your answer, then reveal it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answer": mo.md(r"""
The plot answers **who is nearby** in the embedding; the measurements answer **whether the flower is typical** for its label. Row 133 is geometrically close to versicolor in four dimensions, so t-SNE places it there, while every centimetre value stays inside the virginica whiskers.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(iris_share, knn_tsne, mo, refit_shift, spread_ratio):
    mo.md(f"""
    ## Conclusions

    1. **Five criteria, one table:** PCA, MDS, LLE, Isomap, and t-SNE all read the same standardized four columns yet optimise different properties, so their 2-D groupings disagree while setosa always separates.
    2. **Variance share ≠ separation:** two PCA components keep **{iris_share.sum():.1%}** of the variance, but versicolor and virginica still overlap in the biplot because they overlap in 4-D.
    3. **Standardize first:** raw spreads differ by **{spread_ratio:.2f}×** (petal length 5.9 cm vs 2.4 cm columns), so distances on centimetres would be dominated by petals without scaling.
    4. **t-SNE neighbourhoods vs measurements:** 5-NN accuracy is highest in t-SNE (**{knn_tsne:.3f}** in 2-D) with **no empty gap** between versicolor and virginica; row 133 shows three versicolor neighbours in 4-D yet ordinary virginica sizes.
    5. **Fit, not projection:** dropping row 133 shifts the t-SNE layout only **{refit_shift:.3f}** (Procrustes); new flowers need a full refit, and seed or perplexity move the map more — unlike PCA **`transform`**.
    """)
    return


if __name__ == "__main__":
    app.run()
