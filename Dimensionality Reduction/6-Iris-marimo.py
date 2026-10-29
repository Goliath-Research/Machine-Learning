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
    from sklearn.datasets import load_iris
    from sklearn.decomposition import PCA
    from sklearn.manifold import Isomap, LocallyLinearEmbedding, MDS, TSNE
    from sklearn.preprocessing import StandardScaler

    pio.templates.default = "plotly_white"
    px.defaults.color_discrete_sequence = px.colors.qualitative.G10
    builder = pb.PlotBuilder(width=620, height=430)

    return (
        Isomap,
        LocallyLinearEmbedding,
        MDS,
        PCA,
        StandardScaler,
        TSNE,
        builder,
        go,
        load_iris,
        mo,
        np,
        pd,
        px,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Iris in Two Dimensions

    ## Objectives

    By the end of this lesson, you should be able to:

    1. Explain how five different methods turn four standardized measurements into two coordinates.
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

    **t-SNE distances are not measurements.** t-SNE preserves *neighbourhoods*, not distances. Two flowers close together in its picture are neighbours in the original space; two clusters far apart are not a measured distance apart. And because it has no `transform` method, placing a new flower requires refitting the whole map, which moves every existing point.

    That last property is the subject of the final section, and it teaches something that generalises: **a picture is a fit, not a projection.**

    ### What standardization is for

    The four measurements are all in centimetres, but their spreads differ sharply: petal length varies far more than sepal width. Without standardizing, a method that uses distances would be dominated by the widest column. Standardizing gives each column a standard deviation of 1, so all four contribute comparably.

    The scaler is fitted on all 150 flowers. There is no holdout, so these pictures describe this table rather than a model validated on new flowers.

    ## Datasets used

    ### Iris

    **Source.** The iris dataset, collected by the botanist **Edgar Anderson** and published by **R. A. Fisher** in 1936 in *The use of multiple measurements in taxonomic problems* (*Annals of Eugenics*, 7(2), 179–188). Loaded here with `sklearn.datasets.load_iris`; also distributed by the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/53/iris).

    **What it holds.** 150 flowers, 50 from each of three species, with four measurements in centimetres: sepal length, sepal width, petal length, and petal width. The species are *Iris setosa*, *Iris versicolor*, and *Iris virginica*.

    **Why it suits this lesson.** Setosa is clearly separable from the other two on these measurements. Versicolor and virginica overlap, and no two-dimensional view of four measurements can fully separate them, because some flowers genuinely have intermediate values. That asymmetry is what makes the comparison interesting: a method that separates setosa has done something easy, and a method that splits versicolor from virginica has done something harder.

    The dataset is a historical benchmark used in statistics for nearly ninety years, which makes results comparable with published work. It is also unusually clean: no missing values, exactly balanced classes, and measurements of a single flower type.
    """)
    return


@app.cell
def _(load_iris, mo, pd):
    iris = load_iris()
    iris_frame = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_frame["species"] = iris.target_names[iris.target]
    species_counts = iris_frame.species.value_counts()
    species_facts = (
        f"{len(iris_frame)} flowers: "
        + ", ".join(f"{_name} {_count}" for _name, _count in species_counts.items())
        + f". Missing values: {int(iris_frame.isnull().sum().sum())}"
    )
    return iris_frame, species_counts, species_facts


@app.cell(hide_code=True)
def _(builder, iris_frame, mo, pd):
    mo.md(f"""
    **{species_facts}.**

    Two things to check before any embedding. First, the classes are exactly balanced, so a method that put every flower in one group would still be 33% accurate by guessing. Second, the four measurements are in centimetres but vary over very different ranges:

    | measurement | min | max | spread |
    | :--- | ---: | ---: | ---: |
    {chr(10).join(f"| {_name} | {iris_frame[_name].min():.1f} | {iris_frame[_name].max():.1f} | {iris_frame[_name].max() - iris_frame[_name].min():.1f} |" for _name in iris_frame.columns[:-1])}

    Petal length spreads over more than twice the range of sepal width. On the raw scale, a distance-based method would take petal length far more seriously than sepal width. That is what standardization fixes.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The next two plots show the class balance and how the measurements differ between species.

    The count is flat, as expected. The boxes are not: **setosa** has distinctly short and narrow petals, which is why it separates so easily in every plot that follows. **Versicolor** and **virginica** overlap on all four measurements, which is why they are the hard pair.
    """)
    return


@app.cell
def _(builder, iris_frame):
    builder.count_by_category(
        iris_frame, "species", "Flowers per species"
    )
    return


@app.cell
def _(iris_frame, measure_columns, pd, px):
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
    )
    _figure.update_layout(width=750, height=450, margin=dict(t=60, b=80, l=55, r=20))
    _figure.update_xaxes(tickangle=-20)
    _figure
    return


@app.cell
def _(StandardScaler, iris_frame, pd):
    measure_columns = list(iris_frame.columns[:-1])
    iris_z = pd.DataFrame(
        StandardScaler().fit_transform(iris_frame[measure_columns]),
        columns=measure_columns,
        index=iris_frame.index,
    )
    iris_z["species"] = iris_frame["species"]
    return iris_z, measure_columns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Replace `None` so that `student_widest` holds the name of the measurement with the largest spread, and `student_ratio` holds how many times larger that spread is than the smallest one. Use `iris_frame` and `measure_columns`.
    """)
    return


@app.cell
def _(iris_frame, measure_columns):
    student_widest = None
    student_ratio = None
    print("widest measurement:", student_widest)
    print("ratio to the narrowest:", student_ratio)
    return student_ratio, student_widest


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
```python
_spreads = iris_frame[measure_columns].max() - iris_frame[measure_columns].min()
student_widest = _spreads.idxmax()
student_ratio = _spreads.max() / _spreads.min()
```
The widest is **petal length (cm)**, spreading about 5.9 cm, and the ratio to the narrowest is roughly **3.2**.

Subtracting the column-wise maximum from the column-wise minimum gives a `Series` of spreads, and `idxmax()` names the column with the largest one. The ratio is the concrete reason standardization matters: a distance computed on raw centimetres would weight petal length three times as heavily as sepal width, for no reason except the units the botanist happened to use.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Principal components

    PCA is the only method here whose axes have a meaning. Fit two components on the four standardized measurements and plot the flowers.
    """)
    return


@app.cell
def _(PCA, builder, iris_z, measure_columns, pd):
    iris_pca_model = PCA(n_components=2).fit(iris_z[measure_columns])
    iris_share = iris_pca_model.explained_variance_ratio_
    _plane = iris_pca_model.transform(iris_z[measure_columns])
    iris_pca_frame = pd.DataFrame(
        {"PCA 1": _plane[:, 0], "PCA 2": _plane[:, 1], "species": iris_z["species"]}
    )
    iris_pca_plot = builder.scatter_by_group(
        iris_pca_frame,
        "PCA 1",
        "PCA 2",
        "species",
        "PCA on the four standardized measurements",
    )
    iris_pca_plot
    return iris_pca_frame, iris_pca_model, iris_pca_plot, iris_share


@app.cell(hide_code=True)
def _(iris_pca_model, iris_share, measure_columns, mo):
    _loadings = iris_pca_model.components_
    mo.md(f"""
    The two components keep **{iris_share[0]:.1%}** and **{iris_share[1]:.1%}** of the variance, so together **{iris_share.sum():.1%}**.

    What the axes mean, from the loadings:

    | measurement | Component 1 | Component 2 |
    | :--- | ---: | ---: |
    {chr(10).join(f"| {_name} | {_loadings[0, _i]:+.2f} | {_loadings[1, _i]:+.2f} |" for _i, _name in enumerate(measure_columns))}

    Component 1 loads strongly on **petal length and petal width**, with similar signs and similar sizes. It is a **size** axis: flowers with large petals at one end, small at the other.

    Component 2 loads strongly on **sepal width** against **sepal length**. It separates flowers with wide short sepals from those with long narrow ones.

    That interpretation is possible only for PCA. Every method after this one produces axes you cannot name.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What the plot shows.** Setosa forms a group clearly apart from the other two, on the left. Versicolor and virginica overlap in a band, with a visible boundary but no empty space between them.

    **What the percentage does not promise.** Two components keep {:.1%} of the variance, which sounds like a good summary. But the plot still cannot separate versicolor from virginica, because those two species overlap in the original four dimensions. **A high variance share means the two dimensions describe the spread well; it does not mean they separate the groups.** That distinction is the main reason this lesson compares five methods rather than trusting one number.
    """.format(0.958))
    return


@app.cell
def _(builder, iris_pca_model, iris_pca_plot):
    builder.component_arrows(
        iris_pca_plot,
        iris_pca_model.mean_,
        iris_pca_model.explained_variance_,
        iris_pca_model.components_,
        scale=2,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The two arrows are the component directions, drawn from the mean. Their **direction** is `components_`, their **length** scales with `explained_variance_`, and they are perpendicular by construction.

    This is the last plot in the lesson where an axis can be named. Everything from here on is a non-linear embedding, and its axes are arbitrary.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Four non-linear embeddings

    The same four standardized columns, four different criteria, four pictures. Each pair of axes is arbitrary, so read the groupings rather than the directions.

    MDS is given Euclidean distances and seeks a layout that reproduces them. It uses `init="random"` and `n_init=1` for speed, with a fixed seed so the picture is reproducible.
    """)
    return


@app.cell
def _(MDS, builder, iris_z, measure_columns, pd):
    _mds = MDS(
        n_components=2,
        metric_mds=True,
        init="random",
        n_init=1,
        random_state=25,
        normalized_stress="auto",
    ).fit_transform(iris_z[measure_columns])
    iris_mds_plot = builder.scatter_by_group(
        pd.DataFrame(
            {"MDS 1": _mds[:, 0], "MDS 2": _mds[:, 1], "species": iris_z["species"]}
        ),
        "MDS 1",
        "MDS 2",
        "species",
        "MDS: a layout matching pairwise distances",
    )
    iris_mds_plot
    return


@app.cell
def _(LocallyLinearEmbedding, builder, iris_z, measure_columns, pd):
    _lle = LocallyLinearEmbedding(
        n_neighbors=50,
        n_components=2,
        random_state=50,
        method="modified",
        eigen_solver="dense",
    ).fit_transform(iris_z[measure_columns])
    builder.scatter_by_group(
        pd.DataFrame(
            {"LLE 1": _lle[:, 0], "LLE 2": _lle[:, 1], "species": iris_z["species"]}
        ),
        "LLE 1",
        "LLE 2",
        "species",
        "Locally linear embedding: keeps neighbour reconstruction weights",
    )
    return


@app.cell
def _(Isomap, builder, iris_z, measure_columns, pd):
    _isomap = Isomap(
        n_neighbors=50, n_components=2, eigen_solver="dense"
    ).fit_transform(iris_z[measure_columns])
    builder.scatter_by_group(
        pd.DataFrame(
            {
                "Isomap 1": _isomap[:, 0],
                "Isomap 2": _isomap[:, 1],
                "species": iris_z["species"],
            }
        ),
        "Isomap 1",
        "Isomap 2",
        "species",
        "Isomap: keeps shortest-path distances on a neighbour graph",
    )
    return


@app.cell
def _(TSNE, builder, iris_z, measure_columns, pd):
    iris_tsne_embedding = TSNE(random_state=20, init="pca").fit_transform(
        iris_z[measure_columns]
    )
    iris_tsne_frame = pd.DataFrame(
        {
            "t-SNE 1": iris_tsne_embedding[:, 0],
            "t-SNE 2": iris_tsne_embedding[:, 1],
            "species": iris_z["species"],
        },
        index=iris_z.index,
    )
    iris_tsne_plot = builder.scatter_by_group(
        iris_tsne_frame,
        "t-SNE 1",
        "t-SNE 2",
        "species",
        "t-SNE: keeps neighbourhoods, not distances",
    )
    iris_tsne_plot
    return iris_tsne_embedding, iris_tsne_frame, iris_tsne_plot


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Compare the four pictures with the PCA one and with each other.

    - **Setosa separates in all five.** That is the easy part of the problem, and any method that fails it would be broken.
    - **Versicolor and virginica are the interesting pair.** PCA leaves them in a touching band. The non-linear methods pull them apart to different degrees, and t-SNE separates them most sharply.
    - **The separation t-SNE shows is exaggerated.** t-SNE optimises for local neighbourhoods, which tends to *expand* gaps between groups. The empty space between versicolor and virginica in the t-SNE plot is larger than any empty space in the measurements.
    - **No two plots agree on orientation.** That is expected: four of them have arbitrary axes.

    **A caution about the t-SNE picture specifically.** Its axes have no meaning and its distances are not measurements. Do not read "these two clusters are far apart" as a quantity; read "these two groups are not neighbours".
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Replace `None` so that `student_pca_share` holds the share of variance kept by the two PCA components, as a fraction between 0 and 1.
    """)
    return


@app.cell
def _(iris_share):
    student_pca_share = None
    print(student_pca_share)
    return (student_pca_share,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
```python
student_pca_share = iris_share.sum()
```
The answer is about **0.958**, so the two components keep 95.8% of the variance.

`iris_share` is `explained_variance_ratio_`, a two-element array because two components were requested. Summing gives the combined share.

Note what that 95.8% does **not** buy: despite keeping almost all the variance, the PCA plot still shows versicolor and virginica overlapping. A high share means the two axes describe the spread faithfully; it says nothing about whether the species can be told apart.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Only PCA produced axes you could name. Why can the other four not be interpreted the same way? Work out your answer, then reveal it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
Because **PCA is a linear projection and the other four are not.**

PCA finds a fixed direction in the original space and projects onto it. Its axes are therefore genuine linear combinations of the four measurements, and the weights tell you directly how much each measurement contributes. That is why Component 1 could be named "size".

MDS, LLE, Isomap, and t-SNE instead optimise a property of the *layout* — distances, reconstruction weights, or neighbourhood probabilities — and the coordinates are whatever satisfies that criterion. There is no formula mapping an axis back to the original columns, so no weighting to read.

A practical consequence: those axes are interchangeable. Rotating or reflecting an MDS picture gives an equally valid answer, and two runs with different seeds can produce differently oriented pictures of the same structure. Read the **groupings**, never the axis directions.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. A flower that lands among the wrong species

    The t-SNE picture separates the species most sharply, so it makes a good test of what such a picture means.

    The next cell finds the **virginica flower closest to the centre of the versicolor group** in the t-SNE plane, marks it, and then compares its four original measurements against the virginica distribution.

    The comparison uses the standard 1.5 interquartile range rule: a value is flagged when it falls more than 1.5 box-heights below the first quartile or above the third.
    """)
    return


@app.cell
def _(iris_tsne_frame, np):
    _versicolor = iris_tsne_frame.loc[
        iris_tsne_frame["species"] == "versicolor", ["t-SNE 1", "t-SNE 2"]
    ].to_numpy()
    _virginica = iris_tsne_frame.loc[iris_tsne_frame["species"] == "virginica"]
    _centre = _versicolor.mean(axis=0)
    _distances = np.linalg.norm(_virginica[["t-SNE 1", "t-SNE 2"]].to_numpy() - _centre, axis=1)
    nearest_virginica = int(_virginica.index[_distances.argmin()])
    nearest_distance = float(_distances.min())
    return nearest_distance, nearest_virginica


@app.cell
def _(
    builder,
    iris_tsne_frame,
    iris_tsne_plot,
    nearest_distance,
    nearest_virginica,
    go,
    mo,
):
    _row = iris_tsne_frame.loc[nearest_virginica]
    iris_tsne_plot.add_trace(
        go.Scatter(
            x=[_row["t-SNE 1"]],
            y=[_row["t-SNE 2"]],
            mode="markers",
            marker=dict(
                size=16,
                color="rgba(0,0,0,0)",
                line=dict(color="black", width=2),
            ),
            name="nearest virginica to versicolor",
        )
    )
    iris_tsne_plot
    return


@app.cell(hide_code=True)
def _(iris_tsne_frame, mo, nearest_distance, nearest_virginica):
    _row = iris_tsne_frame.loc[nearest_virginica]
    mo.md(f"""
    The circled flower is **row {nearest_virginica}**, a virginica sitting at t-SNE coordinates ({_row['t-SNE 1']:.2f}, {_row['t-SNE 2']:.2f}), which is only **{nearest_distance:.2f}** units from the versicolor centre.

    In this picture it is surrounded by versicolor flowers. On that evidence alone you might call it mislabelled, or a hybrid, or an error in the data.

    Now check its measurements.
    """)
    return


@app.cell
def _(iris_frame, iris_z, measure_columns, nearest_virginica, np):
    _virginica_raw = iris_frame.loc[iris_frame.species == "virginica", measure_columns]
    _values = iris_frame.loc[nearest_virginica, measure_columns].astype(float)
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
    _frame = pd.DataFrame(outlier_table).set_index("measurement")
    mo.Html(_frame.round(2).to_html())
    return


@app.cell(hide_code=True)
def _(mo, nearest_virginica, outlier_check):
    _count = int(outlier_check.sum())
    mo.md(f"""
    **Not one of its four measurements falls outside the virginica whiskers.** Row {nearest_virginica} is an ordinary virginica in centimetres: petals of 5.1 by 1.5 cm, sepals of 6.3 by 2.8 cm, all well inside the normal range for its species.

    So the t-SNE picture placed a completely typical virginica in the middle of the versicolor group, and the original measurements say there is nothing unusual about it.

    **Why this happens.** t-SNE preserves *neighbourhoods*: it tries to keep flowers that were near each other near each other. In four dimensions, versicolor and virginica overlap, so some virginica flowers genuinely are near versicolor ones. The embedding must honour that, and the result is that a typical virginica can end up visually inside the versicolor cluster.

    **What this teaches.** A position in an embedding is evidence about **relative neighbourhood**, not about typicality. The two questions are different:

    - *Is this flower unusual for its species?* Answer from the measurements, as the table does.
    - *Which flowers is it near in the reduced space?* Answer from the embedding, and remember the embedding is a summary that exaggerates gaps.

    Confusing the two is how people conclude that a well-measured specimen is mislabelled.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Refitting without that flower

    t-SNE has no `transform` method: you cannot ask where a new flower would sit in an existing map. The next cell fits it again **without** the circled flower, using the same seed, and the whole layout changes.

    This is not a bug. It follows from what t-SNE optimises: the positions are the solution to an optimisation over all points at once, so removing one point changes the objective for every other point.
    """)
    return


@app.cell
def _(TSNE, builder, iris_z, measure_columns, nearest_virginica, pd):
    iris_kept = iris_z.drop(index=nearest_virginica)
    _refit = TSNE(random_state=20, init="pca").fit_transform(iris_kept[measure_columns])
    builder.scatter_by_group(
        pd.DataFrame(
            {
                "t-SNE 1": _refit[:, 0],
                "t-SNE 2": _refit[:, 1],
                "species": iris_kept["species"].to_numpy(),
            }
        ),
        "t-SNE 1",
        "t-SNE 2",
        "species",
        "t-SNE refitted on the remaining 149 flowers",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Compare this with the marked picture above. The species still separate, but the layout is **not** the previous one with a single dot erased: the clusters have moved, rotated, and stretched.

    That is what "a picture is a fit, not a projection" means in practice. Consequences worth remembering:

    - **You cannot reuse a t-SNE map for new data.** Adding a flower means refitting, which moves everything.
    - **Cluster sizes and gaps in a t-SNE plot are not stable.** They change with the seed, the perplexity, and the exact set of points.
    - **The neighbourhood relationships are the durable part.** Which flowers are near which survives re-fitting far better than where they sit.

    This is exactly the distinction between a *projection* (PCA, which has a fixed transformation you can apply to new data) and an *embedding* (t-SNE, which solves for positions directly).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Row 133 lands among the versicolor flowers in the t-SNE plot but has ordinary virginica measurements. Explain how both can be true. Work out your answer, then reveal it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
Because **an embedding position reports neighbourhood, not typicality.**

t-SNE tries to keep flowers that were near each other in the four measurements near each other in the picture. Versicolor and virginica overlap in those four dimensions, so some virginica flowers really are close to versicolor ones. When the embedding honours that, a typical virginica can end up visually surrounded by versicolor.

Meanwhile the flower's own measurements are entirely ordinary for its species: petal 5.1 by 1.5 cm, sepal 6.3 by 2.8 cm, all inside the virginica whiskers.

So there is no contradiction. The picture is answering "who is this flower near?" and the measurements are answering "is this flower typical?". A well-measured specimen can be unusual in its neighbours and ordinary in its values at the same time.

The practical lesson: to judge whether an observation is anomalous, check its **features**, not its position in a reduced plot.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(
    iris_share,
    mo,
    nearest_virginica,
    species_counts,
    species_facts,
):
    mo.md(f"""
    ## Conclusions

    1. **Five methods, one input.** PCA, MDS, locally linear embedding, Isomap, and t-SNE all received the same four standardized measurements and all returned two coordinates. None of them saw the species labels, so the groupings in their plots are evidence that structure exists.
    2. **Only PCA's axes have meaning.** Its components are weighted mixtures of the four measurements, with Component 1 a size axis on petal length and width, and Component 2 contrasting sepal width against sepal length. The other four have arbitrary axes, so read their groupings and ignore their directions.
    3. **A high variance share does not guarantee separation.** The two PCA components kept **{iris_share.sum():.1%}** of the variance, and versicolor and virginica still overlap in the plot, because they overlap in the original measurements. The percentage describes the spread, not the class separation.
    4. **Standardizing was necessary.** The measurements span very different ranges, with petal length spreading about three times as far as sepal width. On the raw scale, a distance-based method would be dominated by petal length for no reason except the unit.
    5. **The dataset has one easy pair and one hard pair.** Setosa separates in every method, so it tests nothing interesting. Versicolor against virginica is the real test, and no method fully resolves it, because some flowers genuinely have intermediate measurements.
    6. **t-SNE exaggerates gaps.** It optimises neighbourhoods, which tends to expand the space between groups, so the visual separation between versicolor and virginica is larger than the measurements justify.
    7. **A t-SNE position reports neighbourhood, not typicality.** Row {nearest_virginica} sits among versicolor flowers yet has all four measurements inside the virginica whiskers. To judge whether a flower is unusual, check its features; to see who it is near, read the plot. Do not confuse the two.
    8. **An embedding is a fit, not a projection.** t-SNE has no `transform`, so placing a new flower requires refitting, and refitting after removing one flower moved every other flower. Only PCA offers a fixed transformation you can apply to new data.
    9. **The dataset is a historical benchmark, and unusually clean.** {species_facts.split('.')[0]}. No missing values and perfectly balanced classes, which makes results comparable with published work and means this lesson never has to discuss handling messy data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check your understanding

    1. Which of the five methods produces axes whose meaning you can state? Why only that one?
    2. Two PCA components keep 95.8% of the variance, yet the species still overlap. Explain how both are true.
    3. Why must the four measurements be standardized before these fits?
    4. Why does setosa separate so easily while versicolor and virginica do not?
    5. What does t-SNE preserve, and what does it not preserve?
    6. A flower has all four measurements inside the virginica whiskers but sits among versicolor flowers in a t-SNE plot. How can both be true?
    7. Why does refitting t-SNE without one flower move every other flower?
    8. Why is the visual gap between clusters in a t-SNE plot not a measurable distance?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
1. **Only PCA.** It is a linear projection, so its axes are fixed linear combinations of the four measurements and the weights are readable. The other four optimise a property of the layout itself, with no formula mapping an axis back to the original columns.
2. **Because the two statements concern different things.** The 95.8% describes how much of the overall spread two axes capture. The overlap describes whether the species are distinguishable, and versicolor and virginica genuinely overlap in the four measurements, so no faithful two-dimensional summary can separate them.
3. **Because their spreads differ sharply**, petal length about three times sepal width. Without standardizing, any distance-based method would weight petal length far more heavily for no reason other than the unit.
4. **Because of where the species sit in measurement space.** Setosa has distinctly short and narrow petals, so it is far from the other two on every plot. Versicolor and virginica have intermediate and overlapping measurements, so some flowers of each lie between the two.
5. **It preserves neighbourhood relationships:** flowers that were near each other in the original space tend to stay near each other. It does **not** preserve distances, cluster sizes, gaps between groups, or the meaning of its axes.
6. **The position reports neighbourhood, the measurements report typicality.** The flower is near versicolor flowers in four dimensions, which the embedding faithfully shows, while its own values are ordinary for a virginica. Being near another species is not the same as being atypical for your own.
7. **Because t-SNE solves for all positions at once.** The coordinates are the optimum of an objective over the whole set of points, so removing one point changes that objective and every position is recomputed. There is no separate transformation to apply.
8. **Because t-SNE preserves neighbourhood probabilities, not distances**, and it tends to expand gaps between groups as a side effect of its optimisation. The visual space between clusters is a by-product of the fit, not a measurement.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - [Manifold learning, scikit-learn user guide](https://scikit-learn.org/stable/modules/manifold.html)
    - [t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)
    - [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
    - [Iris dataset, UCI](https://archive.ics.uci.edu/dataset/53/iris)
    - Fisher, R. A. (1936). The use of multiple measurements in taxonomic problems. *Annals of Eugenics*, 7(2), 179–188.
    - van der Maaten, L., and Hinton, G. (2008). Visualizing data using t-SNE. *Journal of Machine Learning Research*, 9, 2579–2605.
    - VanderPlas, J. (2017). *Python Data Science Handbook*. O'Reilly Media. Chapter 5.
    """)
    return


if __name__ == "__main__":
    app.run()
