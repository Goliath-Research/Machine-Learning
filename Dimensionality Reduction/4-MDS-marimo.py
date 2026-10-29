# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
#     "pandas",
#     "pillow",
#     "plotly",
#     "scikit-learn>=1.8",
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
        "PIL": "pillow",
        "plotly": "plotly",
        "sklearn": "scikit-learn>=1.8",
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
    from PIL import Image
    import base64
    import io
    from plotly.subplots import make_subplots
    from sklearn.manifold import MDS
    from sklearn.metrics import pairwise_distances

    pio.templates.default = "plotly_white"
    px.defaults.color_discrete_sequence = px.colors.qualitative.G10
    builder = pb.PlotBuilder(width=640, height=400)

    return (
        MDS,
        Path,
        Image,
        base64,
        builder,
        go,
        io,
        make_subplots,
        mo,
        np,
        pairwise_distances,
        pd,
        px,
    )


@app.cell
def _(go, make_subplots, np, pd, px):
    def sample_goliath(image, per_letter_point_count=1000, seed=42):
        """Sample points from the ink of a rendered word.

        Parameters
        ----------
        image : numpy.ndarray
            The loaded image, with pixel values in the 0 to 1 float range. Only
            the first colour channel is used, so the ink is separated from the
            background by brightness.
        per_letter_point_count : int, default 1000
            How many ink pixels to keep. Four times as many are proposed, and
            only those landing on ink are kept, so the request is oversampled.
        seed : int, default 42
            Seed for the sampler. A fixed seed makes the point cloud reproducible.

        Returns
        -------
        pandas.DataFrame
            Columns `x0`, `x1`, `labels`: the coordinates of each sampled pixel
            and the letter it belongs to. Rows are sorted left to right.
        """
        ink = image[::-1, :, 0].T
        rng = np.random.RandomState(seed)
        proposals = rng.rand(4 * per_letter_point_count, 2)
        rows, columns = (proposals * ink.shape).astype(int).T
        keep = ink[rows, columns] < 1
        points = proposals[keep]
        # Rescale the horizontal axis so the letters keep their proportions.
        points[:, 0] *= ink.shape[0] / ink.shape[1]
        points = points[:per_letter_point_count]
        points = points[np.argsort(points[:, 0])]

        # Label each point by letter, using the gaps between glyphs.
        gaps = np.diff(points[:, 0]) > 0.08
        word = "GOLIATH"
        labels = np.empty(len(points), dtype="<U1")
        letter_index = 0
        for position, is_gap in enumerate(gaps):
            labels[position] = word[letter_index % len(word)]
            if is_gap:
                letter_index += 1
        labels[-1] = word[letter_index % len(word)]

        return pd.DataFrame({"x0": points[:, 0], "x1": points[:, 1], "labels": labels})

    def draw_letters(frame, title):
        """Scatter the letters coloured by glyph, with a 1:1 aspect ratio.

        Parameters
        ----------
        frame : pandas.DataFrame
            Columns `x0`, `x1`, and `labels`.
        title : str
            Figure title.

        Returns
        -------
        plotly.graph_objects.Figure
            The scatter, with one legend entry per glyph.
        """
        figure = px.scatter(
            frame,
            x="x0",
            y="x1",
            color="labels",
            title=title,
            category_orders={"labels": list("GOLIATH")},
        )
        figure.update_traces(marker=dict(size=4, opacity=0.8))
        figure.update_yaxes(scaleanchor="x", scaleratio=1)
        figure.update_layout(
            height=380,
            margin=dict(t=55, b=40, l=45, r=20),
            legend=dict(font=dict(size=9), itemsizing="constant"),
        )
        return figure

    def letters_panel(frames, captions, title="", width=1150):
        """Place several letter clouds side by side for comparison.

        A single figure with one panel per cloud, so the rotated and recovered
        layouts can be compared without scrolling between cells.

        Parameters
        ----------
        frames : list of pandas.DataFrame
            One cloud per panel, each with `x0`, `x1`, and `labels`.
        captions : list of str
            Panel titles, in the same order.
        title : str, default ""
            Figure title.

        Returns
        -------
        plotly.graph_objects.Figure
            The assembled figure, sharing one legend.
        """
        count = len(frames)
        figure = make_subplots(
            rows=1,
            cols=count,
            subplot_titles=captions,
            horizontal_spacing=0.04,
        )
        for index, frame in enumerate(frames, start=1):
            for position, letter in enumerate("GOLIATH"):
                mask = frame["labels"].to_numpy() == letter
                figure.add_trace(
                    go.Scatter(
                        x=frame["x0"].to_numpy()[mask],
                        y=frame["x1"].to_numpy()[mask],
                        mode="markers",
                        name=letter,
                        legendgroup=letter,
                        showlegend=index == 1,
                        marker=dict(size=4, opacity=0.8),
                    ),
                    row=1,
                    col=index,
                )
        figure.update_layout(
            width=width,
            height=340,
            title=title,
            margin=dict(t=70, b=35, l=45, r=20),
            legend=dict(font=dict(size=9), itemsizing="constant", orientation="h"),
        )
        return figure

    def rotate_letters(frame, angle):
        """Rotate the coordinates of a point cloud by `angle` degrees."""
        theta = np.deg2rad(angle)
        rotation = np.array(
            [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
        )
        rotated = frame.copy()
        rotated[["x0", "x1"]] = frame[["x0", "x1"]].to_numpy() @ rotation
        return rotated

    def along_word(position, embedded):
        """Absolute correlation between the word order and an embedding axis.

        The word runs left to right, so the longest axis of a good layout should
        line up with that order. Absolute value is used because MDS may flip the
        sign of the axis.
        """
        embedded = np.asarray(embedded)
        longest = embedded[:, int(np.argmax(np.var(embedded, axis=0)))]
        return abs(float(np.corrcoef(position, longest)[0, 1]))

    return along_word, draw_letters, letters_panel, rotate_letters, sample_goliath


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Multidimensional Scaling

    ## Objectives

    By the end of this lesson, you should be able to:

    1. Explain what MDS is given, and what it tries to preserve.
    2. Explain why a rotation changes every coordinate but no distance, and what that implies for an MDS layout.
    3. Read a distance matrix as a picture of pairwise relationships.
    4. Explain the difference between metric and non-metric MDS, and when to prefer each.
    5. Explain why MDS struggles with a bent surface even though it preserves distances faithfully.

    ## Background

    **Multidimensional scaling** answers a narrow question: given the distances between items, what coordinates would reproduce them?

    That is a different starting point from PCA. PCA is given coordinates and looks for directions of variance. MDS is given, or computes, a **distance matrix** and searches for a layout whose own distances match it. You can run MDS on data with no coordinates at all — judgments of similarity between faces, say — which is its original use in psychology.

    ### What "preserve" means

    The fit minimises **stress**, a measure of how far the distances in the recovered layout are from the distances you supplied. Two consequences follow.

    - **Only distances are preserved.** Absolute position and orientation are not. A layout can be rotated, reflected, or moved, and still reproduce every pairwise distance exactly.
    - **The layout is a picture, not a measurement.** If two points are close in the layout, they were close by the supplied distances. The axes themselves carry no meaning and no units.

    That first point is easy to state and easy to disbelieve, so the lesson tests it numerically: rotate the letters, measure the largest change in any pairwise distance, and see that it is essentially zero.

    ### Metric and non-metric

    There are two versions, and the distinction matters more than it sounds.

    | | Metric MDS | Non-metric MDS |
    | :--- | :--- | :--- |
    | What it preserves | the **values** of the distances | only their **rank order** |
    | Assumes distances are meaningful | yes | no, only that ordering is |
    | When to use | distances are real measurements | distances are similarities or judgments |

    If you measured distances in centimetres, metric MDS is appropriate. If your data are answers to "how similar are these two on a scale of 1 to 10", the numbers are not really interval measurements, and non-metric MDS is the honest choice: it only requires that "more similar" is ordered correctly.

    ### The version note that matters

    The argument names changed in scikit-learn 1.8, and older code will fail or, worse, silently do the wrong thing:

    - A precomputed distance matrix is passed with **`metric="precomputed"`**. The old name was `dissimilarity="precomputed"`.
    - **`metric_mds=True`** selects metric MDS; `metric_mds=False` selects non-metric. The old boolean `metric=False` was a *different* argument that meant non-metric MDS.

    Those two uses of the word "metric" are the source of most confusion in this topic. This lesson keeps them visibly separate.

    ### Why a bend is hard

    MDS preserves **straight-line distance**. On a flat page that is exactly what you want. On a curved surface it is the wrong quantity: two points on opposite sides of an S can be close in a straight line while being far apart *along the surface*.

    So MDS on a bent surface tends to flatten the bend, because flattening reduces the mismatch between straight-line distances. Keeping the bend would require the surface distances instead, which is the job of the manifold methods in the next lesson.

    ## Datasets used

    ### The rendered word GOLIATH

    **Source.** `Goliath.png`, supplied with this lesson: a 561 by 135 image of the word "GOLIATH" rendered in a bold typeface, white on a transparent background. Every coordinate in this notebook comes from the ink in that image, so the data is a set of real pixel positions rather than a synthetic cloud.

    **How the points are made.** The sampler proposes four times as many random positions as needed, keeps only those landing on ink, and rescales the horizontal axis so the letters keep their proportions. It then labels each point with its letter by detecting the gaps between glyphs. The result is **1,000 points**, and rows are sorted left to right, so the column order carries the word order.

    **Also in the folder.** `GOLIATH_S_curve.csv` is the same 1,000 points mapped onto an S-shaped surface in three dimensions. It is what the next lesson unrolls. This lesson builds that surface itself so the formula is visible.

    There is no missing-value or privacy concern: the data are coordinates sampled from a rendered image, and nothing identifies a person.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Points sampled from the image

    Start with the picture the coordinates came from, then the sampled cloud.
    """)
    return


@app.cell
def _(base64, builder, io, np):
    # Goliath.png, embedded as base64 so the notebook needs no external file.
    # The string is the PNG itself, split across lines only to keep it readable.
    # Pixel values are divided by 255 because Pillow returns 0-255 integers while
    # the sampler expects the 0-1 float range that the original code relied on.
    goliath_b64 = (
        "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"
    )
    goliath_image = (
        np.array(Image.open(io.BytesIO(base64.b64decode(goliath_b64)))).astype(float)
        / 255.0
    )
    builder.image(goliath_image, title="Goliath.png, the source image", width=700, height=200)
    return goliath_b64, goliath_image


@app.cell(hide_code=True)
def _(goliath_image, mo):
    mo.md(f"""
    The image is **embedded in this notebook as base64 text**, not read from a file. That is deliberate: a notebook that opens `Goliath.png` works on your own machine and fails anywhere the file is absent, including a browser runtime. The decoded array has shape **{goliath_image.shape}**: rows, columns, and colour channels. The sampler reads a single colour channel, and a pixel counts as ink when it is dark, which is why the word appears as a shape rather than as colours.

    The sampling step converts an **image** into a **point cloud**, which is what clustering and manifold methods can work with. Each ink pixel that is sampled becomes one row with two coordinates.

    ### The sampled cloud
    """)
    return


@app.cell
def _(goliath_image, pd, sample_goliath):
    letters = sample_goliath(goliath_image)
    letter_counts = letters.labels.value_counts().reindex(list("GOLIATH"))
    return letter_counts, letters


@app.cell(hide_code=True)
def _(letter_counts, letters, mo):
    mo.md(f"""
    **{len(letters)} points**, sampled from the ink and sorted left to right.

    | letter | points |
    | :--- | ---: |
    {"".join(f"| {_l} | {int(_c)} |" + chr(10) for _l, _c in letter_counts.items())}
    The letters are not equally wide, so they do not get equal numbers of points. **I** is a thin glyph and **O** is wide, and the counts follow the available ink. That is a property of the image, not of the sampler.

    The row order matters for everything that follows: because the rows are sorted by horizontal position, **row index equals position in the word**. That is what makes it possible to ask later whether a layout still runs left to right.
    """)
    return


@app.cell
def _(draw_letters, letters):
    draw_letters(letters, "Points sampled from the word GOLIATH")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The word is legible from points alone. Each letter is a distinct cluster of ink, and the gaps between letters are visible as empty vertical bands.

    Keep two facts in mind, because the rest of the lesson depends on them:

    - **The data is two-dimensional.** There are only `x0` and `x1`.
    - **The horizontal axis carries the word order.** Moving left to right spells GOLIATH.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. A rotation changes coordinates, not distances

    This section tests a claim that is easy to accept and hard to feel: **rotating a point cloud leaves every pairwise distance unchanged.**

    Two rotated copies of the letters, one by −20° and one by 50°:
    """)
    return


@app.cell
def _(letters, letters_panel, rotate_letters):
    turned_down = rotate_letters(letters, -20)
    turned_up = rotate_letters(letters, 50)
    letters_panel(
        [letters, turned_down, turned_up],
        ["Original", "Rotated by -20 degrees", "Rotated by 50 degrees"],
        title="The same word, three orientations",
    )
    return turned_down, turned_up


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The word is tilted, and no letter changed shape or size. Now the arithmetic behind that visual impression: compute the full distance matrix for each of the three clouds and compare them.
    """)
    return


@app.cell
def _(letters, np, pairwise_distances, turned_down, turned_up):
    distances = pairwise_distances(letters[["x0", "x1"]])
    distances_down = pairwise_distances(turned_down[["x0", "x1"]])
    distances_up = pairwise_distances(turned_up[["x0", "x1"]])
    distance_gap = float(
        max(
            np.max(np.abs(distances - distances_down)),
            np.max(np.abs(distances - distances_up)),
        )
    )
    return distance_gap, distances


@app.cell(hide_code=True)
def _(distance_gap, letters, mo):
    mo.md(f"""
    **The largest change in any pairwise distance is {distance_gap:.2e}.** With {len(letters)} points there are {len(letters) * (len(letters) - 1) // 2:,} pairs, and not one of them moved by more than a rounding error.

    So the three clouds are **the same data**, described by different coordinates. That is the crux of what follows: if a method is only given distances, it cannot possibly recover which of the three orientations produced them.
    """)
    return


@app.cell
def _(distances, distances_down, distances_up, go, make_subplots):
    _figure = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Original", "Rotated -20 degrees", "Rotated 50 degrees"),
        horizontal_spacing=0.05,
    )
    for _index, _matrix in enumerate(
        (distances, distances_down, distances_up), start=1
    ):
        _figure.add_trace(
            go.Heatmap(z=_matrix, colorscale="Blues", showscale=False),
            row=1,
            col=_index,
        )
    _figure.update_layout(
        width=900, height=330, margin=dict(t=60, b=25, l=25, r=20)
    )
    _figure.update_xaxes(visible=False)
    _figure.update_yaxes(visible=False)
    _figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The three distance matrices are indistinguishable, which is the numerical form of the previous sentence.

    **Reading a distance matrix.** It is square, one row and column per point, and symmetric because the distance from A to B is the distance from B to A. The diagonal is zero. Dark cells are small distances, so the dark blocks along the diagonal are the letters themselves: points within a letter are close to each other, and points in different letters are far.

    Note that the matrix contains no orientation information at all. Nothing in it says the word was horizontal.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Replace `None` so that `student_max_distance` holds the largest value in the distance matrix, and `student_diagonal` holds the sum of its diagonal.
    """)
    return


@app.cell
def _(distances):
    student_max_distance = None
    student_diagonal = None
    print("largest distance:", student_max_distance)
    print("sum of the diagonal:", student_diagonal)
    return student_diagonal, student_max_distance


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
```python
student_max_distance = distances.max()
student_diagonal = distances.diagonal().sum()
```
The largest distance is about **11.1**, roughly the span of the whole word, and the diagonal sums to **0.0**.

The diagonal is zero because every point is at distance zero from itself. That is what makes a distance matrix mostly redundant: the useful information sits off the diagonal, and the matrix is symmetric, so only half of it is needed.

`distances.max()` is the diameter of the point cloud in the supplied metric, which is a compact way to describe how spread out the data is.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    MDS is given only the distance matrix. Given that the three rotations above produce identical matrices, what can MDS possibly recover? Work out your answer, then reveal it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
It recovers the **relative arrangement** of the points, up to a rotation, reflection, and translation.

Because all three clouds have the same distance matrix, any of them is a valid answer, and MDS cannot know which one produced it. Whichever orientation it returns, the pairwise distances in its layout will match the supplied distances.

Practically, this means:

- The **shape** is meaningful: clusters, gaps, and neighbours are real.
- The **axes are not**: "x" and "y" in an MDS plot are arbitrary directions.
- The **orientation is arbitrary**: two runs with different seeds can produce layouts that are turned or mirrored relative to each other, and both are correct.

That is why the next section measures agreement with the word order using an absolute correlation, and why it does not expect the layout to look like the original picture.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Recovering a layout from the distances alone

    Now throw away the coordinates and keep only the distance matrix. MDS is asked for two dimensions, given the precomputed distances, and run twice with different seeds.

    Two seeds are used deliberately. If the layout is determined by the distances, both runs should agree on the structure; where they differ is where the search started.
    """)
    return


@app.cell
def _(MDS, along_word, distances, letters, letters_panel, pd):
    flat_layouts = {}
    flat_agreement = {}
    for _name, _seed in (("seed 9", 9), ("seed 5", 5)):
        _embedded = MDS(
            n_components=2,
            metric="precomputed",
            metric_mds=True,
            init="random",
            n_init=1,
            random_state=_seed,
            normalized_stress="auto",
        ).fit_transform(distances)
        flat_layouts[_name] = pd.DataFrame(
            {
                "x0": _embedded[:, 0],
                "x1": _embedded[:, 1],
                "labels": letters.labels.to_numpy(),
            }
        )
        flat_agreement[_name] = along_word(letters.x0.to_numpy(), _embedded)
    letters_panel(
        [flat_layouts["seed 9"], flat_layouts["seed 5"]],
        ["MDS from distances, seed 9", "MDS from distances, seed 5"],
        title="Two seeds, the same distances",
        width=900,
    )
    return flat_agreement, flat_layouts


@app.cell(hide_code=True)
def _(flat_agreement, mo):
    mo.md(f"""
    The word comes back, and it is **not** in the same orientation as the original. One run may be turned or mirrored.

    The `along-word` score measures how well the layout still runs left to right: it correlates the original row order with the longest axis of the layout, and takes the absolute value so a flipped axis is not penalised.

    - seed 9: **{flat_agreement['seed 9']:.3f}**
    - seed 5: **{flat_agreement['seed 5']:.3f}**

    Both are close to 1, so in both layouts the word order is preserved along one axis, even if that axis is not the horizontal one you see.

    That is exactly the behaviour promised in the previous section. The **structure** is recovered; the **orientation** is not, and cannot be, because it was never in the input.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Replace `None` so that `student_precomputed` holds the string that tells MDS the input is already a distance matrix, using the **current** scikit-learn argument.
    """)
    return


@app.cell
def _():
    student_precomputed = None
    print(student_precomputed)
    return (student_precomputed,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
```python
student_precomputed = "precomputed"
```
You would write `MDS(metric="precomputed", ...)`.

The old name for this was `dissimilarity="precomputed"`, and older tutorials still use it. Since scikit-learn 1.8 the argument is `metric`, which is confusing because `metric` is also the name of the *distance function* in other estimators.

Do not confuse it with `metric_mds`, which is a separate switch: `True` for metric MDS, `False` for non-metric. Two different meanings of the word "metric" in adjacent arguments is genuinely poor API design, and being explicit about both is the only defence.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. A word bent into an S

    The letters are now wrapped onto an S-shaped surface in three dimensions, using a simple parametric formula. The **word order along the surface is unchanged**; only the geometry is different.

    This is the case where preserving distances faithfully becomes a problem.
    """)
    return


@app.cell
def _(letters, np, pd):
    _t = (letters.x0 - 2) * 0.78 * np.pi
    curve = pd.DataFrame(
        {
            "x0": np.sin(_t),
            "x1": letters.x1 * 1.5,
            "x2": np.sign(_t) * (np.cos(_t) - 1) * 1.5,
            "labels": letters.labels.to_numpy(),
        }
    )
    return (curve,)


@app.cell
def _(curve, px):
    _figure = px.scatter_3d(
        curve,
        x="x0",
        y="x1",
        z="x2",
        color="labels",
        title="The word bent into an S",
        category_orders={"labels": list("GOLIATH")},
    )
    _figure.update_traces(marker=dict(size=2, opacity=0.85))
    _figure.update_layout(
        width=760,
        height=520,
        margin=dict(t=55, b=20, l=20, r=20),
        legend=dict(font=dict(size=9), itemsizing="constant"),
    )
    _figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The word spirals through three dimensions. **G** and **H** are at opposite ends in word order, and they are also far apart in space, but points in the middle of the S can be close to points in an entirely different letter.

    That is the difficulty. MDS preserves **straight-line** distance, so if two points from distant letters happen to sit close in three dimensions, MDS will faithfully place them close in two dimensions — even though they are far apart along the surface.

    ### Metric MDS against non-metric MDS

    Both are fitted to the same three coordinates. Metric MDS matches the distances themselves; non-metric MDS only matches their rank order.
    """)
    return


@app.cell
def _(MDS, along_word, curve, letters, letters_panel, pd):
    curve_layouts = {}
    curve_agreement = {}
    for _name, _metric_mds in (("metric", True), ("non-metric", False)):
        _embedded = MDS(
            n_components=2,
            metric_mds=_metric_mds,
            init="random",
            n_init=1,
            random_state=19,
            normalized_stress="auto",
        ).fit_transform(curve[["x0", "x1", "x2"]])
        curve_layouts[_name] = pd.DataFrame(
            {
                "x0": _embedded[:, 0],
                "x1": _embedded[:, 1],
                "labels": curve.labels.to_numpy(),
            }
        )
        curve_agreement[_name] = along_word(letters.x0.to_numpy(), _embedded)
    letters_panel(
        [curve_layouts["metric"], curve_layouts["non-metric"]],
        ["Metric MDS on the S", "Non-metric MDS on the S"],
        title="Metric and non-metric MDS on the same bend",
        width=900,
    )
    return curve_agreement, curve_layouts


@app.cell(hide_code=True)
def _(curve_agreement, flat_agreement, mo):
    mo.md(f"""
    Compare the four correlations:

    | Fit | Along-word correlation |
    | :--- | ---: |
    | Flat letters, metric MDS, seed 9 | {flat_agreement['seed 9']:.3f} |
    | Flat letters, metric MDS, seed 5 | {flat_agreement['seed 5']:.3f} |
    | S-bend, metric MDS | {curve_agreement['metric']:.3f} |
    | S-bend, non-metric MDS | {curve_agreement['non-metric']:.3f} |

    All four are high, and the two S-bend fits are close to the flat ones. The word order survives the bend, because the geometry, although curved, is still a gentle S rather than a fold that brings distant letters into contact.

    So this example does **not** show MDS failing dramatically. What it does show is the shape of the problem:

    - **Neither metric nor non-metric MDS unrolls a surface.** They find a flat layout that best matches straight-line distances. When the surface curves gently, that flat layout still works.
    - **A tighter fold would break it.** If the S were folded so that distant letters touched, straight-line distances would say those letters are neighbours, and both versions would faithfully put them next to each other.
    - **Unrolling is a different job.** That is what locally linear embedding and Isomap are for, in the next lesson. They use distances **along the surface** rather than through the air.

    The lesson to take away is not "MDS fails on curves". It is: **know which distance your method preserves, and check that it is the distance your problem means.**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Metric and non-metric MDS give similar correlations here. Describe the situation in which they would be expected to differ, and why. Work out your answer, then reveal it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
They differ when the **numbers are not trustworthy as distances, but their order is**.

- **Metric MDS** tries to reproduce each distance value. If your inputs are real measurements in a unit, that is right, and it uses more of the information.
- **Non-metric MDS** only requires that "closer" stays "closer". It is the correct choice when the numbers came from ratings, rankings, or perceived similarity, where the interval between 3 and 4 need not equal the interval between 7 and 8.

In this lesson the inputs are Euclidean distances computed from real coordinates, so the values are meaningful and metric MDS loses nothing. The two agree because there is nothing for the rank-only version to gain.

They would visibly diverge if you handed both a matrix of subjective similarity scores, or a matrix with systematic distortion, such as distances measured by a stretched ruler. There the values are distorted but the ordering survives, and only non-metric MDS is being honest about what the data supports.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(
    curve_agreement,
    distance_gap,
    flat_agreement,
    letter_counts,
    letters,
    mo,
):
    mo.md(f"""
    ## Conclusions

    1. **MDS is given distances and returns coordinates.** It minimises stress, the mismatch between the distances you supplied and the distances in its layout. It does not need coordinates to start with, which is why it is used on similarity judgments as well as on measurements.
    2. **A rotation changes every coordinate and no distance.** The largest change across all {len(letters) * (len(letters) - 1) // 2:,} pairs was **{distance_gap:.2e}**, so the three clouds of letters are the same data in different orientations.
    3. **Therefore an MDS layout has no meaningful axes.** Any rotation, reflection, or translation of a good layout is equally good. The recovered shape is real; the direction it points is arbitrary, which is why two seeds can give turned or mirrored pictures with the same correlation.
    4. **A distance matrix is symmetric with a zero diagonal.** It encodes which points are close to which, and nothing about orientation. {len(letters)} points means {len(letters) * (len(letters) - 1) // 2:,} distinct distances.
    5. **The word was recovered from distances alone** on the flat data, at along-word correlations of {flat_agreement['seed 9']:.3f} and {flat_agreement['seed 5']:.3f} for the two seeds.
    6. **Metric and non-metric MDS preserve different things.** Metric matches the distance values; non-metric matches only their rank order. Use non-metric when the numbers are ratings or similarities rather than true distances. On the S-bend they scored {curve_agreement['metric']:.3f} and {curve_agreement['non-metric']:.3f}.
    7. **Preserving distances is not the same as unrolling a surface.** Both MDS variants look for a flat layout matching straight-line distances, so a gentle bend survives but a tight fold would not. Unrolling needs distances along the surface, which is the next lesson's topic.
    8. **Watch the two meanings of "metric".** `metric="precomputed"` says the input is already a distance matrix; `metric_mds=True/False` selects the algorithm variant. Older code uses `dissimilarity=` for the first.
    9. **The letters have unequal point counts** ({", ".join(f"{_l}: {int(_c)}" for _l, _c in letter_counts.items())}), because glyph widths differ. That is a property of the image, and it is worth noticing before reading anything into cluster sizes.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check your understanding

    1. What is MDS given, and what does it try to preserve?
    2. A rotation leaves every pairwise distance unchanged. What does that imply about the axes of an MDS plot?
    3. Two MDS runs with different seeds produce layouts that are mirror images of each other. Is one of them wrong?
    4. What does a distance matrix look like: its shape, its symmetry, and its diagonal?
    5. What is the difference between metric and non-metric MDS, and when would you choose the second?
    6. Why does MDS not unroll a tightly folded surface, even though it reproduces distances accurately?
    7. What are the two different meanings of "metric" in `MDS(metric=..., metric_mds=...)`?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
1. **A matrix of pairwise distances**, and it returns coordinates whose own distances match them as closely as possible. It minimises stress, the mismatch between the two sets of distances.
2. **That the axes carry no meaning.** Only relative positions are determined, so a layout can be rotated, reflected, or translated and remain equally correct. The shape is informative; the directions are arbitrary.
3. **No, both are correct.** A mirror image preserves every pairwise distance, so both layouts satisfy the input. MDS cannot distinguish them, because orientation was never in the data.
4. **Square, with one row and column per point; symmetric, because the distance from A to B equals the distance from B to A; and with a zero diagonal**, because every point is at distance zero from itself.
5. **Metric MDS matches the distance values; non-metric matches only their rank order.** Choose non-metric when the inputs are ratings, rankings, or perceived similarities, where the numbers are not trustworthy as interval measurements.
6. **Because it preserves straight-line distance, not distance along the surface.** On a tight fold, points from distant parts of the surface come close in space, and MDS faithfully places them close in the layout. Reproducing those distances accurately is exactly what flattens the fold.
7. **`metric="precomputed"` says the input is already a distance matrix** (the old argument was `dissimilarity`). **`metric_mds=True` or `False` selects metric or non-metric MDS.** The same word names two unrelated settings.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - [MDS](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html)
    - [Manifold learning, scikit-learn user guide](https://scikit-learn.org/stable/modules/manifold.html#multidimensional-scaling)
    - Kruskal, J. B. (1964). Multidimensional scaling by optimizing goodness of fit to a nonmetric hypothesis. *Psychometrika*, 29(1), 1–27.
    - Borg, I., and Groenen, P. J. F. (2005). *Modern Multidimensional Scaling: Theory and Applications* (2nd ed.). Springer.
    - VanderPlas, J. (2017). *Python Data Science Handbook*. O'Reilly Media. Chapter 5.
    """)
    return


if __name__ == "__main__":
    app.run()
