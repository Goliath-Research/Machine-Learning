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
    import base64
    import io
    from PIL import Image
    from plotly.subplots import make_subplots
    from scipy.spatial import procrustes as scipy_procrustes
    from sklearn.manifold import MDS
    from sklearn.metrics import pairwise_distances

    pio.templates.default = "plotly_white"
    px.defaults.color_discrete_sequence = px.colors.qualitative.G10
    LETTER_COLORS = {
        letter: px.colors.qualitative.G10[index]
        for index, letter in enumerate("GOLIATH")
    }

    return (
        LETTER_COLORS,
        MDS,
        Path,
        Image,
        base64,
        go,
        io,
        scipy_procrustes,
        make_subplots,
        mo,
        np,
        pairwise_distances,
        pd,
        px,
    )



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
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What MDS does

    **Multidimensional scaling** answers a narrow question: given the distances between items, what coordinates would reproduce them?

    That is a different starting point from PCA. PCA is given coordinates and looks for directions of variance. MDS is given, or computes, a **distance matrix** and searches for a layout whose own distances match it. You can run MDS on data with no coordinates at all — judgments of similarity between faces, say — which is its original use in psychology.

    The fit minimises **stress**, a measure of how far the distances in the recovered layout are from the distances you supplied. **Only distances are preserved:** position and orientation are not. A layout can be rotated, reflected, or moved, and still reproduce every pairwise distance exactly.
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Metric and non-metric MDS

    | | Metric MDS | Non-metric MDS |
    | :--- | :--- | :--- |
    | What it preserves | the **values** of the distances | only their **rank order** |
    | Assumes distances are meaningful | yes | no, only that ordering is |
    | When to use | distances are real measurements | distances are similarities or judgments |

    If you measured distances in centimetres, metric MDS is appropriate. If your data are answers to "how similar are these two on a scale of 1 to 10", non-metric MDS is the honest choice: it only requires that "more similar" is ordered correctly.
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## scikit-learn version note

    The argument names changed in scikit-learn 1.8:

    - A precomputed distance matrix is passed with **`metric="precomputed"`** (was `dissimilarity="precomputed"`).
    - **`metric_mds=True`** selects metric MDS; `metric_mds=False` selects non-metric. The old boolean `metric=False` meant non-metric MDS but was a different argument.

    Those two uses of the word "metric" are the source of most confusion in this topic.
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Why a bend is hard for MDS

    MDS preserves **straight-line (Euclidean) distance**. On a flat page that is exactly what you want. On a curved surface it is the wrong quantity: two points on opposite arms of an S can be close through the air while being far apart **along the surface**.

    MDS therefore tends to **flatten** bends, because flattening reduces mismatch between straight-line distances. Keeping the bend would require **geodesic** distances along the surface — the topic of Isomap and locally linear embedding in the next lesson.
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Stress and SMACOF (metric MDS)

    **Raw stress** (badness of fit) is often written

    $$
    \sigma = \sum_{i<j} \bigl(d_{ij} - \hat d_{ij}\bigr)^2
    $$

    where \(d_{ij}\) is the input distance and \(\hat d_{ij}\) is the distance in the layout. With **`normalized_stress="auto"`**, **metric** MDS reports this **raw** stress; **non-metric** MDS reports **Kruskal stress-1** (normalised). Do not compare the two numbers directly.

    The algorithm is **SMACOF**: starting from coordinates (here `init="random"` with `n_init=1` so you can see seed sensitivity), it alternates updates that pull embedded distances toward the input distances. Non-metric SMACOF replaces \(d_{ij}\) by **disparities** from **isotonic regression** on the rank order, then embeds those disparities.
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Classical MDS and PCA

    For **Euclidean** input distances, **classical MDS** double-centres the squared distances,

    $$
    B = -\tfrac12 J D^{(2)} J, \quad J = I - \tfrac1n \mathbf 1\mathbf 1^\top,
    $$

    and takes eigenvectors of \(B\). That is the same geometry as **PCA on the original coordinates** when those coordinates exist — a useful link back to lesson 1.
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dataset: rendered GOLIATH

    **Source:** `Goliath.png` (also embedded as base64 below) — a **561 × 135** RGB image (alpha is **1 everywhere**, so there is no transparency). The word is **dark ink on a white background** (low channel-0 values); the sampler keeps pixels with value **below 1** as ink.

    The sampler keeps **1,000** reproducible points (`seed=42`), sorted left to right. **`GOLIATH_S_curve.csv`** in the folder is the same points bent onto an S for the next lesson; here we build that surface with `make_goliath_s_curve` so the formula is visible.
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
def _(np, pd):
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

    return (sample_goliath,)

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `sample_goliath`

    The image is flipped vertically and transposed so `x0` runs along the word. Four times as many random `(x, y)` pairs are drawn as points needed; only pairs landing on **ink** (`brightness < 1`) are kept. The horizontal axis is stretched by `rows/columns` so letters are not squashed. Rows are sorted by `x0`, then **gaps** wider than `0.08` in `x0` mark letter boundaries and assign labels left to right through GOLIATH.
    """)
    return

@app.cell
def _(LETTER_COLORS, px):
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
            color_discrete_map=LETTER_COLORS,
        )
        figure.update_traces(marker=dict(size=4, opacity=0.8))
        figure.update_yaxes(scaleanchor="x", scaleratio=1)
        figure.update_layout(
            height=380,
            margin=dict(t=55, b=40, l=45, r=20),
            legend=dict(font=dict(size=9), itemsizing="constant", title="letter"),
        )
        return figure

    return (draw_letters,)

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `draw_letters`

    Plotly draws one trace per letter. **`color_discrete_map=LETTER_COLORS`** keeps each glyph the same colour everywhere in the lesson. **`scaleanchor="x"`** forces one data unit on `x0` to equal one on `x1`, so circles stay round and the word is not stretched.
    """)
    return

@app.cell(hide_code=True)
def _(Image, base64, io, np):
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
    return goliath_b64, goliath_image

@app.cell
def _(go, goliath_image):
    _channel = goliath_image[:, :, 0]
    _figure = go.Figure(
        go.Heatmap(
            z=_channel,
            colorscale=[[0, "rgb(0,0,0)"], [1, "rgb(255,255,255)"]],
            showscale=False,
        )
    )
    _figure.update_layout(
        width=700,
        height=200,
        title="Goliath.png, the source image",
        margin=dict(t=55, b=10, l=10, r=10),
    )
    _figure.update_xaxes(visible=False)
    _figure.update_yaxes(autorange="reversed", visible=False)
    _figure
    return

@app.cell(hide_code=True)
def _(goliath_image, mo):
    mo.md(f"""
    The image is **embedded as base64** in a hidden cell so the notebook runs without an external file (including browser runtimes). Shape **{goliath_image.shape}** (rows, columns, channels). Channel 0 is shown with **dark pixels = ink** on a white background; the sampler uses the same channel and keeps pixels with value **below 1** as ink.
    """)
    return

@app.cell
def _(goliath_image, sample_goliath):
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
    Letters are not equally wide, so counts differ. **I** is the thinnest glyph; **L** has the fewest sampled points (**{int(letter_counts['L'])}**) because it is narrow. **O** is wide (**{int(letter_counts['O'])}**). Row order equals position in the word.
    """)
    return

@app.cell(hide_code=True)
def _(letters, mo):
    mo.md(f"""
    First rows of the sampled table (two coordinate columns plus label):

    {mo.Html(letters.head(8).to_html(index=False))}
    """)
    return

@app.cell
def _(draw_letters, letters):
    letters_scatter_figure = draw_letters(
        letters, "Points sampled from the word GOLIATH"
    )
    letters_scatter_figure
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The word is legible from points alone. The data are **two-dimensional** (`x0`, `x1`), and **`x0` carries word order** left to right.
    """)
    return

@app.cell
def _(np, pd):
    def rotate_letters(frame, angle):
        """Rotate a 2-D point cloud counter-clockwise by `angle` degrees.

        Row vectors are post-multiplied by a 2×2 rotation matrix: `coords @ R`.
        That convention matches a counter-clockwise turn in a standard
        `x`–`y` plot.

        Parameters
        ----------
        frame : pandas.DataFrame
            Columns `x0`, `x1`, and optionally `labels`.
        angle : float
            Rotation in degrees (positive = counter-clockwise).

        Returns
        -------
        pandas.DataFrame
            Copy of `frame` with rotated `x0` and `x1`.
        """
        theta = np.deg2rad(angle)
        rotation = np.array(
            [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
        )
        rotated = frame.copy()
        rotated[["x0", "x1"]] = frame[["x0", "x1"]].to_numpy() @ rotation
        return rotated

    return (rotate_letters,)

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `rotate_letters`

    For row vector `(x, y)`, the product `(x, y) @ R` applies the usual counter-clockwise rotation. Orthogonal matrices satisfy **RᵀR = I**, so ‖Rx − Ry‖ = ‖x − y‖: distances are unchanged even though every coordinate changes.
    """)
    return

@app.cell
def _(LETTER_COLORS, go, make_subplots):
    def letters_panel(frames, captions, title="", width=1150):
        """Place several letter clouds side by side for comparison.

        Parameters
        ----------
        frames : list of pandas.DataFrame
            One cloud per panel, each with `x0`, `x1`, and `labels`.
        captions : list of str
            Panel titles, in the same order.
        title : str, default ""
            Figure title.
        width : int, default 1150
            Figure width in pixels.

        Returns
        -------
        plotly.graph_objects.Figure
            Side-by-side panels sharing one colour legend.
        """
        count = len(frames)
        figure = make_subplots(
            rows=1,
            cols=count,
            subplot_titles=captions,
            horizontal_spacing=0.04,
        )
        for index, frame in enumerate(frames, start=1):
            for letter in "GOLIATH":
                mask = frame["labels"].to_numpy() == letter
                figure.add_trace(
                    go.Scatter(
                        x=frame["x0"].to_numpy()[mask],
                        y=frame["x1"].to_numpy()[mask],
                        mode="markers",
                        name=letter,
                        legendgroup=letter,
                        showlegend=index == 1,
                        marker=dict(
                            size=4,
                            opacity=0.8,
                            color=LETTER_COLORS[letter],
                        ),
                    ),
                    row=1,
                    col=index,
                )
            axis_suffix = "" if index == 1 else str(index)
            figure.update_yaxes(
                scaleanchor=f"x{axis_suffix}",
                scaleratio=1,
                row=1,
                col=index,
            )
        figure.update_layout(
            width=width,
            height=340,
            title=title,
            margin=dict(t=70, b=35, l=45, r=20),
            legend=dict(font=dict(size=9), itemsizing="constant", orientation="h", title="letter"),
        )
        return figure

    return (letters_panel,)

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `letters_panel`

    Several layouts are drawn in **one** figure so you can compare them without scrolling. Each letter always uses the same colour from `LETTER_COLORS`. Every panel gets its own **`scaleanchor`**, so a unit on `x0` matches a unit on `x1` in that panel — otherwise a recovered MDS layout would look sheared next to the original.
    """)
    return

@app.cell
def _(go, make_subplots, np):
    def distance_matrix_figure(matrices, captions, title="Pairwise distance matrices"):
        """Draw heatmaps for one or more distance matrices.

        Parameters
        ----------
        matrices : list of numpy.ndarray
            Square distance matrices, points ordered left-to-right.
        captions : list of str
            Subplot titles.
        title : str, default "Pairwise distance matrices"
            Overall figure title.

        Returns
        -------
        plotly.graph_objects.Figure
            Heatmaps with a shared colour bar and reversed row axis.
        """
        count = len(matrices)
        figure = make_subplots(
            rows=1,
            cols=count,
            subplot_titles=captions,
            horizontal_spacing=0.06,
        )
        zmax = max(float(m.max()) for m in matrices)
        for index, matrix in enumerate(matrices, start=1):
            figure.add_trace(
                go.Heatmap(
                    z=matrix,
                    colorscale="Blues",
                    zmin=0,
                    zmax=zmax,
                    colorbar=dict(title="distance") if index == count else None,
                    showscale=index == count,
                ),
                row=1,
                col=index,
            )
            figure.update_yaxes(autorange="reversed", row=1, col=index)
            figure.update_xaxes(visible=False, row=1, col=index)
        figure.update_layout(
            width=min(320 * count + 120, 980),
            height=330,
            title=title,
            margin=dict(t=60, b=25, l=25, r=20),
        )
        return figure

    return (distance_matrix_figure,)

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `distance_matrix_figure`

    With the **`Blues`** scale, **darker cells are larger distances** and lighter cells are smaller. Reversing the **y axis** puts index 0 at the top so the light **within-letter blocks** sit on the diagonal. One colour bar keeps the scale comparable across panels.
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. A rotation changes coordinates, not distances

    Rotating a point cloud leaves every pairwise distance unchanged. Below, drag the slider to rotate the cloud and watch the **largest pairwise distance change** stay at rounding error.
    """)
    return

@app.cell
def _(mo):
    rotation_slider = mo.ui.slider(
        -90,
        90,
        value=25,
        step=1,
        label="Rotation angle (degrees)",
        show_value=True,
    )
    rotation_slider
    return (rotation_slider,)

@app.cell
def _(
    distance_matrix_figure,
    letters,
    letters_panel,
    np,
    pairwise_distances,
    rotate_letters,
    rotation_slider,
):
    _angle = float(rotation_slider.value)
    _rotated = rotate_letters(letters, _angle)
    _base = pairwise_distances(letters[["x0", "x1"]])
    _rot = pairwise_distances(_rotated[["x0", "x1"]])
    _gap = float(np.max(np.abs(_base - _rot)))
    rotation_slider_figure = letters_panel(
        [letters, _rotated],
        ["Original", f"Rotated { _angle:.0f}°"],
        title=f"Rotation preserves shape (max distance change {_gap:.2e})",
        width=780,
    )
    rotation_slider_figure
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Orthogonal rotations satisfy **RᵀR = I**, hence ‖Rx − Ry‖ = ‖x − y‖. MDS never sees orientation — only the distance matrix — so any rotated copy is equally valid input.
    """)
    return

@app.cell
def _(letters, letters_panel, rotate_letters):
    turned_down = rotate_letters(letters, -20)
    turned_up = rotate_letters(letters, 50)
    rotation_triple_figure = letters_panel(
        [letters, turned_down, turned_up],
        ["Original", "Rotated by -20°", "Rotated by 50°"],
        title="Three orientations, identical distances",
    )
    rotation_triple_figure
    return turned_down, turned_up

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
    Across {len(letters) * (len(letters) - 1) // 2:,} pairs, the **largest distance change** under rotation is **{distance_gap:.2e}** — essentially zero. The three clouds are the same data in different coordinates.
    """)
    return

@app.cell
def _(distance_matrix_figure, distances, distances_down, distances_up):
    distance_matrices_figure = distance_matrix_figure(
        [distances, distances_down, distances_up],
        ["Original", "Rotated −20°", "Rotated 50°"],
        title="Identical distance matrices (dark = far, light = near)",
    )
    distance_matrices_figure
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Each heatmap is symmetric with a **zero diagonal**. With **`Blues`**, **darker cells are larger distances**; the **light blocks on the diagonal** are within-letter pairs (small distance). The y axis is reversed so index 0 is at the top. Nothing in the matrix encodes orientation.
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
The largest distance is about **4.02**, roughly the width of the word in the rescaled coordinates, and the diagonal sums to **0.0**.

The diagonal is zero because every point is at distance zero from itself. `distances.max()` is the diameter of the cloud — a compact summary of how spread out the ink is.
""")
        },
        expanded=False,
    )
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    MDS is given only the distance matrix. Given that rotations produce identical matrices, what can MDS possibly recover? Work out your answer, then reveal it.
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
It recovers the **relative arrangement** up to rotation, reflection, and translation.

The **shape** (clusters, gaps, neighbours) is meaningful; **axis directions are not**. Two seeds may return turned or mirrored layouts — both correct.

That is why this lesson uses **along_word** (order along one axis) and **Procrustes alignment** (shape after removing orientation), not pixel overlap with the original image.
""")
        },
        expanded=False,
    )
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Recovering a layout from distances alone

    Coordinates are discarded; MDS sees only the **precomputed** distance matrix. Use the seed slider to see that **structure** repeats while **orientation** changes.
    """)
    return

@app.cell
def _(mo):
    mds_seed_slider = mo.ui.slider(
        0,
        99,
        value=9,
        step=1,
        label="MDS random seed",
        show_value=True,
    )
    mds_seed_slider
    return (mds_seed_slider,)

@app.cell
def _(np):
    def along_word(position, embedded):
        """Absolute correlation between word order and an embedding axis.

        Parameters
        ----------
        position : array-like of shape (n,)
            Original left-to-right order (here, sorted `x0` or row index).
        embedded : array-like of shape (n, k)
            MDS coordinates.

        Returns
        -------
        float
            |correlation| between `position` and the embedded axis with largest
            variance (sign is ignored because MDS may flip an axis).
        """
        embedded = np.asarray(embedded)
        longest = embedded[:, int(np.argmax(np.var(embedded, axis=0)))]
        return abs(float(np.corrcoef(position, longest)[0, 1]))

    return (along_word,)

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `along_word`

    The word runs left to right, so a faithful layout should spread along one axis in that order. This score picks the **longest-variance axis** of the embedding and reports the absolute correlation with the original horizontal order. High values mean the **sequence** of letters survived; they do **not** mean each letter is still readable (see the S-bend section).
    """)
    return

@app.cell
def _(MDS, along_word, distances, letters, letters_panel, mds_seed_slider, pd):
    _seed = int(mds_seed_slider.value)
    _model = MDS(
        n_components=2,
        metric="precomputed",
        metric_mds=True,
        init="random",
        n_init=1,
        random_state=_seed,
        normalized_stress="auto",
    )
    _embedded = _model.fit_transform(distances)
    _layout = pd.DataFrame(
        {
            "x0": _embedded[:, 0],
            "x1": _embedded[:, 1],
            "labels": letters.labels.to_numpy(),
        }
    )
    _score = along_word(letters.x0.to_numpy(), _embedded)
    mds_seed_figure = letters_panel(
        [letters, _layout],
        ["Original coordinates", f"MDS seed {_seed} (stress {_model.stress_:.4f})"],
        title=f"Flat word from distances alone (along-word {_score:.3f})",
        width=820,
    )
    mds_seed_figure
    return

@app.cell
def _(MDS, along_word, distances, letters, letters_panel, pd):
    flat_layouts = {}
    flat_agreement = {}
    flat_stress = {}
    for _name, _seed in (("seed 9", 9), ("seed 5", 5)):
        _model = MDS(
            n_components=2,
            metric="precomputed",
            metric_mds=True,
            init="random",
            n_init=1,
            random_state=_seed,
            normalized_stress="auto",
        )
        _embedded = _model.fit_transform(distances)
        flat_layouts[_name] = pd.DataFrame(
            {
                "x0": _embedded[:, 0],
                "x1": _embedded[:, 1],
                "labels": letters.labels.to_numpy(),
            }
        )
        flat_agreement[_name] = along_word(letters.x0.to_numpy(), _embedded)
        flat_stress[_name] = float(_model.stress_)
    flat_seeds_figure = letters_panel(
        [flat_layouts["seed 9"], flat_layouts["seed 5"]],
        [
            f"MDS seed 9 (raw stress {flat_stress['seed 9']:.2f})",
            f"MDS seed 5 (raw stress {flat_stress['seed 5']:.2f})",
        ],
        title="Two seeds, the same flat distances",
        width=900,
    )
    flat_seeds_figure
    return flat_agreement, flat_layouts, flat_stress

@app.cell(hide_code=True)
def _(flat_agreement, flat_stress, mo):
    mo.md(f"""
    Both layouts recover the word with **along-word** correlations **{flat_agreement['seed 9']:.3f}** (seed 9) and **{flat_agreement['seed 5']:.3f}** (seed 5). **Seed 5 is mirrored** relative to seed 9 here — equally valid.

    **Raw stress** differs (**{flat_stress['seed 9']:.2f}** vs **{flat_stress['seed 5']:.2f}**) because `n_init=1` lets SMACOF stop in different local minima; the layouts still match after Procrustes alignment. Production code would raise **`n_init`** for stability.
    """)
    return

@app.cell
def _(MDS, distances, go, np):
    _dims = list(range(1, 6))
    _stress_values = []
    for _d in _dims:
        _model = MDS(
            n_components=_d,
            metric="precomputed",
            metric_mds=True,
            init="random",
            n_init=4,
            random_state=0,
            normalized_stress="auto",
        )
        _model.fit(distances)
        _stress_values.append(float(_model.stress_))
    _figure = go.Figure(
        go.Scatter(x=_dims, y=_stress_values, mode="lines+markers")
    )
    _figure.update_layout(
        width=520,
        height=380,
        title="Stress vs number of dimensions (flat word)",
        xaxis_title="n_components",
        yaxis_type="log",
        yaxis_title="raw stress (metric MDS)",
        margin=dict(t=55, b=45, l=55, r=20),
    )
    _figure
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Raw stress** (metric MDS, **`n_init=4`**) is **lowest at two dimensions** for this flat word — the log axis shows the 1-D penalty clearly. Stress at 3–5 dimensions is **clearly higher, about 8 times** the 2-D value here when SMACOF finds other local minima; that does not mean the cloud is “really” four-dimensional.
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Linear 3-D embedding: MDS succeeds

    The flat word is embedded in 3-D by a **random orthogonal mix** (`random_projection`) — distances unchanged. MDS on those 3-D distances should recover the word, contrasting with the S-bend failure later.
    """)
    return

@app.cell
def _(np, pd):
    def random_projection(frame, n=3, random_state=42):
        """Embed flat coordinates in `n` dimensions by a random orthogonal mix.

        Parameters
        ----------
        frame : pandas.DataFrame
            Columns `x0`, `x1`, and `labels`.
        n : int, default 3
            Target dimension (must be at least the input dimension).
        random_state : int, default 42
            Seed for the random orthogonal matrix.

        Returns
        -------
        pandas.DataFrame
            Columns `x0` … `x{n-1}` plus `labels`.
        """
        labels = frame["labels"]
        coords = frame[["x0", "x1"]].to_numpy()
        assert n >= coords.shape[1]
        rng = np.random.RandomState(random_state)
        random_matrix = rng.randn(n, n)
        _, vectors = np.linalg.eigh(random_matrix @ random_matrix.T)
        projected = coords @ vectors[: coords.shape[1]]
        columns = {f"x{j}": projected[:, j] for j in range(projected.shape[1])}
        columns["labels"] = labels.to_numpy()
        return pd.DataFrame(columns)

    return (random_projection,)

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `random_projection`

    A random orthogonal matrix **`V`** mixes the flat `(x0, x1)` into three coordinates **`(x0, x1, x2) = (x0, x1) @ V[:2]`** without changing any pairwise Euclidean distance. MDS fed those 3-D distances should recover a flat word — the success case before the S-bend failure.
    """)
    return

@app.cell
def _(letters, random_projection):
    letters_3d = random_projection(letters, n=3, random_state=42)
    return (letters_3d,)

@app.cell(hide_code=True)
def _(letters_3d, mo):
    mo.md(f"""
    First rows of the 3-D table (four columns — coordinates plus label):

    {mo.Html(letters_3d.head(8).to_html(index=False))}
    """)
    return

@app.cell
def _(
    MDS,
    along_word,
    letters,
    letters_3d,
    letters_panel,
    pairwise_distances,
    pd,
):
    dist_3d = pairwise_distances(letters_3d[["x0", "x1", "x2"]])
    layouts_3d = {}
    agree_3d = {}
    for _label, _seed in (("seed 9", 9), ("seed 5", 5)):
        _model = MDS(
            n_components=2,
            metric="precomputed",
            metric_mds=True,
            init="random",
            n_init=1,
            random_state=_seed,
            normalized_stress="auto",
        )
        _emb = _model.fit_transform(dist_3d)
        layouts_3d[_label] = pd.DataFrame(
            {
                "x0": _emb[:, 0],
                "x1": _emb[:, 1],
                "labels": letters.labels.to_numpy(),
            }
        )
        agree_3d[_label] = along_word(letters.x0.to_numpy(), _emb)
    linear_3d_figure = letters_panel(
        [layouts_3d["seed 9"], layouts_3d["seed 5"]],
        [
            f"3-D distances → MDS, seed 9 ({agree_3d['seed 9']:.3f})",
            f"3-D distances → MDS, seed 5 ({agree_3d['seed 5']:.3f})",
        ],
        title="Linear 3-D embedding recovered in 2-D",
        width=900,
    )
    linear_3d_figure
    return agree_3d, dist_3d, layouts_3d

@app.cell
def _(np, scipy_procrustes):
    def procrustes_align(source, target):
        """Orthogonally align `target` to `source`, allowing reflection.

        Wraps ``scipy.spatial.procrustes``: both clouds are centred and scaled
        to unit Frobenius norm, then `target` is rotated/reflected to match
        `source`.

        Parameters
        ----------
        source, target : array-like of shape (n, 2)
            Two layouts of the same points.

        Returns
        -------
        aligned : numpy.ndarray
            Procrustes-transformed `target` coordinates.
        disparity : float
            Procrustes disparity (0 = perfect match up to similarity transform).
        """
        _, aligned, disparity = scipy_procrustes(
            np.asarray(source, dtype=float),
            np.asarray(target, dtype=float),
        )
        return aligned, float(disparity)

    return (procrustes_align,)

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `procrustes_align`

    MDS layouts differ by rotation and reflection. **`scipy.spatial.procrustes`** superimposes two layouts (rotation **and** reflection allowed, both normalised to unit scale) so you can compare **shape** after removing arbitrary orientation.
    """)
    return

@app.cell(hide_code=True)
def _(flat_layouts, layouts_3d, mo, procrustes_align):
    _, procrustes_3d_gap = procrustes_align(
        layouts_3d["seed 9"][["x0", "x1"]].to_numpy(),
        layouts_3d["seed 5"][["x0", "x1"]].to_numpy(),
    )
    _, flat_vs_3d_gap = procrustes_align(
        flat_layouts["seed 9"][["x0", "x1"]].to_numpy(),
        layouts_3d["seed 9"][["x0", "x1"]].to_numpy(),
    )
    mo.md(f"""
    Distances after a **linear orthogonal lift** are unchanged, so these panels match the **flat** MDS recovery up to rotation/reflection (Procrustes disparity **{flat_vs_3d_gap:.4f}** vs flat seed 9; seed 5 mirrored with disparity **{procrustes_3d_gap:.4f}**). The S-bend breaks this because straight-line 3-D distances no longer match **along-surface** neighbours.
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Replace `None` so that `student_precomputed` holds the string that tells MDS the input is already a distance matrix (scikit-learn ≥ 1.8).
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
Use `MDS(metric="precomputed", ...)`. Do not confuse with `metric_mds`, which selects metric vs non-metric **algorithm**.
""")
        },
        expanded=False,
    )
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    For the flat distance matrix, fill in the loop so `student_stresses[d]` holds `model.stress_` after metric MDS with `n_components=d` (try `d = 1, 2, 3`).
    """)
    return

@app.cell
def _(MDS, distances):
    student_stresses = {}
    for _d in (1, 2, 3):
        student_stresses[_d] = None
    print(student_stresses)
    return (student_stresses,)

@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
```python
student_stresses = {}
for d in (1, 2, 3):
    model = MDS(
        n_components=d,
        metric="precomputed",
        metric_mds=True,
        init="random",
        n_init=4,
        random_state=0,
        normalized_stress="auto",
    )
    model.fit(distances)
    student_stresses[d] = model.stress_
```
With **`n_init=4`**, stress is **lowest at 2** dimensions; the 1-D value is vastly larger on a log plot. Values at 3+ can bump up slightly — local minima — but the word is still intrinsically **two-dimensional**.
""")
        },
        expanded=False,
    )
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Use `procrustes_align` to register the seed-5 layout onto the seed-9 layout. Store the post-alignment Frobenius **`disparity`** in `student_procrustes_gap`.
    """)
    return

@app.cell
def _(flat_layouts, procrustes_align):
    student_procrustes_gap = None
    print("Procrustes gap:", student_procrustes_gap)
    return (student_procrustes_gap,)

@app.cell(hide_code=True)
def _(flat_layouts, mo, procrustes_align):
    _, _expected = procrustes_align(
        flat_layouts["seed 9"][["x0", "x1"]].to_numpy(),
        flat_layouts["seed 5"][["x0", "x1"]].to_numpy(),
    )
    mo.accordion(
        {
            "Show answers": mo.md(f"""
```python
_, student_procrustes_gap = procrustes_align(
    flat_layouts["seed 9"][["x0", "x1"]].to_numpy(),
    flat_layouts["seed 5"][["x0", "x1"]].to_numpy(),
)
```
After allowing rotation **and reflection** (`scipy.spatial.procrustes`), disparity ≈ **{_expected:.4f}** — the shapes match; seed 5 was mirrored, not wrong. A disparity near 1 would mean genuinely different shapes.
""")
        },
        expanded=False,
    )
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Ordinal distances: where non-metric MDS wins

    The next demo **raises** flat distances to a power (`monotonic_distortion`). Order is preserved but intervals are wrong. Compare stress and Shepard diagrams for metric vs non-metric MDS on the same matrix.
    """)
    return

@app.cell
def _(np):
    def monotonic_distortion(distance_matrix, exponent=1.7):
        """Distort distance values while preserving their rank order.

        Parameters
        ----------
        distance_matrix : numpy.ndarray
            Symmetric matrix of distances.
        exponent : float, default 1.7
            Power applied to off-diagonal entries (1 preserves values).

        Returns
        -------
        numpy.ndarray
            Distorted symmetric matrix with the same ordering but wrong intervals.
        """
        distorted = distance_matrix.astype(float).copy()
        mask = ~np.eye(distorted.shape[0], dtype=bool)
        distorted[mask] = distorted[mask] ** exponent
        return distorted

    return (monotonic_distortion,)

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `monotonic_distortion`

    Raising distances to a power **`> 1`** stretches larger gaps more than small ones. **Order is unchanged**, but the numbers are no longer true Euclidean distances — a case where **non-metric** MDS is honest and **metric** MDS overfits the wrong scale.
    """)
    return

@app.cell
def _(go, np):
    def shepard_figure(
        input_distances,
        embedded_coords,
        title="Shepard diagram",
        max_pairs=6000,
        seed=42,
        show_reference_line=True,
    ):
        """Plot input distances against distances in the embedding.

        Parameters
        ----------
        input_distances : numpy.ndarray
            Square matrix of input distances.
        embedded_coords : numpy.ndarray of shape (n, k)
            Coordinates returned by MDS.
        title : str, default "Shepard diagram"
            Figure title.
        max_pairs : int, default 6000
            Maximum number of upper-triangle pairs to draw (random subsample).
        seed : int, default 42
            Subsample seed.
        show_reference_line : bool, default True
            If True, draw the ``y = x`` line (meaningful for metric MDS only).

        Returns
        -------
        plotly.graph_objects.Figure
            Scatter of input vs embedded distances.
        """
        embedded_distances = np.linalg.norm(
            embedded_coords[:, None, :] - embedded_coords[None, :, :],
            axis=-1,
        )
        indices = np.triu_indices(input_distances.shape[0], k=1)
        original = input_distances[indices]
        embedded = embedded_distances[indices]
        if original.size > max_pairs:
            rng = np.random.RandomState(seed)
            pick = rng.choice(original.size, max_pairs, replace=False)
            original = original[pick]
            embedded = embedded[pick]
        limit = float(max(original.max(), embedded.max()))
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=original,
                y=embedded,
                mode="markers",
                marker=dict(size=3, opacity=0.35),
                name="pairs",
            )
        )
        if show_reference_line:
            figure.add_trace(
                go.Scatter(
                    x=[0, limit],
                    y=[0, limit],
                    mode="lines",
                    name="y = x (metric target)",
                    line=dict(dash="dash"),
                )
            )
        figure.update_layout(
            width=520,
            height=420,
            title=title,
            xaxis_title="input distance",
            yaxis_title="embedded distance",
            margin=dict(t=55, b=45, l=55, r=20),
        )
        return figure

    return (shepard_figure,)

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `shepard_figure`

    Each dot is one pair of points. On **metric** MDS the cloud **hugs y = x** when distances are trustworthy; on the S-bend, **short** pairs often sit **below** y = x (embedded distance smaller than input straight-line distance) while the bulk still tracks the diagonal — letters flatten in the layout.
    """)
    return

@app.cell
def _(distances, monotonic_distortion):
    ordinal_subset_size = 250
    ordinal_distances = monotonic_distortion(
        distances[:ordinal_subset_size, :ordinal_subset_size],
        exponent=1.8,
    )
    return ordinal_distances, ordinal_subset_size

@app.cell
def _(MDS, ordinal_distances, shepard_figure):
    ordinal_metric_model = MDS(
        n_components=2,
        metric="precomputed",
        metric_mds=True,
        init="random",
        n_init=1,
        random_state=3,
        normalized_stress="auto",
    )
    ordinal_nonmetric_model = MDS(
        n_components=2,
        metric="precomputed",
        metric_mds=False,
        init="random",
        n_init=1,
        random_state=3,
        normalized_stress="auto",
    )
    ordinal_plane_m = ordinal_metric_model.fit_transform(ordinal_distances)
    ordinal_plane_n = ordinal_nonmetric_model.fit_transform(ordinal_distances)
    ordinal_metric_shepard = shepard_figure(
        ordinal_distances,
        ordinal_plane_m,
        title=f"Shepard — metric MDS (raw stress {ordinal_metric_model.stress_:.2f})",
        show_reference_line=True,
    )
    ordinal_metric_shepard
    return (
        ordinal_metric_model,
        ordinal_nonmetric_model,
        ordinal_plane_m,
        ordinal_plane_n,
    )

@app.cell
def _(ordinal_distances, ordinal_plane_n, ordinal_nonmetric_model, shepard_figure):
    ordinal_nonmetric_shepard = shepard_figure(
        ordinal_distances,
        ordinal_plane_n,
        title=(
            f"Shepard — non-metric MDS "
            f"(Stress-1 {ordinal_nonmetric_model.stress_:.4f})"
        ),
        show_reference_line=False,
    )
    ordinal_nonmetric_shepard
    return

@app.cell(hide_code=True)
def _(mo, ordinal_metric_model, ordinal_nonmetric_model):
    mo.md(f"""
    **Metric raw stress:** {ordinal_metric_model.stress_:.2f}. **Non-metric Stress-1:** {ordinal_nonmetric_model.stress_:.4f} — different scales by design. Read the **metric** Shepard plot against the **y = x** line; for **non-metric**, look for a **monotonic** cloud (no y = x reference — embedded distances follow disparities, not input units).
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. A word bent into an S — MDS fails to unroll

    The same word order is wrapped onto an S-shaped surface. MDS still minimises straight-line distance mismatch, so the layout collapses letters into **thin curved bands**: order survives, **legibility does not**.
    """)
    return

@app.cell
def _(np, pd):
    def make_goliath_s_curve(frame, fold_tightness=1.0):
        """Wrap the flat word onto an S-shaped surface in three dimensions.

        Parameters
        ----------
        frame : pandas.DataFrame
            Flat cloud with `x0`, `x1`, and `labels`.
        fold_tightness : float, default 1.0
            Multiplier on the parameter along the word; values above 1 wind the
            S more tightly (letters can come closer in space through the air).

        Returns
        -------
        pandas.DataFrame
            Columns `x0`, `x1`, `x2`, and `labels` on the bent surface.
        """
        parameter = (frame.x0 - 2) * 0.78 * np.pi * fold_tightness
        return pd.DataFrame(
            {
                "x0": np.sin(parameter),
                "x1": frame.x1 * 1.5,
                "x2": np.sign(parameter) * (np.cos(parameter) - 1) * 1.5,
                "labels": frame.labels.to_numpy(),
            }
        )

    return (make_goliath_s_curve,)

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `make_goliath_s_curve`

    The horizontal position along the word becomes an angle **`t`**. **`x0 = sin(t)`** and **`x2 = sign(t)(cos(t) − 1)`** bend the sheet into an S while **`x1`** still tracks glyph height. Increasing **`fold_tightness`** winds the same word order through more of the S, bringing distant letters closer in straight-line distance — the regime where MDS misleads.
    """)
    return

@app.cell
def _(letters, make_goliath_s_curve):
    curve = make_goliath_s_curve(letters, fold_tightness=1.0)
    return (curve,)

@app.cell(hide_code=True)
def _(curve, mo):
    mo.md(f"""
    Bent cloud (four columns):

    {mo.Html(curve.head(8).to_html(index=False))}
    """)
    return

@app.cell
def _(LETTER_COLORS, curve, px):
    _figure = px.scatter_3d(
        curve,
        x="x0",
        y="x1",
        z="x2",
        color="labels",
        title="The word bent into an S (word order along the curve)",
        category_orders={"labels": list("GOLIATH")},
        color_discrete_map=LETTER_COLORS,
    )
    _figure.update_traces(marker=dict(size=2, opacity=0.85))
    _figure.update_layout(
        width=760,
        height=480,
        margin=dict(t=55, b=20, l=20, r=20),
        legend=dict(font=dict(size=9), itemsizing="constant", title="letter"),
        scene=dict(
            camera=dict(eye=dict(x=2.2, y=1.5, z=1.0)),
            aspectmode="cube",
        ),
    )
    _figure
    return

@app.cell
def _(np):
    def letter_shape_ratio(frame, embedded):
        """Mean within-letter ``sigma_2 / sigma_1`` from an SVD of each glyph.

        Parameters
        ----------
        frame : pandas.DataFrame
            Table with a `labels` column (same row order as the embedding).
        embedded : array-like of shape (n, 2)
            Two-dimensional coordinates to measure.

        Returns
        -------
        float
            Average over letters of ``sigma_2 / sigma_1`` (0 = line, 1 = round
            blob). Versus a flat layout the ratio **drops sharply** on a
            moderate S-bend; at **extreme** folds overlapping letters can
            **raise** it again, so it is not monotonic in tightness.
        """
        embedded = np.asarray(embedded)
        labels = frame["labels"].to_numpy()
        ratios = []
        for letter in "GOLIATH":
            mask = labels == letter
            if mask.sum() < 3:
                continue
            centered = embedded[mask] - embedded[mask].mean(axis=0)
            singular = np.linalg.svd(centered, compute_uv=False)
            if singular[0] > 0:
                ratios.append(float(singular[1] / singular[0]))
        return float(np.mean(ratios)) if ratios else 0.0

    return (letter_shape_ratio,)

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `letter_shape_ratio`

    Legible letters occupy **2-D blobs** in the layout; S-bend MDS squashes them into **ribbons** (one long direction, one thin direction). For each letter we SVD its centered points and record ``sigma_2 / sigma_1`` (near **1** = fat blob, near **0** = line). The ratio **falls vs flat** at moderate bends but can **rise again** when letters overlap in projection — not a monotonic legibility dial.
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Explore fold tightness, metric vs non-metric, and stress

    Increase **fold tightness** to bring distant letters closer through the air. **Along-word** correlation can stay high at moderate folds but **falls** when the S is very tight. **`letter_shape_ratio`** **drops vs flat** then can **rise again** as letters overlap (watch the stat line under the plot). **Raw stress** is **not monotonic** in tightness — use it as a diagnostic, not a single trend line.
    """)
    return

@app.cell
def _(mo):
    fold_slider = mo.ui.slider(
        0.6,
        2.2,
        value=1.0,
        step=0.1,
        label="S fold tightness",
        show_value=True,
    )
    metric_checkbox = mo.ui.checkbox(value=True, label="Metric MDS (off = non-metric)")
    mo.hstack([fold_slider, metric_checkbox], gap=2)
    return fold_slider, metric_checkbox

@app.cell
def _(
    MDS,
    along_word,
    fold_slider,
    letter_shape_ratio,
    letters,
    letters_panel,
    make_goliath_s_curve,
    metric_checkbox,
    pd,
):
    live_tightness = float(fold_slider.value)
    live_curve = make_goliath_s_curve(letters, fold_tightness=live_tightness)
    live_model = MDS(
        n_components=2,
        metric_mds=bool(metric_checkbox.value),
        init="random",
        n_init=1,
        random_state=19,
        normalized_stress="auto",
    )
    live_emb = live_model.fit_transform(live_curve[["x0", "x1", "x2"]])
    live_layout = pd.DataFrame(
        {
            "x0": live_emb[:, 0],
            "x1": live_emb[:, 1],
            "labels": live_curve.labels.to_numpy(),
        }
    )
    live_shape_ratio = letter_shape_ratio(live_curve, live_emb)
    live_kind = "Metric" if metric_checkbox.value else "Non-metric"
    live_stress_label = (
        "raw stress"
        if metric_checkbox.value
        else "Stress-1"
    )
    live_s_summary = {
        "kind": live_kind,
        "tightness": live_tightness,
        "stress": float(live_model.stress_),
        "stress_label": live_stress_label,
        "shape_ratio": live_shape_ratio,
        "along_word": along_word(letters.x0.to_numpy(), live_emb),
    }
    live_s_figure = letters_panel(
        [live_layout],
        [f"{live_kind} MDS (tightness {live_tightness:.1f})"],
        title="Live S-bend layout",
        width=520,
    )
    live_s_figure
    return (live_s_summary,)

@app.cell(hide_code=True)
def _(live_s_summary, mo):
    mo.md(f"""
    **{live_s_summary['kind']}** at tightness **{live_s_summary['tightness']:.1f}:** {live_s_summary['stress_label']} **{live_s_summary['stress']:.2f}**, along-word **{live_s_summary['along_word']:.3f}**, mean shape ratio **{live_s_summary['shape_ratio']:.3f}** (σ₂/σ₁ per letter; lower = thinner ribbons).
    """)
    return

@app.cell
def _(
    MDS,
    along_word,
    curve,
    flat_layouts,
    letter_shape_ratio,
    letters,
    letters_panel,
    pairwise_distances,
    pd,
    shepard_figure,
):
    curve_metric_model = MDS(
        n_components=2,
        metric_mds=True,
        init="random",
        n_init=1,
        random_state=19,
        normalized_stress="auto",
    )
    curve_nm_model = MDS(
        n_components=2,
        metric_mds=False,
        init="random",
        n_init=1,
        random_state=19,
        normalized_stress="auto",
    )
    curve_emb_metric = curve_metric_model.fit_transform(curve[["x0", "x1", "x2"]])
    curve_emb_nonmetric = curve_nm_model.fit_transform(curve[["x0", "x1", "x2"]])
    curve_layouts = {
        "metric": pd.DataFrame(
            {
                "x0": curve_emb_metric[:, 0],
                "x1": curve_emb_metric[:, 1],
                "labels": curve.labels.to_numpy(),
            }
        ),
        "non-metric": pd.DataFrame(
            {
                "x0": curve_emb_nonmetric[:, 0],
                "x1": curve_emb_nonmetric[:, 1],
                "labels": curve.labels.to_numpy(),
            }
        ),
    }
    curve_agreement = {
        "metric": along_word(letters.x0.to_numpy(), curve_emb_metric),
        "non-metric": along_word(letters.x0.to_numpy(), curve_emb_nonmetric),
    }
    curve_shape = {
        "metric": letter_shape_ratio(curve, curve_emb_metric),
        "non-metric": letter_shape_ratio(curve, curve_emb_nonmetric),
    }
    flat_shape = letter_shape_ratio(
        letters, flat_layouts["seed 9"][["x0", "x1"]].to_numpy()
    )
    curve_input_distances = pairwise_distances(curve[["x0", "x1", "x2"]])
    collapse_comparison_figure = letters_panel(
        [flat_layouts["seed 9"], curve_layouts["metric"], curve_layouts["non-metric"]],
        [
            f"Flat MDS (shape ratio {flat_shape:.3f})",
            f"S metric (ratio {curve_shape['metric']:.3f})",
            f"S non-metric (ratio {curve_shape['non-metric']:.3f})",
        ],
        title="Flat recovery vs S-bend collapse",
        width=1150,
    )
    collapse_comparison_figure
    return (
        curve_agreement,
        curve_shape,
        curve_layouts,
        flat_shape,
        curve_input_distances,
        curve_emb_metric,
        curve_emb_nonmetric,
        curve_metric_model,
        curve_nm_model,
    )

@app.cell
def _(
    curve_emb_metric,
    curve_emb_nonmetric,
    curve_input_distances,
    curve_metric_model,
    curve_nm_model,
    shepard_figure,
):
    sbend_metric_shepard = shepard_figure(
        curve_input_distances,
        curve_emb_metric,
        title=(
            f"Shepard — metric MDS on the S-bend "
            f"(raw stress {curve_metric_model.stress_:.0f})"
        ),
        show_reference_line=True,
    )
    sbend_metric_shepard
    return

@app.cell
def _(
    curve_emb_nonmetric,
    curve_input_distances,
    curve_nm_model,
    shepard_figure,
):
    sbend_nonmetric_shepard = shepard_figure(
        curve_input_distances,
        curve_emb_nonmetric,
        title=(
            f"Shepard — non-metric MDS on the S-bend "
            f"(Stress-1 {curve_nm_model.stress_:.4f})"
        ),
        show_reference_line=False,
    )
    sbend_nonmetric_shepard
    return

@app.cell(hide_code=True)
def _(curve_agreement, curve_shape, flat_shape, mo):
    mo.md(f"""
    **MDS fails to unroll the S.** Along-word correlation is still **{curve_agreement['metric']:.3f}** (metric) — order along the curve survives — but **mean shape ratio** falls from **{flat_shape:.3f}** on the flat layout to **{curve_shape['metric']:.3f}** (metric) and **{curve_shape['non-metric']:.3f}** (non-metric). Letters merge into **illegible ribbons** (the S seen edge-on), not an opened word.

    The **metric** Shepard plot **hugs y = x** overall, with **short** pairs often **below** the line — straight-line input distances are a poor stand-in for **along-surface** neighbours. Neither MDS variant invents geodesics; that is why the next lesson turns to Isomap and LLE.
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Metric and non-metric MDS give similar correlations on the S-bend here. Describe a situation where they would differ, and why.
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
They differ when **values are untrustworthy but order is**.

Metric MDS chases every numeric gap; non-metric MDS fits **rank order** via isotonic disparities. See the **ordinal distortion** section: powering distances breaks interval meaning while preserving order.

On the S-bend, inputs are true Euclidean distances in 3-D, so both methods chase the same wrong straight-line objective — hence similar collapsed layouts.
""")
        },
        expanded=False,
    )
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    If **fold tightness** increases, what should happen to (a) along-word correlation and (b) mean shape ratio in the MDS layout? Reason first, then reveal.
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
**(a) Along-word** can stay fairly high at moderate tightness but **falls** when the fold is extreme (try 1.6–2.2 on the slider): the global sequence is harder to keep on one axis when unrelated letters become neighbours through the air.

**(b) Shape ratio** (σ₂/σ₁ per letter) **drops vs the flat layout** when glyphs become **ribbons**, but can **rise again** at extreme tightness when letters **overlap** in the 2-D map — check the stat line under the live plot.

**Raw stress** is **not monotonic** in tightness; it reports misfit to straight-line distances, not “how bent” the S looks.
""")
        },
        expanded=False,
    )
    return

@app.cell(hide_code=True)
def _(
    curve_agreement,
    curve_shape,
    distance_gap,
    flat_agreement,
    flat_shape,
    mo,
):
    mo.md(f"""
    ## Conclusions

    1. **MDS fits coordinates to a distance matrix** by lowering stress — the mismatch between input and embedded distances (Goal 1).
    2. **Rotations change coordinates, not distances** (max change **{distance_gap:.2e}** here), so MDS layouts have no meaningful axes — only shape (Goal 2).
    3. **Heatmaps summarise pairs:** symmetric, zero diagonal, dark = far with the Blues scale (Goal 3).
    4. **Metric MDS matches values; non-metric fits order** — use non-metric for ratings or monotonically distorted inputs (Goal 4; ordinal demo above).
    5. **On the S-bend, MDS preserves straight-line distances but fails to unroll:** along-word **{curve_agreement['metric']:.3f}** vs flat **{flat_agreement['seed 9']:.3f}**, but mean shape ratio **{curve_shape['metric']:.3f}** vs **{flat_shape:.3f}** on the flat layout — ribbons, not a readable word (Goal 5).
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
