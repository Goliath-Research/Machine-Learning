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
            import micropip as _micropip
            await _micropip.install(_missing)
        else:
            import subprocess as _subprocess
            _subprocess.check_call([_sys.executable, "-m", "pip", "install", *_missing])

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
    from PIL import Image
    from plotly.subplots import make_subplots
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
        go,
        make_subplots,
        mo,
        np,
        pairwise_distances,
        pd,
        px,
    )



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
def _(np, pd):
    def letter_height_spread(flat_frame, embedded):
        """Mean within-letter spread along the original height direction.

        Parameters
        ----------
        flat_frame : pandas.DataFrame
            Original flat cloud with `x1` (glyph height) and `labels`.
        embedded : array-like of shape (n, 2)
            Two-dimensional MDS layout.

        Returns
        -------
        float
            Average over letters of the standard deviation, along the embedded
            axis most aligned with flat `x1`, computed within each letter.
        """
        embedded = np.asarray(embedded)
        height = flat_frame["x1"].to_numpy()
        axis_index = int(
            np.argmax(
                [
                    abs(np.corrcoef(height, embedded[:, j])[0, 1])
                    for j in range(embedded.shape[1])
                ]
            )
        )
        values = embedded[:, axis_index]
        labels = flat_frame["labels"].to_numpy()
        spreads = []
        for letter in "GOLIATH":
            mask = labels == letter
            if mask.sum() > 1:
                spreads.append(float(np.std(values[mask])))
        return float(np.mean(spreads)) if spreads else 0.0

    return (letter_height_spread,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `letter_height_spread`

    On the flat word, **`x1` is stroke height** — loops and bars need spread in that direction to stay legible. After MDS on a bent 3-D cloud, layouts often collapse each letter into a thin ribbon: word order can stay high on `along_word`, but **height spread inside each letter** collapses. This number tracks that legibility loss.
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


@app.cell
def _(go, np):
    def shepard_figure(input_distances, embedded_coords, title="Shepard diagram"):
        """Plot original distances against distances in the embedding.

        Parameters
        ----------
        input_distances : numpy.ndarray
            Square matrix of input distances.
        embedded_coords : numpy.ndarray of shape (n, k)
            Coordinates returned by MDS.
        title : str, default "Shepard diagram"
            Figure title.

        Returns
        -------
        plotly.graph_objects.Figure
            Scatter of input vs embedded distance plus the y = x line.
        """
        embedded_distances = np.linalg.norm(
            embedded_coords[:, None, :] - embedded_coords[None, :, :],
            axis=-1,
        )
        indices = np.triu_indices(input_distances.shape[0], k=1)
        original = input_distances[indices]
        embedded = embedded_distances[indices]
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
        figure.add_trace(
            go.Scatter(
                x=[0, limit],
                y=[0, limit],
                mode="lines",
                name="perfect fit",
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

    Each dot is one pair of points. On **metric** MDS the cloud hugs the diagonal when straight-line input distances are trustworthy. When the surface is bent, many pairs are **too close in the input** (through-the-air distance) compared with distance along the word — the cloud bends **below** the diagonal and letters flatten in the layout.
    """)
    return


@app.cell
def _(np):
    def procrustes_align(source, target):
        """Orthogonally align `target` to `source` (same scale, rotation only).

        Parameters
        ----------
        source, target : array-like of shape (n, 2)
            Two layouts of the same points.

        Returns
        -------
        aligned : numpy.ndarray
            Rotated/reflected `target` registered to `source`.
        disparity : float
            Frobenius norm of `source - aligned` after alignment.
        """
        source = np.asarray(source, dtype=float)
        target = np.asarray(target, dtype=float)
        source = source - source.mean(axis=0)
        target = target - target.mean(axis=0)
        cross = target.T @ source
        u, _, vt = np.linalg.svd(cross)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0:
            u[:, -1] *= -1
            rotation = u @ vt
        aligned = target @ rotation
        scale = np.trace(source.T @ aligned) / np.trace(aligned.T @ aligned)
        aligned = aligned * scale
        disparity = float(np.linalg.norm(source - aligned))
        return aligned, disparity

    return (procrustes_align,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `procrustes_align`

    MDS layouts differ by rotation and reflection. Procrustes analysis finds the orthogonal map that best superimposes two layouts so you can compare **shape** after removing arbitrary orientation.
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

    \\[
    \\sigma = \\sum_{i<j} \\bigl(d_{ij} - \\hat d_{ij}\\bigr)^2
    \\]

    where \\(d_{ij}\\) is the input distance and \\(\\hat d_{ij}\\) is the distance in the layout. **Kruskal stress-1** normalises that sum for comparability; scikit-learn reports a normalised stress when `normalized_stress="auto"`.

    The algorithm is **SMACOF**: starting from coordinates (here `init="random"` with `n_init=1` so you can see seed sensitivity), it alternates updates that pull embedded distances toward the input distances. Non-metric SMACOF replaces \\(d_{ij}\\) by **disparities** from **isotonic regression** on the rank order, then embeds those disparities.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Classical MDS and PCA

    For **Euclidean** input distances, **classical MDS** double-centres the squared distances,

    \\[
    B = -\\tfrac12 J D^{(2)} J, \\quad J = I - \\tfrac1n \\mathbf 1\\mathbf 1^\\top,
    \\]

    and takes eigenvectors of \\(B\\). That is the same geometry as **PCA on the original coordinates** when those coordinates exist — a useful link back to lesson 1.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dataset: rendered GOLIATH

    **Source:** `Goliath.png` in this folder — a **561 × 135** RGB image (alpha is **1 everywhere**, so there is no transparency). The word is **light ink on a white background**; the sampler treats **dark pixels** as ink.

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


@app.cell(hide_code=True)
def _(Image, Path, np):
    _image_path = Path(__file__).resolve().parent / "Goliath.png"
    goliath_image = np.array(Image.open(_image_path)).astype(float) / 255.0
    return (goliath_image,)


@app.cell
def _(go, goliath_image):
    _ink = 1.0 - goliath_image[:, :, 0]
    _figure = go.Figure(
        go.Heatmap(z=_ink, colorscale="Gray", showscale=False)
    )
    _figure.update_layout(
        width=700,
        height=200,
        title="Goliath.png, the source image (ink shown dark)",
        margin=dict(t=55, b=10, l=10, r=10),
    )
    _figure.update_xaxes(visible=False)
    _figure.update_yaxes(visible=False)
    _figure
    return


@app.cell(hide_code=True)
def _(goliath_image, mo):
    mo.md(f"""
    The PNG is read from **`Goliath.png`** next to this notebook (shape **{goliath_image.shape}**: rows, columns, channels). Channel 0 is inverted for display so ink reads clearly on white. The sampler uses the same channel and keeps pixels with value **below 1** as ink.
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
    Letters are not equally wide, so counts differ. **L** is the thinnest glyph (**{int(letter_counts['L'])}** points) and **O** is wide (**{int(letter_counts['O'])}**). Row order equals position in the word.
    """)
    return


@app.cell(hide_code=True)
def _(letters, mo):
    mo.md(f"""
    First rows of the sampled table (three coordinate columns):

    {mo.Html(letters.head(8).to_html(index=False))}
    """)
    return


@app.cell
def _(draw_letters, letters):
    _fig = draw_letters(letters, "Points sampled from the word GOLIATH")
    _fig
    return (_fig,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The word is legible from points alone. The data are **two-dimensional** (`x0`, `x1`), and **`x0` carries word order** left to right.
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
    letters_panel(
        [letters, _rotated],
        ["Original", f"Rotated { _angle:.0f}°"],
        title=f"Rotation preserves shape (max distance change {_gap:.2e})",
        width=780,
    )
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
    letters_panel(
        [letters, turned_down, turned_up],
        ["Original", "Rotated by -20°", "Rotated by 50°"],
        title="Three orientations, identical distances",
    )
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
    distance_matrix_figure(
        [distances, distances_down, distances_up],
        ["Original", "Rotated −20°", "Rotated 50°"],
        title="Identical distance matrices (dark = far, light = near)",
    )
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
    letters_panel(
        [letters, _layout],
        ["Original coordinates", f"MDS seed {_seed} (stress {_model.stress_:.4f})"],
        title=f"Flat word from distances alone (along-word {_score:.3f})",
        width=820,
    )
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
    letters_panel(
        [flat_layouts["seed 9"], flat_layouts["seed 5"]],
        [
            f"MDS seed 9 (stress {flat_stress['seed 9']:.4f})",
            f"MDS seed 5 (stress {flat_stress['seed 5']:.4f})",
        ],
        title="Two seeds, the same flat distances",
        width=900,
    )
    return flat_agreement, flat_layouts, flat_stress


@app.cell(hide_code=True)
def _(flat_agreement, mo):
    mo.md(f"""
    Both layouts recover the word with **along-word** correlations **{flat_agreement['seed 9']:.3f}** (seed 9) and **{flat_agreement['seed 5']:.3f}** (seed 5). Seed 5 is commonly **mirrored** relative to seed 9 — equally valid.

    **`init="random"`** with **`n_init=1`** makes the seed visible; production code would raise `n_init` for stability.
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
            n_init=1,
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
        yaxis_title="normalized stress",
        margin=dict(t=55, b=45, l=55, r=20),
    )
    _figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Stress **falls** as you allow more dimensions, then levels off once the layout has enough freedom to match the input distances. For this 2-D word, **two components** already capture essentially all structure.
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
def _(letters, pd, random_projection):
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
    _dist_3d = pairwise_distances(letters_3d[["x0", "x1", "x2"]])
    _layouts_3d = {}
    _agree_3d = {}
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
        _emb = _model.fit_transform(_dist_3d)
        _layouts_3d[_label] = pd.DataFrame(
            {
                "x0": _emb[:, 0],
                "x1": _emb[:, 1],
                "labels": letters.labels.to_numpy(),
            }
        )
        _agree_3d[_label] = along_word(letters.x0.to_numpy(), _emb)
    letters_panel(
        [_layouts_3d["seed 9"], _layouts_3d["seed 5"]],
        [
            f"3-D distances → MDS, seed 9 ({_agree_3d['seed 9']:.3f})",
            f"3-D distances → MDS, seed 5 ({_agree_3d['seed 5']:.3f})",
        ],
        title="Linear 3-D embedding recovered in 2-D",
        width=900,
    )
    return _agree_3d, _dist_3d, _layouts_3d


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Distances after a linear orthogonal lift are unchanged, so MDS still finds a **legible** flat word — only orientation differs by seed. The S-bend breaks this because straight-line 3-D distances no longer match **along-surface** neighbours.
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
        n_init=1,
        random_state=0,
        normalized_stress="auto",
    )
    model.fit(distances)
    student_stresses[d] = model.stress_
```
Stress drops sharply from 1 to 2 dimensions, then barely changes at 3 — the cloud is intrinsically two-dimensional.
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
After removing rotation/reflection, the two layouts match closely (disparity ≈ **{_expected:.3f}** on this scale). Large gaps would mean different **shapes**, not just orientation.
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
def _(distances, monotonic_distortion):
    ordinal_subset_size = 250
    ordinal_distances = monotonic_distortion(
        distances[:ordinal_subset_size, :ordinal_subset_size],
        exponent=1.8,
    )
    return ordinal_distances, ordinal_subset_size


@app.cell
def _(MDS, ordinal_distances, shepard_figure):
    _metric = MDS(
        n_components=2,
        metric="precomputed",
        metric_mds=True,
        init="random",
        n_init=1,
        random_state=3,
        normalized_stress="auto",
    )
    _nonmetric = MDS(
        n_components=2,
        metric="precomputed",
        metric_mds=False,
        init="random",
        n_init=1,
        random_state=3,
        normalized_stress="auto",
    )
    _plane_m = _metric.fit_transform(ordinal_distances)
    _plane_n = _nonmetric.fit_transform(ordinal_distances)
    shepard_figure(
        ordinal_distances,
        _plane_m,
        title=f"Shepard — metric MDS (stress {_metric.stress_:.4f})",
    )
    return _metric, _nonmetric, _plane_m, _plane_n


@app.cell
def _(ordinal_distances, shepard_figure, _plane_n, _nonmetric):
    shepard_figure(
        ordinal_distances,
        _plane_n,
        title=f"Shepard — non-metric MDS (stress {_nonmetric.stress_:.4f})",
    )
    return


@app.cell(hide_code=True)
def _(mo, _metric, _nonmetric):
    mo.md(f"""
    **Metric stress:** {_metric.stress_:.4f}. **Non-metric stress:** {_nonmetric.stress_:.4f}. Non-metric MDS fits the **rank order** without pretending the powered values are true distances — the Shepard plot for non-metric typically hugs the monotonic trend more honestly when intervals are distorted.
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
def _(LETTER_COLORS, curve, go, px):
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
            camera=dict(eye=dict(x=1.6, y=1.3, z=0.55)),
            aspectmode="data",
        ),
    )
    _figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Explore fold tightness, metric vs non-metric, and stress

    Increase **fold tightness** to bring distant letters closer through the air. Watch **along-word** correlation stay high while **letter height spread** collapses — and stress rise as the 2-D layout fights the bend.
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
    letter_height_spread,
    letters,
    letters_panel,
    make_goliath_s_curve,
    metric_checkbox,
    pd,
):
    _tight = float(fold_slider.value)
    _curve = make_goliath_s_curve(letters, fold_tightness=_tight)
    _model = MDS(
        n_components=2,
        metric_mds=bool(metric_checkbox.value),
        init="random",
        n_init=1,
        random_state=19,
        normalized_stress="auto",
    )
    _emb = _model.fit_transform(_curve[["x0", "x1", "x2"]])
    _layout = pd.DataFrame(
        {
            "x0": _emb[:, 0],
            "x1": _emb[:, 1],
            "labels": _curve.labels.to_numpy(),
        }
    )
    _order = along_word(letters.x0.to_numpy(), _emb)
    _height = letter_height_spread(letters, _emb)
    _kind = "metric" if metric_checkbox.value else "non-metric"
    letters_panel(
        [_layout],
        [
            f"{_kind} MDS, tightness {_tight:.1f} — stress {_model.stress_:.4f}, "
            f"along-word {_order:.3f}, height spread {_height:.3f}",
        ],
        title="Live S-bend layout",
        width=520,
    )
    return


@app.cell
def _(
    MDS,
    along_word,
    curve,
    flat_layouts,
    letter_height_spread,
    letters,
    letters_panel,
    pairwise_distances,
    pd,
    shepard_figure,
):
    _metric_model = MDS(
        n_components=2,
        metric_mds=True,
        init="random",
        n_init=1,
        random_state=19,
        normalized_stress="auto",
    )
    _nm_model = MDS(
        n_components=2,
        metric_mds=False,
        init="random",
        n_init=1,
        random_state=19,
        normalized_stress="auto",
    )
    _m = _metric_model.fit_transform(curve[["x0", "x1", "x2"]])
    _n = _nm_model.fit_transform(curve[["x0", "x1", "x2"]])
    curve_layouts = {
        "metric": pd.DataFrame(
            {
                "x0": _m[:, 0],
                "x1": _m[:, 1],
                "labels": curve.labels.to_numpy(),
            }
        ),
        "non-metric": pd.DataFrame(
            {
                "x0": _n[:, 0],
                "x1": _n[:, 1],
                "labels": curve.labels.to_numpy(),
            }
        ),
    }
    curve_agreement = {
        "metric": along_word(letters.x0.to_numpy(), _m),
        "non-metric": along_word(letters.x0.to_numpy(), _n),
    }
    curve_height = {
        "metric": letter_height_spread(letters, _m),
        "non-metric": letter_height_spread(letters, _n),
    }
    flat_height = letter_height_spread(
        letters, flat_layouts["seed 9"][["x0", "x1"]].to_numpy()
    )
    letters_panel(
        [flat_layouts["seed 9"], curve_layouts["metric"], curve_layouts["non-metric"]],
        [
            f"Flat MDS (height spread {flat_height:.3f})",
            f"S metric (spread {curve_height['metric']:.3f})",
            f"S non-metric (spread {curve_height['non-metric']:.3f})",
        ],
        title="Flat recovery vs S-bend collapse",
        width=1150,
    )
    _input_d = pairwise_distances(curve[["x0", "x1", "x2"]])
    shepard_figure(
        _input_d,
        _m,
        title=f"Shepard — metric MDS on the S-bend (stress {_metric_model.stress_:.4f})",
    )
    return (
        curve_agreement,
        curve_height,
        curve_layouts,
        flat_height,
        _input_d,
        _m,
        _n,
        _metric_model,
        _nm_model,
    )


@app.cell
def _(shepard_figure, _input_d, _n, _nm_model):
    shepard_figure(
        _input_d,
        _n,
        title=f"Shepard — non-metric MDS on the S-bend (stress {_nm_model.stress_:.4f})",
    )
    return


@app.cell(hide_code=True)
def _(curve_agreement, curve_height, flat_height, mo):
    mo.md(f"""
    **MDS fails to unroll the S.** Along-word correlations stay high (**{curve_agreement['metric']:.3f}** metric, **{curve_agreement['non-metric']:.3f}** non-metric), so letter **order** survives, but **height spread** drops from **{flat_height:.3f}** on the flat layout to **{curve_height['metric']:.3f}** (metric) and **{curve_height['non-metric']:.3f}** (non-metric). Letters merge into illegible ribbons — the S viewed edge-on, not opened flat.

    The Shepard plots bend away from the diagonal: straight-line input distances are a poor stand-in for **along-surface** neighbours. Neither MDS variant invents geodesics; that is why the next lesson turns to Isomap and LLE.
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

    If **fold tightness** increases, what should happen to (a) along-word correlation and (b) letter height spread in the MDS layout? Reason first, then reveal.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
**(a) Along-word** can stay high: left-to-right order along the curve is still in the data, and MDS can keep that sequence on one axis.

**(b) Height spread** should fall further and layouts grow more ribbon-like: tighter folds put unrelated letters closer **through the air**, so MDS pulls them together in 2-D and squashes within-letter variation.

Use the **fold tightness slider** above: stress typically rises as the 2-D map fights tighter 3-D straight-line distances.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(curve_agreement, distance_gap, flat_agreement, mo):
    mo.md(f"""
    ## Conclusions

    1. **MDS fits coordinates to a distance matrix** by lowering stress — the mismatch between input and embedded distances (Goal 1).
    2. **Rotations change coordinates, not distances** (max change **{distance_gap:.2e}** here), so MDS layouts have no meaningful axes — only shape (Goal 2).
    3. **Heatmaps summarise pairs:** symmetric, zero diagonal, dark = far with the Blues scale (Goal 3).
    4. **Metric MDS matches values; non-metric fits order** — use non-metric for ratings or monotonically distorted inputs (Goal 4; ordinal demo above).
    5. **On the S-bend, MDS preserves straight-line distances but fails to unroll:** high along-word correlation (**{curve_agreement['metric']:.3f}**) with collapsed height spread vs the flat recovery (**{flat_agreement['seed 9']:.3f}** along-word, legible letters) shows the limit (Goal 5).
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
