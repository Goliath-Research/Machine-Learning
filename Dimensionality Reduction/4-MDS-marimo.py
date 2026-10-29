# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "scikit-learn>=1.8",
#     "seaborn",
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
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
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
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from matplotlib.image import imread
    from sklearn.manifold import MDS
    from sklearn.metrics import pairwise_distances

    plt.rcParams["figure.max_open_warning"] = 0
    sns.set_theme(style="whitegrid")

    return MDS, Path, imread, mo, np, pairwise_distances, pd, plt, sns


@app.cell
def _(imread, np, pd):
    def sample_goliath(image, per_letter_point_count=1000, seed=42):
        """Sample points from the ink of a rendered word.

        Parameters
        ----------
        image : numpy.ndarray
            The loaded image, as returned by `matplotlib.image.imread`. Only the
            first colour channel is used, so the ink is separated from the
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

    def draw_letters(axes, frame, title):
        """Scatter the letters coloured by glyph, with equal axis scaling."""
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
        axes.set_aspect("equal", adjustable="datalim")
        axes.set_title(title)
        axes.legend(fontsize=7, markerscale=0.7, ncol=2)

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

    return along_word, draw_letters, rotate_letters, sample_goliath


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
def _(Path, imread, plt):
    goliath_image = imread(Path(__file__).resolve().parent / "Goliath.png")
    _figure, _axes = plt.subplots(figsize=(7, 2.2))
    _axes.imshow(goliath_image)
    _axes.set_xticks([])
    _axes.set_yticks([])
    _axes.set_title("Goliath.png, the source image")
    _figure.tight_layout()
    _figure
    return (goliath_image,)


@app.cell(hide_code=True)
def _(goliath_image, mo):
    mo.md(f"""
    The image is a real file in this folder, and it is what produces every coordinate below. `imread` returns an array of shape **{goliath_image.shape}**: rows, columns, and colour channels. The sampler reads a single colour channel from it, and a pixel counts as ink when it is dark, which is why the word appears as a shape rather than as colours.

    The sampling step converts an **image** into a **point cloud**, which is what clustering and manifold methods can work with. Each ink pixel that is sampled becomes one row with two coordinates.

    ### The sampled cloud
    """)
    return


@app.cell
def _(Path, imread, pd, sample_goliath):
    goliath_path = Path(__file__).resolve().parent / "Goliath.png"
    letters = sample_goliath(imread(goliath_path))
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
def _(draw_letters, letters, plt):
    _figure, _axes = plt.subplots(figsize=(9, 2.8))
    draw_letters(_axes, letters, "Points sampled from the word GOLIATH")
    _figure.tight_layout()
    _figure
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
def _(draw_letters, letters, plt, rotate_letters):
    turned_down = rotate_letters(letters, -20)
    turned_up = rotate_letters(letters, 50)
    _figure, _axes = plt.subplots(1, 3, figsize=(12, 2.9))
    draw_letters(_axes[0], letters, "Original")
    draw_letters(_axes[1], turned_down, "Rotated by −20°")
    draw_letters(_axes[2], turned_up, "Rotated by 50°")
    _figure.tight_layout()
    _figure
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
def _(distances, distances_down, distances_up, np, plt):
    _figure, _axes = plt.subplots(1, 3, figsize=(10, 3.2))
    for _ax, _matrix, _title in zip(
        _axes,
        (distances, distances_down, distances_up),
        ("Original", "Rotated −20°", "Rotated 50°"),
    ):
        _ax.imshow(_matrix, cmap="Blues", aspect="auto")
        _ax.set_title(_title)
        _ax.set_xticks([])
        _ax.set_yticks([])
    _figure.tight_layout()
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
def _(MDS, along_word, distances, draw_letters, letters, pd, plt):
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
    _figure, _axes = plt.subplots(1, 2, figsize=(10, 3.2))
    draw_letters(_axes[0], flat_layouts["seed 9"], "MDS from distances, seed 9")
    draw_letters(_axes[1], flat_layouts["seed 5"], "MDS from distances, seed 5")
    _figure.tight_layout()
    _figure
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
def _(curve, plt):
    _figure = plt.figure(figsize=(7, 4.5))
    _axes = _figure.add_subplot(111, projection="3d")
    for _letter in "GOLIATH":
        _rows = curve.labels.to_numpy() == _letter
        _axes.scatter(
            curve.loc[_rows, "x0"],
            curve.loc[_rows, "x1"],
            curve.loc[_rows, "x2"],
            s=6,
            label=_letter,
        )
    _axes.set_title("The word bent into an S")
    _axes.legend(fontsize=7, markerscale=0.6)
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
def _(MDS, along_word, curve, draw_letters, letters, pd, plt):
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
    _figure, _axes = plt.subplots(1, 2, figsize=(10, 3.2))
    draw_letters(_axes[0], curve_layouts["metric"], "Metric MDS on the S")
    draw_letters(_axes[1], curve_layouts["non-metric"], "Non-metric MDS on the S")
    _figure.tight_layout()
    _figure
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
