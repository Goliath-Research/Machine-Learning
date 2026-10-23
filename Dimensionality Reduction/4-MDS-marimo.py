# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "scikit-learn",
#     "seaborn",
# ]
# ///

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns

    plt.rcParams["figure.max_open_warning"] = 0
    sns.set_theme(style="whitegrid")
    return mo, np, pd, plt, sns


@app.cell
def _():
    from sklearn.manifold import MDS
    from sklearn.metrics import pairwise_distances

    return MDS, pairwise_distances


@app.cell
def _(np, pd, plt, sns):
    def sample_goliath(per_letter=143, seed=42):
        rng = np.random.RandomState(seed)
        frames = []
        cursor = 0.0
        for letter in "GOLIATH":
            glyph = plt.figure(figsize=(1.4, 1.6), dpi=90)
            axes = glyph.add_axes((0, 0, 1, 1))
            axes.set_axis_off()
            axes.text(0.08, 0.08, letter, fontsize=90, fontweight="bold", color="black")
            glyph.canvas.draw()
            image = np.asarray(glyph.canvas.buffer_rgba())
            plt.close(glyph)
            data = (image[:, :, 0] < 200)[::-1].T.astype(float)
            raw = rng.rand(40 * per_letter, 2)
            columns, rows = (raw * np.array(data.shape)).astype(int).T
            columns = np.clip(columns, 0, data.shape[0] - 1)
            rows = np.clip(rows, 0, data.shape[1] - 1)
            chosen = raw[data[columns, rows] > 0][:per_letter]
            chosen[:, 0] *= data.shape[0] / data.shape[1]
            chosen[:, 0] = chosen[:, 0] - chosen[:, 0].min() + cursor
            cursor = chosen[:, 0].max() + 0.18
            frames.append(pd.DataFrame({"x0": chosen[:, 0], "x1": chosen[:, 1], "labels": letter}))
        return pd.concat(frames, ignore_index=True)

    def draw_letters(axes, frame, title):
        sns.scatterplot(
            data=frame, x="x0", y="x1", hue="labels", hue_order=list("GOLIATH"), s=12, ax=axes, linewidth=0
        )
        axes.set_aspect("equal", adjustable="datalim")
        axes.set_title(title)
        axes.legend(fontsize=7, markerscale=0.7, ncol=2)

    def rotate_letters(frame, angle):
        theta = np.deg2rad(angle)
        rotation = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        rotated = frame.copy()
        rotated[["x0", "x1"]] = frame[["x0", "x1"]].to_numpy() @ rotation
        return rotated

    def along_word(position, embedded):
        embedded = np.asarray(embedded)
        axis = embedded[:, int(np.argmax(np.var(embedded, axis=0)))]
        return abs(float(np.corrcoef(position, axis)[0, 1]))

    return along_word, draw_letters, rotate_letters, sample_goliath


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Multidimensional Scaling

    ## Objectives

    - Sample the letters in GOLIATH and show that rotation leaves the pairwise distances unchanged.
    - Recover a two-dimensional layout from those distances, and from a random three-dimensional projection.
    - Bend the same letters into an S and compare metric MDS with non-metric MDS.

    ## Background

    Multidimensional scaling is given distances, or it computes them. It looks for coordinates whose distances match. A rotation does not change distances, so the recovered picture can be turned or flipped relative to the original.

    Since scikit-learn 1.8, a precomputed distance matrix is passed with `metric="precomputed"`. The old name was `dissimilarity`. Metric MDS is `metric_mds=True`. Non-metric MDS is `metric_mds=False`. The old boolean `metric=False` meant the non-metric fit. These fits use `init="random"` and `n_init=1`.

    ## Datasets Used

    - **Rendered letters.** `Goliath.png` is not in this folder. Each letter is drawn, 143 ink pixels are sampled with `random_state=42`, and the letters are placed in order with a gap. That is 1,001 points.
    - **An S built from those letters.** The notebook's saved cloud is `GOLIATH_S_curve.csv`, sampled from the missing image. The S below is built from the rendered letters with the same formula, so this lesson does not rewrite that file.
    """)
    return


@app.cell
def _(draw_letters, plt, rotate_letters, sample_goliath):
    letters = sample_goliath()
    turned_down = rotate_letters(letters, -20)
    turned_up = rotate_letters(letters, 50)
    print(f"{len(letters)} points")
    print(letters.labels.value_counts().reindex(list("GOLIATH")).to_string())
    _figure, _axes = plt.subplots(1, 3, figsize=(11, 3.4))
    draw_letters(_axes[0], letters, "Sampled letters")
    draw_letters(_axes[1], turned_down, "Rotated by −20°")
    draw_letters(_axes[2], turned_up, "Rotated by 50°")
    _figure.tight_layout()
    return _figure


@app.cell
def _(letters, np, pairwise_distances, plt, turned_down, turned_up):
    distances = pairwise_distances(letters[["x0", "x1"]])
    distances_down = pairwise_distances(turned_down[["x0", "x1"]])
    distances_up = pairwise_distances(turned_up[["x0", "x1"]])
    distance_gap = float(
        max(np.max(np.abs(distances - distances_down)), np.max(np.abs(distances - distances_up)))
    )
    print(f"Largest absolute change in a pairwise distance: {distance_gap:.3e}")
    _figure, _axes = plt.subplots(1, 3, figsize=(9, 3))
    for _ax, _matrix, _title in zip(
        _axes, (distances, distances_down, distances_up), ("Original", "Rotated −20°", "Rotated 50°")
    ):
        _ax.imshow(_matrix, cmap="Blues", aspect="auto")
        _ax.set_title(_title)
        _ax.set_xticks([])
        _ax.set_yticks([])
    _figure.tight_layout()
    return _figure


@app.cell
def _(MDS, along_word, distances, draw_letters, letters, pd, plt):
    flat_agreement = {}
    flat_layouts = {}
    for _name, _seed in (("random_state 9", 9), ("random_state 5", 5)):
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
            {"x0": _embedded[:, 0], "x1": _embedded[:, 1], "labels": letters.labels.to_numpy()}
        )
        flat_agreement[_name] = along_word(letters.x0.to_numpy(), _embedded)
        print(f"{_name}: absolute correlation with the original horizontal position {flat_agreement[_name]:.3f}")
    _figure, _axes = plt.subplots(1, 2, figsize=(8, 3.6))
    draw_letters(_axes[0], flat_layouts["random_state 9"], "MDS from distances, seed 9")
    draw_letters(_axes[1], flat_layouts["random_state 5"], "MDS from distances, seed 5")
    _figure.tight_layout()
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A straight trip through three dimensions

    The letters are projected into three dimensions with a random orthogonal map (`random_state=42`). MDS then sees those three coordinates, computes Euclidean distances itself, and returns two coordinates. The notebook text mentions `random_state=1` for an earlier fit. The calls in that notebook use 9 and 5. The two fits here use 28 and 5.
    """)
    return


@app.cell
def _(MDS, along_word, draw_letters, letters, np, pd, plt):
    _labels = letters.labels
    _xy = letters[["x0", "x1"]].to_numpy()
    _rng = np.random.RandomState(42)
    _draw = _rng.randn(3, 3)
    _eigenvalues, _vectors = np.linalg.eigh(_draw @ _draw.T)
    projected = _xy @ _vectors[: _xy.shape[1]]
    straight_agreement = {}
    straight_layouts = {}
    for _name, _seed in (("random_state 28", 28), ("random_state 5", 5)):
        _embedded = MDS(
            n_components=2,
            metric_mds=True,
            init="random",
            n_init=1,
            random_state=_seed,
            normalized_stress="auto",
        ).fit_transform(projected)
        straight_layouts[_name] = pd.DataFrame(
            {"x0": _embedded[:, 0], "x1": _embedded[:, 1], "labels": _labels.to_numpy()}
        )
        straight_agreement[_name] = along_word(letters.x0.to_numpy(), _embedded)
        print(f"{_name}: absolute correlation {straight_agreement[_name]:.3f}")
    _figure = plt.figure(figsize=(10, 3.6))
    _cloud = _figure.add_subplot(1, 3, 1, projection="3d")
    for _letter in "GOLIATH":
        _rows = _labels.to_numpy() == _letter
        _cloud.scatter(projected[_rows, 0], projected[_rows, 1], projected[_rows, 2], s=6, label=_letter)
    _cloud.set_title("Random 3D projection")
    _flat = _figure.add_subplot(1, 3, 2)
    _other = _figure.add_subplot(1, 3, 3)
    draw_letters(_flat, straight_layouts["random_state 28"], "MDS, seed 28")
    draw_letters(_other, straight_layouts["random_state 5"], "MDS, seed 5")
    _figure.tight_layout()
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## An S-shaped bend

    The horizontal position is wrapped with the notebook's formula. Metric MDS keeps the distance values. Non-metric MDS keeps their rank order. Neither one is built to unroll a bend. The absolute correlation compares the longer recovered axis with the original horizontal position.
    """)
    return


@app.cell
def _(MDS, along_word, draw_letters, letters, np, pd, plt):
    _t = (letters.x0 - 2) * 0.78 * np.pi
    curve = pd.DataFrame(
        {
            "x0": np.sin(_t),
            "x1": letters.x1 * 1.5,
            "x2": np.sign(_t) * (np.cos(_t) - 1) * 1.5,
            "labels": letters.labels.to_numpy(),
        }
    )
    curve_agreement = {}
    curve_layouts = {}
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
            {"x0": _embedded[:, 0], "x1": _embedded[:, 1], "labels": curve.labels.to_numpy()}
        )
        curve_agreement[_name] = along_word(letters.x0.to_numpy(), _embedded)
        print(f"{_name}: absolute correlation {curve_agreement[_name]:.3f}")
    _figure = plt.figure(figsize=(10, 3.6))
    _cloud = _figure.add_subplot(1, 3, 1, projection="3d")
    for _letter in "GOLIATH":
        _rows = curve.labels.to_numpy() == _letter
        _cloud.scatter(curve.loc[_rows, "x0"], curve.loc[_rows, "x1"], curve.loc[_rows, "x2"], s=6)
    _cloud.set_title("Letters bent into an S")
    _metric = _figure.add_subplot(1, 3, 2)
    _nonmetric = _figure.add_subplot(1, 3, 3)
    draw_letters(_metric, curve_layouts["metric"], "Metric MDS")
    draw_letters(_nonmetric, curve_layouts["non-metric"], "Non-metric MDS")
    _figure.tight_layout()
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Which value of `metric` tells MDS that the input is already a distance matrix? Replace `None` with that string.
    """)
    return


@app.cell
def _():
    student_metric = None
    print(student_metric)
    return (student_metric,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r'''
```python
student_metric = "precomputed"
```
**precomputed.** The notebook called this `dissimilarity="precomputed"`. That argument name is the old one. `metric_mds` is a separate switch: true for metric MDS, false for non-metric MDS.
''')
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(curve_agreement, distance_gap, flat_agreement, mo, straight_agreement):
    mo.md(f"""
    ## Conclusions

    **Key takeaways:**

    - After rotations of −20° and 50°, the largest change in a pairwise distance is {distance_gap:.3e}.
    - MDS from that distance matrix, with seeds 9 and 5, has absolute correlations {flat_agreement["random_state 9"]:.3f} and {flat_agreement["random_state 5"]:.3f} with the original horizontal position. The picture can be rotated or reflected.
    - After a random projection into three dimensions, seeds 28 and 5 give correlations {straight_agreement["random_state 28"]:.3f} and {straight_agreement["random_state 5"]:.3f}.
    - On the S-shaped bend, metric MDS has correlation {curve_agreement["metric"]:.3f} and non-metric MDS has correlation {curve_agreement["non-metric"]:.3f}. Both stay near the correlations from the straight layouts, so the longer axis still follows most of the left-to-right order of these rendered letters.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check your understanding

    1. What does a rotation change, the coordinates or the pairwise distances?
    2. Why can two MDS pictures of the same distances look turned relative to each other?
    3. What did `metric=False` mean in the notebook, and what is the argument now?
    4. Why does the S-shaped bend give MDS less of the original left-to-right order?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
1. **The coordinates:** Each point moves. Distances between points stay the same, so the three distance matrices match.
2. **Only distances were kept:** MDS can turn or flip the layout. The seed changes the start of the search. The letters can still sit in the same order along the word.
3. **Non-metric MDS:** `metric_mds=False` keeps the rank order of the distances. `metric="precomputed"` is the current way to pass a distance matrix.
4. **The path is bent:** Points that were far apart along the word can sit closer in the S. On these rendered letters the correlation drops only a little, and the longer axis still tracks most of the letter order. A tighter fold would drop it further.
""")
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - [MDS](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html)
    - [Manifold learning](https://scikit-learn.org/stable/modules/manifold.html)
    - VanderPlas, J. (2017). *Python Data Science Handbook*. O'Reilly Media. Chapter 5.
    """)
    return


if __name__ == "__main__":
    app.run()
