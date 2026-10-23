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
    from sklearn.datasets import make_blobs, make_circles, make_moons
    from sklearn.decomposition import PCA, KernelPCA

    return KernelPCA, PCA, make_blobs, make_circles, make_moons


@app.cell
def _(np, pd, sns):
    def ranges_overlap(values, labels):
        values = np.asarray(values)
        labels = np.asarray(labels)
        left = values[labels == 0]
        right = values[labels == 1]
        return bool(left.max() >= right.min() and right.max() >= left.min())

    def draw_groups(axes, x, y, labels, title, equal=True):
        frame = pd.DataFrame(
            {"x": np.asarray(x), "y": np.asarray(y), "group": np.asarray(labels).astype(str)}
        )
        sns.scatterplot(
            data=frame, x="x", y="y", hue="group", hue_order=["0", "1"], s=22, ax=axes, linewidth=0, legend=False
        )
        axes.set_title(title)
        if equal:
            axes.set_aspect("equal", adjustable="datalim")

    return draw_groups, ranges_overlap


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Kernel Principal Component Analysis

    ## Objectives

    - Compare a linear component with an RBF kernel component on blobs, moons, rings, and two hearts.
    - Judge separation on the first component by whether the two groups' ranges overlap.
    - Keep the kernel fixed at `rbf` and `gamma=15`, and read the pictures against that choice.

    ## Background

    Linear PCA rotates the columns. If two groups are tangled, a straight direction may not pull them apart. Kernel PCA first compares points with a kernel. The RBF kernel gives a large value to nearby points and a small value to distant ones. Linear components are then taken in that comparison. `gamma` sets how fast the comparison falls off with distance.

    The colors are drawn after the fit. Neither PCA nor kernel PCA sees the group labels.

    ## Datasets Used

    - **Blobs.** 200 points, two centers, `cluster_std=0.5`, `random_state=0`.
    - **Moons.** 100 points, `noise=0.015`, `random_state=0`.
    - **Rings.** 400 points, `noise=0.1`, `factor=0.2`, `random_state=123`.
    - **Hearts.** Two curves of 100 points each. The second heart is shifted by 20 and −10. There is no random seed.
    """)
    return


@app.cell
def _(PCA, draw_groups, make_blobs, np, plt, ranges_overlap):
    blobs, blob_labels = make_blobs(n_samples=200, centers=2, cluster_std=0.5, random_state=0)
    blob_pca = PCA(n_components=2).fit(blobs)
    blob_plane = blob_pca.transform(blobs)
    blob_overlap = ranges_overlap(blob_plane[:, 0], blob_labels)
    print(f"Explained variance: {blob_pca.explained_variance_.round(3)}")
    print(f"Explained variance ratio: {blob_pca.explained_variance_ratio_.round(3)}")
    print(f"First linear component overlaps: {blob_overlap}")
    _figure, _axes = plt.subplots(1, 3, figsize=(10, 3.2))
    draw_groups(_axes[0], blobs[:, 0], blobs[:, 1], blob_labels, "Blobs")
    draw_groups(_axes[1], blob_plane[:, 0], blob_plane[:, 1], blob_labels, "PCA, two components")
    draw_groups(
        _axes[2], blob_plane[:, 0], np.zeros(len(blob_plane)), blob_labels, "PCA, first component", equal=False
    )
    _figure.tight_layout()
    return _figure


@app.cell
def _(
    KernelPCA,
    PCA,
    draw_groups,
    make_moons,
    np,
    plt,
    ranges_overlap,
):
    moons, moon_labels = make_moons(100, noise=0.015, random_state=0)
    moon_plane = PCA(n_components=2).fit_transform(moons)
    moon_kernel = KernelPCA(n_components=2, kernel="rbf", gamma=15).fit_transform(moons)
    moon_pca_overlap = ranges_overlap(moon_plane[:, 0], moon_labels)
    moon_kernel_overlap = ranges_overlap(moon_kernel[:, 0], moon_labels)
    print(f"PCA first component overlaps: {moon_pca_overlap}")
    print(f"RBF first component overlaps: {moon_kernel_overlap}")
    _figure, _axes = plt.subplots(1, 5, figsize=(14, 3.1))
    draw_groups(_axes[0], moons[:, 0], moons[:, 1], moon_labels, "Moons")
    draw_groups(_axes[1], moon_plane[:, 0], moon_plane[:, 1], moon_labels, "PCA, two")
    draw_groups(_axes[2], moon_plane[:, 0], np.zeros(len(moons)), moon_labels, "PCA, first", equal=False)
    draw_groups(_axes[3], moon_kernel[:, 0], moon_kernel[:, 1], moon_labels, "RBF, two")
    draw_groups(_axes[4], moon_kernel[:, 0], np.zeros(len(moons)), moon_labels, "RBF, first", equal=False)
    _figure.tight_layout()
    return _figure


@app.cell
def _(
    KernelPCA,
    PCA,
    draw_groups,
    make_circles,
    np,
    plt,
    ranges_overlap,
):
    rings, ring_labels = make_circles(n_samples=400, random_state=123, noise=0.1, factor=0.2)
    ring_plane = PCA(n_components=2).fit_transform(rings)
    ring_kernel = KernelPCA(n_components=2, kernel="rbf", gamma=15).fit_transform(rings)
    ring_pca_overlap = ranges_overlap(ring_plane[:, 0], ring_labels)
    ring_kernel_overlap = ranges_overlap(ring_kernel[:, 0], ring_labels)
    print(f"PCA first component overlaps: {ring_pca_overlap}")
    print(f"RBF first component overlaps: {ring_kernel_overlap}")
    _figure, _axes = plt.subplots(1, 5, figsize=(14, 3.1))
    draw_groups(_axes[0], rings[:, 0], rings[:, 1], ring_labels, "Rings")
    draw_groups(_axes[1], ring_plane[:, 0], ring_plane[:, 1], ring_labels, "PCA, two")
    draw_groups(_axes[2], ring_plane[:, 0], np.zeros(len(rings)), ring_labels, "PCA, first", equal=False)
    draw_groups(_axes[3], ring_kernel[:, 0], ring_kernel[:, 1], ring_labels, "RBF, two")
    draw_groups(_axes[4], ring_kernel[:, 0], np.zeros(len(rings)), ring_labels, "RBF, first", equal=False)
    _figure.tight_layout()
    return _figure


@app.cell
def _(KernelPCA, PCA, draw_groups, np, plt, ranges_overlap):
    def _heart(a, b):
        _t = np.linspace(0, 2 * np.pi, 100)
        _x = 16 * np.sin(_t) ** 3
        _y = 13 * np.cos(_t) - 5 * np.cos(2 * _t) - 2 * np.cos(3 * _t) - np.cos(4 * _t)
        return a + _x, b + _y

    _x1, _y1 = _heart(0, 0)
    _x2, _y2 = _heart(20, -10)
    hearts = np.column_stack((np.hstack((_x1, _x2)), np.hstack((_y1, _y2))))
    heart_labels = np.array([0] * len(_x1) + [1] * len(_x2))
    heart_plane = PCA(n_components=2).fit_transform(hearts)
    heart_kernel = KernelPCA(n_components=2, kernel="rbf", gamma=15).fit_transform(hearts)
    heart_pca_overlap = ranges_overlap(heart_plane[:, 0], heart_labels)
    heart_kernel_overlap = ranges_overlap(heart_kernel[:, 0], heart_labels)
    print(f"PCA first component overlaps: {heart_pca_overlap}")
    print(f"RBF first component overlaps: {heart_kernel_overlap}")
    _figure, _axes = plt.subplots(1, 5, figsize=(14, 3.1))
    draw_groups(_axes[0], hearts[:, 0], hearts[:, 1], heart_labels, "Hearts")
    draw_groups(_axes[1], heart_plane[:, 0], heart_plane[:, 1], heart_labels, "PCA, two")
    draw_groups(_axes[2], heart_plane[:, 0], np.zeros(len(hearts)), heart_labels, "PCA, first", equal=False)
    draw_groups(_axes[3], heart_kernel[:, 0], heart_kernel[:, 1], heart_labels, "RBF, two")
    draw_groups(_axes[4], heart_kernel[:, 0], np.zeros(len(hearts)), heart_labels, "RBF, first", equal=False)
    _figure.tight_layout()
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Which string selects the Gaussian RBF kernel? Replace `None` with that string.
    """)
    return


@app.cell
def _():
    student_kernel = None
    print(student_kernel)
    return (student_kernel,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r'''
```python
student_kernel = "rbf"
```
**rbf.** The radial basis function compares nearby points strongly. `gamma=15` makes that comparison fall off quickly, which suits these small shapes. Another cloud can need another gamma.
''')
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(
    blob_overlap,
    heart_kernel_overlap,
    heart_pca_overlap,
    mo,
    moon_kernel_overlap,
    moon_pca_overlap,
    ring_kernel_overlap,
    ring_pca_overlap,
):
    def _line(overlap):
        return "the two ranges overlap" if overlap else "the two ranges do not overlap"

    mo.md(f"""
    ## Conclusions

    **Key takeaways:**

    - On the blobs, the first linear component has this result: {_line(blob_overlap)}.
    - On the moons, the first linear component has this result: {_line(moon_pca_overlap)}. The first RBF component, with `gamma=15`, has this result: {_line(moon_kernel_overlap)}.
    - On the rings, the linear result is: {_line(ring_pca_overlap)}. The RBF result is: {_line(ring_kernel_overlap)}.
    - On the hearts, the linear result is: {_line(heart_pca_overlap)}. The RBF result is: {_line(heart_kernel_overlap)}.
    - Overlap is measured on the first component only. A picture with two kernel components can still show structure when that one axis does not split the labels. The labels were not used in any fit.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check your understanding

    1. Why can a straight component fail on the moons?
    2. What does the RBF kernel compare?
    3. Why is a large gamma a different fit from a small gamma?
    4. Why are the colors absent from the fit?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
1. **The groups bend around each other:** The largest straight direction still mixes the two arms. Projecting onto it leaves their ranges on top of each other.
2. **Nearby points:** The Gaussian kernel is large when two points are close and small when they are far. Kernel PCA then takes linear components of those comparisons.
3. **The neighborhood shrinks:** A larger gamma makes the kernel ignore all but the closest points. The same number can separate one shape and tear another apart.
4. **Both methods are unsupervised:** The group number is used only to color the points and to test whether the ranges on the first component overlap.
""")
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - [KernelPCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html)
    - [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
    - VanderPlas, J. (2017). *Python Data Science Handbook*. O'Reilly Media. Chapter 5.
    """)
    return


if __name__ == "__main__":
    app.run()
