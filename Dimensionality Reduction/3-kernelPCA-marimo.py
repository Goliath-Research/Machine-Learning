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
    from plotly.subplots import make_subplots
    from sklearn.datasets import make_blobs, make_circles, make_moons
    from sklearn.decomposition import KernelPCA, PCA

    pio.templates.default = "plotly_white"
    px.defaults.color_discrete_sequence = px.colors.qualitative.G10
    builder = pb.PlotBuilder(width=1150, height=330)

    return (
        KernelPCA,
        PCA,
        builder,
        go,
        make_blobs,
        make_circles,
        make_moons,
        make_subplots,
        mo,
        np,
        pd,
        px,
    )


@app.cell
def _(go, make_subplots, np, pd):
    def ranges_overlap(values, labels):
        """True when the two groups' ranges on one axis overlap.

        Used as a blunt test of separation: if the ranges overlap, at least one
        value from each group interleaves, so that single axis cannot separate
        them.

        Parameters
        ----------
        values : array-like
            One-dimensional scores, such as the first component.
        labels : array-like
            Group label per value, with two distinct values.

        Returns
        -------
        bool
            Whether the two ranges share any interval.
        """
        values = np.asarray(values)
        labels = np.asarray(labels)
        left = values[labels == 0]
        right = values[labels == 1]
        return bool(left.max() >= right.min() and right.max() >= left.min())

    def panel_of_scatters(specs, title="", height=330):
        """Draw several two-dimensional scatters side by side in one figure.

        Each spec is one panel: the two coordinate arrays, the group labels, and
        a caption. Panels share a legend, since the group colours are the same
        throughout.

        Parameters
        ----------
        specs : list of dict
            Each entry has `x`, `y`, `labels`, and `caption`. Add `flat=True` to
            draw the panel as a one-dimensional strip, which is how a single
            component is shown.
        title : str, default ""
            Figure title.
        height : int, default 330
            Figure height in pixels.

        Returns
        -------
        plotly.graph_objects.Figure
            The assembled figure.
        """
        count = len(specs)
        figure = make_subplots(
            rows=1,
            cols=count,
            subplot_titles=[spec["caption"] for spec in specs],
            horizontal_spacing=0.05,
        )
        for index, spec in enumerate(specs, start=1):
            x = np.asarray(spec["x"])
            labels = np.asarray(spec["labels"]).astype(str)
            y = np.zeros(len(x)) if spec.get("flat") else np.asarray(spec["y"])
            for group in ("0", "1"):
                mask = labels == group
                figure.add_trace(
                    go.Scatter(
                        x=x[mask],
                        y=y[mask],
                        mode="markers",
                        name=f"group {group}",
                        legendgroup=group,
                        showlegend=index == 1,
                        marker=dict(size=6, opacity=0.75),
                    ),
                    row=1,
                    col=index,
                )
            if not spec.get("flat"):
                figure.update_xaxes(scaleanchor=f"y{index if index > 1 else ''}", row=1, col=index)
        figure.update_layout(
            width=1150,
            height=height,
            title=title,
            margin=dict(t=70, b=40, l=50, r=20),
        )
        return figure

    return panel_of_scatters, ranges_overlap


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
def _(PCA, make_blobs, panel_of_scatters, ranges_overlap):
    blobs, blob_labels = make_blobs(
        n_samples=200, centers=2, cluster_std=0.5, random_state=0
    )
    blob_pca = PCA(n_components=2).fit(blobs)
    blob_plane = blob_pca.transform(blobs)
    blob_overlap = ranges_overlap(blob_plane[:, 0], blob_labels)
    blob_ratio = blob_pca.explained_variance_ratio_
    panel_of_scatters(
        [
            {"x": blobs[:, 0], "y": blobs[:, 1], "labels": blob_labels, "caption": "Blobs"},
            {
                "x": blob_plane[:, 0],
                "y": blob_plane[:, 1],
                "labels": blob_labels,
                "caption": "PCA, two components",
            },
            {
                "x": blob_plane[:, 0],
                "labels": blob_labels,
                "caption": "PCA, first component only",
                "flat": True,
            },
        ],
        title="Blobs: a linear component is enough",
    )
    return blob_overlap, blob_ratio


@app.cell
def _(KernelPCA, PCA, make_moons, panel_of_scatters, ranges_overlap):
    moons, moon_labels = make_moons(100, noise=0.015, random_state=0)
    moon_plane = PCA(n_components=2).fit_transform(moons)
    moon_kernel = KernelPCA(n_components=2, kernel="rbf", gamma=15).fit_transform(moons)
    moon_pca_overlap = ranges_overlap(moon_plane[:, 0], moon_labels)
    moon_kernel_overlap = ranges_overlap(moon_kernel[:, 0], moon_labels)
    panel_of_scatters(
        [
            {"x": moons[:, 0], "y": moons[:, 1], "labels": moon_labels, "caption": "Moons"},
            {
                "x": moon_plane[:, 0],
                "y": moon_plane[:, 1],
                "labels": moon_labels,
                "caption": "PCA, two components",
            },
            {
                "x": moon_plane[:, 0],
                "labels": moon_labels,
                "caption": "PCA, first component",
                "flat": True,
            },
            {
                "x": moon_kernel[:, 0],
                "y": moon_kernel[:, 1],
                "labels": moon_labels,
                "caption": "RBF, two components",
            },
            {
                "x": moon_kernel[:, 0],
                "labels": moon_labels,
                "caption": "RBF, first component",
                "flat": True,
            },
        ],
        title="Moons: the linear component mixes the arms",
    )
    return moon_kernel_overlap, moon_pca_overlap


@app.cell
def _(KernelPCA, PCA, make_circles, panel_of_scatters, ranges_overlap):
    rings, ring_labels = make_circles(
        n_samples=400, random_state=123, noise=0.1, factor=0.2
    )
    ring_plane = PCA(n_components=2).fit_transform(rings)
    ring_kernel = KernelPCA(n_components=2, kernel="rbf", gamma=15).fit_transform(rings)
    ring_pca_overlap = ranges_overlap(ring_plane[:, 0], ring_labels)
    ring_kernel_overlap = ranges_overlap(ring_kernel[:, 0], ring_labels)
    panel_of_scatters(
        [
            {"x": rings[:, 0], "y": rings[:, 1], "labels": ring_labels, "caption": "Rings"},
            {
                "x": ring_plane[:, 0],
                "y": ring_plane[:, 1],
                "labels": ring_labels,
                "caption": "PCA, two components",
            },
            {
                "x": ring_plane[:, 0],
                "labels": ring_labels,
                "caption": "PCA, first component",
                "flat": True,
            },
            {
                "x": ring_kernel[:, 0],
                "y": ring_kernel[:, 1],
                "labels": ring_labels,
                "caption": "RBF, two components",
            },
            {
                "x": ring_kernel[:, 0],
                "labels": ring_labels,
                "caption": "RBF, first component",
                "flat": True,
            },
        ],
        title="Rings: a ring is not a linear direction",
    )
    return ring_kernel_overlap, ring_pca_overlap


@app.cell
def _(KernelPCA, PCA, np, panel_of_scatters, ranges_overlap):
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
    heart_kernel = KernelPCA(n_components=2, kernel="rbf", gamma=15).fit_transform(
        hearts
    )
    heart_pca_overlap = ranges_overlap(heart_plane[:, 0], heart_labels)
    heart_kernel_overlap = ranges_overlap(heart_kernel[:, 0], heart_labels)
    panel_of_scatters(
        [
            {"x": hearts[:, 0], "y": hearts[:, 1], "labels": heart_labels, "caption": "Two hearts"},
            {
                "x": heart_plane[:, 0],
                "y": heart_plane[:, 1],
                "labels": heart_labels,
                "caption": "PCA, two components",
            },
            {
                "x": heart_plane[:, 0],
                "labels": heart_labels,
                "caption": "PCA, first component",
                "flat": True,
            },
            {
                "x": heart_kernel[:, 0],
                "y": heart_kernel[:, 1],
                "labels": heart_labels,
                "caption": "RBF, two components",
            },
            {
                "x": heart_kernel[:, 0],
                "labels": heart_labels,
                "caption": "RBF, first component",
                "flat": True,
            },
        ],
        title="Two hearts: separated in space, so a linear component works",
    )
    return heart_kernel_overlap, heart_pca_overlap


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
