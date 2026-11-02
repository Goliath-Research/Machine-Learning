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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Kernel Principal Component Analysis

    ## Objectives

    By the end of this lesson, you should be able to:

    1. Explain why a linear component cannot separate groups that curve around each other.
    2. Explain what a kernel does: it replaces the inner product with a similarity, so components are computed in a richer space without ever building that space.
    3. Explain what `gamma` controls, and predict which failure a value that is too small or too large produces.
    4. Explain why the same kernel can rescue data that winds around a centre yet *lose* ground on data a straight boundary already separates, and predict which shape falls into which case.
    5. Explain why kernel PCA is **not** automatically better, using a case where it does worse than linear PCA.

    ## Background

    The previous lessons found directions in the data. **Principal component analysis** looks for the direction of greatest variance, and its components are weighted mixtures of the original columns. That works well when groups are separated by a straight boundary.

    This lesson is about the case where they are not.

    ### The problem with a straight direction

    Two crescents that interlock, or a ring around a circle, share the same region of space. A straight line through that space cuts through both groups. Since PCA can only find straight directions, projecting onto its components leaves the groups interleaved. No choice of component count fixes that, because the limitation is the shape of the boundary, not the number of dimensions.

    Notice what is *not* wrong: PCA has done its job correctly. It found the directions of greatest variance. Those directions simply do not happen to separate the groups, and PCA was never asked to make them, because **it does not see the labels**. PCA is unsupervised; the colours in these plots are added afterwards.

    ### The kernel idea

    **Kernel PCA** keeps PCA's machinery but changes one ingredient. Instead of measuring how the columns co-vary, it measures how **similar each pair of points is**, using a kernel function.

    The **radial basis function** (RBF) kernel is the default choice:

    \[
    k(\mathbf{x}, \mathbf{y}) = \exp(-\gamma \lVert \mathbf{x} - \mathbf{y} \rVert^2)
    \]

    Two consequences follow from that formula, and they are worth reading off the shape of it:

    - A pair of **nearby points** has a small distance, so the exponent is near zero and the kernel value is **near 1**.
    - A pair of **distant points** has a large distance, so the exponent is large and negative, and the kernel value is **near 0**.

    So the kernel converts distances into a similarity that saturates: beyond a certain distance, points count as "not related" and getting farther changes nothing.

    PCA is then run on that similarity matrix rather than on the raw columns. The result is a set of components in a space the kernel implies, which is typically much higher-dimensional than the original data — sometimes infinitely so — without ever being constructed explicitly. That is the trick: **you get the benefit of a richer space without paying to represent it.**

    This is the same idea as the kernel trick in support vector machines, which is why the vocabulary is shared.

    ### Gamma, and its two failure modes

    `gamma` sets **how quickly similarity falls off with distance**. It is the single parameter that matters, and it fails in both directions.

    - **Small `gamma`**: even distant points still count as somewhat similar. The kernel is too permissive, so the induced space barely differs from the original one, and kernel PCA behaves much like linear PCA. It fails to separate the shapes that need separating.
    - **Large `gamma`**: only very close points count as similar. The kernel becomes too strict, so each point's similarity is dominated by its immediate neighbours. The components then describe **local** structure, and a group can fragment rather than separate.
    - **A suitable `gamma`** sits between those, and depends on the spacing of the data.

    This lesson fixes `gamma=15`, which suits the small clouds below. That choice is not portable: on data with a different scale, 15 could be far too small or far too large. The next section shows what happens when it is wrong.

    ### Why kernel PCA is not automatically better

    It is tempting to conclude "linear PCA is limited, so kernel PCA is an upgrade". That is false, and this lesson tests it directly.

    Kernel PCA is a **different** method, not a strictly better one. On data that a straight boundary already separates, a kernel can make things worse: by emphasising local similarity, it can fold a clean split into a tangled one. The honest rule is to try both and look, which is why four clouds are used here and the results are tabulated rather than described.

    ### Reading these plots

    Every figure in this lesson follows one pattern: the **original cloud**, then the **two-component projection**, then the **first component alone** drawn as a flat strip.

    The strip is the crucial panel. A classifier that used one component would split that strip at a threshold, so if the two colours **interleave** along it, one component cannot separate them. That is a much sharper test than looking at the two-dimensional cloud, where overlap can hide.

    The kernel and PCA fits are computed on the coordinates only. The group labels are used afterwards, for colouring and for the overlap measurement. Neither method sees them.

    ## Datasets used

    Four clouds, chosen so that the same pair of methods succeeds and fails in different places. All are generated in the notebook.

    ### Linearly separable blobs

    From `sklearn.datasets.make_blobs`: 200 points, two centres, `cluster_std=0.5`, `random_state=0`. Two round, well-separated groups. **This is the case where linear PCA should win**, and it is included precisely to keep the lesson honest.

    ### Half moons

    From `make_moons`: 100 points, `noise=0.015`, `random_state=0`. Two crescents that interlock. A straight boundary cuts across both arms, so this is the classic case for a kernel.

    ### Concentric circles

    From `make_circles`: 400 points, `noise=0.1`, `factor=0.2`, `random_state=123`. A small circle inside a larger ring. The inner group is *surrounded*, so no straight line can separate it, and this is the hardest of the four for a linear method.

    ### Two heart-shaped curves

    Two parametric hearts of 100 points each, the second offset by 20 in x and −10 in y. **A word of warning about this one, which the reference notebook gets wrong:** because the second heart is *shifted* rather than interlocking, the two shapes do not overlap at all. They are separated in the original coordinates, so linear PCA handles them easily, and the RBF kernel at `gamma=15` actually makes the separation *worse*. This dataset is included because it corrects a common misconception, not because it shows a kernel winning.
    """)
    return


@app.cell
def _(go, make_subplots, np, pd):
    def separation_scores(axes, labels):
        """Score how well a single axis separates two groups.

        Two measures are returned because they answer different questions. AUC
        uses every point and measures ranking quality, so it is robust to a few
        stragglers. The overlap percentage counts how many points physically sit
        inside the shared range, which is what a reader sees in a strip plot.

        Parameters
        ----------
        axes : dict
            Maps a method name to its one-dimensional scores.
        labels : array-like
            Group label per score, with two distinct values.

        Returns
        -------
        pandas.DataFrame
            One row per method, with columns `auc` and `overlap`, plus the raw
            ranges of each group.
        """
        from sklearn.metrics import roc_auc_score

        labels = np.asarray(labels)
        rows = []
        for name, values in axes.items():
            values = np.asarray(values)
            left = values[labels == 0]
            right = values[labels == 1]
            raw = float(roc_auc_score(labels, values))
            low = max(left.min(), right.min())
            high = min(left.max(), right.max())
            if high <= low:
                overlap = 0.0
            else:
                overlap = float(((values >= low) & (values <= high)).sum()) / len(values) * 100
            rows.append(
                {
                    "method": name,
                    "auc": max(raw, 1 - raw),
                    "overlap": overlap,
                    "group 0 range": f"{left.min():.2f} to {left.max():.2f}",
                    "group 1 range": f"{right.min():.2f} to {right.max():.2f}",
                }
            )
        return pd.DataFrame(rows).set_index("method")

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

    return panel_of_scatters, ranges_overlap, separation_scores


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Blobs: the case linear PCA handles

    Two round, well-separated groups. Start with the original cloud, then PCA, then the same data under a kernel.

    The purpose of this section is a **baseline**. Before seeing a kernel do something impressive, it is worth seeing it do something unnecessary.
    """)
    return


@app.cell
def _(KernelPCA, PCA, make_blobs, panel_of_scatters, separation_scores):
    blobs, blob_labels = make_blobs(
        n_samples=200, centers=2, cluster_std=0.5, random_state=0
    )
    blob_pca_model = PCA(n_components=2).fit(blobs)
    blob_plane = blob_pca_model.transform(blobs)
    blob_kernel = KernelPCA(n_components=2, kernel="rbf", gamma=15).fit_transform(blobs)
    blob_pca_share = blob_pca_model.explained_variance_ratio_
    blob_scores = separation_scores(
        {"PCA": blob_plane[:, 0], "RBF kernel": blob_kernel[:, 0]}, blob_labels
    )
    panel_of_scatters(
        [
            {"x": blobs[:, 0], "y": blobs[:, 1], "labels": blob_labels, "caption": "Original blobs"},
            {
                "x": blob_plane[:, 0],
                "y": blob_plane[:, 1],
                "labels": blob_labels,
                "caption": "PCA, two components",
            },
            {
                "x": blob_plane[:, 0],
                "labels": blob_labels,
                "caption": "PCA, first component",
                "flat": True,
            },
            {
                "x": blob_kernel[:, 0],
                "labels": blob_labels,
                "caption": "RBF, first component",
                "flat": True,
            },
        ],
        title="Blobs: both methods separate, one of them more cleanly",
    )
    return blob_pca_share, blob_plane, blob_scores


@app.cell(hide_code=True)
def _(blob_pca_share, blob_plane, blob_scores, mo):
    mo.md(f"""
    ### What the panels show

    The original cloud has two clearly separated groups. PCA puts them on either side of zero along its first component, and the strip shows **no interleaving at all**: every orange point is on one side, every blue point on the other.

    The fourth panel is the same data under the RBF kernel. The groups are still broadly separated but the boundary is **no longer clean** — a few points sit on the wrong side.

    ### The numbers

    | Method | First-component AUC | Points in the overlap zone |
    | :--- | ---: | ---: |
    | PCA | {blob_scores.loc['PCA', 'auc']:.3f} | {blob_scores.loc['PCA', 'overlap']:.1f}% |
    | RBF kernel | {blob_scores.loc['RBF kernel', 'auc']:.3f} | {blob_scores.loc['RBF kernel', 'overlap']:.1f}% |

    The AUC measures how well a single threshold on that axis separates the groups: **1.000 is perfect, 0.500 is chance**. PCA reaches {blob_scores.loc['PCA', 'auc']:.3f} and the kernel reaches {blob_scores.loc['RBF kernel', 'auc']:.3f}.

    PCA also explains **{blob_pca_share[0]:.1%}** of the variance with its first component, and because the groups are round the direction of greatest variance *is* the direction that separates them.

    **So here the kernel is a slight downgrade.** That is the point of starting with it. A kernel is a change of representation, and a change is not automatically an improvement — this dataset is already easy for the method it replaces.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Half moons: where the straight direction fails

    Two crescents that interlock. Before looking at any component, look at the original cloud and ask a simple question: **is there any straight line that puts one colour on each side?**

    The answer is no. Each arm wraps around the other, so any line through the middle cuts both.
    """)
    return


@app.cell
def _(KernelPCA, PCA, make_moons, panel_of_scatters, separation_scores):
    moons, moon_labels = make_moons(100, noise=0.015, random_state=0)
    moon_plane = PCA(n_components=2).fit_transform(moons)
    moon_kernel = KernelPCA(n_components=2, kernel="rbf", gamma=15).fit_transform(moons)
    moon_scores = separation_scores(
        {"PCA": moon_plane[:, 0], "RBF kernel": moon_kernel[:, 0]}, moon_labels
    )
    panel_of_scatters(
        [
            {"x": moons[:, 0], "y": moons[:, 1], "labels": moon_labels, "caption": "Original moons"},
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
                "labels": moon_labels,
                "caption": "RBF, first component",
                "flat": True,
            },
        ],
        title="Moons: the linear component mixes the arms, the kernel does not",
    )
    return moon_kernel, moon_plane, moon_scores


@app.cell(hide_code=True)
def _(moon_plane, moon_scores, mo):
    mo.md(f"""
    ### What the panels show

    **The PCA strip is a mess.** Orange and blue points interleave along the entire axis, with orange on both sides of zero and blue in between. There is no threshold that separates them, and adding a second component does not help: the two-dimensional panel shows the groups as one combined blob.

    This is the failure the background section predicted. PCA found the direction of greatest variance, and for two crescents that direction runs *along* the pair rather than between them. The limitation is the shape of the boundary, not the number of dimensions.

    **The kernel strip is clean.** Every orange point is on one side and every blue point on the other, with a gap between them.

    ### The numbers

    | Method | First-component AUC | Points in the overlap zone |
    | :--- | ---: | ---: |
    | PCA | {moon_scores.loc['PCA', 'auc']:.3f} | {moon_scores.loc['PCA', 'overlap']:.1f}% |
    | RBF kernel | {moon_scores.loc['RBF kernel', 'auc']:.3f} | {moon_scores.loc['RBF kernel', 'overlap']:.1f}% |

    PCA's AUC of {moon_scores.loc['PCA', 'auc']:.3f} looks better than it is: ranking quality stays decent while {moon_scores.loc['PCA', 'overlap']:.1f}% of the points sit inside the shared range, which is what blocks a threshold. The kernel reaches {moon_scores.loc['RBF kernel', 'auc']:.3f} with {moon_scores.loc['RBF kernel', 'overlap']:.1f}% overlap.

    **Why the kernel can do this.** Along one crescent, points are close to their neighbours, so the kernel gives them high similarity. Between the crescents the gap means low similarity. The kernel therefore encodes **connectivity**, and after the transformation the arms occupy different parts of the component space even though they share a region of the original one.

    This is the same effect that made spectral clustering work on these shapes in the clustering lessons: both methods use a neighbourhood notion rather than a distance to a centre.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Concentric circles: the hardest case for a straight boundary

    A small circle inside a larger ring. Here the inner group is **surrounded**: no line can put it on one side, because it is enclosed on every side. If a linear direction fails anywhere, it should fail here.
    """)
    return


@app.cell
def _(KernelPCA, PCA, make_circles, panel_of_scatters, separation_scores):
    rings, ring_labels = make_circles(
        n_samples=400, random_state=123, noise=0.1, factor=0.2
    )
    ring_plane = PCA(n_components=2).fit_transform(rings)
    ring_kernel = KernelPCA(n_components=2, kernel="rbf", gamma=15).fit_transform(rings)
    ring_ratio = PCA(n_components=2).fit(rings).explained_variance_ratio_
    ring_scores = separation_scores(
        {"PCA": ring_plane[:, 0], "RBF kernel": ring_kernel[:, 0]}, ring_labels
    )
    panel_of_scatters(
        [
            {"x": rings[:, 0], "y": rings[:, 1], "labels": ring_labels, "caption": "Original rings"},
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
                "labels": ring_labels,
                "caption": "RBF, first component",
                "flat": True,
            },
        ],
        title="Rings: a linear direction cannot enclose",
    )
    return ring_ratio, ring_scores


@app.cell(hide_code=True)
def _(ring_ratio, ring_scores, mo):
    mo.md(f"""
    ### The most complete failure in the lesson

    | Method | First-component AUC | Points in the overlap zone |
    | :--- | ---: | ---: |
    | PCA | {ring_scores.loc['PCA', 'auc']:.3f} | {ring_scores.loc['PCA', 'overlap']:.1f}% |
    | RBF kernel | {ring_scores.loc['RBF kernel', 'auc']:.3f} | {ring_scores.loc['RBF kernel', 'overlap']:.1f}% |

    PCA's AUC of **{ring_scores.loc['PCA', 'auc']:.3f}** is indistinguishable from chance, which is `0.500`. That is not a near miss; on a single component the method carries **no information at all** about which group a point belongs to. With {ring_scores.loc['PCA', 'overlap']:.1f}% of the points inside the shared range, no threshold can help.

    The reason is geometric and worth stating plainly: the first component is a direction, and the inner circle lies on both sides of every direction through the centre. There is no direction along which "inside" differs from "outside".

    The kernel reaches **{ring_scores.loc['RBF kernel', 'auc']:.3f}** with **{ring_scores.loc['RBF kernel', 'overlap']:.1f}%** overlap, because similarity is high within each ring and low across the gap, and that is a distinction a component can represent.

    **Note what PCA still does correctly.** Its two components explain {ring_ratio[0]:.1%} and {ring_ratio[1]:.1%} of the variance — almost equal, because the ring is symmetric. PCA has faithfully described the spread of this cloud. It simply was never asked to separate the groups, and could not have done so with a straight direction even if it had been.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Two hearts: the case that refutes the usual story

    This dataset is often used to show a kernel winning. **It does not, and understanding why is worth more than the demonstration.**

    Look at the coordinates before the plots. Each heart spans `x` from −16 to 16, and the second is shifted by 20, so it spans 4 to 36. The two shapes **barely touch** — they are offset, not interlocking. The picture looks intricate, but in the original coordinates a vertical line separates them almost perfectly.

    That is the trap. A shape can look complicated and still be easy, and a method can look sophisticated and still make it harder.
    """)
    return


@app.cell
def _(KernelPCA, PCA, np, panel_of_scatters, separation_scores):
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
    heart_scores = separation_scores(
        {"PCA": heart_plane[:, 0], "RBF kernel": heart_kernel[:, 0]}, heart_labels
    )
    panel_of_scatters(
        [
            {"x": hearts[:, 0], "y": hearts[:, 1], "labels": heart_labels, "caption": "Two hearts, offset"},
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
                "labels": heart_labels,
                "caption": "RBF, first component",
                "flat": True,
            },
        ],
        title="Two hearts: the linear component separates them, the kernel does not",
    )
    return hearts, heart_scores


@app.cell(hide_code=True)
def _(heart_scores, hearts, mo, np):
    mo.md(f"""
    ### The result is the opposite of the usual story

    | Method | First-component AUC | Points in the overlap zone |
    | :--- | ---: | ---: |
    | PCA | {heart_scores.loc['PCA', 'auc']:.3f} | {heart_scores.loc['PCA', 'overlap']:.1f}% |
    | RBF kernel | {heart_scores.loc['RBF kernel', 'auc']:.3f} | {heart_scores.loc['RBF kernel', 'overlap']:.1f}% |

    The range of x in the data is {hearts[:, 0].min():.0f} to {hearts[:, 0].max():.0f}, and the two groups occupy mostly disjoint intervals of it. A straight boundary works, so **PCA reaches {heart_scores.loc['PCA', 'auc']:.3f}** on its first component.

    The kernel drops to **{heart_scores.loc['RBF kernel', 'auc']:.3f}**, with {heart_scores.loc['RBF kernel', 'overlap']:.1f}% of the points in the shared range. Here the kernel is a **clear downgrade**.

    **Why the kernel hurts.** A kernel compares points by distance, and within a heart the distance along the outline is large even though the shape is one group. At `gamma=15` the similarity between two points at opposite tips of the same heart is essentially zero. The kernel therefore sees each heart as two or more loose clusters rather than one, and the components describe that fragmentation instead of the split between hearts.

    In short: **the kernel is not measuring what this dataset needs.** The groups differ by *location*, and a kernel that emphasises *local neighbourhood* is answering a different question.

    ### The general lesson

    This is why the lesson tests four clouds instead of one. The honest summary is not "kernel PCA is better than PCA". It is:

    - When groups **interlock or enclose**, a kernel wins by a wide margin (moons and rings).
    - When groups are **already separated by a straight boundary**, a kernel can lose (blobs and hearts).
    - The parameter `gamma` decides how strongly the kernel looks at neighbourhoods, and therefore how far the result departs from linear PCA.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. The parameter that decides everything

    Every result so far used `gamma=15`. That number was chosen to suit these small clouds, and it is worth seeing what happens when it is wrong, because `gamma` is the setting you actually have to get right.

    The same moons and rings are refitted at `gamma` values spanning four orders of magnitude. Only the first component is scored.
    """)
    return


@app.cell
def _(KernelPCA, make_circles, make_moons, pd, separation_scores):
    _moons, _moon_labels = make_moons(100, noise=0.015, random_state=0)
    _rings, _ring_labels = make_circles(
        n_samples=400, random_state=123, noise=0.1, factor=0.2
    )
    _rows = []
    for _gamma in (0.01, 0.1, 1, 15, 100):
        _moon_axis = KernelPCA(
            n_components=2, kernel="rbf", gamma=_gamma
        ).fit_transform(_moons)[:, 0]
        _ring_axis = KernelPCA(
            n_components=2, kernel="rbf", gamma=_gamma
        ).fit_transform(_rings)[:, 0]
        _rows.append(
            {
                "gamma": _gamma,
                "moons AUC": separation_scores({"a": _moon_axis}, _moon_labels).loc["a", "auc"],
                "rings AUC": separation_scores({"a": _ring_axis}, _ring_labels).loc["a", "auc"],
            }
        )
    gamma_scores = pd.DataFrame(_rows).set_index("gamma")
    return (gamma_scores,)


@app.cell
def _(mo, gamma_scores):
    mo.Html(gamma_scores.round(3).to_html())
    return


@app.cell(hide_code=True)
def _(gamma_scores, mo):
    _best_moons = float(gamma_scores["moons AUC"].max())
    _worst_moons = float(gamma_scores["moons AUC"].min())
    mo.md(f"""
    Read the table as a curve with a usable middle.

    - **At `gamma=0.01`**, the kernel is so permissive that almost every pair of points counts as similar. The similarity matrix carries little information beyond the overall scale, so the components behave much like linear PCA. On the moons that shows up immediately: the AUC is **{gamma_scores.loc[0.01, 'moons AUC']:.3f}**, close to the 0.500 of a linear fit.
    - **At `gamma=15`**, each crescent holds together while the gap between them stays dark. The AUC is **{gamma_scores.loc[15, 'moons AUC']:.3f}**.
    - **At `gamma=100`**, the kernel has collapsed to "am I the same point or not". Similarity is high only between immediate neighbours, so the components describe tiny local patches and the two crescents fragment. The AUC falls to **{gamma_scores.loc[100, 'moons AUC']:.3f}**.

    The useful range spans {_worst_moons:.3f} to {_best_moons:.3f} on this data, and it is a **plateau, not a peak**: several values around 1 to 15 work well.

    **The rings behave differently**, which is the interesting part. A ring needs the kernel to reach across its own width, so a `gamma` tuned for the moons is not automatically right here. That is why `gamma` cannot be copied between datasets any more than a distance threshold can.

    **How you would choose it in practice.** There is no formula. You try a range, look at the result, and keep the value that separates groups that you believe should be separate. With labelled data you could score it, as the table does; without labels, this is a judgement made by inspecting the projection.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Replace `None` so that `student_kernel` holds the string that selects the Gaussian RBF kernel, using the argument name `kernel`.
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
            "Show answers": mo.md(r"""
```python
student_kernel = "rbf"
```
You would write `KernelPCA(kernel="rbf", gamma=15, n_components=2)`.

`"rbf"` stands for radial basis function, the Gaussian kernel in the background section. The alternatives include `"poly"`, `"sigmoid"`, `"cosine"`, and `"linear"` — and `"linear"` is worth knowing, because kernel PCA with a linear kernel reproduces ordinary PCA. That makes it a useful baseline in the same code path.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Replace `None` so that `student_moons_auc` holds the first-component AUC of kernel PCA on the moons at `gamma=15`, and `student_gap` holds how much better that is than linear PCA's AUC on the same data. Both are already in `moon_scores`.
    """)
    return


@app.cell
def _(moon_scores):
    student_moons_auc = None
    student_gap = None
    print("kernel AUC:", student_moons_auc)
    print("improvement over PCA:", student_gap)
    return student_gap, student_moons_auc


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
```python
student_moons_auc = moon_scores.loc["RBF kernel", "auc"]
student_gap = moon_scores.loc["RBF kernel", "auc"] - moon_scores.loc["PCA", "auc"]
```
The kernel AUC is **1.000** and the improvement is about **0.135**.

`moon_scores` is indexed by method name, and its `auc` column is already folded so that 1.0 always means perfect separation. Two `loc` lookups take the two rows; subtracting gives the gain.

The 0.135 improvement understates the difference. PCA's 0.865 means its ranking is decent, but with 45% of points in the shared range, no threshold can use that ranking. AUC alone would hide the problem, which is why the overlap column is reported beside it.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(
    blob_scores,
    gamma_scores,
    heart_scores,
    mo,
    moon_scores,
    ring_scores,
):
    _pairs = [
        ("Blobs", blob_scores),
        ("Half moons", moon_scores),
        ("Concentric circles", ring_scores),
        ("Two hearts", heart_scores),
    ]
    _lines = "\n".join(
        f"    - **{_name}:** PCA {_table.loc['PCA', 'auc']:.3f}, "
        f"RBF kernel {_table.loc['RBF kernel', 'auc']:.3f} — "
        + ("the kernel is better" if _table.loc['RBF kernel', 'auc'] > _table.loc['PCA', 'auc'] else "PCA is better")
        for _name, _table in _pairs
    )
    mo.md(f"""
    ## Conclusions

    1. **A linear component cannot separate groups that curve around each other.** PCA finds the direction of greatest variance, and for interlocking or enclosing shapes that direction runs *along* the structure rather than between the groups. The limit is the shape of the boundary, not the number of components.
    2. **A kernel replaces the inner product with a similarity.** The RBF kernel gives nearby points a value near 1 and distant points a value near 0, and PCA is then computed from that matrix. The implied space is never built explicitly, which is what makes the approach practical.
    3. **`gamma` controls how local the kernel is, and it fails in both directions.** Too small and the kernel is too permissive, so the result approaches linear PCA. Too large and only immediate neighbours count, so groups fragment. On the moons the AUC ranged from {gamma_scores['moons AUC'].min():.3f} to {gamma_scores['moons AUC'].max():.3f} across five values of `gamma`.
    4. **First-component separation, measured two ways:**

    {_lines}

    5. **Kernel PCA is not automatically better.** It wins decisively where groups interlock or enclose, and it **loses** on data a straight boundary already handles. The hearts case is the clearest: the two hearts are offset rather than interlocking, PCA reaches {heart_scores.loc['PCA', 'auc']:.3f} and the kernel only {heart_scores.loc['RBF kernel', 'auc']:.3f}.
    6. **The appearance of complexity is not complexity.** The hearts look intricate and are easy; the rings look simple and defeat a linear direction completely, at an AUC of {ring_scores.loc['PCA', 'auc']:.3f} — indistinguishable from chance.
    7. **Neither method sees the labels.** Both fits use coordinates only. The colours and every score in this lesson are computed afterwards, which is what makes the comparison a fair test of structure.
    8. **The overlap measure matters as much as the score.** An AUC near 0.87 with 45% of points in the shared range describes a ranking that cannot be turned into a decision. Always ask whether a threshold exists, not just whether the ordering is good.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check your understanding

    1. Why can a straight component not separate the two moons, and why would adding another component not fix it?
    2. What does the RBF kernel compare, and what two values does it give for very close and very distant points?
    3. What does `gamma` control, and what happens at very small and very large values?
    4. Why can kernel PCA be computed without building the higher-dimensional space?
    5. On which of the four clouds did kernel PCA do **worse** than linear PCA, and why?
    6. A method reports a first-component AUC of 0.87 with 45% of points inside the shared range. Can a single threshold separate the groups?
    7. Why do the two hearts separate so easily, despite looking intricate?
    8. Why are the group labels not used in either fit, and what does that make the comparison a test of?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
1. **Because the arms interlock.** Any straight line through the middle crosses both crescents, so the projection mixes them. A second component does not help, because the problem is the *shape* of the boundary rather than the number of directions kept.
2. **The similarity between pairs of points.** It returns a value near **1** for points that are very close and near **0** for points that are far apart. PCA is then run on that similarity rather than on the raw columns.
3. **How quickly similarity falls off with distance.** Very small `gamma` makes the kernel too permissive, so distant points still count as similar and the result approaches linear PCA. Very large `gamma` makes only immediate neighbours count, so groups fragment into local patches.
4. **Because only inner products are needed.** The kernel gives the similarity that points would have in the higher-dimensional space, without any coordinate being computed there. This is the kernel trick, shared with support vector machines.
5. **On the blobs and on the two hearts.** In both cases a straight boundary already separates the groups, so the linear component works. The kernel emphasises local neighbourhood, which for these datasets breaks a clean split rather than making one — on the hearts the AUC dropped from about 0.95 to 0.60.
6. **No.** An overlap of 45% means 45% of the points lie inside the range shared by both groups, so no cut-off can place them correctly. The AUC describes ranking quality, which can be good while a usable threshold does not exist. Both numbers must be read together.
7. **Because they are offset, not interlocking.** The second heart is shifted 20 units in x, so the two shapes barely touch and a vertical line separates them almost perfectly. The outline is intricate, but the *groups* are not entangled.
8. **Both methods are unsupervised.** They see coordinates only; a label is never an input. That makes the comparison a test of whether the fitted structure happens to line up with the real groups, which is exactly the question that matters when you later apply the method without labels.
""")
        },
        expanded=False,
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
