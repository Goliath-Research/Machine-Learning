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
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    pio.templates.default = "plotly_white"
    px.defaults.color_discrete_sequence = px.colors.qualitative.G10
    builder = pb.PlotBuilder(width=900, height=520)
    GROUP_COLOR_0 = "#332288"
    GROUP_COLOR_1 = "#CC6677"
    GROUP_COLORS = {"0": GROUP_COLOR_0, "1": GROUP_COLOR_1}

    return (
        GROUP_COLORS,
        KernelPCA,
        PCA,
        StandardScaler,
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
        roc_auc_score,
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
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Background

    The previous lessons found directions in the data. **Principal component analysis** looks for the direction of greatest variance, and its components are weighted mixtures of the original columns. That works well when groups are separated by a straight boundary.

    This lesson is about the case where they are not.

    ### The problem with a straight direction

    Two crescents that interlock, or a ring around a circle, share the same region of space. A straight line through that space cuts through both groups. Since PCA can only find straight directions, projecting onto its components leaves the groups interleaved. No choice of component count fixes that, because the limitation is the shape of the boundary, not the number of dimensions.

    Notice what is *not* wrong: PCA has done its job correctly. It found the directions of greatest variance. Those directions simply do not happen to separate the groups, and PCA was never asked to make them, because **it does not see the labels**. PCA is unsupervised; the colours in these plots are added afterwards.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
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
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Gamma, and its two failure modes

    `gamma` sets **how quickly similarity falls off with distance**. It is the single parameter that matters, and it fails in both directions.

    - **Small `gamma`**: even distant points still count as somewhat similar. The kernel is too permissive, so the induced space barely differs from the original one, and kernel PCA behaves much like linear PCA. It fails to separate the shapes that need separating.
    - **Large `gamma`**: only very close points count as similar. The kernel becomes too strict, so each point's similarity is dominated by its immediate neighbours. The components then describe **local** structure, and a group can fragment rather than separate.
    - **A suitable `gamma`** sits between those, and depends on the spacing of the data.

    This lesson fixes `gamma=15`, which suits the small clouds below. That choice is not portable: on data with a different scale, 15 could be far too small or far too large. The next section shows what happens when it is wrong.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Why kernel PCA is not automatically better

    It is tempting to conclude "linear PCA is limited, so kernel PCA is an upgrade". That is false, and this lesson tests it directly.

    Kernel PCA is a **different** method, not a strictly better one. On data that a straight boundary already separates, a kernel can make things worse: by emphasising local similarity, it can fold a clean split into a tangled one. The honest rule is to try both and look, which is why four clouds are used here and the results are tabulated rather than described.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reading these plots

    Every figure in this lesson follows one pattern: the **original cloud**, then **two-component** projections for PCA and for the RBF kernel, then the **first component alone** for each method drawn as a flat strip with a hidden vertical axis.

    The strip is the crucial panel. A classifier that used one component would split that strip at a threshold, so if the two colours **interleave** along it, one component cannot separate them. That is a much sharper test than looking at the two-dimensional cloud, where overlap can hide.

    The kernel and PCA fits are computed on the coordinates only. The group labels are used afterwards, for colouring and for the overlap measurement. Neither method sees them. Group 0 is **blue** and group 1 is **coral** in every panel.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Datasets used

    Four clouds, chosen so that the same pair of methods succeeds and fails in different places. All are generated in the notebook.

    ### Linearly separable blobs

    From `sklearn.datasets.make_blobs`: 200 points, two centres, `cluster_std=0.5`, `random_state=0`. Two round, well-separated groups. **This is the case where linear PCA should win**, and it is included precisely to keep the lesson honest.

    ### Half moons

    From `make_moons`: 100 points, `noise=0.015`, `random_state=0`. Two crescents that interlock. A straight boundary cuts across both arms, so this is the classic case for a kernel.

    ### Concentric circles

    From `make_circles`: 400 points, `noise=0.1`, `factor=0.2`, `random_state=123`. A small circle inside a larger ring. The inner group is *surrounded*, so no straight line can separate it, and this is the hardest of the four for a linear method.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Two heart-shaped curves

    Two parametric hearts of 100 points each, the second offset by 20 in x and −10 in y. The lobes **do overlap** in the original plane: each heart spans about −16 to 16 in x, so after the shift the shared band runs from about 4 to 16. Linear PCA still separates them well on its first component, but the RBF kernel at `gamma=15` **fails** here because the coordinates are large — with scale around ±30, that gamma makes every pair except immediate neighbours essentially unrelated, so the kernel matrix is nearly an identity and the RBF scores collapse. Standardising the coordinates or choosing a much smaller `gamma` (roughly \(1/(2\sigma^2)\) for a typical neighbour distance \(\sigma\)) is what restores a usable kernel.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The mathematics (what the code is doing)

    **Kernel matrix.** For points \(\mathbf{x}_1,\ldots,\mathbf{x}_n\), build \(K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)\). For the RBF kernel this is an \(n\times n\) similarity matrix.

    **Centring.** Kernel PCA centres \(K\) in feature space (scikit-learn handles this internally). Centring is what makes the components describe variance *around the cloud* rather than raw similarities.

    **Eigenproblem.** Find eigenvectors of the centred kernel matrix. Those are principal directions in the implicit feature space \(\phi(\mathbf{x})\) where \(\langle \phi(\mathbf{x}), \phi(\mathbf{y}) \rangle = k(\mathbf{x}, \mathbf{y})\). You never form \(\phi\); only inner products matter.

    **Scores.** A point's component value is built from its kernel row against the training points — a weighted combination of similarities, projected onto the leading eigenvectors.

    **Why RBF is infinite-dimensional.** The Gaussian kernel corresponds to an inner product in a space of all smooth functions; keeping two components still means truncating that space.

    **Gamma and scale.** Rule of thumb: choose \(\gamma \approx 1/(2\sigma^2)\) where \(\sigma\) is a typical distance between neighbours. If coordinates span tens of units, `gamma=15` is enormous unless you **standardise** first.

    **AUC (used later).** For a one-dimensional score, AUC is the probability that a random point from group 1 ranks above a random point from group 0. Values near 0.5 mean no useful ordering; 1.0 means perfect separation. When the score direction is flipped, the code uses \(\max(\text{AUC}, 1-\text{AUC})\) so that 1.0 always means separable.
    """)
    return


@app.cell
def _(np, pd, roc_auc_score):
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

    return (separation_scores,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **`separation_scores`.** For each method, the overlap percentage counts every point whose score lies between the two groups' inner range limits (the intersection of their min–max intervals). That matches what you see when two coloured clouds share a strip. AUC uses scikit-learn's `roc_auc_score` on the raw scores, then the code keeps \(\max(\text{AUC}, 1-\text{AUC})\) so a reversed axis still counts as perfect separation.
    """)
    return


@app.cell
def _(GROUP_COLORS, go, make_subplots, np):
    def panel_of_scatters(specs, title="", width=900):
        """Draw comparison panels for kernel PCA lessons.

        Each spec is one panel: coordinate arrays, group labels, and a caption.
        Non-flat panels keep a 1:1 aspect ratio. Flat panels show one component
        with vertical jitter and a hidden y axis.

        Parameters
        ----------
        specs : list of dict
            Each entry has `x`, `labels`, and `caption`. Non-flat entries also
            need `y`. Set `flat=True` for a one-dimensional strip.
        title : str, default ""
            Figure title.
        width : int, default 900
            Figure width in pixels.

        Returns
        -------
        plotly.graph_objects.Figure
            The assembled figure.
        """
        positions = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2)]
        figure = make_subplots(
            rows=2,
            cols=3,
            subplot_titles=[spec["caption"] for spec in specs],
            horizontal_spacing=0.06,
            vertical_spacing=0.14,
            row_heights=[0.62, 0.38],
        )
        for panel_index, spec in enumerate(specs):
            row, col = positions[panel_index]
            x = np.asarray(spec["x"])
            labels = np.asarray(spec["labels"]).astype(str)
            is_flat = spec.get("flat", False)
            for group in ("0", "1"):
                mask = labels == group
                if is_flat:
                    jitter = ((np.arange(mask.sum()) % 5) - 2) * 0.035
                    y_vals = jitter
                else:
                    y_vals = np.asarray(spec["y"])[mask]
                figure.add_trace(
                    go.Scatter(
                        x=x[mask],
                        y=y_vals,
                        mode="markers",
                        name=f"group {group}",
                        legendgroup=group,
                        showlegend=panel_index == 0,
                        marker=dict(
                            size=6,
                            opacity=0.75,
                            color=GROUP_COLORS[group],
                        ),
                    ),
                    row=row,
                    col=col,
                )
            if not is_flat:
                anchor = f"y{col}" if row == 1 and col > 1 else "y"
                figure.update_xaxes(
                    scaleanchor=anchor,
                    scaleratio=1,
                    row=row,
                    col=col,
                )
            else:
                figure.update_yaxes(
                    visible=False,
                    showticklabels=False,
                    range=[-0.14, 0.14],
                    row=row,
                    col=col,
                )
                figure.update_xaxes(
                    row=row,
                    col=col,
                    ticklabelstandoff=6,
                )
        figure.update_layout(
            width=width,
            height=620,
            title=title,
            margin=dict(t=80, b=55, l=50, r=20),
        )
        return figure

    return (panel_of_scatters,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **`panel_of_scatters`.** Each dict in `specs` supplies `x`, `labels`, and `caption`; scatter panels also need `y`, while strip panels set `flat=True` so all points sit on the x axis with a small deterministic jitter (so stacked scores remain visible). Panels are arranged in a 2×3 grid: three scatter views on the top row and two strips on the bottom. `scaleanchor` ties each scatter's x and y units so circles stay round.
    """)
    return


@app.cell
def _(np):
    def heart_curve(a, b):
        """Parametric coordinates for one heart outline.

        Parameters
        ----------
        a : float
            Horizontal shift added to every x coordinate.
        b : float
            Vertical shift added to every y coordinate.

        Returns
        -------
        x : numpy.ndarray
            x coordinates along the curve.
        y : numpy.ndarray
            y coordinates along the curve.
        """
        t = np.linspace(0, 2 * np.pi, 100)
        x = 16 * np.sin(t) ** 3
        y = (
            13 * np.cos(t)
            - 5 * np.cos(2 * t)
            - 2 * np.cos(3 * t)
            - np.cos(4 * t)
        )
        return a + x, b + y

    return (heart_curve,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **`heart_curve`.** The heart uses the classic parametric form: \(x = 16\sin^3 t\) and a sum of cosines in \(y\), sampled at 100 equally spaced angles \(t \in [0, 2\pi]\). Shifts `(a, b)` place the second heart away from the first without changing its shape.
    """)
    return


@app.cell
def _(np):
    def rbf_kernel_matrix(points, gamma):
        """Build the RBF kernel matrix for a set of points.

        Parameters
        ----------
        points : array-like, shape (n_samples, n_features)
            Input coordinates.
        gamma : float
            RBF sharpness parameter.

        Returns
        -------
        numpy.ndarray
            Symmetric matrix with entries exp(-gamma * ||x_i - x_j||^2).
        """
        points = np.asarray(points)
        diff = points[:, None, :] - points[None, :, :]
        sq_dist = np.sum(diff * diff, axis=2)
        return np.exp(-gamma * sq_dist)

    return (rbf_kernel_matrix,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **`rbf_kernel_matrix`.** Each entry compares one pair of points by squared Euclidean distance, then applies the RBF formula. The result is the same matrix kernel PCA uses before centring and the eigen-decomposition.
    """)
    return


@app.cell
def _(go, np, px):
    def kernel_heatmap(kernel_matrix, title=""):
        """Plot a kernel similarity matrix as a heatmap.

        Parameters
        ----------
        kernel_matrix : array-like, shape (n, n)
            Pairwise similarities.
        title : str, default ""
            Figure title.

        Returns
        -------
        plotly.graph_objects.Figure
            Heatmap figure.
        """
        figure = px.imshow(
            kernel_matrix,
            color_continuous_scale="Viridis",
            origin="lower",
            title=title,
        )
        figure.update_layout(
            width=420,
            height=380,
            margin=dict(t=60, b=40, l=50, r=20),
            coloraxis_colorbar=dict(title="k"),
        )
        figure.update_xaxes(title_text="point index")
        figure.update_yaxes(title_text="point index")
        return figure

    return (kernel_heatmap,)


@app.cell
def _(go, np):
    def kernel_vs_distance_curve(gammas, max_distance=3.0):
        """Plot RBF kernel value against distance for several gammas.

        Parameters
        ----------
        gammas : sequence of float
            Values of gamma to compare.
        max_distance : float, default 3.0
            Largest distance on the horizontal axis.

        Returns
        -------
        plotly.graph_objects.Figure
            Line plot of k(d) for each gamma.
        """
        distances = np.linspace(0, max_distance, 200)
        figure = go.Figure()
        for gamma in gammas:
            values = np.exp(-gamma * distances**2)
            figure.add_trace(
                go.Scatter(
                    x=distances,
                    y=values,
                    mode="lines",
                    name=f"gamma={gamma}",
                )
            )
        figure.update_layout(
            width=520,
            height=380,
            title="RBF value vs distance",
            xaxis_title="distance",
            yaxis_title="k(x, y)",
            margin=dict(t=60, b=45, l=55, r=20),
        )
        return figure

    return (kernel_vs_distance_curve,)


@app.cell
def _(go, np, pd):
    def auc_vs_gamma_figure(gamma_values, moons_auc, rings_auc, title=""):
        """Plot first-component AUC against gamma on a log axis.

        Parameters
        ----------
        gamma_values : array-like
            Gamma settings tried.
        moons_auc, rings_auc : array-like
            AUC values for each gamma on moons and rings.
        title : str, default ""
            Figure title.

        Returns
        -------
        plotly.graph_objects.Figure
            Line plot with log-scaled gamma axis.
        """
        frame = pd.DataFrame(
            {
                "gamma": np.asarray(gamma_values, dtype=float),
                "moons AUC": np.asarray(moons_auc, dtype=float),
                "rings AUC": np.asarray(rings_auc, dtype=float),
            }
        )
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=frame["gamma"],
                y=frame["moons AUC"],
                mode="lines+markers",
                name="moons",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=frame["gamma"],
                y=frame["rings AUC"],
                mode="lines+markers",
                name="rings",
            )
        )
        figure.update_xaxes(type="log", title_text="gamma")
        figure.update_yaxes(title_text="first-component AUC", range=[0.45, 1.02])
        figure.update_layout(
            width=620,
            height=380,
            title=title,
            margin=dict(t=60, b=45, l=55, r=20),
        )
        return figure

    return (auc_vs_gamma_figure,)


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
                "x": blob_kernel[:, 0],
                "y": blob_kernel[:, 1],
                "labels": blob_labels,
                "caption": "RBF, two components",
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
    return blob_kernel, blob_pca_share, blob_plane, blob_scores, blobs, blob_labels


@app.cell(hide_code=True)
def _(blob_pca_share, blob_plane, blob_scores, mo):
    mo.md(f"""
    ### What the panels show

    The original cloud has two clearly separated groups. PCA puts them on either side of zero along its first component, and the strip shows **no interleaving at all**: every blue point is on one side, every coral point on the other.

    The RBF strip is the same data under the kernel. The groups are still broadly separated but the boundary is **no longer clean** — **{blob_scores.loc['RBF kernel', 'overlap']:.0f}%** of the points sit in the overlap zone, even though the AUC stays high.

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


@app.cell
def _(blob_scores, mo):
    mo.Html(blob_scores.round(3).to_html())
    return


@app.cell(hide_code=True)
def _(KernelPCA, PCA, blob_kernel, blob_plane, blobs, mo, np):
    _linear_kernel = KernelPCA(n_components=2, kernel="linear").fit_transform(blobs)
    _max_diff = float(np.max(np.abs(_linear_kernel - blob_plane)))
    mo.md(f"""
    ### Linear kernel sanity check

    `KernelPCA(kernel="linear")` on the blobs matches ordinary PCA up to numerical noise: the largest absolute difference between any coordinate in the two two-component projections is **{_max_diff:.2e}**. A linear kernel is the same inner product as PCA; only non-linear kernels change the geometry.
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
                "x": moon_kernel[:, 0],
                "y": moon_kernel[:, 1],
                "labels": moon_labels,
                "caption": "RBF, two components",
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
    return moon_kernel, moon_labels, moon_plane, moon_scores, moons


@app.cell(hide_code=True)
def _(moon_plane, moon_scores, mo):
    mo.md(f"""
    ### What the panels show

    **The PCA strip is a mess.** Blue and coral points interleave along the entire axis, with coral on both sides of zero and blue in between. There is no threshold that separates them, and adding a second component does not help: the two-dimensional panel shows the groups as one combined blob.

    This is the failure the background section predicted. PCA found the direction of greatest variance, and for two crescents that direction runs *along* the pair rather than between them. The limitation is the shape of the boundary, not the number of dimensions.

    **The kernel strip is clean.** Every coral point is on one side and every blue point on the other, with a gap between them.

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


@app.cell
def _(mo, moon_scores):
    mo.Html(moon_scores.round(3).to_html())
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
                "x": ring_kernel[:, 0],
                "y": ring_kernel[:, 1],
                "labels": ring_labels,
                "caption": "RBF, two components",
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
    return ring_labels, ring_ratio, ring_scores, rings


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


@app.cell
def _(mo, ring_scores):
    mo.Html(ring_scores.round(3).to_html())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Two hearts: the case that refutes the usual story

    This dataset is often used to show a kernel winning. **It does not, and understanding why is worth more than the demonstration.**

    Look at the coordinates before the plots. Each heart spans about −16 to 16 in `x`, and the second is shifted by 20, so it spans about 4 to 36. The lobes **overlap** between about 4 and 16 — the picture looks intricate, and the groups are not cleanly disjoint in the plane.

    That is the trap. A shape can look complicated while still being separable by a straight direction in PCA space, and a kernel can still fail when `gamma` does not match the coordinate scale.
    """)
    return


@app.cell
def _(KernelPCA, PCA, heart_curve, np, panel_of_scatters, separation_scores):
    _x1, _y1 = heart_curve(0, 0)
    _x2, _y2 = heart_curve(20, -10)
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
                "x": heart_kernel[:, 0],
                "y": heart_kernel[:, 1],
                "labels": heart_labels,
                "caption": "RBF, two components",
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
        title="Two hearts: PCA still wins; RBF collapses at this scale",
    )
    return heart_kernel, heart_labels, heart_plane, heart_scores, hearts


@app.cell(hide_code=True)
def _(heart_scores, hearts, mo, np):
    mo.md(f"""
    ### The result is the opposite of the usual story

    | Method | First-component AUC | Points in the overlap zone |
    | :--- | ---: | ---: |
    | PCA | {heart_scores.loc['PCA', 'auc']:.3f} | {heart_scores.loc['PCA', 'overlap']:.1f}% |
    | RBF kernel | {heart_scores.loc['RBF kernel', 'auc']:.3f} | {heart_scores.loc['RBF kernel', 'overlap']:.1f}% |

    The range of x in the data is {hearts[:, 0].min():.0f} to {hearts[:, 0].max():.0f}. The groups overlap on the first component ({heart_scores.loc['PCA', 'overlap']:.0f}% of points in the shared strip range), yet **PCA still reaches {heart_scores.loc['PCA', 'auc']:.3f}** AUC — a straight direction in the projected space works far better than chance.

    The kernel drops to **{heart_scores.loc['RBF kernel', 'auc']:.3f}**, with {heart_scores.loc['RBF kernel', 'overlap']:.1f}% of the points in the shared range. Most RBF scores pile up near zero in the strip because the kernel matrix is almost an identity at this scale.
    """)
    return


@app.cell(hide_code=True)
def _(heart_scores, mo):
    mo.md(rf"""
    **Why the kernel hurts.** With coordinates spanning roughly ±30, `gamma=15` makes \(\exp(-15\\,d^2)\) vanish unless \(d\) is tiny. Points at opposite tips of the same heart look unrelated, so the kernel fragments each heart instead of separating the two. **Fix:** standardise the coordinates or choose a much smaller `gamma` (on the order of \(1/(2\\sigma^2)\) for a typical neighbour distance \(\sigma\)).

    When groups **interlock or enclose**, a well-tuned kernel wins (moons and rings). When a straight boundary already works in PCA space, a mismatched kernel can lose (blobs and hearts here).
    """)
    return


@app.cell
def _(heart_scores, mo):
    mo.Html(heart_scores.round(3).to_html())
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


@app.cell
def _(auc_vs_gamma_figure, gamma_scores):
    auc_vs_gamma_figure(
        gamma_scores.index.to_numpy(),
        gamma_scores["moons AUC"].to_numpy(),
        gamma_scores["rings AUC"].to_numpy(),
        title="First-component AUC vs gamma (log scale)",
    )
    return


@app.cell
def _(kernel_vs_distance_curve):
    kernel_vs_distance_curve([0.01, 1, 15, 100], max_distance=1.5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Kernel connectivity at three gammas (moons sample)

    Each heatmap shows the RBF similarity matrix on the first 40 moon points. **Small gamma** washes every pair towards 1; **gamma = 15** separates within-arm similarity from across-gap darkness; **large gamma** leaves only the diagonal bright — the matrix looks like an identity, and kernel PCA has almost nothing global to work with.
    """)
    return


@app.cell
def _(kernel_heatmap, make_moons, mo, rbf_kernel_matrix):
    _moons, _ = make_moons(100, noise=0.015, random_state=0)
    _sample = _moons[:40]
    mo.vstack(
        [
            kernel_heatmap(
                rbf_kernel_matrix(_sample, 0.01),
                title="Moons kernel matrix, gamma=0.01",
            ),
            kernel_heatmap(
                rbf_kernel_matrix(_sample, 15),
                title="Moons kernel matrix, gamma=15",
            ),
            kernel_heatmap(
                rbf_kernel_matrix(_sample, 100),
                title="Moons kernel matrix, gamma=100",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Explore gamma, dataset, and kernel live

    Use the controls below to refit kernel PCA. The scatter is the two-component projection; the strip is the first component with the same jitter as the main figures; the line reports first-component AUC and overlap for the current setting.
    """)
    return


@app.cell
def _(mo, np):
    dataset_dropdown = mo.ui.dropdown(
        options={
            "Half moons": "moons",
            "Concentric rings": "rings",
            "Two hearts": "hearts",
            "Blobs": "blobs",
        },
        value="Half moons",
        label="Dataset",
    )
    _gamma_steps = np.logspace(-2, 2, 25)
    gamma_slider = mo.ui.slider(
        steps=_gamma_steps,
        value=float(_gamma_steps[12]),
        label="gamma",
        show_value=True,
    )
    kernel_dropdown = mo.ui.dropdown(
        options={
            "RBF": "rbf",
            "Linear": "linear",
            "Polynomial": "poly",
            "Cosine": "cosine",
        },
        value="RBF",
        label="Kernel",
    )
    mo.vstack([dataset_dropdown, gamma_slider, kernel_dropdown])
    return dataset_dropdown, gamma_slider, kernel_dropdown


@app.cell
def _(
    GROUP_COLORS,
    KernelPCA,
    blob_labels,
    blobs,
    dataset_dropdown,
    gamma_slider,
    go,
    heart_labels,
    hearts,
    kernel_dropdown,
    make_subplots,
    mo,
    moon_labels,
    moons,
    np,
    ring_labels,
    rings,
    separation_scores,
):
    _catalog = {
        "moons": (moons, moon_labels),
        "rings": (rings, ring_labels),
        "hearts": (hearts, heart_labels),
        "blobs": (blobs, blob_labels),
    }
    _points, _labels = _catalog[dataset_dropdown.value]
    _kernel = kernel_dropdown.value
    _gamma = float(gamma_slider.value)
    _model = KernelPCA(n_components=2, kernel=_kernel, gamma=_gamma)
    _plane = _model.fit_transform(_points)
    _scores = separation_scores({"kernel": _plane[:, 0]}, _labels)
    _figure = make_subplots(rows=1, cols=2, subplot_titles=("Two components", "First component"))
    for _group in ("0", "1"):
        _mask = _labels.astype(str) == _group
        _figure.add_trace(
            go.Scatter(
                x=_plane[_mask, 0],
                y=_plane[_mask, 1],
                mode="markers",
                name=f"group {_group}",
                marker=dict(size=7, opacity=0.75, color=GROUP_COLORS[_group]),
            ),
            row=1,
            col=1,
        )
        _jitter = ((np.arange(_mask.sum()) % 5) - 2) * 0.035
        _figure.add_trace(
            go.Scatter(
                x=_plane[_mask, 0],
                y=_jitter,
                mode="markers",
                showlegend=False,
                marker=dict(size=7, opacity=0.75, color=GROUP_COLORS[_group]),
            ),
            row=1,
            col=2,
        )
    _figure.update_xaxes(scaleanchor="y", scaleratio=1, row=1, col=1)
    _figure.update_yaxes(visible=False, range=[-0.14, 0.14], row=1, col=2)
    _figure.update_layout(
        width=820,
        height=360,
        title=f"{dataset_dropdown.value}, kernel={_kernel}, gamma={_gamma:g}",
        margin=dict(t=70, b=45, l=50, r=20),
    )
    mo.vstack(
        [
            _figure,
            mo.md(
                f"First-component AUC **{_scores.loc['kernel', 'auc']:.3f}**, "
                f"overlap **{_scores.loc['kernel', 'overlap']:.1f}%**."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(gamma_scores, mo, moon_scores):
    _pca_moons = float(moon_scores.loc["PCA", "auc"])
    mo.md(f"""
    Read the table and the AUC curve together.

    - **At `gamma=0.01`**, the kernel is so permissive that almost every pair of points counts as similar. On the moons the AUC is **{gamma_scores.loc[0.01, 'moons AUC']:.3f}**, essentially the same as linear PCA's **{_pca_moons:.3f}** on this shape — not chance (0.500).
    - **At `gamma=15`**, each crescent holds together while the gap between them stays dark. The moons AUC is **{gamma_scores.loc[15, 'moons AUC']:.3f}** — the only value in the table that reaches perfect separation for the moons.
    - **At `gamma=100`**, similarity is high only between immediate neighbours, so the crescents fragment. The moons AUC falls to **{gamma_scores.loc[100, 'moons AUC']:.3f}**.

    There is **no broad plateau**: intermediate values such as `gamma=1` only reach **{gamma_scores.loc[1, 'moons AUC']:.3f}** on the moons. You need a sharp enough kernel for this spacing, but not so sharp that arms break apart.

    **Rings follow the same peak at `gamma=15`** in this table (**{gamma_scores.loc[15, 'rings AUC']:.3f}**), so a gamma that works on the moons can still work elsewhere — the lesson is that scale and spacing must be checked on *each* dataset, not that a good moons gamma is automatically wrong on rings.

    **How you would choose it in practice.** There is no formula. You try a log-spaced range, inspect projections and (if labels exist) score the first component as here.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    For the half moons, predict qualitatively what the first-component strip will look like under RBF kernel PCA at **`gamma=0.01`** and at **`gamma=100`**. Work out your answer, then reveal it.
    """)
    return


@app.cell(hide_code=True)
def _(gamma_scores, mo, moon_scores):
    mo.accordion(
        {
            "Show answers": mo.md(
                f"""
At **`gamma=0.01`** the kernel is almost flat, so kernel PCA behaves like linear PCA: blue and coral stay interleaved on the strip, with AUC about **{gamma_scores.loc[0.01, 'moons AUC']:.3f}** (compare linear PCA **{moon_scores.loc['PCA', 'auc']:.3f}**).

At **`gamma=100`** only immediate neighbours share similarity, so each crescent fragments into tiny patches and the strip mixes colours again; AUC falls to **{gamma_scores.loc[100, 'moons AUC']:.3f}**.

Only the middle regime (here **`gamma=15`**, AUC **{gamma_scores.loc[15, 'moons AUC']:.3f}**) separates the arms cleanly.
"""
            )
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    The two hearts use large coordinates and `gamma=15`. In one sentence, explain **why** the RBF first component fails (hint: scale and the kernel matrix). Reveal when ready.
    """)
    return


@app.cell(hide_code=True)
def _(heart_scores, mo):
    mo.accordion(
        {
            "Show answers": mo.md(
                f"""
With distances on the order of tens, **`gamma=15`** makes exp(-15 d²) essentially zero unless points are almost coincident, so the kernel matrix is nearly an **identity** and most first-component scores collapse near zero (AUC **{heart_scores.loc['RBF kernel', 'auc']:.3f}**, overlap **{heart_scores.loc['RBF kernel', 'overlap']:.0f}%**). The kernel sees local fragments of each outline, not the split between hearts. **Standardising** or using a much smaller `gamma` aligned with neighbour spacing fixes the scale mismatch.
"""
            )
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Replace `None` so that `student_hearts_auc` holds the first-component AUC of RBF kernel PCA on the **standardised** hearts (`StandardScaler` then `gamma=15`), using the same labels as `heart_labels`.
    """)
    return


@app.cell
def _():
    student_hearts_auc = None
    print("scaled hearts RBF AUC:", student_hearts_auc)
    return (student_hearts_auc,)


@app.cell(hide_code=True)
def _(KernelPCA, StandardScaler, heart_labels, hearts, mo, separation_scores):
    _scaled = StandardScaler().fit_transform(hearts)
    _axis = KernelPCA(n_components=2, kernel="rbf", gamma=15).fit_transform(_scaled)[
        :, 0
    ]
    _answer = float(separation_scores({"RBF": _axis}, heart_labels).loc["RBF", "auc"])
    mo.accordion(
        {
            "Show answers": mo.md(
                f"""
```python
_scaled = StandardScaler().fit_transform(hearts)
_axis = KernelPCA(n_components=2, kernel="rbf", gamma=15).fit_transform(_scaled)[:, 0]
student_hearts_auc = separation_scores({{"RBF": _axis}}, heart_labels).loc["RBF", "auc"]
```

After scaling, the AUC rises to about **{_answer:.3f}** because neighbour distances match the chosen `gamma`.
"""
            )
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Replace `None` so that `student_linear_match` holds the largest absolute difference between `KernelPCA(kernel="linear")` and ordinary PCA on the blobs (two components each), as in the sanity check above.
    """)
    return


@app.cell
def _(KernelPCA, PCA, blob_plane, blobs):
    student_linear_match = None
    print("your answer:", student_linear_match)
    return (student_linear_match,)


@app.cell(hide_code=True)
def _(KernelPCA, PCA, blob_plane, blobs, mo, np):
    _linear = KernelPCA(n_components=2, kernel="linear").fit_transform(blobs)
    _max_diff = float(np.max(np.abs(_linear - blob_plane)))
    mo.accordion(
        {
            "Show answers": mo.md(
                f"""
```python
_linear = KernelPCA(n_components=2, kernel="linear").fit_transform(blobs)
student_linear_match = np.max(np.abs(_linear - blob_plane))
```

The largest difference is **{_max_diff:.2e}** — effectively zero, so a linear kernel reproduces PCA on this cloud.
"""
            )
        },
        expanded=False,
    )
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
def _(mo, moon_scores):
    _kernel_auc = float(moon_scores.loc["RBF kernel", "auc"])
    _gap = float(
        moon_scores.loc["RBF kernel", "auc"] - moon_scores.loc["PCA", "auc"]
    )
    _pca_overlap = float(moon_scores.loc["PCA", "overlap"])
    mo.accordion(
        {
            "Show answers": mo.md(
                f"""
```python
student_moons_auc = moon_scores.loc["RBF kernel", "auc"]
student_gap = moon_scores.loc["RBF kernel", "auc"] - moon_scores.loc["PCA", "auc"]
```
The kernel AUC is **{_kernel_auc:.3f}** and the improvement over PCA is **{_gap:.3f}**.

`moon_scores` is indexed by method name, and its `auc` column is already folded so that 1.0 always means perfect separation. Two `loc` lookups take the two rows; subtracting gives the gain.

That AUC gap understates the strip picture: PCA's overlap is **{_pca_overlap:.0f}%**, so no single threshold can exploit PCA's ranking even when AUC looks moderately high.
"""
            )
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(gamma_scores, heart_scores, mo, moon_scores, ring_scores):
    mo.md(f"""
    ## Conclusions

    1. **Goal 1 — curved groups:** Linear PCA cannot separate interlocking moons (PCA AUC **{moon_scores.loc['PCA', 'auc']:.3f}**, overlap **{moon_scores.loc['PCA', 'overlap']:.0f}%**) or enclosed rings (PCA AUC **{ring_scores.loc['PCA', 'auc']:.3f}**); the boundary shape limits a single straight direction.
    2. **Goal 2 — kernels:** RBF kernel PCA builds a similarity matrix and finds components in the implied feature space without constructing coordinates; RBF lifts moons/rings to AUC **{moon_scores.loc['RBF kernel', 'auc']:.3f}** / **{ring_scores.loc['RBF kernel', 'auc']:.3f}**.
    3. **Goal 3 — gamma:** On the moons, only **`gamma=15`** hits AUC **{gamma_scores.loc[15, 'moons AUC']:.3f}**; **`gamma=0.01`** matches linear PCA (**{gamma_scores.loc[0.01, 'moons AUC']:.3f}**), and **`gamma=100`** fragments the shape (**{gamma_scores.loc[100, 'moons AUC']:.3f}**).
    4. **Goal 4 — winding vs straight:** Kernels rescue moons and rings but lose on blobs and on hearts when PCA already reaches **{heart_scores.loc['PCA', 'auc']:.3f}** while a mismatched RBF drops to **{heart_scores.loc['RBF kernel', 'auc']:.3f}** at large coordinate scale.
    5. **Goal 5 — not automatically better:** Compare both methods and read **AUC with overlap**; neither fit uses labels, so the scores test whether unsupervised structure aligns with the groups you care about.
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
    7. Why does linear PCA still beat the RBF kernel on the two hearts at `gamma=15`, despite the intricate outline?
    8. Why are the group labels not used in either fit, and what does that make the comparison a test of?
    """)
    return


@app.cell(hide_code=True)
def _(heart_scores, mo, moon_scores):
    _heart_drop = float(
        heart_scores.loc["PCA", "auc"] - heart_scores.loc["RBF kernel", "auc"]
    )
    mo.accordion(
        {
            "Show answers": mo.md(
                f"""
1. **Because the arms interlock.** Any straight line through the middle crosses both crescents, so the projection mixes them. A second component does not help, because the problem is the *shape* of the boundary rather than the number of directions kept.
2. **The similarity between pairs of points.** It returns a value near **1** for points that are very close and near **0** for points that are far apart. PCA is then run on that similarity rather than on the raw columns.
3. **How quickly similarity falls off with distance.** Very small `gamma` makes the kernel too permissive, so distant points still count as similar and the result approaches linear PCA. Very large `gamma` makes only immediate neighbours count, so groups fragment into local patches.
4. **Because only inner products are needed.** The kernel gives the similarity that points would have in the higher-dimensional space, without any coordinate being computed there. This is the kernel trick, shared with support vector machines.
5. **On the blobs and on the two hearts.** In both cases linear PCA already separates well; the kernel emphasises local neighbourhood and can break that split — on the hearts the AUC drops from **{heart_scores.loc['PCA', 'auc']:.3f}** to **{heart_scores.loc['RBF kernel', 'auc']:.3f}** (a gap of **{_heart_drop:.3f}**).
6. **No.** An overlap of **{moon_scores.loc['PCA', 'overlap']:.0f}%** means that fraction of the points lie inside the range shared by both groups, so no cut-off can place them correctly. The AUC describes ranking quality, which can be good while a usable threshold does not exist. Both numbers must be read together.
7. **Scale mismatch, not lack of overlap.** PCA still reaches **{heart_scores.loc['PCA', 'auc']:.3f}** because a straight direction in its projection separates the groups; at **`gamma=15`** the RBF matrix on large coordinates is nearly an identity, so the kernel first component collapses (**{heart_scores.loc['RBF kernel', 'auc']:.3f}**).
8. **Both methods are unsupervised.** They see coordinates only; a label is never an input. That makes the comparison a test of whether the fitted structure happens to line up with the real groups, which is exactly the question that matters when you later apply the method without labels.
"""
            )
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
