# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "matplotlib",
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
        "matplotlib": "matplotlib",
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
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.io as pio
    import PlotBuilder as pb
    from sklearn.datasets import load_digits
    from sklearn.decomposition import PCA

    pio.templates.default = "plotly_white"
    px.defaults.color_discrete_sequence = px.colors.qualitative.G10
    builder = pb.PlotBuilder(width=780, height=540)

    return PCA, builder, load_digits, mo, np, pd, plt, px


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # PCA for Pictures and Noise

    ## Objectives

    By the end of this lesson, you should be able to:

    1. Explain what PCA assumes about an image: the 64 pixels are treated as independent coordinates, so a shifted digit is a different point. Say what that blindness costs on images and what it buys in generality.
    2. Explain why a two-component plot supports conclusions about *grouping* but not about distances, and tie the loss to the share of variance those components keep.
    3. Choose a component count from the cumulative variance curve, and explain why the choice is always a prefix and never an arbitrary subset.
    4. Explain why shared stroke structure lands in the high-variance components while independent pixel noise does not.
    5. Explain why keeping *more* components can make a reconstruction worse, and why comparing the output with its noisy input cannot reveal that.

    ## Background

    The previous lesson introduced PCA as a way of finding directions of variance. This lesson uses it for the two jobs it is most useful for in practice: **looking** at high-dimensional data, and **cleaning** it.

    ### Pictures as points

    A digit image here is 8 by 8 pixels, and each pixel is a grey level from 0 to 16. That is 64 numbers per image.

To PCA, an image is simply **a point in 64 dimensions**. Nothing about the grid survives: the 64 values are just coordinates, in the same way that two columns gave a point in a plane last lesson. Flattening the grid into a row of 64 keeps the order fixed, and that order is all PCA sees.

**That last point is an assumption with consequences.** PCA measures closeness with ordinary straight-line distance, and that distance has no idea that pixel 1 sits next to pixel 2. Every pixel is an independent, interchangeable coordinate. Three things follow:

- **A shifted digit is a different point.** Move every stroke one pixel to the right and the distance to the original is large, even though it is the same digit. PCA cannot represent translation.
- **PCA cannot exploit spatial structure.** It has no notion of a stroke, an edge, or a region. That is why its components look like ghostly whole-image templates rather than anything you would draw, and it is the reason image-specific methods beat it on real pictures.
- **The same blindness is what makes it general.** Because it never assumes pixels are adjacent, the identical fit works on cars, flowers, or word counts. Nothing about images is hard-coded.

With 1,797 images, the data is a table of **1,797 rows and 64 columns**. Two principal components then give two columns, which you can plot. That is the entire trick behind "visualising high-dimensional data": you find the two directions that spread the cloud most, and accept whatever is lost.

    **What the plot shows and does not show.** Two components keep only part of the variance, so distances in the picture are shorter and some overlap is exaggerated. What you can trust is *relative* placement: digits that form separate islands really are different in the 64 original dimensions. What you cannot trust is the axes, or the exact distances between islands.

    ### Choosing how many components

    The cumulative explained variance ratio answers a practical question: **how many components do I need to keep a given share of the variance?**

    Read it as a running total, not per component. If the first component holds 15% and the second 13%, the cumulative value after two is 28%. You stop when the total reaches your target.

    The target itself is a judgement. There is no universal answer, and 70% to 95% are all defensible depending on what the components are for. A plot needs very little; a filter needs enough to hold the signal; a compressed representation needs whatever your downstream model tolerates.

    ### Noise filtering, and why it works

    This is the subtler idea, and it rests on how variance is distributed.

    Real structure in the digits — the stroke of a 3, the loop of an 8 — is **shared across images**, so it appears in directions where the data genuinely varies. That is, in the components with large variance.

    Random pixel noise is, by construction, **independent from pixel to pixel and image to image**. It has no shared direction, so it spreads thinly across all 64 components, including the ones with almost no variance.

    So if you reconstruct an image from only its largest components, you keep the shared structure and drop the directions where the noise was hiding. That is the filter, and it is why PCA is used for denoising as well as for compression.

    **The catch, which is the real lesson here.** More components is not monotonically better:

    - **Too few components** discard noise *and* real stroke detail, so the digit blurs.
    - **Too many components** keep almost everything, so the noise comes back with the signal.

    There is a middle range that removes the most noise without destroying the digit. This lesson measures it rather than asserting it.

    ## Datasets used

    ### Handwritten digits

    **Source.** The digits dataset bundled with scikit-learn, from the **UCI Machine Learning Repository's** optical recognition of handwritten digits collection. It contains 1,797 images of handwritten digits 0 to 9, each pre-processed to an 8 by 8 grid of grey levels from 0 to 16. [Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html).

    The images are derived from handwritten samples collected from around 44 writers. They are small and clean, which is why the dataset is used for teaching rather than for benchmarking modern recognition systems.

    **The noise is synthetic.** Every noisy image in this lesson is the clean pixels plus a normal draw with standard deviation 4 and `random_state=42`. That means the clean original is available for comparison, so the filtering effect can be *measured* rather than judged by eye.

    There are no missing values and no privacy concerns in what this lesson uses.
    """)
    return


@app.cell
def _(load_digits):
    digits = load_digits()
    digit_shapes = (
        f"{digits.images.shape[0]} images, each {digits.images.shape[1]} by "
        f"{digits.images.shape[2]} pixels; the table is {digits.data.shape[0]} rows "
        f"by {digits.data.shape[1]} columns"
    )
    return digit_shapes, digits


@app.cell(hide_code=True)
def _(digit_shapes, digits, mo):
    mo.md(f"""
    **{digit_shapes}.**

    That sentence contains the whole transformation. `digits.images` is a three-dimensional array, one 8 by 8 grid per image. `digits.data` is the same information with each grid flattened into a row of 64 values. PCA works on the flattened form, because it needs a table of points and features.

    | | shape | what it is |
    | :--- | :--- | :--- |
    | `digits.images` | {digits.images.shape} | image, row, column |
    | `digits.data` | {digits.data.shape} | image, pixel |
    | `digits.target` | {digits.target.shape} | the digit each image shows |

    The labels in `digits.target` are used **only** to colour the plots and to check results. No fit in this lesson is given them.
    """)
    return


@app.cell
def _(np, plt):
    def plot_digits(images, rows, cols, title=None, labels=None, clim=(0, 16)):
        """Draw a grid of 8 by 8 digit images with matplotlib.

        Parameters
        ----------
        images : array-like
            One row of 64 pixel values per image, or an (n, 8, 8) array.
        rows, cols : int
            Grid shape for the panel.
        title : str or None
            Optional heading above the grid.
        labels : array-like or None, default None
            Optional digit label per image, written in blue near the
            bottom-left of each cell. At 8 by 8 pixels the pattern alone is
            hard to identify, so the label is what lets a reader tell which
            digit they are looking at. The label is placed in *axes*
            coordinates, so it sits inside the cell whatever the pixel values
            happen to be.
        clim : tuple of float or None, default (0, 16)
            Fixed grey-level range shared by every panel in the lesson. A
            shared range is what makes a noisy panel comparable with a
            reconstructed one; `None` lets each panel autoscale.

        Returns
        -------
        matplotlib.figure.Figure
            The grid. Axes are left tick-free because pixel coordinates carry
            no meaning for a reader.
        """
        data = np.asarray(images)
        if data.ndim == 3:
            data = data.reshape(len(data), -1)
        figure, axes = plt.subplots(
            rows,
            cols,
            figsize=(8, rows),
            subplot_kw={"xticks": [], "yticks": []},
            gridspec_kw=dict(hspace=0.1, wspace=0.1),
        )
        for index, ax in enumerate(axes.flat):
            if index >= len(data):
                ax.set_visible(False)
                continue
            ax.imshow(
                data[index].reshape(8, 8),
                cmap="binary",
                interpolation="nearest",
                clim=clim,
            )
            if labels is not None:
                ax.text(
                    0.10,
                    0.05,
                    str(labels[index]),
                    transform=ax.transAxes,
                    color="blue",
                    verticalalignment="bottom",
                )
        if title:
            figure.suptitle(title)
        return figure

    return (plot_digits,)


@app.cell
def _(digits, plot_digits):
    _figure = plot_digits(
        digits.images,
        4,
        10,
        "The first 40 images, with the true digit written in each cell",
        labels=digits.target,
    )
    _figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Each panel is one image. The digits are legible even at 8 by 8, which tells you that 64 numbers carry a lot of information — but not how those numbers vary across the 1,797 images, which is the question PCA answers.

    ## 1. The 64-dimensional cloud, viewed in two dimensions

    Fit two components and plot every image at its two component values. The colour shows the true digit, added afterwards.
    """)
    return


@app.cell
def _(PCA, builder, digits, pd):
    digit_pca2 = PCA(n_components=2).fit(digits.data)
    digit_plane = digit_pca2.transform(digits.data)
    digit_plane_share = float(digit_pca2.explained_variance_ratio_.sum())
    _frame = pd.DataFrame(
        {
            "Component 1": digit_plane[:, 0],
            "Component 2": digit_plane[:, 1],
            "digit": digits.target.astype(str),
        }
    )
    _figure = builder.scatter_by_group(
        _frame,
        "Component 1",
        "Component 2",
        "digit",
        "All 1,797 digits on the first two components",
        size=5,
        equal=False,
    )
    _figure
    return digit_plane, digit_plane_share


@app.cell(hide_code=True)
def _(digit_plane_share, mo):
    mo.md(f"""
    Two components out of 64 keep **{digit_plane_share:.1%}** of the total variance. Everything else — {1 - digit_plane_share:.1%} — is not in this picture.

    What you can read from it:

    - **The digits are not randomly scattered.** Structure is visible: several digits form separable regions, which means they differ systematically in how their pixels vary.
    - **The labels were not used.** PCA found this layout from the 64 pixel values alone; the digit names were attached afterwards. The fact that named digits land in coherent areas is evidence the layout reflects something real.

    What you must **not** read from it:

    - **The axes are directions, not measurements.** Component 1 is a weighted mixture of 64 pixel values, not "the size of the writing".
    - **Overlap is exaggerated.** With {1 - digit_plane_share:.0%} of the variance missing, groups that are distinct in 64 dimensions can appear to touch here.
    - **Distance is not literal.** Two digits far apart on the page are far apart in this 28.5% summary, which is not the same as far apart in the original image space.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. How many components are needed?

    Fit all 64 components and record two quantities for each one:

    - **its own share** of the variance — how much that component alone accounts for;
    - **the running total** — how much the first k components account for together.

    These are shown as two separate figures rather than side by side, because **their vertical axes measure different things**. The individual shares are all small fractions, while the running total climbs to 1. Plotting them side by side, with two different scales, invites the false impression that the two curves are comparable.

    ### Each component's own share
    """)
    return


@app.cell
def _(PCA, builder, digits, np, px):
    digit_pca = PCA().fit(digits.data)
    digit_ratio = digit_pca.explained_variance_ratio_
    digit_cumulative = np.cumsum(digit_ratio)
    components_for_80 = int(np.searchsorted(digit_cumulative, 0.80) + 1)
    components_for_90 = int(np.searchsorted(digit_cumulative, 0.90) + 1)
    tiny_components = int(np.sum(digit_ratio < 0.0005))
    _components = np.arange(1, len(digit_ratio) + 1)
    _figure = px.scatter(
        x=_components,
        y=digit_ratio,
        title="Share of the variance held by each component, on its own",
        labels={
            "x": "Component number",
            "y": "Share of variance held by this component",
        },
    )
    _figure.update_traces(
        marker=dict(size=6, color="#C0392B"),
        hovertemplate="Component %{x}<br>Share: %{y:.4f}<extra></extra>",
    )
    _figure.update_layout(
        width=builder.width, height=380, margin=dict(t=60, b=55, l=70, r=20)
    )
    _figure
    return (
        components_for_80,
        components_for_90,
        digit_cumulative,
        digit_pca,
        digit_ratio,
        tiny_components,
    )


@app.cell(hide_code=True)
def _(digit_ratio, mo):
    mo.md(f"""
    This curve answers **"which single component matters most?"**

    The first component alone holds **{digit_ratio[0]:.1%}** of the total variance — one sixty-fourth of the axes carrying more than a seventh of the variation. The second holds {digit_ratio[1]:.1%}. Then the curve collapses quickly: by around the twentieth component each one adds only a fraction of a percent.

    That collapse is what makes compression possible. Most of the 64 directions carry almost nothing.

    **How to read it:** a tall point near the left means that component is worth keeping. The height here is a share **of the total**, so the values do not add up to 1 unless you sum every one of them.

    **What it cannot tell you:** how many components you need in total. For that you need the running total, in the next figure.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The running total

    This is the curve you actually use to choose a number of components. Each point answers: **if I stop here, how much of the variance have I kept?**
    """)
    return


@app.cell
def _(builder, digit_cumulative, digit_ratio, np, px):
    _components = np.arange(1, len(digit_ratio) + 1)
    _figure = px.line(
        x=_components,
        y=digit_cumulative,
        title="Variance kept by the first k components",
        labels={
            "x": "Number of components kept (k)",
            "y": "Cumulative share of variance kept",
        },
    )
    _figure.update_traces(
        line=dict(color="#2E86C1", width=2.5),
        hovertemplate="k = %{x}<br>Kept: %{y:.3f}<extra></extra>",
    )
    for _level in (0.80, 0.90):
        _figure.add_hline(
            y=_level,
            line_dash="dash",
            line_color="grey",
            annotation_text=f"{_level:.0%}",
            annotation_position="right",
        )
    _figure.update_layout(
        width=builder.width,
        height=380,
        margin=dict(t=60, b=55, l=70, r=40),
        yaxis_range=[0, 1.02],
    )
    _figure
    return


@app.cell(hide_code=True)
def _(
    components_for_80,
    components_for_90,
    digit_ratio,
    mo,
    tiny_components,
):
    mo.md(f"""
    The vertical axis runs from **0 to 1**, so every point is a fraction of the whole. That is what makes the two dashed lines meaningful: they mark the 80% and 90% levels.

    | Target share | Components needed | What is left out |
    | :--- | ---: | ---: |
    | 80% | **{components_for_80}** | {64 - components_for_80} components, holding {1 - 0.80:.0%} |
    | 90% | **{components_for_90}** | {64 - components_for_90} components, holding {1 - 0.90:.0%} |

    So **{components_for_80} components out of 64** — about a fifth of them — carry 80% of the variance. That is real compression: the remaining {64 - components_for_80} components together hold only 20%.

    **How to read it:** find your target share on the vertical axis, move right to the curve, then read the component count down on the horizontal axis. The `np.searchsorted` calls in the code above do exactly that arithmetic.

    **Why the tail is nearly flat.** **{tiny_components}** components have a share below 0.0005 each. Those directions vary so little that they are, in practice, constant. This is where the noise-filtering argument in the next section will come from: whatever lives in those directions is not the shared structure of a digit.

    **The target itself is a judgement.** 70% to 95% are all defensible depending on the purpose: a picture needs little, a filter needs enough to hold the signal, and a compressed representation needs whatever the downstream task tolerates. There is no universal threshold.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Replace `None` so that `student_running_total` holds the cumulative share after the first **13** components, and `student_share_13` holds just the 13th component's own share. Use `digit_cumulative` and `digit_ratio`.
    """)
    return


@app.cell
def _(digit_cumulative, digit_ratio):
    student_running_total = None
    student_share_13 = None
    print("cumulative after 13 components:", student_running_total)
    print("the 13th component's own share:", student_share_13)
    return student_running_total, student_share_13


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
```python
student_running_total = digit_cumulative[12]
student_share_13 = digit_ratio[12]
```
The running total after 13 components is about **0.801**, and the 13th component alone contributes only about **0.005**.

The index is **12**, not 13, because Python counts from zero: `digit_cumulative[0]` is the total after one component. That off-by-one is the most common mistake with these arrays, which is why `np.searchsorted(...) + 1` is used elsewhere in the notebook.

The two numbers together show why the curve flattens: by the 13th component each new one adds only half a percent, so reaching 90% takes {8} more of them.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    The two panels plot the same numbers. Explain why only the right one is used to choose a number of components. Work out your answer, then reveal it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
Because **the left panel answers a different question.**

- The **per-component share** says how much *one* component adds. Comparing the 13th to the 14th tells you which is more important, not how much of the data you have kept.
- The **cumulative share** says how much you keep *in total* if you stop there. That is exactly what "keeping k components" means, so it is the curve that maps a component count to retained information.

A concrete trap: a component in the middle of the left curve might keep only 1%, which looks negligible. But you cannot skip it — components must be taken in order, because each is defined as perpendicular to all the earlier ones. You cannot keep component 40 without 39.

So the choice is always "how many, from the front", and that is a cumulative question.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Adding noise, so the filtering can be measured

    The clean images are kept aside, a noisy copy is made, and PCA is fit on the noisy copy. Because both versions exist, the effect of the filter can be measured against the truth rather than judged by eye.

    The noise is a normal draw with standard deviation 4, using a fixed seed.
    """)
    return


@app.cell
def _(digits, np, plot_digits):
    noisy_digits = np.random.RandomState(42).normal(digits.data, 4)
    _figure = plot_digits(
        noisy_digits,
        4,
        10,
        "The same 40 images with pixel noise added, labelled with their true digit",
        labels=digits.target,
    )
    _figure
    return (noisy_digits,)


@app.cell(hide_code=True)
def _(digits, mo, noisy_digits, np):
    noisy_rmse = float(np.sqrt(((noisy_digits - digits.data) ** 2).mean()))
    mo.md(f"""
    The noise is obvious: speckled pixels, broken strokes, and shapes that are harder to name.

    **Measured damage: root-mean-square error {noisy_rmse:.2f} grey levels** per pixel, against a signal that ranges over 0 to 16. This number is the baseline the reconstructions must beat.

    Two things to note before filtering:

    - The noise is **independent per pixel and per image**. That independence is what makes it land in the low-variance components rather than in the shared structure.
    - The **clean originals still exist**, so every reconstruction below is scored against them. That is a luxury you do not have with real noisy data, and the lesson uses it to check the intuition.
    """)
    return (noisy_rmse,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Reconstructing from four shares

    For each target share, PCA is fitted on the noisy table, the images are reduced and then rebuilt. Two errors are recorded for each: against the noisy input, and against the clean original.

    The second one is the one that matters. An error against the noisy input only says how much you changed it; an error against the clean original says whether the change was in the right direction.
    """)
    return


@app.cell
def _(PCA, digits, noisy_digits, np):
    rebuilt = {}
    noise_components = {}
    _rows = []
    for _share in (0.25, 0.50, 0.60, 0.75):
        _model = PCA(_share).fit(noisy_digits)
        _restored = _model.inverse_transform(_model.transform(noisy_digits))
        rebuilt[_share] = _restored
        noise_components[_share] = int(_model.n_components_)
        _rows.append(
            {
                "share": f"{_share:.0%}",
                "components": int(_model.n_components_),
                "RMSE vs noisy input": float(
                    np.sqrt(((_restored - noisy_digits) ** 2).mean())
                ),
                "RMSE vs clean original": float(
                    np.sqrt(((_restored - digits.data) ** 2).mean())
                ),
            }
        )
    recon_scores = pd.DataFrame(_rows).set_index("share")
    return noise_components, rebuilt, recon_scores


@app.cell
def _(mo, recon_scores):
    mo.Html(recon_scores.round(3).to_html())
    return


@app.cell
def _(digits, plot_digits, rebuilt):
    _figure = plot_digits(
        rebuilt[0.25],
        4,
        10,
        "Rebuilt from 25% of the noisy variance",
        labels=digits.target,
    )
    _figure
    return


@app.cell
def _(digits, plot_digits, rebuilt):
    _figure = plot_digits(
        rebuilt[0.50],
        4,
        10,
        "Rebuilt from 50% of the noisy variance",
        labels=digits.target,
    )
    _figure
    return


@app.cell
def _(digits, plot_digits, rebuilt):
    _figure = plot_digits(
        rebuilt[0.60],
        4,
        10,
        "Rebuilt from 60% of the noisy variance",
        labels=digits.target,
    )
    _figure
    return


@app.cell
def _(digits, plot_digits, rebuilt):
    _figure = plot_digits(
        rebuilt[0.75],
        4,
        10,
        "Rebuilt from 75% of the noisy variance",
        labels=digits.target,
    )
    _figure
    return


@app.cell(hide_code=True)
def _(mo, noise_components):
    mo.md(f"""
    Compare the four panels against the noisy one above, then against the clean one at the top of the lesson.

    - **25%** ({noise_components[0.25]} components) removes almost all the speckle, and also softens the strokes. Some digits lose the detail that made them identifiable. This is **over-smoothing**: real signal went out with the noise.
    - **50%** ({noise_components[0.50]} components) is a clear improvement: the speckle is largely gone and the digits are still clearly themselves.
    - **60%** ({noise_components[0.60]} components) is the interesting comparison. It keeps six components more than 50%, so it should preserve finer stroke detail while letting a little more noise through. Look for whether the extra detail helps or whether it is mostly speckle coming back.
    - **75%** ({noise_components[0.75]} components) looks close to the noisy input, because most of the components were kept. This is **under-filtering**.

    That is the trade the background section described, made visible. The next cell turns it into numbers.
    """)
    return


@app.cell(hide_code=True)
def _(mo, noisy_rmse, recon_scores):
    _best = recon_scores["RMSE vs clean original"].idxmin()
    _best_value = recon_scores.loc[_best, "RMSE vs clean original"]
    _rows = "\n".join(
        f"    | {_share} | {int(recon_scores.loc[_share, 'components'])} | "
        f"{recon_scores.loc[_share, 'RMSE vs noisy input']:.3f} | "
        f"{recon_scores.loc[_share, 'RMSE vs clean original']:.3f} |"
        for _share in recon_scores.index
    )
    _r25 = recon_scores.loc["25%", "RMSE vs clean original"]
    _r50 = recon_scores.loc["50%", "RMSE vs clean original"]
    _r60 = recon_scores.loc["60%", "RMSE vs clean original"]
    _r75 = recon_scores.loc["75%", "RMSE vs clean original"]
    mo.md(f"""
    Read the two error columns together, and against the noise baseline of **{noisy_rmse:.3f}**.

    | Share | Components | vs noisy | vs clean |
    | :--- | ---: | ---: | ---: |
    {_rows}
    | (noisy input) | none | — | {noisy_rmse:.3f} |

    **Every reconstruction beats the noisy input**, which is the filter working. But they are not equally good: the best is **{_best}**, at an error of **{_best_value:.3f}**.

    The shape of that result is the lesson:

    - **At 25%**, the error against the clean original is **{_r25:.3f}**, higher than at 50%. Removing more components did not help; it removed real stroke detail. The error against the *noisy input* is large, which is why "how much did it change?" is the wrong question.
    - **At 50%**, the error against the clean original is **{_r50:.3f}**.
    - **At 60%**, it is **{_r60:.3f}**, against **{_r50:.3f}** at 50%. The two are **effectively tied** — a difference of {abs(_r60 - _r50):.3f}, far smaller than the gaps to the 25% or 75% rows. The extra components brought back some stroke detail and some noise, and the two roughly cancelled.

    - **At 75%**, the error against the noisy input is smallest — the output is closest to its input — but the error against the clean original is **{_r75:.3f}**, higher again, because the noise was kept.

    The 50% and 60% rows are therefore worth reading together. **The best share is a region, not a single value**: anywhere from about half to about two thirds of the variance gives a similar result, and the choice inside that band hardly matters. The 25% and 75% rows are the clear failures on either side.

    That last row is the trap in miniature: **the reconstruction that changed the data least was not the most accurate one.** Judging a filter by how faithful it is to its noisy input rewards doing nothing.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Replace `None` so that `student_best_share` holds the share with the **lowest error against the clean original**, as a string matching the table index.
    """)
    return


@app.cell
def _(recon_scores):
    student_best_share = None
    print(student_best_share)
    return (student_best_share,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
```python
student_best_share = recon_scores["RMSE vs clean original"].idxmin()
```
The answer is **"50%"** — but only just. Its error is 2.692 and the 60% fit scores 2.693, a difference of 0.001.

`idxmin()` returns the index label of the smallest value, which here is one of the strings `"25%"`, `"50%"`, `"60%"`, or `"75%"`. That is why the answer is a string rather than a number.

That near-tie is the point worth taking away: **`idxmin()` picks a winner, but the table shows a plateau.** Reporting "50% is best" without noting that 60% is indistinguishable would overstate what the numbers support.

Note also which column is used. `idxmin()` on the `"RMSE vs noisy input"` column would return `"75%"`, because reconstructing with most components changes the input least. That column measures fidelity to the noise, not accuracy.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    A reconstruction from 75% of the variance changes the noisy input less than one from 50%, yet it is less accurate. Explain how both can be true. Work out your answer, then reveal it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
Because **the two measurements compare against different things.**

- **Changing the input a little** means the output stays close to what you fed in. With 31 components out of 64, only a third of the directions are discarded, so most of the noise survives untouched. The output is a lightly softened copy of the noisy image.
- **Being accurate** means the output is close to the *clean* image. The noisy input is itself far from the clean image, so staying close to it means inheriting its errors.

So a filter that does almost nothing scores well on "faithfulness to the input" and poorly on "closeness to the truth". The two goals point in opposite directions, and only the second one is the point of filtering.

This is a general principle beyond PCA: **when you evaluate a denoiser, you need a reference that is not the noisy input.** That is why this lesson keeps the clean digits. Without them, you could only compare reconstructions to each other, and the one that changed least would look best.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(
    components_for_80,
    components_for_90,
    digit_plane_share,
    mo,
    noise_components,
    noisy_rmse,
    recon_scores,
    tiny_components,
):
    mo.md(f"""
    ## Conclusions

    1. **The fit assumes the pixels are independent coordinates.** The distance PCA maximises has no notion of adjacency, so a digit shifted by one pixel is as different, to the fit, as a different digit. It cannot represent translation and cannot see a stroke. That is the real cost of using a general purpose method on images: not that the arithmetic is wrong, but that the geometry it assumes is not the geometry of a picture. The same indifference is why one fit serves cars, flowers and text equally well.
    2. **Two components give a usable picture, but only a partial one.** The plot keeps {digit_plane_share:.1%} of the variance, and the digits do form coherent regions without the labels being used. What it does not give you is meaningful axes, literal distances, or an honest impression of overlap.
    3. **Read the cumulative curve, not the per-component one, to choose a count.** {components_for_80} components reach 80% of the variance and {components_for_90} reach 90%. You cannot skip components, because each is defined as perpendicular to all previous ones; the choice is always "how many from the front".
    4. **The tail of the curve is nearly flat.** {tiny_components} components have a share below 0.0005 each, so those directions are almost constant. Whatever varies there is not the shared structure of a digit.
    5. **Noise filtering works because noise has no shared direction.** Real stroke structure appears across images, so it lands in the high-variance components. Independent pixel noise spreads thinly through all 64 directions, including the negligible ones that a reconstruction discards.
    6. **There is an optimum, and it is not at either end.** Against the clean original the error was {recon_scores.loc['25%', 'RMSE vs clean original']:.3f} at 25%, {recon_scores.loc['50%', 'RMSE vs clean original']:.3f} at 50%, {recon_scores.loc['60%', 'RMSE vs clean original']:.3f} at 60% and {recon_scores.loc['75%', 'RMSE vs clean original']:.3f} at 75%, against a noisy baseline of {noisy_rmse:.3f}. Too few components ({noise_components[0.25]}) blur away real stroke detail; too many ({noise_components[0.75]}) carry the noise back in. The 50% and 60% fits differ by only {abs(recon_scores.loc['50%', 'RMSE vs clean original'] - recon_scores.loc['60%', 'RMSE vs clean original']):.3f}, so the honest reading is a **band** from about half to two thirds of the variance, not a single winning number.
    7. **Judge a filter by the truth, never by its faithfulness to its input.** The 75% reconstruction changes the noisy input least of the four, and it is worse than either middle fit. Scoring against the noisy input rewards doing nothing, which is why the clean originals were kept: without them this lesson could not have found its own optimum.
    8. **Choosing the share is a judgement, not a calculation.** The target depends on the purpose: a picture needs little, a filter needs enough to hold the signal, and a compressed representation needs whatever the downstream task tolerates. No threshold such as 90% is universal.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check your understanding

    1. An image becomes a point in 64 dimensions. What does PCA's distance measure assume about those 64 coordinates, and what can it therefore not represent about a picture?
    2. Two components keep only part of the variance. What can you still trust in that plot, and what can you not?
    3. Why is the cumulative ratio, rather than the per-component ratio, the curve you use to choose a number of components?
    4. Why can you not keep component 40 while discarding component 39?
    5. Why does independent pixel noise end up in the low-variance components?
    6. The 75% reconstruction changes the noisy input least, yet it is not the most accurate. Explain the apparent contradiction.
    7. Which share performed best on this data, and what does "best" mean here?
    8. Why is there no universal threshold, such as 90%, for the variance to keep?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
1. **It assumes the coordinates are independent.** Straight-line distance treats each pixel as its own axis with no relation to its neighbours, so the fit has no idea that pixel 1 is next to pixel 2. It therefore cannot represent **translation**: the same digit shifted one pixel right is a distant point, because every one of its 64 coordinates moved. Nor can it see a stroke or an edge, only whole-image patterns. The trade is that the same assumption makes the method work on any table of numbers.
2. **Trust the relative placement:** digits forming separate islands really do differ in the 64 original dimensions. **Do not trust the axes**, which are mixtures rather than measurements, the exact distances, or the apparent overlap, which is exaggerated because most of the variance is missing.
3. **Because keeping k components is a cumulative question.** The per-component ratio says how much one component adds; the cumulative ratio says how much you retain in total if you stop there. Only the second maps a component count to retained information.
4. **Because each component is defined as perpendicular to all the earlier ones.** The 40th direction depends on the previous 39 having been removed from consideration, so the set of components is ordered by construction. You always keep a prefix.
5. **Because it has no shared direction.** Real stroke structure appears consistently across images and therefore lines up in a direction where the data genuinely varies. Independent noise varies differently in every image, so it spreads thinly across all 64 directions, most of which have negligible variance.
6. **The two statements compare against different references.** Changing the input little means staying close to the noisy image; being accurate means staying close to the clean image. The noisy input is itself far from the clean one, so fidelity to it means inheriting its errors.
7. **50% and 60% are tied**, at 2.692 and 2.693 against the **clean original**. "Best" here means closest to the truth, not closest to the input, and it is measurable only because the clean digits were kept. A difference of 0.001 does not pick a winner: the honest answer is that anything from about half to two thirds of the variance works equally well on this data.
8. **Because the right share depends on what the components are for.** A visualisation tolerates a lot of loss; a filter needs enough to hold the signal; a compressed representation needs whatever the downstream model can work with. The required share also depends on the noise level and on how the variance is distributed in the particular data.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
    - [Digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
    - [UCI optical recognition of handwritten digits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits)
    - Müller, A. C., and Guido, S. (2017). *Introduction to Machine Learning with Python*. O'Reilly Media. Chapter 3.
    - VanderPlas, J. (2017). *Python Data Science Handbook*. O'Reilly Media. Chapter 5.
    """)
    return


if __name__ == "__main__":
    app.run()

