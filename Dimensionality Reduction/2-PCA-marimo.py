# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "scikit-learn>=1.2",
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
    import seaborn as sns
    from sklearn.datasets import load_digits
    from sklearn.decomposition import PCA

    plt.rcParams["figure.max_open_warning"] = 0
    sns.set_theme(style="whitegrid")

    return PCA, load_digits, mo, np, pd, plt, sns


@app.cell
def _(np, plt):
    def plot_digits(images, rows, cols, title=None):
        """Draw a grid of 8 by 8 digit images.

        Parameters
        ----------
        images : array-like
            One row of 64 pixel values per image, or an (n, 8, 8) array.
        rows, cols : int
            Grid shape for the panel.
        title : str or None
            Optional heading above the grid.

        Returns
        -------
        matplotlib.figure.Figure
            The grid. Axes are hidden, because pixel coordinates carry no
            meaning for a reader.
        """
        figure, axes = plt.subplots(
            rows,
            cols,
            figsize=(8, 0.72 * rows + 0.4),
            gridspec_kw={"hspace": 0.15, "wspace": 0.05},
        )
        for ax, image in zip(np.ravel(axes), images):
            ax.imshow(
                np.asarray(image).reshape(8, 8),
                cmap="binary",
                interpolation="nearest",
                vmin=0,
                vmax=16,
            )
            ax.set_xticks([])
            ax.set_yticks([])
        if title:
            figure.suptitle(title)
        return figure

    return (plot_digits,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # PCA for Pictures and Noise

    ## Objectives

    By the end of this lesson, you should be able to:

    1. Explain why an 8 by 8 image becomes a point with 64 coordinates.
    2. Project that 64-dimensional cloud onto two principal components and explain what the picture does and does not show.
    3. Use the cumulative explained variance ratio to choose a number of components for a target share.
    4. Explain why a component with tiny variance can hold mostly noise, and why that makes reconstruction a filter.
    5. Measure how much noise a reconstruction removes, and explain why a *higher* share is not always better.

    ## Background

    The previous lesson introduced PCA as a way of finding directions of variance. This lesson uses it for the two jobs it is most useful for in practice: **looking** at high-dimensional data, and **cleaning** it.

    ### Pictures as points

    A digit image here is 8 by 8 pixels, and each pixel is a grey level from 0 to 16. That is 64 numbers per image.

    To PCA, an image is simply **a point in 64 dimensions**. Nothing about the grid survives: the 64 values are just coordinates, in the same way that two columns gave a point in a plane last lesson. Flattening the grid into a row of 64 keeps the order fixed, and that order is all PCA sees.

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
def _(digits, plot_digits):
    plot_digits(digits.images, 4, 10, "The first 40 images, as 8 by 8 grids")
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
def _(PCA, digits, pd, plt, sns):
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
    _figure, _axes = plt.subplots(figsize=(8, 5.5))
    sns.scatterplot(
        data=_frame,
        x="Component 1",
        y="Component 2",
        hue="digit",
        s=16,
        ax=_axes,
        linewidth=0,
    )
    _axes.set_title("All 1,797 digits on the first two components")
    _axes.legend(title="Digit", fontsize=7, markerscale=0.8, ncol=2)
    _figure.tight_layout()
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

    Fit all 64 components and look at the shares. The left panel shows each component's own share; the right panel shows the running total, which is the curve you actually use to choose.
    """)
    return


@app.cell
def _(PCA, digits, np, plt):
    digit_pca = PCA().fit(digits.data)
    digit_ratio = digit_pca.explained_variance_ratio_
    digit_cumulative = np.cumsum(digit_ratio)
    components_for_80 = int(np.searchsorted(digit_cumulative, 0.80) + 1)
    components_for_90 = int(np.searchsorted(digit_cumulative, 0.90) + 1)
    tiny_components = int(np.sum(digit_ratio < 0.0005))
    _components = np.arange(1, len(digit_ratio) + 1)
    _figure, _axes = plt.subplots(1, 2, figsize=(10, 3.8))
    _axes[0].plot(_components, digit_ratio, color="crimson", lw=1.4)
    _axes[0].set_title("Share of one component")
    _axes[0].set_ylabel("Share of variance")
    _axes[1].plot(_components, digit_cumulative, color="C0", lw=1.4)
    _axes[1].axhline(0.80, color="0.5", lw=0.8, ls="--")
    _axes[1].axhline(0.90, color="0.5", lw=0.8, ls="--")
    _axes[1].set_title("Running total")
    _axes[1].set_ylabel("Cumulative share")
    for _ax in _axes:
        _ax.set_xlabel("Number of components")
    _figure.tight_layout()
    _figure
    return (
        components_for_80,
        components_for_90,
        digit_cumulative,
        digit_ratio,
        tiny_components,
        digit_pca,
    )


@app.cell(hide_code=True)
def _(
    components_for_80,
    components_for_90,
    digit_ratio,
    mo,
    tiny_components,
):
    mo.md(f"""
    The two curves say different things, and confusing them is a common error.

    - **Left, the per-component share**, falls steeply and then flattens. The first component keeps {digit_ratio[0]:.1%} on its own.
    - **Right, the running total**, rises quickly at first and then levels off. This is the curve you read a component count from.

    | Target share | Components needed |
    | :--- | ---: |
    | 80% | **{components_for_80}** |
    | 90% | **{components_for_90}** |

    So 13 components out of 64 — about a fifth of them — carry 80% of the variance. That is real compression: the remaining 51 components together hold only 20%.

    **The tail is nearly flat.** **{tiny_components}** components have a share below 0.0005 each. Those directions vary so little that they are, for practical purposes, constant. This is where the noise-filtering argument will come from: whatever lives in those directions is not the shared structure of a digit.
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
    plot_digits(noisy_digits, 4, 10, "The same 40 images, with pixel noise added")
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
    ## 4. Reconstructing from three shares

    For each target share, PCA is fitted on the noisy table, the images are reduced and then rebuilt. Two errors are recorded for each: against the noisy input, and against the clean original.

    The second one is the one that matters. An error against the noisy input only says how much you changed it; an error against the clean original says whether the change was in the right direction.
    """)
    return


@app.cell
def _(PCA, digits, noisy_digits, np):
    rebuilt = {}
    noise_components = {}
    _rows = []
    for _share in (0.25, 0.50, 0.75):
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
def _(plot_digits, rebuilt):
    plot_digits(rebuilt[0.25], 4, 10, "Rebuilt from 25% of the noisy variance")
    return


@app.cell
def _(plot_digits, rebuilt):
    plot_digits(rebuilt[0.50], 4, 10, "Rebuilt from 50% of the noisy variance")
    return


@app.cell
def _(plot_digits, rebuilt):
    plot_digits(rebuilt[0.75], 4, 10, "Rebuilt from 75% of the noisy variance")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Compare the three panels against the noisy one above, then against the clean one at the top of the lesson.

    - **25%** removes almost all the speckle, and also softens the strokes. Some digits lose the detail that made them identifiable. This is **over-smoothing**: real signal went out with the noise.
    - **50%** is the middle. The speckle is largely gone and the digits are still clearly themselves.
    - **75%** looks close to the noisy input, because most of the components were kept. This is **under-filtering**.

    That is the trade the background section described, made visible. The next cell turns it into numbers.
    """)
    return


@app.cell(hide_code=True)
def _(mo, noisy_rmse, recon_scores):
    _best = recon_scores["RMSE vs clean original"].idxmin()
    _best_value = recon_scores.loc[_best, "RMSE vs clean original"]
    _r25 = recon_scores.loc["25%", "RMSE vs clean original"]
    _r50 = recon_scores.loc["50%", "RMSE vs clean original"]
    _r75 = recon_scores.loc["75%", "RMSE vs clean original"]
    mo.md(f"""
    Read the two error columns together, and against the noise baseline of **{noisy_rmse:.3f}**.

    | Share | Components | vs noisy | vs clean |
    | :--- | ---: | ---: | ---: |
    | 25% | {int(recon_scores.loc['25%', 'components'])} | {recon_scores.loc['25%', 'RMSE vs noisy input']:.3f} | {_r25:.3f} |
    | 50% | {int(recon_scores.loc['50%', 'components'])} | {recon_scores.loc['50%', 'RMSE vs noisy input']:.3f} | {_r50:.3f} |
    | 75% | {int(recon_scores.loc['75%', 'components'])} | {recon_scores.loc['75%', 'RMSE vs noisy input']:.3f} | {_r75:.3f} |
    | (noisy input) | none | — | {noisy_rmse:.3f} |

    **All three reconstructions beat the noisy input**, which is the filter working. But they are not equally good: the best is **{_best}**, at an error of **{_best_value:.3f}**.

    The shape of that result is the lesson:

    - **At 25%**, the error against the clean original is **higher** than at 50%. Removing more components did not help; it removed real stroke detail. The error against the *noisy input* is large, which is why "how much did it change?" is the wrong question.
    - **At 50%**, the error against the clean original is lowest of the three. This is the balance point for this particular noise level.
    - **At 75%**, the error against the noisy input is smallest — the output is closest to its input — but the error against the clean original is **higher** again, because the noise was kept.

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
The answer is **"50%"**.

`idxmin()` returns the index label of the smallest value, which here is one of the strings `"25%"`, `"50%"`, or `"75%"`. That is why the answer is a string rather than a number.

Note which column is used. `idxmin()` on the `"RMSE vs noisy input"` column would return `"75%"`, because reconstructing with most components changes the input least. That column measures fidelity to the noise, not accuracy.
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
    digit_shapes,
    digit_ratio,
    mo,
    noise_components,
    noisy_rmse,
    recon_scores,
    tiny_components,
):
    _best = recon_scores["RMSE vs clean original"].idxmin()
    _best_value = recon_scores.loc[_best, "RMSE vs clean original"]
    mo.md(f"""
    ## Conclusions

    1. **An image is a point.** Each 8 by 8 digit becomes 64 coordinates, so the dataset is a table of {digit_shapes.split(";")[0].split(",")[0]}. Flattening the grid keeps the pixel order and discards nothing; the geometry of the grid plays no part in the fit.
    2. **Two components give a usable picture, but only a partial one.** The plot keeps {digit_plane_share:.1%} of the variance, and the digits do form coherent regions without the labels being used. What it does not give you is meaningful axes, literal distances, or an honest impression of overlap.
    3. **Read the cumulative curve, not the per-component one, to choose a count.** {components_for_80} components reach 80% of the variance and {components_for_90} reach 90%. You cannot skip components, because each is defined as perpendicular to all previous ones; the choice is always "how many from the front".
    4. **The tail of the curve is nearly flat.** {tiny_components} components have a share below 0.0005 each, so those directions are almost constant. Whatever varies there is not the shared structure of a digit.
    5. **Noise filtering works because noise has no shared direction.** Real stroke structure appears across images, so it lands in the high-variance components. Independent pixel noise spreads thinly through all 64 directions, including the negligible ones that a reconstruction discards.
    6. **More components is not better.** On this data the reconstruction error against the clean original was {recon_scores.loc['25%', 'RMSE vs clean original']:.3f} at 25%, {recon_scores.loc['50%', 'RMSE vs clean original']:.3f} at 50%, and {recon_scores.loc['75%', 'RMSE vs clean original']:.3f} at 75%. The best was **{_best}**, at {_best_value:.3f}, against a noisy baseline of {noisy_rmse:.3f}.
    7. **Too few components destroys signal, too many keeps noise.** At 25% ({noise_components[0.25]} components) the strokes soften and detail is lost. At 75% ({noise_components[0.75]} components) most of the noise survives. The middle is where the filter does its job.
    8. **Never judge a filter by its faithfulness to its input.** Reconstructed from 75%, the output changes the input least, and it is the second-least accurate. Judging by closeness to the noisy input rewards doing nothing. That is why the clean originals were kept in this lesson.
    9. **Choosing the share is a judgement, not a calculation.** The target depends on the purpose: a picture needs little, a filter needs enough to hold the signal, and a compressed representation needs whatever the downstream task tolerates. No threshold is universal.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check your understanding

    1. How does an 8 by 8 image become a point, and how many coordinates does it have?
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
1. **Each pixel becomes a feature.** The 8 by 8 grid is flattened into a row of **64** values, so the dataset is a table with one row per image and 64 columns. The grid layout plays no part in the fit.
2. **Trust the relative placement:** digits forming separate islands really do differ in the 64 original dimensions. **Do not trust the axes**, which are mixtures rather than measurements, the exact distances, or the apparent overlap, which is exaggerated because most of the variance is missing.
3. **Because keeping k components is a cumulative question.** The per-component ratio says how much one component adds; the cumulative ratio says how much you retain in total if you stop there. Only the second maps a component count to retained information.
4. **Because each component is defined as perpendicular to all the earlier ones.** The 40th direction depends on the previous 39 having been removed from consideration, so the set of components is ordered by construction. You always keep a prefix.
5. **Because it has no shared direction.** Real stroke structure appears consistently across images and therefore lines up in a direction where the data genuinely varies. Independent noise varies differently in every image, so it spreads thinly across all 64 directions, most of which have negligible variance.
6. **The two statements compare against different references.** Changing the input little means staying close to the noisy image; being accurate means staying close to the clean image. The noisy input is itself far from the clean one, so fidelity to it means inheriting its errors.
7. **50%**, with the lowest root-mean-square error against the **clean original**. "Best" here means closest to the truth, not closest to the input, and it is measurable only because the clean digits were kept.
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

