# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
#     "pandas",
#     "pillow",
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
        "PIL": "pillow",
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

    import ClusterVisualizer as cv
    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.io as pio
    from PIL import Image
    from sklearn.cluster import MeanShift, estimate_bandwidth
    from sklearn.datasets import make_blobs

    pio.templates.default = "plotly_white"
    px.defaults.color_discrete_sequence = px.colors.qualitative.G10

    return (
        Image,
        MeanShift,
        Path,
        cv,
        estimate_bandwidth,
        make_blobs,
        mo,
        np,
        pd,
        px,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Mean Shift

    ## Objectives

    By the end of this lesson, you should be able to:

    1. Explain how a mean-shift window walks toward a density peak and stops there.
    2. Explain what `bandwidth` controls, and why the method does not take an `n_clusters` argument.
    3. Explain what `MeanShift()` does when you pass no bandwidth, and why "the default is 1" is wrong.
    4. Predict how the cluster count responds to a smaller or larger bandwidth, and verify it on two clouds.
    5. Explain how `estimate_bandwidth` turns a quantile of nearest-neighbour distances into a radius.
    6. Apply the same method to pixel colours and read the result as image segmentation.

    ## Background

    **Mean shift** is a clustering method that finds groups by looking for **peaks in density**. Like affinity propagation, it decides the number of groups itself. Unlike DBSCAN, which labels points as noise, mean shift assigns every point to a peak.

    ### The walking window

    Imagine placing a circular window on a cloud of points. The algorithm then repeats one step:

    1. Look at all the points inside the window.
    2. Move the window to the **mean** of those points.
    3. Repeat until the window barely moves.

    When the window stops moving, it has climbed to a dense region: a place where the points around it pull equally in all directions. That stopping place is a **peak**, and it becomes a cluster centre. Every point that climbs to the same peak belongs to the same cluster.

    The name is literal. The window *shifts* to a *mean*, over and over.

    Two consequences follow from that mechanism.

    - **The number of clusters is an output.** You never say how many groups to find. The number of peaks that exist in the data, at the scale of your window, is the number you get.
    - **The result depends entirely on the window size.** This is the parameter the whole lesson turns on.

    ### Bandwidth

    `bandwidth` is the **radius of the window**. It sets the scale at which the algorithm examines the data.

    - **A small bandwidth** means a short window, so only very close points pull on it. Peaks that are close together stay separate, and a single wide blob can be split into several. You get **more, smaller clusters**.
    - **A large bandwidth** means a long reach, so far-away points also pull. Neighbouring peaks merge into one, and you get **fewer, larger clusters**. Push it far enough and everything collapses into a single cluster.

    Because the bandwidth is a radius in the units of your data, it is **not a universal number**. A bandwidth of 1 means one unit on whatever scale you are using. On coordinates ranging over tens of units, 1 is a small window; on pixel colours ranging over 0 to 255, 1 is an extremely small one.

    ### What the default actually is

    This is the point most often stated incorrectly, so it is worth being precise.

    **There is no default bandwidth of 1.** In scikit-learn, if you write `MeanShift()` with no `bandwidth`, the estimator calls `estimate_bandwidth` internally on your data and uses the value it computes. That function takes a **quantile of the distances to each point's nearest neighbours** (the default quantile is 0.3) and returns a radius near that quantile.

    So `MeanShift()` is a data-driven choice, not a constant. On the three-centre cloud below the estimate comes out at **1.39**, which happens to be close enough to 1 that both produce the same three clusters. They agree by coincidence here, not because 1 is the default.

    That coincidence is a useful trap. If you assume "1" and then apply the same reasoning to the pixel data, where colour distances run into the hundreds, you will be badly wrong.

    ### Estimating a bandwidth yourself

    `estimate_bandwidth` is public, so you can call it and inspect the value, which is what this lesson does:

    - **`quantile`** is which proportion of the nearest-neighbour distances to land on. A **smaller quantile** gives a **smaller radius**, because you are taking a lower cut of the distribution.
    - **`n_samples`** is how many rows to sample when estimating. Fewer rows means a noisier estimate.
    - **`random_state`** fixes the sample so the estimate is reproducible.

    Both parameters change the radius, and the radius changes the cluster count. That chain is the practical skill in this lesson.

    ### Images as points

    A colour image gives a natural clustering problem with no coordinates in the usual sense. Each pixel is a point with **three features**: its red, green, and blue values. Cluster those points and you are grouping colours rather than locations.

    The procedure is to flatten the image from `(height, width, 3)` into `(height × width, 3)` so each row is a pixel, cluster the rows, then fold the labels back into the original grid. The result is the image redrawn using only a few colours.

    `bin_seeding=True` is worth understanding here. By default mean shift starts a window at **every point**, which for an image means 200,000 windows. With bin seeding, the starting positions are snapped to a coarse grid of the feature space, so many nearby pixels share one starting point. The answer is nearly the same and the fit is far faster. On large images this is not an optimisation but a requirement.

    ## Datasets used

    All data here comes from the notebook or from files beside it. Nothing is downloaded.

    ### Three centres

    300 points from `sklearn.datasets.make_blobs` around (1, 1), (3, 3), and (5, 1), `cluster_std=0.5`, `random_state=0`. Small and well separated, so the bandwidth sweeps are easy to read. [Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html).

    ### Four blobs

    400 points around four centres, `cluster_std=0.5`, `random_state=0`. Same generator, more groups, and a scale where the estimated bandwidth is noticeably above 1.

    ### A built image

    Four coloured squares, 80 by 80 pixels, with a little noise. Constructed in the notebook so the correct answer is known in advance: exactly four colours, one per quadrant. This is the case to reason about, because you can check the segmentation against the truth.

    ### Two photographs

    **`toy.jpg`** and **`Lake_Tahoe.png`**, both in this folder. They are teaching copies of ordinary photographs, used here only to show the method on real colour variation. They are not a benchmark and carry no particular licence beyond the course material.

    - `toy.jpg` is 696 by 297 pixels: about 207,000 pixels.
    - `Lake_Tahoe.png` is 596 by 916 with an alpha channel, so it is first reduced to three colour channels.

    **A practical note.** Mean shift is slow on many points, because it runs a window from every seed. The Lake Tahoe image is therefore **downsampled by taking every second pixel**, which quarters the number of points and keeps the fit to a few seconds. The toy image is used at full size.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Three centres: how the window behaves

    300 points around three well-separated peaks. Start by looking at the data alone: you can see three groups, and the algorithm has not been told there are three.
    """)
    return


@app.cell
def _(MeanShift, cv, estimate_bandwidth, make_blobs, pd):
    centers = [(1, 1), (3, 3), (5, 1)]
    blobs, _blob_truth = make_blobs(
        n_samples=300, centers=centers, cluster_std=0.5, random_state=0
    )
    blob_view = cv.ClusterVisualizer(pd.DataFrame(blobs, columns=["x", "y"]))
    estimated = float(
        estimate_bandwidth(blobs, quantile=0.3, n_samples=len(blobs), random_state=0)
    )
    default_model = MeanShift().fit(blobs)
    blob_fits = {
        "estimate": MeanShift(bandwidth=estimated).fit(blobs),
        "bandwidth 1": MeanShift(bandwidth=1).fit(blobs),
        "bandwidth 0.55": MeanShift(bandwidth=0.55).fit(blobs),
        "bandwidth 2": MeanShift(bandwidth=2).fit(blobs),
    }
    blob_counts = {
        _name: len(_model.cluster_centers_) for _name, _model in blob_fits.items()
    }
    default_count = len(default_model.cluster_centers_)
    blob_view.plot_data(title="Three hundred points around three peaks")
    return (
        blob_counts,
        blob_fits,
        blob_view,
        centers,
        default_count,
        default_model,
        estimated,
    )


@app.cell(hide_code=True)
def _(default_count, estimated, mo):
    mo.md(f"""
    ### The estimate, and the default

    Calling `estimate_bandwidth` with the default quantile of 0.3 gives **{estimated:.2f}** on this cloud.

    Calling `MeanShift()` with **no bandwidth at all** produces **{default_count} clusters**, and it arrives at that by computing essentially the same estimate internally. There is no hard-coded 1 anywhere in that path.

    Keep the number {estimated:.2f} in mind, because the next plot uses a bandwidth of exactly **1** and gets the same answer. That agreement is a coincidence of this particular dataset, not a rule.
    """)
    return


@app.cell
def _(blob_counts, blob_fits, blob_view, estimated):
    blob_view.plot_clusters(
        blob_fits["estimate"].labels_,
        blob_fits["estimate"].cluster_centers_,
        title=f"Estimated bandwidth {estimated:.2f}: {blob_counts['estimate']} clusters",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Three peaks found, and where they are

    Three clusters, matching the three visible groups, with no `n_clusters` argument anywhere. The black markers are the fitted centres.

    They should land close to the generating centres: (1, 1), (3, 3), and (5, 1). The next table puts them side by side.
    """)
    return


@app.cell
def _(blob_fits, centers, pd):
    center_compare = pd.DataFrame(
        {
            "generating centre": [str(_c) for _c in centers],
            "fitted centre": [
                str(tuple(_row))
                for _row in sorted(
                    blob_fits["estimate"].cluster_centers_.round(2).tolist(),
                    key=lambda _v: _v[1],
                )
            ],
        }
    )
    return (center_compare,)


@app.cell
def _(center_compare, mo):
    mo.Html(center_compare.to_html())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The fitted centres are within a few hundredths of the true peaks. That is what mean shift does well on compact, well-separated data: it finds the density maxima without being told how many exist.

    The small offsets are not errors in the algorithm. A cluster of 100 points scattered with standard deviation 0.5 does not have its sample mean exactly at the generating centre, and mean shift converges to the mean of the points, not to the point you used to generate them.

    ### Try it yourself

    Replace `None` so that `student_centers` holds the fitted cluster centres at the **estimated** bandwidth, and `student_count` holds how many there are.
    """)
    return


@app.cell
def _(blob_fits):
    student_centers = None
    student_count = None
    print(student_count)
    print(student_centers)
    return student_centers, student_count


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
```python
student_centers = blob_fits["estimate"].cluster_centers_
student_count = len(student_centers)
```
The count is **3** and the centres are approximately (0.99, 1.08), (2.94, 2.93), and (4.91, 1.00), near the generating (1, 1), (3, 3), and (5, 1).

`cluster_centers_` is an array with one row per cluster, so its length is the cluster count. Note that the rows come out in the order the algorithm discovered the peaks, not in any meaningful order, which is why the comparison table sorts them.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Sweeping the bandwidth on the same data

    Nothing about the data changes here. Only the window size does, and it decides how many peaks survive.

    ### Smaller than the estimate: 0.55
    """)
    return


@app.cell
def _(blob_counts, blob_fits, blob_view):
    blob_view.plot_clusters(
        blob_fits["bandwidth 0.55"].labels_,
        blob_fits["bandwidth 0.55"].cluster_centers_,
        title=f"Bandwidth 0.55: {blob_counts['bandwidth 0.55']} clusters",
    )
    return


@app.cell(hide_code=True)
def _(blob_counts, mo):
    mo.md(f"""
    **{blob_counts['bandwidth 0.55']} clusters instead of 3.** A window of 0.55 is smaller than the spread inside each blob, so a single blob offers more than one local peak for the windows to climb.

    Look at where the extra centres sit: they are **inside** the original blobs, not between them. That is the signature of too small a bandwidth. It does not merge groups, it fragments them. Nothing has been done incorrectly by the algorithm; the scale of observation is simply finer than the scale of the groups.

    This matters for interpretation. If you saw this plot without knowing the bandwidth, you might conclude the data has five natural groups. The five "groups" are an artefact of the window size.

    ### Larger than the estimate: 2
    """)
    return


@app.cell
def _(blob_counts, blob_fits, blob_view):
    blob_view.plot_clusters(
        blob_fits["bandwidth 2"].labels_,
        blob_fits["bandwidth 2"].cluster_centers_,
        title=f"Bandwidth 2: {blob_counts['bandwidth 2']} clusters",
    )
    return


@app.cell(hide_code=True)
def _(blob_counts, mo):
    mo.md(f"""
    **{blob_counts['bandwidth 2']} clusters instead of 3.** A window of radius 2 reaches across the gaps between blobs, so the points from two groups pull the same window and their peaks merge.

    Notice how the failure differs from the previous one. Too small a bandwidth *added* clusters that were not real; too large a bandwidth *removed* a cluster that was real. Both give a confident-looking plot, and neither raises a warning.

    ### Bandwidth 1, and why it matched

    Bandwidth 1 is neither the estimate (1.39) nor an extreme. It sits between 0.55 and 1.39, and at this scale the density landscape still has exactly three peaks.
    """)
    return


@app.cell
def _(blob_counts, blob_fits, blob_view):
    blob_view.plot_clusters(
        blob_fits["bandwidth 1"].labels_,
        blob_fits["bandwidth 1"].cluster_centers_,
        title=f"Bandwidth 1: {blob_counts['bandwidth 1']} clusters",
    )
    return


@app.cell(hide_code=True)
def _(blob_counts, default_count, estimated, mo):
    mo.md(f"""
    | Fit | Bandwidth | Clusters |
    | :--- | ---: | ---: |
    | `estimate_bandwidth` (quantile 0.3) | {estimated:.2f} | {blob_counts['estimate']} |
    | `MeanShift()` with no bandwidth | internally {estimated:.2f} | {default_count} |
    | explicit `bandwidth=1` | 1.00 | {blob_counts['bandwidth 1']} |
    | explicit `bandwidth=0.55` | 0.55 | {blob_counts['bandwidth 0.55']} |
    | explicit `bandwidth=2` | 2.00 | {blob_counts['bandwidth 2']} |

    The two rows that matter for understanding the default are the first two: they are the **same fit**, reached two ways. The third row is a different bandwidth that happens to give the same count.

    That is why "the default is 1" is misleading even though it looks right on this data. Set the same three fits loose on a dataset with a different scale, such as pixel colours running from 0 to 255, and 1 becomes an absurdly small window while the estimate stays sensible.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Which argument sets the radius of the window, and why is a bandwidth of 1 **not** the default even though `MeanShift(bandwidth=1)` matches `MeanShift()` here? Work out your answer, then reveal it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
The argument is **`bandwidth`**.

It is not the default because `MeanShift()` with no bandwidth calls `estimate_bandwidth` internally, using a quantile of 0.3 of the nearest-neighbour distances. On this cloud that returns **1.39**, not 1.

The two match here only because the estimated 1.39 and an explicit 1 both sit inside the same range where the density landscape has three peaks. Below about 0.55 the blobs fragment into five; above about 2 two of them merge. Between those limits the count stays at three, so many bandwidths give the same answer.

The mistake would show up immediately on a different scale. Pixel colours span 0 to 255, and the estimated bandwidth there is **around 43**, so a bandwidth of 1 would shatter the image into thousands of colour groups.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Four blobs, and estimating the bandwidth

    A second cloud, with four groups rather than three. This section concentrates on the estimation step, because that is where the parameters are easiest to confuse.

    `estimate_bandwidth` has two settings that both change the radius:

    - **`quantile`**, which proportion of the nearest-neighbour distances to land on;
    - **`n_samples`**, how many rows to sample for the estimate.

    The next two fits change `quantile` from 0.1 to 0.3 while also reducing the sample to 100 rows.
    """)
    return


@app.cell
def _(MeanShift, cv, estimate_bandwidth, make_blobs, pd):
    wide, _wide_truth = make_blobs(
        n_samples=400, centers=4, cluster_std=0.5, random_state=0
    )
    wide_view = cv.ClusterVisualizer(pd.DataFrame(wide, columns=["x", "y"]))
    fine = float(estimate_bandwidth(wide, quantile=0.1, n_samples=400, random_state=0))
    coarse = float(
        estimate_bandwidth(wide, quantile=0.3, n_samples=100, random_state=0)
    )
    wide_fits = {
        "quantile 0.1": MeanShift(bandwidth=fine).fit(wide),
        "quantile 0.3": MeanShift(bandwidth=coarse).fit(wide),
        "bandwidth 0.6": MeanShift(bandwidth=0.6).fit(wide),
        "bandwidth 2.5": MeanShift(bandwidth=2.5).fit(wide),
    }
    wide_counts = {
        _name: len(_model.cluster_centers_) for _name, _model in wide_fits.items()
    }
    wide_view.plot_data(title="Four hundred points around four peaks")
    return coarse, fine, wide_counts, wide_fits, wide_view


@app.cell(hide_code=True)
def _(coarse, fine, mo):
    mo.md(f"""
    The two estimates are far apart, and the reason is the quantile.

    - **quantile 0.1** lands on the 10th percentile of nearest-neighbour distances, giving **{fine:.2f}**.
    - **quantile 0.3** lands on the 30th percentile, giving **{coarse:.2f}**.

    The larger quantile is a higher cut of the same distribution of distances, so it returns a bigger radius. That is the mechanism: the quantile is not a quality setting, it is a position in the distance distribution.

    A bandwidth of {coarse:.2f} is nearly four times {fine:.2f}, which is a large enough change to alter the number of peaks. The next two plots show the result.
    """)
    return


@app.cell
def _(fine, wide_counts, wide_fits, wide_view):
    wide_view.plot_clusters(
        wide_fits["quantile 0.1"].labels_,
        wide_fits["quantile 0.1"].cluster_centers_,
        title=f"quantile 0.1, bandwidth {fine:.2f}: {wide_counts['quantile 0.1']} clusters",
    )
    return


@app.cell
def _(coarse, wide_counts, wide_fits, wide_view):
    wide_view.plot_clusters(
        wide_fits["quantile 0.3"].labels_,
        wide_fits["quantile 0.3"].cluster_centers_,
        title=f"quantile 0.3, bandwidth {coarse:.2f}: {wide_counts['quantile 0.3']} clusters",
    )
    return


@app.cell(hide_code=True)
def _(coarse, fine, wide_counts, mo):
    mo.md(f"""
    - **quantile 0.1, bandwidth {fine:.2f}:** **{wide_counts['quantile 0.1']} clusters**, which is correct. The window is small enough to keep the four blobs apart but large enough not to fragment them.
    - **quantile 0.3, bandwidth {coarse:.2f}:** **{wide_counts['quantile 0.3']} clusters**, one too few. The window is now wide enough to bridge two neighbouring blobs.

    This is the single most useful caution in the lesson. The estimation step **looks** automatic: you call a function, it gives you a number, you pass it on. But the number it gives depends on a quantile you choose, and here a change from 0.1 to 0.3 turns a correct answer into an incorrect one.

    "Let scikit-learn choose the bandwidth" therefore means "let scikit-learn choose it **with the default quantile of 0.3**", which is not guaranteed to suit your data.

    ### Explicit bandwidths, for comparison

    Two hand-picked values, one below the estimates and one above.
    """)
    return


@app.cell
def _(wide_counts, wide_fits, wide_view):
    wide_view.plot_clusters(
        wide_fits["bandwidth 0.6"].labels_,
        wide_fits["bandwidth 0.6"].cluster_centers_,
        title=f"Bandwidth 0.6: {wide_counts['bandwidth 0.6']} clusters",
    )
    return


@app.cell
def _(wide_counts, wide_fits, wide_view):
    wide_view.plot_clusters(
        wide_fits["bandwidth 2.5"].labels_,
        wide_fits["bandwidth 2.5"].cluster_centers_,
        title=f"Bandwidth 2.5: {wide_counts['bandwidth 2.5']} clusters",
    )
    return


@app.cell(hide_code=True)
def _(wide_counts, mo):
    mo.md(f"""
    **Bandwidth 0.6 gives {wide_counts['bandwidth 0.6']} clusters**: one too many. It is close to the quantile-0.1 estimate of 0.74 but small enough that one of the four blobs develops two peaks. This is a good reminder that the boundary between "right" and "one too many" is narrow.

    **Bandwidth 2.5 gives {wide_counts['bandwidth 2.5']} clusters**: one too few, for the same merging reason as the quantile-0.3 fit.

    A useful habit when you see a mean-shift result with an unexpected count: check whether the extra clusters sit **inside** a visible group (bandwidth too small) or whether visible groups have **merged** (bandwidth too large). The two failures have different signatures in the plot.

    ### Try it yourself

    Replace `None` so that `student_sweep` holds a DataFrame with one row per fit in `wide_fits`, showing the cluster count. The index should be the fit names.
    """)
    return


@app.cell
def _(pd, wide_fits):
    student_sweep = None
    student_sweep
    return (student_sweep,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
```python
student_sweep = pd.DataFrame(
    {"clusters": {name: len(m.cluster_centers_) for name, m in wide_fits.items()}}
)
student_sweep
```
The counts are **4** (quantile 0.1), **3** (quantile 0.3), **5** (bandwidth 0.6), and **3** (bandwidth 2.5).

Building the dictionary with the fit names as keys makes them the DataFrame index, which reads better than a numbered index. `len(m.cluster_centers_)` is the cluster count, the same expression used throughout the lesson.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. A built image, where the answer is known

    Now the method is applied to pixels instead of \(x, y\) coordinates. Each pixel is a point with three features: red, green, and blue. Clustering those points groups **colours**, not locations.

    This first image is built in the notebook, so the right answer is known: four coloured squares, one per quadrant, 80 by 80 pixels, with a little noise added.
    """)
    return


@app.cell
def _(MeanShift, estimate_bandwidth, np, px):
    rng = np.random.default_rng(0)
    image = np.zeros((80, 80, 3), dtype=float)
    image[:40, :40] = (210, 40, 40)
    image[:40, 40:] = (40, 160, 70)
    image[40:, :40] = (40, 80, 200)
    image[40:, 40:] = (230, 190, 40)
    image = np.clip(image + rng.normal(0, 8, image.shape), 0, 255)
    built_pixels = image.reshape(-1, 3)
    image_bandwidth = float(
        estimate_bandwidth(built_pixels, quantile=0.2, n_samples=500, random_state=0)
    )
    image_model = MeanShift(
        bandwidth=image_bandwidth, bin_seeding=True
    ).fit(built_pixels)
    image_clusters = len(image_model.cluster_centers_)
    image_labels = image_model.labels_.reshape(80, 80)
    _fig = px.imshow(image.astype(np.uint8), title="Built image: four colour quadrants")
    _fig.update_layout(width=420, height=420)
    _fig
    return (
        built_pixels,
        image_bandwidth,
        image_clusters,
        image_labels,
        image_model,
    )


@app.cell(hide_code=True)
def _(image_bandwidth, mo):
    mo.md(f"""
    The image is 80 by 80, which is **6,400 pixels**, and each becomes one row of three numbers. `estimate_bandwidth` returns **{image_bandwidth:.2f}**.

    Compare that with the blobs sections, where the bandwidths were between 0.55 and 2.75. Here it is an order of magnitude larger, because colour values run from 0 to 255 while blob coordinates ran over a handful of units. **The bandwidth always has the units of the data**, and that is why it cannot be reused across datasets.

    The next plot shows the segmentation: each pixel recoloured by the cluster it was assigned to.
    """)
    return


@app.cell
def _(image_clusters, image_labels, px):
    _fig = px.imshow(image_labels, title=f"Segmentation: {image_clusters} colour clusters")
    _fig.update_layout(width=420, height=420)
    _fig
    return


@app.cell(hide_code=True)
def _(image_clusters, image_model, mo):
    mo.md(f"""
    **{image_clusters} colour clusters**, which is exactly the four colours that were painted. Mean shift recovered the segmentation without being told there were four.

    The fitted centres land on the four pure colours:
    {", ".join(str(tuple(int(_v) for _v in _row)) for _row in image_model.cluster_centers_.round(0).tolist())}
    That is the ideal case. The colours are far apart relative to the noise, so every pixel climbs to the peak of its own quadrant.

    ### Why `bin_seeding` matters here

    Without `bin_seeding`, mean shift starts a window at **every one of the 6,400 pixels** and shifts each one. With it, starting positions snap to a coarse grid of the colour space, so many pixels share one starting point. On an image this small both are fast, but the difference becomes decisive at photo resolution, which is the next section.

    ### Try it yourself

    Replace `None` so that `student_image_centers` holds the four fitted colour centres as integers, and `student_image_count` holds how many there are.
    """)
    return


@app.cell
def _(image_model, np):
    student_image_centers = None
    student_image_count = None
    print(student_image_count)
    print(student_image_centers)
    return student_image_centers, student_image_count


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
```python
student_image_centers = image_model.cluster_centers_.round(0).astype(int)
student_image_count = len(student_image_centers)
```
The count is **4**, and the centres are approximately (210, 40, 40), (40, 80, 200), (230, 190, 40), and (40, 160, 70), the four colours used to paint the quadrants.

`.round(0).astype(int)` converts the floating-point means back into pixel values, which is what you would use to reconstruct the image. This is the same `cluster_centers_` attribute used on the blobs; the only difference is that each row has three numbers instead of two.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Real photographs

    The built image was designed to be easy. These are ordinary photographs, with gradients, texture, and thousands of subtly different colours. Both files sit in this folder and are loaded directly.

    ### A toy photograph, at full size

    696 by 297 pixels, about **207,000** pixels. Every one of them becomes a row of three numbers.
    """)
    return


@app.cell
def _(Image, Path, np, px):
    toy_array = np.array(Image.open(Path(__file__).resolve().parent / "toy.jpg"))
    _fig = px.imshow(toy_array, title="Original photograph")
    _fig.update_layout(width=460, height=380)
    _fig
    return (toy_array,)


@app.cell
def _(toy_array):
    toy_flat = toy_array.reshape(-1, 3)
    toy_shape = toy_array.shape
    toy_pixels = f"{toy_shape[0]} by {toy_shape[1]} pixels, flattened to {toy_flat.shape[0]:,} rows"
    return toy_flat, toy_pixels, toy_shape


@app.cell
def _(toy_pixels):
    toy_pixels
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Flattening is the whole trick. The image is a `(696, 297, 3)` block of numbers; `reshape(-1, 3)` turns it into a table of **206,712 rows and 3 columns**, where each row is one pixel's red, green, and blue. That table is what the clustering sees.

    The reverse operation, `reshape(696, 297)`, turns the resulting labels back into a picture.

    Now the fit. This takes several seconds even with `bin_seeding`, because there are over 200,000 points.
    """)
    return


@app.cell
def _(MeanShift, estimate_bandwidth, toy_flat):
    toy_bandwidth = float(
        estimate_bandwidth(toy_flat, quantile=0.2, n_samples=500, random_state=0)
    )
    toy_model = MeanShift(bandwidth=toy_bandwidth, bin_seeding=True).fit(toy_flat)
    toy_clusters = len(toy_model.cluster_centers_)
    toy_labels = toy_model.labels_.reshape(toy_shape[0], toy_shape[1])
    return toy_bandwidth, toy_clusters, toy_labels


@app.cell
def _(px, toy_clusters, toy_labels):
    _fig = px.imshow(toy_labels, title=f"Segmented into {toy_clusters} colour clusters")
    _fig.update_layout(width=460, height=380)
    _fig
    return


@app.cell(hide_code=True)
def _(mo, toy_bandwidth, toy_clusters):
    mo.md(f"""
    Estimated bandwidth **{toy_bandwidth:.2f}**, and **{toy_clusters} colour clusters** from 206,712 pixels.

    Compare the two pictures. The segmented version keeps the shapes that matter, the figure and its main colour regions, and throws away the subtle shading. That is exactly what clustering does: it replaces many nearly-identical colours with one representative.

    Two observations worth making.

    - **The bandwidth is large in absolute terms** ({toy_bandwidth:.1f}), because colour space spans 0 to 255. This is the same lesson from the built image, at a scale where a bandwidth of 1 would produce tens of thousands of clusters.
    - **Fewer clusters does not mean less information about shape.** The segmented image still shows the outline clearly. The lost information is *tonal*: the number of distinct colours, not the boundaries between objects.

    ### Try it yourself

    Replace `None` so that `student_toy_pixels` holds the number of pixel rows that were clustered, and `student_toy_clusters` holds the number of colour clusters found.
    """)
    return


@app.cell
def _(toy_flat, toy_model):
    student_toy_pixels = None
    student_toy_clusters = None
    print("pixels:", student_toy_pixels)
    print("clusters:", student_toy_clusters)
    return student_toy_clusters, student_toy_pixels


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
```python
student_toy_pixels = toy_flat.shape[0]
student_toy_clusters = len(toy_model.cluster_centers_)
```
There are **206,712** pixels and **4** colour clusters.

`toy_flat.shape[0]` is the number of rows, which equals height times width. Each was clustered as a point in three dimensions, so the fit processed over 200,000 points to produce 4 groups. That is a striking ratio, and it is why `bin_seeding=True` matters: without it the algorithm starts a window at every one of those points.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A lake photograph, downsampled

    The second photograph is **596 by 916** with an alpha channel, about 546,000 pixels. Mean shift on that many points takes close to half a minute, which is too slow for a lesson that reruns.

    So it is **downsampled by taking every second pixel in both directions**, which quarters the count to about 136,000. The picture is slightly softer, and the fit takes a few seconds.

    This is a real trade-off you will meet in practice: the algorithm's cost grows with the number of points, and downsampling is the usual way to make image segmentation tractable.
    """)
    return


@app.cell
def _(Image, Path, np, px):
    lake_array = np.array(Image.open(Path(__file__).resolve().parent / "Lake_Tahoe.png"))[
        :, :, :3
    ]
    lake_small = lake_array[::2, ::2]
    _fig = px.imshow(lake_small, title="Lake photograph, downsampled by 2")
    _fig.update_layout(width=520, height=360)
    _fig
    return lake_array, lake_small


@app.cell
def _(lake_array, lake_small):
    lake_flat = lake_small.reshape(-1, 3)
    lake_facts = (
        f"original {lake_array.shape[0]} by {lake_array.shape[1]} with 4 channels; "
        f"downsampled to {lake_small.shape[0]} by {lake_small.shape[1]}; "
        f"clustered rows: {lake_flat.shape[0]:,}"
    )
    lake_facts
    return lake_facts, lake_flat


@app.cell
def _(MeanShift, estimate_bandwidth, lake_flat):
    lake_bandwidth = float(
        estimate_bandwidth(lake_flat, quantile=0.2, n_samples=500, random_state=0)
    )
    lake_model = MeanShift(bandwidth=lake_bandwidth, bin_seeding=True).fit(lake_flat)
    lake_clusters = len(lake_model.cluster_centers_)
    return lake_bandwidth, lake_clusters, lake_model


@app.cell
def _(lake_model, lake_small):
    lake_labels = lake_model.labels_.reshape(lake_small.shape[0], lake_small.shape[1])
    return (lake_labels,)


@app.cell
def _(lake_clusters, lake_labels, px):
    _fig = px.imshow(
        lake_labels, title=f"Lake photograph segmented into {lake_clusters} colour clusters"
    )
    _fig.update_layout(width=520, height=360)
    _fig
    return


@app.cell(hide_code=True)
def _(lake_bandwidth, lake_clusters, lake_model, mo):
    _centres = lake_model.cluster_centers_.round(0).astype(int).tolist()
    _sizes = __import__("numpy").bincount(lake_model.labels_)
    _share = (_sizes / _sizes.sum() * 100).round(1)
    _lines = "\n".join(
        f"    - RGB {tuple(_c)}: {_s:.1f}% of the pixels"
        for _c, _s in zip(_centres, _share)
    )
    mo.md(f"""
    Estimated bandwidth **{lake_bandwidth:.2f}**, and **{lake_clusters} colour clusters** from about 136,000 pixels.

    The three cluster centres, and how much of the image each covers:

    {_lines}

    Read the actual colours rather than assuming what they "should" be. A pale near-white region dominates, which is the bright sky; a dark teal region captures the water and shadowed land; and a mid brown region carries the warmer tones. Three clusters for a landscape is a strong simplification, and that is the point: mean shift has kept the large colour regions and discarded the shading within them.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    The lake photograph was downsampled to keep the fit fast. Explain what downsampling costs and why it is still worth doing here. Work out your answer, then reveal it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
**It costs resolution, and with it some colour detail.** Taking every second pixel keeps a quarter of the points, so small features and fine gradients are lost or averaged away. The segmentation therefore describes broader regions than the full-resolution image would.

**It is still worth it because mean shift's cost grows with the number of points.** The full image has about 546,000 pixels and takes close to half a minute on a normal machine; the downsampled version has about 136,000 and takes a few seconds. For a lesson that reruns often, that difference matters.

The trade-off is reasonable because **the question being asked is about large colour regions, not fine detail.** If you needed to segment individual objects at the pixel level, downsampling would be the wrong choice and you would pay the cost, or use a faster method.

This is also why `bin_seeding=True` is used: it reduces the number of starting points, which is the other lever on cost besides reducing the number of data points.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(
    blob_counts,
    coarse,
    default_count,
    estimated,
    fine,
    image_bandwidth,
    image_clusters,
    lake_bandwidth,
    lake_clusters,
    mo,
    toy_bandwidth,
    toy_clusters,
    wide_counts,
):
    mo.md(f"""
    ## Conclusions

    1. **A mean-shift window walks to a peak.** It repeatedly moves to the mean of the points inside it and stops where that mean no longer moves. That stopping point is a cluster centre, and every point climbing to the same peak shares a cluster.
    2. **There is no `n_clusters` argument, because the peaks are the output.** On the three-centre cloud the method found {blob_counts['estimate']} clusters matching the {blob_counts['estimate']} visible groups, and the fitted centres landed within a few hundredths of the generating centres (1, 1), (3, 3), and (5, 1).
    3. **`bandwidth` is the window radius, and it is not a universal number.** It carries the units of the data: 0.55 to 2.75 were the useful range for blob coordinates, {image_bandwidth:.2f} for 0-to-255 colours in the built image, {toy_bandwidth:.1f} for the photograph, and {lake_bandwidth:.1f} for the downsampled lake.
    4. **The default is not 1.** `MeanShift()` with no bandwidth calls `estimate_bandwidth` internally, which uses a quantile of 0.3. On the three-centre cloud that gives {estimated:.2f}, and both that value and an explicit 1 produce {default_count} clusters. They agree by coincidence at this scale, not because 1 is the default.
    5. **Too small a bandwidth fragments; too large merges.** On the three-centre cloud, 0.55 produced {blob_counts['bandwidth 0.55']} clusters by splitting blobs from the inside, while 2 produced {blob_counts['bandwidth 2']} by merging two of them. The two failures look different in the plot and neither raises a warning.
    6. **`estimate_bandwidth` depends on choices you make.** A quantile of 0.1 gave {fine:.2f} and found {wide_counts['quantile 0.1']} clusters, the correct count for four blobs. A quantile of 0.3 on 100 sampled rows gave {coarse:.2f} and found {wide_counts['quantile 0.3']}. The estimator is not neutral, so always inspect the value it returns.
    7. **Pixels are points in colour space.** Each pixel contributes one row of red, green, and blue, so the same algorithm groups colours rather than positions. Folding the labels back into the original grid turns the result into an image.
    8. **`bin_seeding=True` makes image segmentation practical.** It snaps starting windows to a coarse grid instead of placing one at every pixel. The toy photograph had over 200,000 pixels; starting a window at each would be far slower.
    9. **Downsampling buys speed at the cost of detail.** The lake image was reduced to every second pixel, cutting the point count to about 136,000 and the runtime to a few seconds, which is acceptable because the question concerns large colour regions.
    10. **Three different simplifications gave the answers above**: {image_clusters} colours for the built image with known truth, {toy_clusters} for the toy photograph, and {lake_clusters} for the lake. Reading how much was thrown away is part of interpreting the result.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check your understanding

    1. What does the window move toward, and when does it stop?
    2. Why does the method have no `n_clusters` argument?
    3. What does `bandwidth` control, and in what units?
    4. What does `MeanShift()` do when no bandwidth is given, and why is a bandwidth of 1 not the default?
    5. What happens to the cluster count as the bandwidth shrinks, and what happens as it grows?
    6. What does a smaller `quantile` in `estimate_bandwidth` return, and why?
    7. How is an image turned into data for clustering, and how is the result turned back into an image?
    8. What does `bin_seeding=True` do, and why does it matter for photographs?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
1. **Toward the mean of the points inside the window.** It stops when that mean stops moving appreciably. The stopping point is a local density peak, and it becomes a cluster centre.
2. **Because the peaks are discovered, not requested.** The number of clusters is the number of density peaks at the scale of the window, so it is an output of the fit.
3. **The radius of the window**, which sets the scale of observation. It is measured in the units of the data, so it differs entirely between coordinates near 1 and colour values up to 255.
4. **It calls `estimate_bandwidth` internally**, using a quantile of 0.3 of the nearest-neighbour distances. A bandwidth of 1 is therefore not the default; on the three-centre cloud the estimate is 1.39, and the two agree only because both sit in a range that yields three peaks.
5. **It rises as the bandwidth shrinks** (blobs split from within) **and falls as it grows** (groups merge). On the three-centre cloud, 0.55 gave 5 clusters and 2 gave 2.
6. **A smaller radius.** The quantile is a position in the distribution of nearest-neighbour distances, so a lower quantile takes a lower cut and returns a shorter distance.
7. **Flatten it** from height by width by three channels into one row per pixel with three columns, then cluster the rows. **Reshape the labels** back to height by width to get an image.
8. **It snaps the starting windows to a coarse grid** of the feature space instead of one per point. On a photograph with over 200,000 pixels that is the difference between a few seconds and far longer.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - [MeanShift](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html)
    - [estimate_bandwidth](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.estimate_bandwidth.html)
    - [Mean shift, scikit-learn user guide](https://scikit-learn.org/stable/modules/clustering.html#mean-shift)
    - Comaniciu, D., and Meer, P. (2002). Mean shift: A robust approach toward feature space analysis. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 24(5), 603–619.
    - VanderPlas, J. (2017). *Python Data Science Handbook*. O'Reilly Media. Chapter 5.
    """)
    return


if __name__ == "__main__":
    app.run()
