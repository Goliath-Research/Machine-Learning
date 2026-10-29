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
    import base64
    import io
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
        base64,
        cv,
        estimate_bandwidth,
        io,
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

    ### Two photographs

    **`toy.JPG`** and **`Lake_Tahoe.png`**, both in this folder. They are teaching copies of ordinary photographs, used here only to show the method on real colour variation. They are not a benchmark and carry no particular licence beyond the course material.

    - `toy.JPG` is 696 by 297 pixels: about 207,000 pixels. Note the **uppercase extension**; the file on disk is `toy.JPG`, not `toy.jpg`, and file names are case-sensitive in a browser.
    - `Lake_Tahoe.png` is 596 by 916 with an alpha channel, so it is first reduced to three colour channels.

    **A practical note.** Mean shift is slow on many points, because it runs a window from every seed. The Lake Tahoe image is therefore **downsampled, keeping every third pixel in each direction**, which cuts the number of points to about a ninth and keeps the fit to a couple of seconds. The toy image is used at full size.
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
    ## 4. Real photographs

    Now the method is applied to pixels instead of \(x, y\) coordinates. Each pixel is a point with three features: red, green, and blue. Clustering those points groups **colours**, not locations.

    These are ordinary photographs, with gradients, texture, and thousands of subtly different colours.

    **The images are embedded in this notebook as base64 text**, not read from files. 
    """)
    return


@app.cell(hide_code=True)
def _(np):
    # Two photographs, embedded so the notebook needs no external files.
    # Each block below is base64-encoded image data, split across lines only to
    # keep it readable. Do not edit these strings.
    toy_b64 = (
        "/9j/4AAQSkZJRgABAQEAwADAAAD/2wBDAAcFBQYFBAcGBQYIBwcIChELCgkJChUPEAwRGBUaGRgV""GBcbHichGx0lHRcYIi4iJSgpKywrGiAvMy8qMicqKyr/2wBDAQcICAoJChQLCxQqHBgcKioqKioq""KioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKir/wAARCAK4ASkDASIA""AhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQA""AAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3""ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWm""p6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEA""AwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSEx"
        "BhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElK""U1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3""uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD6Rooo""oAKKKKACiiigAooooAKKKKACiiigAooooAKKCcdeKqvqVlGxV7yBT6GQf40rpDSb2LVFVV1Oxbpe""W5+kg/xpwv7M/wDL3B/38FHMh8suxYoqD7daf8/UP/fwVIs8T/clRvowNF0LlfYfRRkUUxBRRRQA""UUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFBopGOB""k0AcJ8SvE76VYpYWchSecZZh1Va8da4dmJZiSepJrb8cat/a3ii6lVsxq2xPoK50V83jKznUfZH2""GAw8aVFd2WVlJ71etIWnYDms+BdzCuk0q2yAcVzQXMzrqNRRPbaKjjLs350XlmbCEyQ3EsZH91yK"
        "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"
    )
    lake_b64 = (
        "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""vK51oJTKRk+df//Z"
    )
    return lake_b64, toy_b64


@app.cell
def _(Image, base64, io, np, px, toy_b64):
    toy_array = np.array(
        Image.open(io.BytesIO(base64.b64decode(toy_b64)))
    )
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

    - **The bandwidth is large in absolute terms** ({toy_bandwidth:.1f}), because colour space spans 0 to 255. This is exactly the point made in the blobs sections, at a scale where a bandwidth of 1 would produce tens of thousands of clusters.
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

    The second photograph is 596 by 916 with an alpha channel, about 546,000 pixels. Mean shift on that many points takes close to half a minute, which is too slow for a lesson that reruns.

    So it is **downsampled by taking every third pixel in both directions**, which cuts the count to about 61,000. The picture is softer, and the fit takes a couple of seconds.

    This is a real trade-off you will meet in practice: the algorithm's cost grows with the number of points, and downsampling is the usual way to make image segmentation tractable. The embedded copy of this image was saved after downsampling, so the notebook carries the smaller version rather than the original.
    """)
    return


@app.cell
def _(Image, base64, io, np, px, lake_b64):
    lake_array = np.array(Image.open(io.BytesIO(base64.b64decode(lake_b64))))
    lake_small = lake_array
    _fig = px.imshow(lake_small, title="Lake photograph, downsampled")
    _fig.update_layout(width=520, height=360)
    _fig
    return lake_array, lake_small


@app.cell
def _(lake_array, lake_small):
    lake_flat = lake_small.reshape(-1, 3)
    lake_facts = (
        f"embedded copy {lake_array.shape[0]} by {lake_array.shape[1]} pixels; "
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
    Estimated bandwidth **{lake_bandwidth:.2f}**, and **{lake_clusters} colour clusters** from about 61,000 pixels.

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
**It costs resolution, and with it some colour detail.** Taking every third pixel keeps about a ninth of the points, so small features and fine gradients are lost or averaged away. The segmentation therefore describes broader regions than the full-resolution image would.

**It is still worth it because mean shift's cost grows with the number of points.** The full image has about 546,000 pixels and takes close to half a minute on a normal machine; the downsampled version has about 61,000 and takes a couple of seconds. For a lesson that reruns often, that difference matters.

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
    3. **`bandwidth` is the window radius, and it is not a universal number.** It carries the units of the data: 0.55 to 2.75 were the useful range for blob coordinates, {toy_bandwidth:.1f} for the photograph whose colours run from 0 to 255, and {lake_bandwidth:.1f} for the downsampled lake. A bandwidth of 1 would shatter an image into thousands of colour groups.
    4. **The default is not 1.** `MeanShift()` with no bandwidth calls `estimate_bandwidth` internally, which uses a quantile of 0.3. On the three-centre cloud that gives {estimated:.2f}, and both that value and an explicit 1 produce {default_count} clusters. They agree by coincidence at this scale, not because 1 is the default.
    5. **Too small a bandwidth fragments; too large merges.** On the three-centre cloud, 0.55 produced {blob_counts['bandwidth 0.55']} clusters by splitting blobs from the inside, while 2 produced {blob_counts['bandwidth 2']} by merging two of them. The two failures look different in the plot and neither raises a warning.
    6. **`estimate_bandwidth` depends on choices you make.** A quantile of 0.1 gave {fine:.2f} and found {wide_counts['quantile 0.1']} clusters, the correct count for four blobs. A quantile of 0.3 on 100 sampled rows gave {coarse:.2f} and found {wide_counts['quantile 0.3']}. The estimator is not neutral, so always inspect the value it returns.
    7. **Pixels are points in colour space.** Each pixel contributes one row of red, green, and blue, so the same algorithm groups colours rather than positions. Folding the labels back into the original grid turns the result into an image.
    8. **`bin_seeding=True` makes image segmentation practical.** It snaps starting windows to a coarse grid instead of placing one at every pixel. The toy photograph had over 200,000 pixels; starting a window at each would be far slower.
    9. **Downsampling buys speed at the cost of detail.** The lake image was reduced to every third pixel, cutting the point count to about 61,000 and the runtime to a couple of seconds, which is acceptable because the question concerns large colour regions. The embedded copy is already the downsampled version, so the notebook carries no large file.
    10. **Two different simplifications gave the answers above**: {toy_clusters} colour clusters for the toy photograph and {lake_clusters} for the lake. Reading how much was thrown away is part of interpreting the result.
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
