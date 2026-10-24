# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
#     "pandas",
#     "plotly",
#     "scikit-learn",
# ]
# ///

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))

    import ClusterVisualizer as cv
    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.express as px
    from sklearn.cluster import MeanShift, estimate_bandwidth
    from sklearn.datasets import make_blobs

    return MeanShift, cv, estimate_bandwidth, make_blobs, mo, np, pd, px


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Mean Shift

    ## Objectives

    - Watch a window walk toward a dense region and stop at a peak.
    - Change the bandwidth and count the clusters that remain.
    - Estimate the bandwidth from the data, then use the same idea on pixel colors.

    ## Background

    Mean shift places a window on the points and moves it to the mean of the points inside it. The window stops when that mean no longer moves. Each stopping place is a cluster center. Points that climb to the same peak share a cluster.

    The bandwidth is the radius of the window. In current scikit-learn, leaving `bandwidth` unset estimates it from nearest-neighbor distances, using a quantile of 0.3. A bandwidth of 1 is a choice you pass in. It is not that default. A smaller window can split a blob. A larger window can merge two blobs.

    The clouds are drawn with `ClusterVisualizer`, including the centers. The color picture is `plotly.express.imshow`, because it is an image rather than a two-column scatter.

    ## Datasets Used

    - **Three centers.** 300 points around (1, 1), (3, 3), and (5, 1), `cluster_std=0.5`, `random_state=0`.
    - **Four blobs.** 400 points, four centers, `cluster_std=0.5`, `random_state=0`.
    - **A built image.** Four colored squares, 80 by 80 pixels, with a little noise. The notebook also opens `toy.jpg` and `Lake_Tahoe.png`. Those files are not in this folder.
    """)
    return


@app.cell
def _(MeanShift, cv, estimate_bandwidth, make_blobs, pd):
    centers = [(1, 1), (3, 3), (5, 1)]
    blobs, _blob_labels = make_blobs(
        n_samples=300, centers=centers, cluster_std=0.5, random_state=0
    )
    blob_view = cv.ClusterVisualizer(pd.DataFrame(blobs, columns=["x", "y"]))
    estimated = float(
        estimate_bandwidth(blobs, quantile=0.3, n_samples=len(blobs), random_state=0)
    )
    blob_fits = {
        "estimate": MeanShift(bandwidth=estimated).fit(blobs),
        "bandwidth 1": MeanShift(bandwidth=1).fit(blobs),
        "bandwidth 0.55": MeanShift(bandwidth=0.55).fit(blobs),
        "bandwidth 2": MeanShift(bandwidth=2).fit(blobs),
    }
    blob_counts = {
        _name: len(_model.cluster_centers_) for _name, _model in blob_fits.items()
    }
    print(f"Estimated bandwidth at quantile 0.3 = {estimated:.2f}")
    print(blob_counts)
    print("Centers at bandwidth 1")
    print(blob_fits["bandwidth 1"].cluster_centers_.round(2))
    blob_view.plot_data(title="Three centers")
    return blob_counts, blob_fits, blob_view, estimated


@app.cell
def _(blob_counts, blob_fits, blob_view, estimated):
    blob_view.plot_clusters(
        blob_fits["estimate"].labels_,
        blob_fits["estimate"].cluster_centers_,
        title=f"Mean Shift, estimated bandwidth {estimated:.2f}, {blob_counts['estimate']} clusters",
    )
    return


@app.cell
def _(blob_counts, blob_fits, blob_view):
    blob_view.plot_clusters(
        blob_fits["bandwidth 1"].labels_,
        blob_fits["bandwidth 1"].cluster_centers_,
        title=f"Mean Shift Clustering (bandwidth = 1), {blob_counts['bandwidth 1']} clusters",
    )
    return


@app.cell
def _(blob_counts, blob_fits, blob_view):
    blob_view.plot_clusters(
        blob_fits["bandwidth 0.55"].labels_,
        blob_fits["bandwidth 0.55"].cluster_centers_,
        title=f"Mean Shift Clustering (bandwidth = 0.55), {blob_counts['bandwidth 0.55']} clusters",
    )
    return


@app.cell
def _(blob_counts, blob_fits, blob_view):
    blob_view.plot_clusters(
        blob_fits["bandwidth 2"].labels_,
        blob_fits["bandwidth 2"].cluster_centers_,
        title=f"Mean Shift Clustering (bandwidth = 2), {blob_counts['bandwidth 2']} clusters",
    )
    return


@app.cell
def _(MeanShift, cv, estimate_bandwidth, make_blobs, pd):
    wide, _wide_labels = make_blobs(
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
    print(f"quantile 0.1 bandwidth = {fine:.2f}, clusters {wide_counts['quantile 0.1']}")
    print(
        f"quantile 0.3 on 100 points, bandwidth = {coarse:.2f}, clusters {wide_counts['quantile 0.3']}"
    )
    wide_view.plot_data(title="Four blobs")
    return coarse, fine, wide_counts, wide_fits, wide_view


@app.cell
def _(fine, wide_counts, wide_fits, wide_view):
    wide_view.plot_clusters(
        wide_fits["quantile 0.1"].labels_,
        wide_fits["quantile 0.1"].cluster_centers_,
        title=f"quantile 0.1, bandwidth {fine:.2f}, {wide_counts['quantile 0.1']} clusters",
    )
    return


@app.cell
def _(coarse, wide_counts, wide_fits, wide_view):
    wide_view.plot_clusters(
        wide_fits["quantile 0.3"].labels_,
        wide_fits["quantile 0.3"].cluster_centers_,
        title=f"quantile 0.3 on 100 points, bandwidth {coarse:.2f}, {wide_counts['quantile 0.3']} clusters",
    )
    return


@app.cell
def _(wide_counts, wide_fits, wide_view):
    wide_view.plot_clusters(
        wide_fits["bandwidth 0.6"].labels_,
        wide_fits["bandwidth 0.6"].cluster_centers_,
        title=f"Mean Shift Clustering (bandwidth = 0.6), {wide_counts['bandwidth 0.6']} clusters",
    )
    return


@app.cell
def _(wide_counts, wide_fits, wide_view):
    wide_view.plot_clusters(
        wide_fits["bandwidth 2.5"].labels_,
        wide_fits["bandwidth 2.5"].cluster_centers_,
        title=f"Mean Shift Clustering (bandwidth = 2.5), {wide_counts['bandwidth 2.5']} clusters",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `estimate_bandwidth` uses a quantile of the nearest-neighbor distances. A smaller quantile, or a sample of the rows, changes that radius. `random_state=0` fixes the sample.

    The picture below is built in memory: four flat colors and a little noise. Each pixel is one row of red, green, and blue. `bin_seeding=True` starts the windows on a grid of the colors instead of on every pixel. `toy.jpg` and `Lake_Tahoe.png` are the files named in the notebook. They are not in this folder, so this lesson uses the built image.
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
    pixels = image.reshape(-1, 3)
    image_bandwidth = float(
        estimate_bandwidth(pixels, quantile=0.2, n_samples=500, random_state=0)
    )
    image_model = MeanShift(bandwidth=image_bandwidth, bin_seeding=True).fit(pixels)
    image_clusters = len(image_model.cluster_centers_)
    image_labels = image_model.labels_.reshape(80, 80)
    print(f"Image shape {image.shape}, pixels {pixels.shape}")
    print(f"Estimated bandwidth = {image_bandwidth:.2f}")
    print(f"{image_clusters} color clusters")
    _fig = px.imshow(image.astype(np.uint8), title="Built image")
    _fig.update_layout(width=420, height=420)
    _fig
    return image_bandwidth, image_clusters, image_labels


@app.cell
def _(image_clusters, image_labels, px):
    _fig = px.imshow(image_labels, title=f"{image_clusters} color clusters")
    _fig.update_layout(width=420, height=420)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Which argument is the radius of the mean-shift window? Replace `None` with its name.
    """)
    return


@app.cell
def _():
    student_bandwidth = None
    print(student_bandwidth)
    return (student_bandwidth,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r'''
```python
student_bandwidth = "bandwidth"
```
**bandwidth.** On the three-center cloud, 0.55, the quantile-0.3 estimate, 1, and 2 do not find the same number of peaks. A bandwidth of 1 is one of those choices. It is not what `MeanShift()` estimates on its own.
''')
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(
    blob_counts,
    coarse,
    estimated,
    fine,
    image_bandwidth,
    image_clusters,
    mo,
    wide_counts,
):
    mo.md(f"""
    ## Conclusions

    **Key takeaways:**

    - On the three-center cloud, the quantile-0.3 estimate is {estimated:.2f} and it finds {blob_counts["estimate"]} clusters. Bandwidth 1 finds {blob_counts["bandwidth 1"]} clusters. Bandwidth 0.55 finds {blob_counts["bandwidth 0.55"]}. Bandwidth 2 finds {blob_counts["bandwidth 2"]}.
    - The centers used to build that cloud are (1, 1), (3, 3), and (5, 1). The printed centers from bandwidth 1 are the estimate of those peaks.
    - On the four-blob cloud, quantile 0.1 gives bandwidth {fine:.2f} and {wide_counts["quantile 0.1"]} clusters. Quantile 0.3 on 100 points gives bandwidth {coarse:.2f} and {wide_counts["quantile 0.3"]} clusters. Bandwidth 0.6 finds {wide_counts["bandwidth 0.6"]} clusters. Bandwidth 2.5 finds {wide_counts["bandwidth 2.5"]} clusters.
    - The built image, bandwidth {image_bandwidth:.2f} with bin seeding, forms {image_clusters} color clusters. Each pixel was a row of red, green, and blue. `toy.jpg` and `Lake_Tahoe.png` are not in this folder.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check your understanding

    1. What does the window walk toward?
    2. What is the bandwidth?
    3. Why is a bandwidth of 1 not the current default?
    4. What does a smaller quantile do to the estimated radius?
    5. What is one row in the image clustering?
    6. What does `bin_seeding=True` change?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
1. **The mean of the points inside it:** The window repeats that step until the mean stops moving. That stopping place is a peak.
2. **The radius of the window:** Points farther away than the bandwidth do not pull the mean.
3. **The estimator computes a radius from the data:** `estimate_bandwidth` with quantile 0.3 is the current default path. Passing `bandwidth=1` is a separate fit, and this run prints a different radius for the estimate.
4. **It gets smaller:** The quantile is a cut of the nearest-neighbor distances. A lower cut is a shorter distance.
5. **One pixel:** Three numbers, red, green, and blue. The picture is those rows folded back into an 80 by 80 grid.
6. **Where the windows start:** They start on a grid of colors instead of on every pixel, which keeps the fit from placing a window on each of the 6,400 pixels.
""")
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - [MeanShift](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html)
    - [estimate_bandwidth](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.estimate_bandwidth.html)
    - Comaniciu, D., and Meer, P. (2002). Mean shift: A robust approach toward feature space analysis. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 24(5), 603–619.
    """)
    return


if __name__ == "__main__":
    app.run()
