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
    from sklearn.cluster import MeanShift, estimate_bandwidth
    from sklearn.datasets import make_blobs

    return MeanShift, estimate_bandwidth, make_blobs


@app.cell
def _(np, pd, sns):
    def draw_clusters(axes, x, y, labels, title, centers=None):
        frame = pd.DataFrame(
            {"x": np.asarray(x), "y": np.asarray(y), "cluster": np.asarray(labels).astype(str)}
        )
        sns.scatterplot(data=frame, x="x", y="y", hue="cluster", s=22, ax=axes, legend=False)
        if centers is not None:
            axes.scatter(centers[:, 0], centers[:, 1], c="black", marker="x", s=70)
        axes.set_title(title)

    return (draw_clusters,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Mean-Shift Clustering

    ## Objectives

    - Let a window walk toward dense regions until the number of clusters appears on its own.
    - Compare a bandwidth estimated from the points with smaller and larger fixed windows.
    - Segment a small color image by clustering its pixels.

    ## Background

    Mean shift places a window on the points and moves it toward the average of the points inside it. The walk stops at a dense peak. Points that climb to the same peak share a cluster. The bandwidth is the radius of that window.

    In current scikit-learn, leaving `bandwidth` unset estimates it from the nearest-neighbor distances, using a quantile of 0.3. A bandwidth of 1 is a choice you pass in. It is not that default. A smaller window can split a blob. A larger window can merge two blobs.

    ## Datasets Used

    - **Three centers.** 300 points around (1, 1), (3, 3), and (5, 1), `cluster_std=0.5`, `random_state=0`.
    - **Four blobs.** 400 points, four centers, `cluster_std=0.5`, `random_state=0`.
    - **A built image.** Four colored squares, 80 by 80 pixels, with a little noise. The notebook also opens `toy.jpg` and `Lake_Tahoe.png`. Those files are not in this folder.
    """)
    return


@app.cell
def _(MeanShift, draw_clusters, estimate_bandwidth, make_blobs, plt):
    centers = [(1, 1), (3, 3), (5, 1)]
    blobs, _blob_labels = make_blobs(
        n_samples=300, centers=centers, cluster_std=0.5, random_state=0
    )
    estimated = float(
        estimate_bandwidth(blobs, quantile=0.3, n_samples=len(blobs), random_state=0)
    )
    fits = {
        f"estimate {estimated:.2f}": MeanShift(bandwidth=estimated).fit(blobs),
        "bandwidth 1": MeanShift(bandwidth=1).fit(blobs),
        "bandwidth 0.55": MeanShift(bandwidth=0.55).fit(blobs),
        "bandwidth 2": MeanShift(bandwidth=2).fit(blobs),
    }
    blob_counts = {name: len(model.cluster_centers_) for name, model in fits.items()}
    _figure, _axes = plt.subplots(2, 2, figsize=(9, 8))
    for _ax, (_name, _model) in zip(_axes.ravel(), fits.items()):
        draw_clusters(
            _ax,
            blobs[:, 0],
            blobs[:, 1],
            _model.labels_,
            f"{_name}: {blob_counts[_name]} clusters",
            centers=_model.cluster_centers_,
        )
    _figure.tight_layout()
    print(f"Estimated bandwidth = {estimated:.2f}")
    print(f"Centers used to build the cloud: {centers}")
    print("Centers at bandwidth 1")
    print(_model_centers := fits["bandwidth 1"].cluster_centers_.round(2))
    return _figure


@app.cell
def _(MeanShift, draw_clusters, estimate_bandwidth, make_blobs, plt):
    wide, _wide_labels = make_blobs(
        n_samples=400, centers=4, cluster_std=0.5, random_state=0
    )
    fine = float(estimate_bandwidth(wide, quantile=0.1, n_samples=400, random_state=0))
    coarse = float(estimate_bandwidth(wide, quantile=0.3, n_samples=100, random_state=0))
    wide_fits = {
        f"quantile 0.1 ({fine:.2f})": MeanShift(bandwidth=fine).fit(wide),
        f"quantile 0.3 ({coarse:.2f})": MeanShift(bandwidth=coarse).fit(wide),
        "bandwidth 0.6": MeanShift(bandwidth=0.6).fit(wide),
        "bandwidth 2.5": MeanShift(bandwidth=2.5).fit(wide),
    }
    wide_counts = {name: len(model.cluster_centers_) for name, model in wide_fits.items()}
    _figure, _axes = plt.subplots(2, 2, figsize=(9, 8))
    for _ax, (_name, _model) in zip(_axes.ravel(), wide_fits.items()):
        draw_clusters(
            _ax,
            wide[:, 0],
            wide[:, 1],
            _model.labels_,
            f"{_name}: {wide_counts[_name]}",
            centers=_model.cluster_centers_,
        )
    _figure.tight_layout()
    print(f"quantile 0.1 bandwidth = {fine:.2f}")
    print(f"quantile 0.3 bandwidth, 100 points = {coarse:.2f}")
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `estimate_bandwidth` uses a quantile of the nearest-neighbor distances. A smaller quantile, or a sample of the rows, changes that radius. `random_state=0` fixes the sample.

    The picture below is built in memory: four flat colors and a little noise. Each pixel is one row of red, green, and blue. `bin_seeding=True` starts the windows on a grid of the colors instead of on every pixel.
    """)
    return


@app.cell
def _(MeanShift, estimate_bandwidth, np, plt):
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
    _figure, _axes = plt.subplots(1, 2, figsize=(8, 4))
    _axes[0].imshow(image.astype(np.uint8))
    _axes[0].set_title("Built image")
    _axes[0].axis("off")
    _axes[1].imshow(image_model.labels_.reshape(image.shape[0], image.shape[1]), cmap="tab10")
    _axes[1].set_title(f"{image_clusters} color clusters")
    _axes[1].axis("off")
    _figure.tight_layout()
    print(f"Image shape {image.shape}, pixels {pixels.shape}")
    print(f"Estimated bandwidth = {image_bandwidth:.2f}")
    return _figure


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
**bandwidth.** A smaller radius follows smaller peaks. A larger radius can pull two peaks into one cluster.
''')
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(blob_counts, coarse, estimated, fine, image_bandwidth, image_clusters, mo, wide_counts):
    mo.md(f"""
    ## Conclusions

    **Key takeaways:**

    - On the three-center cloud, the quantile-0.3 estimate is {estimated:.2f} and it finds {blob_counts[f"estimate {estimated:.2f}"]} clusters. Bandwidth 1 finds {blob_counts["bandwidth 1"]} clusters. Bandwidth 0.55 finds {blob_counts["bandwidth 0.55"]}. Bandwidth 2 finds {blob_counts["bandwidth 2"]}.
    - The centers used to build that cloud are (1, 1), (3, 3), and (5, 1). The printed centers from bandwidth 1 are the estimate of those peaks.
    - On the four-blob cloud, quantile 0.1 gives bandwidth {fine:.2f} and {wide_counts[f"quantile 0.1 ({fine:.2f})"]} clusters. Quantile 0.3 on 100 points gives bandwidth {coarse:.2f} and {wide_counts[f"quantile 0.3 ({coarse:.2f})"]} clusters.
    - The built image, bandwidth {image_bandwidth:.2f} with bin seeding, forms {image_clusters} color clusters. Each pixel was a row of red, green, and blue.
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
1. **The dense part of the points inside it:** The window moves to their mean and repeats until it sits on a peak.
2. **The radius of the window:** Only points inside that radius pull the mean.
3. **The default estimates a radius from neighbor distances:** Quantile 0.3 of those distances is the usual estimate. Passing `bandwidth=1` overrides it.
4. **It usually makes the radius smaller:** The estimate uses a nearer part of the neighbor-distance distribution.
5. **One pixel:** The columns are the color channels. Location on the grid is restored only when the labels are reshaped.
6. **Where the windows start:** Seeds are placed on a coarse grid of the points instead of on every pixel, which shortens the fit.
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
    - VanderPlas, J. (2017). *Python Data Science Handbook*. O'Reilly Media. Chapter 5.
    """)
    return


if __name__ == "__main__":
    app.run()
