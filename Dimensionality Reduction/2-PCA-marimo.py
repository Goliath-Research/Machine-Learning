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
    from sklearn.datasets import load_digits
    from sklearn.decomposition import PCA

    return PCA, load_digits


@app.cell
def _(np, plt):
    def plot_digits(images, rows, cols):
        figure, axes = plt.subplots(
            rows,
            cols,
            figsize=(8, 0.72 * rows + 0.4),
            gridspec_kw={"hspace": 0.15, "wspace": 0.05},
        )
        for ax, image in zip(np.ravel(axes), images):
            ax.imshow(np.asarray(image).reshape(8, 8), cmap="binary", interpolation="nearest", vmin=0, vmax=16)
            ax.set_xticks([])
            ax.set_yticks([])
        return figure

    return (plot_digits,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # PCA for Pictures and Noise

    ## Objectives

    - Draw the digits in the plane of the first two principal components.
    - Read how many components are needed for 80% and for 90% of the variance.
    - Rebuild noisy digits from 25%, 50%, and 75% of the variance in the noisy images.

    ## Background

    Each digit is an 8 by 8 grid, so each image is 64 numbers. Two principal components are a view of that 64-dimensional cloud. The labels can color the points. The fit never sees the labels.

    A float passed as `n_components`, such as `0.50`, asks for enough components to keep that share of the variance. Rebuilding the images from those components keeps the large patterns and drops the small ones. Added pixel noise tends to land in the small ones.

    ## Datasets Used

    - **Digits.** 1,797 handwritten digits from `load_digits`. Each image has 64 pixel values. The noise added below uses a normal draw with standard deviation 4 and `random_state=42`.
    """)
    return


@app.cell
def _(load_digits, plot_digits):
    digits = load_digits()
    print(f"Images: {digits.images.shape}")
    print(f"Table: {digits.data.shape}")
    _figure = plot_digits(digits.images, 10, 10)
    _figure.suptitle("First 100 digits")
    return _figure


@app.cell
def _(PCA, digits, pd, plt, sns):
    digit_plane = PCA(n_components=2).fit_transform(digits.data)
    _frame = pd.DataFrame(
        {
            "Component 1": digit_plane[:, 0],
            "Component 2": digit_plane[:, 1],
            "digit": digits.target.astype(str),
        }
    )
    _figure, _axes = plt.subplots(figsize=(7, 5))
    sns.scatterplot(
        data=_frame, x="Component 1", y="Component 2", hue="digit", s=16, ax=_axes, linewidth=0
    )
    _axes.set_title("Digits in the first two components")
    _axes.legend(title="Digit", fontsize=7, markerscale=0.8, ncol=2)
    print(f"Original shape: {digits.data.shape}")
    print(f"Transformed shape: {digit_plane.shape}")
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How many components

    The curve of shares is the explained variance ratio of each component. The running total of those shares is the cumulative ratio. A component whose share rounds to zero still has a tiny remainder. The counts below use the cumulative total, not a rounded bar.
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
    _figure, _axes = plt.subplots(1, 2, figsize=(9, 3.6))
    _axes[0].plot(_components, digit_ratio, color="crimson", lw=1.4)
    _axes[0].set_title("Explained variance ratio")
    _axes[0].set_ylabel("Share of variance")
    _axes[1].plot(_components, digit_cumulative, color="C0", lw=1.4)
    _axes[1].axhline(0.80, color="0.5", lw=0.8, ls="--")
    _axes[1].axhline(0.90, color="0.5", lw=0.8, ls="--")
    _axes[1].set_title("Cumulative share")
    _axes[1].set_ylabel("Running total")
    for _ax in _axes:
        _ax.set_xlabel("Number of components")
    _figure.tight_layout()
    print(f"Components for 80% of the variance: {components_for_80}")
    print(f"Components for 90% of the variance: {components_for_90}")
    print(f"Components with a share below 0.0005: {tiny_components}")
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Noise

    The noisy images are the original pixels plus a normal draw with standard deviation 4. The generator is `RandomState(42)`. Each reconstruction is fit on the noisy table, then transformed and inverted. The pixel grid is drawn from 0 to 16, so values outside that range are clipped by the color scale.
    """)
    return


@app.cell
def _(PCA, digits, np, plot_digits):
    noisy_digits = np.random.RandomState(42).normal(digits.data, 4)
    noise_components = {}
    rebuilt = {}
    for _share in (0.25, 0.50, 0.75):
        _model = PCA(_share).fit(noisy_digits)
        noise_components[_share] = int(_model.n_components_)
        rebuilt[_share] = _model.inverse_transform(_model.transform(noisy_digits))
        print(f"Share {_share:.0%} uses {noise_components[_share]} components.")
    _figure = plot_digits(noisy_digits, 4, 10)
    _figure.suptitle("Noisy digits")
    return _figure


@app.cell
def _(plot_digits, rebuilt):
    _figure = plot_digits(rebuilt[0.25], 4, 10)
    _figure.suptitle("Rebuilt from 25% of the noisy variance")
    return _figure


@app.cell
def _(plot_digits, rebuilt):
    _figure = plot_digits(rebuilt[0.50], 4, 10)
    _figure.suptitle("Rebuilt from 50% of the noisy variance")
    return _figure


@app.cell
def _(plot_digits, rebuilt):
    _figure = plot_digits(rebuilt[0.75], 4, 10)
    _figure.suptitle("Rebuilt from 75% of the noisy variance")
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    What float, passed as `n_components`, asks for half of the variance? Replace `None` with that number.
    """)
    return


@app.cell
def _():
    student_share = None
    print(student_share)
    return (student_share,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r'''
```python
student_share = 0.5
```
**0.5.** An integer asks for that many components. A float between 0 and 1 asks for enough components to reach that share of the variance.
''')
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(components_for_80, components_for_90, mo, noise_components, tiny_components):
    mo.md(f"""
    ## Conclusions

    **Key takeaways:**

    - The first two components are a view of 64 pixel values. The digit labels color the points and are not inputs to the fit.
    - On the clean digits, {components_for_80} components reach 80% of the variance and {components_for_90} components reach 90%. {tiny_components} components have a share below 0.0005.
    - On the noisy digits, 25% of the variance uses {noise_components[0.25]} components, 50% uses {noise_components[0.50]}, and 75% uses {noise_components[0.75]}.
    - A lower share drops more of the pixel noise and also drops more of the digit. A higher share keeps more of both.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check your understanding

    1. Why can the labels color the scatter when PCA did not use them?
    2. How does the cumulative share differ from the share of one component?
    3. Why can a rebuilt image look cleaner than the noisy one?
    4. What is given up when the kept share moves from 75% down to 25%?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
1. **The coloring is a view:** PCA sees only the 64 pixel values. The digit is attached afterward so the groups can be compared by eye.
2. **One share versus a running total:** The ratio says how much that component adds. The cumulative share adds those ratios from the first component through the one you stop at.
3. **The noise is a small direction:** A component with little variance is left out of the rebuild. Random pixel noise often lives there. So does some of the real stroke.
4. **More of the stroke goes with it:** Fewer components remove more of the noise and more of the digit. The pictures at 25% and 75% are the comparison.
""")
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
    - [Digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
    - VanderPlas, J. (2017). *Python Data Science Handbook*. O'Reilly Media. Chapter 5.
    """)
    return


if __name__ == "__main__":
    app.run()
