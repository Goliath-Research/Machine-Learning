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
    from pathlib import Path

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    return PCA, Path, StandardScaler


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Principal Component Analysis

    ## Objectives

    - Fit principal components to two miles-per-gallon columns and to eight automobile measurements.
    - Read the variance each component keeps, and the share that variance is of the total.
    - Put a one-component summary back into the original plane and see the line it lands on.

    ## Background

    Principal component analysis looks for orthogonal directions. The first direction is the one along which the numbers vary the most. The next direction is the largest variation that is left, at a right angle to the first. The length of each direction comes from the variance along it.

    Asking for fewer components drops the directions with the least variance. The inverse transform puts that shorter summary back into the original columns. What was dropped does not come back.

    ## Datasets Used

    - **Automobile data.** The UCI imports file, 205 cars, with `?` stored as a missing value. City and highway miles per gallon are on the same scale. A second fit uses wheel base, curb weight, horsepower, length, width, height, and the two mileage columns. A copy of the file sits in the Regression folder.
    """)
    return


@app.cell
def _(Path, pd):
    auto_headers = [
        "symboling",
        "normalized_losses",
        "make",
        "fuel_type",
        "aspiration",
        "num_doors",
        "body_style",
        "drive_wheels",
        "engine_location",
        "wheel_base",
        "length",
        "width",
        "height",
        "curb_weight",
        "engine_type",
        "num_cylinders",
        "engine_size",
        "fuel_system",
        "bore",
        "stroke",
        "compression_ratio",
        "horsepower",
        "peak_rpm",
        "city_mpg",
        "highway_mpg",
        "price",
    ]
    local_auto = Path(__file__).resolve().parents[1] / "Regression" / "imports-85.data"
    auto_source = (
        local_auto
        if local_auto.exists()
        else "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
    )
    cars = pd.read_csv(auto_source, header=None, names=auto_headers, na_values="?")
    mileage = cars[["city_mpg", "highway_mpg"]]
    print(f"{len(cars)} cars, from {auto_source}")
    print(f"Mileage rows: {len(mileage)}")
    print(mileage.isnull().sum().to_string())
    return


@app.cell
def _(mileage, plt, sns):
    _figure, _axes = plt.subplots(figsize=(5, 4.5))
    sns.scatterplot(data=mileage, x="city_mpg", y="highway_mpg", s=28, ax=_axes, legend=False)
    _axes.set_aspect("equal", adjustable="datalim")
    _axes.set_title("City and highway miles per gallon")
    _axes.set_xlabel("City (miles per gallon)")
    _axes.set_ylabel("Highway (miles per gallon)")
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Two components

    City and highway mileage share a scale, so this fit skips standardization. The fit is not a regression of one column on the other. It describes how the two columns vary together.
    """)
    return


@app.cell
def _(PCA, mileage):
    mileage_pca = PCA(n_components=2).fit(mileage)
    mileage_variance = mileage_pca.explained_variance_
    mileage_share = mileage_pca.explained_variance_ratio_
    print(f"Explained variance: {mileage_variance.round(3)}")
    print(f"Explained variance ratio: {mileage_share.round(3)}")
    print(f"Components:\n{mileage_pca.components_.round(4)}")
    return


@app.cell
def _(mileage, mileage_pca, np, plt, sns):
    _figure, _axes = plt.subplots(figsize=(5, 4.5))
    sns.scatterplot(data=mileage, x="city_mpg", y="highway_mpg", s=28, ax=_axes, legend=False)
    for _length, _vector in zip(mileage_pca.explained_variance_, mileage_pca.components_):
        _start = mileage_pca.mean_
        _end = _start + 3 * _vector * np.sqrt(_length)
        _axes.annotate(
            "",
            xy=_end,
            xytext=_start,
            arrowprops={"arrowstyle": "->", "lw": 1.6, "color": "black"},
        )
    _axes.set_aspect("equal", adjustable="datalim")
    _axes.set_title("Directions of the two components")
    _axes.set_xlabel("City (miles per gallon)")
    _axes.set_ylabel("Highway (miles per gallon)")
    return _figure


@app.cell
def _(mileage, mileage_pca, pd, plt, sns):
    mileage_plane = pd.DataFrame(
        mileage_pca.transform(mileage), columns=["Component 1", "Component 2"]
    )
    _figure, _axes = plt.subplots(figsize=(5, 4.5))
    sns.scatterplot(data=mileage_plane, x="Component 1", y="Component 2", s=28, ax=_axes, legend=False)
    _axes.set_aspect("equal", adjustable="datalim")
    _axes.set_title("The same cars in the component plane")
    _axes.axhline(0, color="0.7", lw=0.8)
    _axes.axvline(0, color="0.7", lw=0.8)
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## One component

    Keeping one component drops the shorter direction. The inverse transform draws the kept component back as points in the original two columns. Those points fall on a line.
    """)
    return


@app.cell
def _(PCA, mileage, plt, sns):
    mileage_one = PCA(n_components=1).fit(mileage)
    mileage_one_share = float(mileage_one.explained_variance_ratio_[0])
    _reduced = mileage_one.transform(mileage)
    _restored = mileage_one.inverse_transform(_reduced)
    print(f"Explained variance: {mileage_one.explained_variance_.round(3)}")
    print(f"Explained variance ratio: {mileage_one_share:.3f}")
    print(f"Component: {mileage_one.components_.round(4)}")
    print(f"Original shape: {mileage.shape}")
    print(f"Transformed shape: {_reduced.shape}")
    _figure, _axes = plt.subplots(figsize=(5, 4.5))
    sns.scatterplot(
        data=mileage, x="city_mpg", y="highway_mpg", s=28, ax=_axes, label="Original", legend=True
    )
    _axes.scatter(_restored[:, 0], _restored[:, 1], s=28, c="darkorange", label="One component", zorder=3)
    _axes.set_aspect("equal", adjustable="datalim")
    _axes.set_title("One component, drawn back in the original plane")
    _axes.set_xlabel("City (miles per gallon)")
    _axes.set_ylabel("Highway (miles per gallon)")
    _axes.legend(fontsize=8)
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Eight measurements

    These columns do not share a unit. Curb weight is thousands of pounds, and mileage is miles per gallon. Standardization gives each column a mean of 0 and a standard deviation of 1 before the components are fit. The scaler is fit on every remaining row. There is no holdout in this description.
    """)
    return


@app.cell
def _(PCA, StandardScaler, cars, pd, plt, sns):
    measure_columns = [
        "wheel_base",
        "curb_weight",
        "horsepower",
        "length",
        "width",
        "height",
        "city_mpg",
        "highway_mpg",
    ]
    measures = cars[measure_columns]
    print(measures.isnull().sum().to_string())
    measures_complete = measures.dropna()
    measures_dropped = len(measures) - len(measures_complete)
    print(f"{len(measures_complete)} complete rows. {measures_dropped} rows dropped.")
    measures_z = pd.DataFrame(
        StandardScaler().fit_transform(measures_complete), columns=measure_columns
    )
    auto_pca = PCA(n_components=3).fit(measures_z)
    auto_share = auto_pca.explained_variance_ratio_
    print(f"Explained variance: {auto_pca.explained_variance_.round(2)}")
    print(f"Explained variance ratio: {auto_share.round(4)}")
    _long_raw = measures_complete.melt(var_name="measure", value_name="value")
    _long_z = measures_z.melt(var_name="measure", value_name="value")
    _figure, _axes = plt.subplots(2, 1, figsize=(8, 6))
    sns.boxplot(data=_long_raw, x="measure", y="value", ax=_axes[0], color="C0")
    sns.boxplot(data=_long_z, x="measure", y="value", ax=_axes[1], color="C1")
    _axes[0].set_title("Original units")
    _axes[1].set_title("After standardization")
    _axes[0].tick_params(axis="x", rotation=30)
    _axes[1].tick_params(axis="x", rotation=30)
    _figure.tight_layout()
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Which attribute holds each component's share of the total variance? Replace `None` with that attribute name.
    """)
    return


@app.cell
def _():
    student_ratio = None
    print(student_ratio)
    return (student_ratio,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r'''
```python
student_ratio = "explained_variance_ratio_"
```
**explained_variance_ratio_.** `explained_variance_` is the variance itself, in the units of the columns that were fit. The ratio divides each of those variances by their total.
''')
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(auto_share, mileage_share, mileage_one_share, mileage_variance, mo, measures_dropped):
    mo.md(f"""
    ## Conclusions

    **Key takeaways:**

    - On city and highway mileage, the two component variances are {mileage_variance[0]:.1f} and {mileage_variance[1]:.1f}. Their shares of the total are {mileage_share[0]:.1%} and {mileage_share[1]:.1%}.
    - Keeping only the first component leaves {mileage_one_share:.1%} of the variance. Drawn back into the original plane, those cars sit on a line.
    - Among the eight measurements, {measures_dropped} rows have a missing value and are left out. After standardization, the three components keep {auto_share[0]:.1%}, {auto_share[1]:.1%}, and {auto_share[2]:.1%}. Together that is {auto_share.sum():.1%}.
    - The scaler was fit on every complete row. These components describe the table. They are not a model checked on held-out cars.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check your understanding

    1. What do the component vectors point along?
    2. Why can the two mileage columns skip standardization?
    3. What does the inverse transform of one component put back?
    4. Why are the eight measurements standardized before the fit?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
1. **Directions of large variance:** The first vector follows the largest spread. The next vector is the largest spread that remains, at a right angle to the first.
2. **They share a unit:** Both columns are miles per gallon. A column with a much larger unit would pull the first component toward itself.
3. **A line in the original plane:** One number per car is expanded back into city and highway miles per gallon. The second direction is gone, so the restored points have no scatter off that line.
4. **The units disagree:** Curb weight and miles per gallon are not comparable as raw numbers. Standardization gives each column the same standard deviation before the directions are chosen.
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
    - [Automobile data](https://archive.ics.uci.edu/ml/datasets/automobile)
    - VanderPlas, J. (2017). *Python Data Science Handbook*. O'Reilly Media. Chapter 5.
    """)
    return


if __name__ == "__main__":
    app.run()
