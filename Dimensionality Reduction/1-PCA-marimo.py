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
    import plotly.io as pio
    import plotly.graph_objects as go
    import PlotBuilder as pb
    from plotly.subplots import make_subplots
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    pio.templates.default = "plotly_white"
    px.defaults.color_discrete_sequence = px.colors.qualitative.G10
    builder = pb.PlotBuilder(width=620, height=470)

    return PCA, Path, StandardScaler, builder, go, make_subplots, mo, np, pd, px


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Principal Component Analysis

    ## Objectives

    By the end of this lesson, you should be able to:

    1. Explain what a principal component is: a direction through the data, not one of the original variables.
    2. Distinguish the variance along a component, measured in the squared units of the data, from the share of the total variance it explains, and use that share to judge how much structure a reduced description keeps.
    3. Explain how the units and the spread of the measurements decide which directions PCA finds, and decide when standardizing is needed.
    4. Explain what a component's *sign* does and does not mean.
    5. Describe what is lost when data are rebuilt from fewer components, and how large that loss is.
    6. Interpret a component from its weights, and separate what the arithmetic states from what the analyst names.
    7. Choose how many components to keep from the running total of the shares, knowing that the shares decrease in order.
    8. Explain why components fitted to one table describe that table, and not new data.

    ## Background

    **Principal component analysis** is a way of describing a cloud of points using new axes. It is the foundation for everything else in this folder: kernel PCA, multidimensional scaling, and the manifold methods all build on the same idea.

    ### Finding the directions

    Given a cloud of points, PCA looks for **directions** in order of how much the data spreads along them.

    1. The **first component** is the direction along which the points vary most.
    2. The **second component** is the direction of greatest remaining variation, and it must be at a right angle to the first.
    3. Each further component continues that pattern.

    Two consequences follow immediately.

    - **Components are directions, not columns.** A component is a mixture of the original variables. It is a line through the cloud, described by a vector of weights.
    - **Components are uncorrelated by construction.** Because each new one must be perpendicular to all the earlier ones, they share no linear trend.

    ### The two variance attributes

    Fit a PCA model and it reports three things that are easy to confuse:

    | Attribute | What it holds | Units |
    | :--- | :--- | :--- |
    | `components_` | the direction vectors, one row per component | weights, no units |
    | `explained_variance_` | the variance along each component | **squared units of your data** |
    | `explained_variance_ratio_` | each variance divided by the total | a fraction, 0 to 1 |

    The distinction between the last two matters. If you fit on miles per gallon, `explained_variance_` is measured in miles-per-gallon squared, which is hard to interpret. `explained_variance_ratio_` is a share, so it is comparable across datasets. When you report "this component keeps 98.6% of the variance", you are quoting the ratio.

    The ratios always sum to 1 across **all** components, including ones you did not keep.

    ### Standardization, and when to skip it

    PCA finds directions of maximum variance, so **a column with a big numerical range pulls the first component toward itself**. If you mix curb weight (thousands of pounds) with fuel economy (tens of miles per gallon), the weight column will dominate.

    Be careful with numbers that merely *look* similar. Wheel base (around 100 inches) and horsepower (around 100) happen to have values of the same size, but that is a coincidence of the units chosen: measure wheel base in centimetres and it becomes about 250. What matters is whether the columns share a unit and a similar spread, not whether their numbers look alike.

    That gives the rule:

    - **Same units, similar spread → standardizing is unnecessary.** It does no harm, but it adds nothing. Two mileage columns are both miles per gallon; converting them to z-scores would not change the geometry of the cloud, only its scale.
    - **Different units → standardize.** Otherwise the components describe your units, not your data.

    This lesson shows both cases, in that order, so the contrast is visible.

    ### Dimensionality reduction

    Keeping all the components is a rotation, not a reduction: the cloud has the same shape, viewed from a different angle. To reduce, you keep the first few and discard the rest.

    The discarded directions are the ones with least variance. The `inverse_transform` method puts the shortened description back into the original columns. What was dropped does not come back: the rebuilt points all land **on a line** (or a plane), because the missing direction no longer exists.

    That is the trade in one sentence: you keep the largest patterns and lose the small ones, and the size of what you lost is exactly the variance you threw away. (Strictly, the average squared error equals the discarded variance times (n − 1)/n, because `explained_variance_` divides by n − 1 while a plain average divides by n. With 205 cars the difference is half a percent.)

    ### One caution about signs

    A component's direction is only defined up to its sign: flipping every weight in a vector gives the same line. So `components_[0]` might read `[0.688, 0.726]` or `[-0.688, -0.726]` depending on the implementation. **Never interpret the sign of a component** unless you fix it yourself. The magnitudes, and the variance each component explains, are unaffected.

    ## Datasets used

    ### Automobile data

    **Source.** The Automobile dataset from the **UCI Machine Learning Repository**, donated in 1985 and widely used as a teaching set for mixed-type data. 205 cars, 26 columns: price, insurance risk rating, make, and technical specifications such as wheel base, curb weight, horsepower, and fuel economy. Available at the [UCI repository](https://archive.ics.uci.edu/ml/datasets/automobile).

    **What this lesson uses.** Two fits:

    - **Two mileage columns**, `city_mpg` and `highway_mpg`, both in miles per gallon. No missing values.
    - **Eight measurements**: wheel base, curb weight, horsepower, length, width, height, city mpg, and highway mpg.

    **A caution.** The file stores missing values as the string `?`, which must be converted to NaN or the affected statistics are silently wrong. It also encodes categorical columns as text (`alfa-romero`, `two`, `rwd`), so those columns cannot enter a distance computation without encoding. This lesson uses only the numeric columns.

    A copy, `auto_imports85.csv`, sits beside this notebook, with the column names attached and `?` already converted.

    If you want to build that copy yourself from the original file, this is the step that matters. The file has no header row, so the names are supplied with `names=`, and `na_values="?"` tells pandas that a question mark means "missing":

    ```python
    cars = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data",
        header=None,
        names=column_names,
        na_values="?",
    )
    ```

    Without `na_values="?"`, a column such as `horsepower` would be read as text, because a single `?` makes pandas treat the whole column as strings.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Loading the data

    The next cell reads the CSV that sits beside this notebook. `Path(__file__).resolve().parent` is the folder this file lives in, so the notebook finds the data wherever you run it from. It then keeps the two mileage columns in a smaller table called `mileage`, because the first part of the lesson uses only those.
    """)
    return


@app.cell
def _(Path, pd):
    # Automobile data, UCI Machine Learning Repository (1985), 205 cars, 26 columns.
    # https://archive.ics.uci.edu/ml/datasets/automobile
    # The original file has no header row and stores missing values as "?".
    # auto_imports85.csv here already has the column names and NaN for missing values.
    cars = pd.read_csv(Path(__file__).resolve().parent / "auto_imports85.csv")
    mileage = cars[["city_mpg", "highway_mpg"]]
    car_counts = f"{len(cars)} cars, {len(mileage)} with both mileage columns"
    return car_counts, cars, mileage


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Before any analysis, look at the rows themselves. The table below shows the first ten cars. Each row is one car and each column one recorded property. Scroll sideways to see all 27 columns, including the text columns such as `make` and `body_style` that this lesson leaves out.
    """)
    return


@app.cell
def _(cars, mileage, mo):
    mo.vstack(
        [
            mo.md(f"Full table: **{cars.shape[0]} rows x {cars.shape[1]} columns**. Mileage table: **{mileage.shape[0]} rows x {mileage.shape[1]} columns**."),
            mo.ui.table(cars.head(10), selection=None),
        ]
    )
    return


@app.cell(hide_code=True)
def _(car_counts, mileage, mo):
    mo.md(f"""
    **{car_counts}.** Neither mileage column has a missing value, so all 205 cars can be used for the first fit.

    | | missing values |
    | :--- | ---: |
    | city_mpg | {mileage.city_mpg.isnull().sum()} |
    | highway_mpg | {mileage.highway_mpg.isnull().sum()} |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Two columns that share a unit

    City and highway fuel economy, both in miles per gallon. Before any components, look at the raw cloud.

    **How the plot is made.** `px.scatter` draws one dot per car, with city mileage across and highway mileage up. `labels` replaces the column names with readable axis titles. `update_yaxes(scaleanchor="x", scaleratio=1)` forces one mile per gallon to take the same length on both axes. That matters for PCA: on stretched axes a direction would look steeper or flatter than it really is, and the right angle between the components would not look like a right angle.
    """)
    return


@app.cell
def _(builder, mileage, px):
    _figure = px.scatter(
        mileage,
        x="city_mpg",
        y="highway_mpg",
        title="City and highway miles per gallon",
        labels={
            "city_mpg": "City (miles per gallon)",
            "highway_mpg": "Highway (miles per gallon)",
        },
    )
    _figure.update_traces(marker=dict(size=7, opacity=0.75))
    _figure.update_yaxes(scaleanchor="x", scaleratio=1)
    _figure.update_layout(
        width=builder.width, height=builder.height, showlegend=False,
        margin=dict(t=55, b=50, l=60, r=20),
    )
    _figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The cloud is a narrow diagonal band. Highway mileage runs higher than city mileage for every car, and the two move together tightly.

    **This is not a regression.** A regression would ask "given city mileage, what is the expected highway mileage?". PCA asks a different, unsupervised question: *in which direction does this cloud spread most?* No variable is treated as the target.

    Notice also that both axes have similar ranges, roughly 10 to 50. Standardizing would rescale both by nearly the same factor, so it would not change the **shape** of this cloud. That is why this fit skips it, and the eight-measurement fit later does not.

    ### The arithmetic behind PCA

    PCA can be stated in three steps. Write each car as a point **x** with two coordinates.

    1. **Centre** the data: subtract the mean **μ** of each column, so the cloud sits around the origin.
    2. Compute the **covariance matrix** **C**. For two columns it is a 2 × 2 table: the variance of each column on the diagonal, and their covariance off it.
    3. Find the **eigenvectors** and **eigenvalues** of **C**. The eigenvectors are the component directions (the rows of `components_`), and each eigenvalue is the variance along its direction (`explained_variance_`).

    Once the directions are known, two formulas do all the work. Collect the kept directions as the rows of a matrix **W**.

    - **Projection** (`transform`): z = (x − μ) Wᵀ. Each score is how far a car lies along a direction.
    - **Reconstruction** (`inverse_transform`): x̂ = μ + z W. The scores are turned back into the original columns.

    The next cell carries out these steps by hand on the mileage data with NumPy, so you can compare the result with scikit-learn's. `np.cov(..., rowvar=False)` computes the covariance matrix, treating each column as a variable. `np.linalg.eigh` returns the eigenvalues of a symmetric matrix in increasing order, so `[::-1]` puts the largest first, as PCA does.
    """)
    return


@app.cell
def _(mileage, np):
    mileage_covariance = np.cov(mileage.to_numpy(), rowvar=False)
    _values, _vectors = np.linalg.eigh(mileage_covariance)
    hand_variances = _values[::-1]
    hand_directions = _vectors[:, ::-1].T
    return hand_directions, hand_variances, mileage_covariance


@app.cell(hide_code=True)
def _(hand_directions, hand_variances, mileage_covariance, mo):
    mo.md(f"""
    The covariance matrix of the two mileage columns, in miles-per-gallon squared:

    | | city_mpg | highway_mpg |
    | :--- | ---: | ---: |
    | city_mpg | {mileage_covariance[0, 0]:.2f} | {mileage_covariance[0, 1]:.2f} |
    | highway_mpg | {mileage_covariance[1, 0]:.2f} | {mileage_covariance[1, 1]:.2f} |

    The off-diagonal value is almost as large as the diagonal ones: the two columns move together very strongly.

    Its eigenvalues are **{hand_variances[0]:.2f}** and **{hand_variances[1]:.2f}**, and the matching eigenvectors are ({hand_directions[0, 0]:+.4f}, {hand_directions[0, 1]:+.4f}) and ({hand_directions[1, 0]:+.4f}, {hand_directions[1, 1]:+.4f}). Keep these numbers in mind: scikit-learn will report the same values below, perhaps with the signs flipped. PCA is nothing more than this eigen-decomposition, plus bookkeeping.

    The eigenvalues add up to **{hand_variances.sum():.2f}**, which is the sum of the two diagonal entries, {mileage_covariance[0, 0]:.2f} + {mileage_covariance[1, 1]:.2f}. Rotating the axes moves variance between directions but never creates or destroys it. That is why the shares can sum to 1.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The fit, without standardizing

    Now the same thing with scikit-learn. `PCA(n_components=2)` creates a model that will keep two components, and `.fit(mileage)` performs the three steps above: it centres the columns, computes the covariance matrix and finds its eigenvectors. Nothing is transformed yet; the model only learns the mean, the directions and the variances. The cell then stores the three attributes under shorter names for the tables that follow.
    """)
    return


@app.cell
def _(PCA, mileage):
    mileage_pca = PCA(n_components=2).fit(mileage)
    mileage_variance = mileage_pca.explained_variance_
    mileage_share = mileage_pca.explained_variance_ratio_
    mileage_components = mileage_pca.components_
    return mileage_components, mileage_pca, mileage_share, mileage_variance


@app.cell(hide_code=True)
def _(mo, mileage_components):
    mo.md(f"""
    The fit reports two components, each a pair of weights over the two original columns:

    | Component | city_mpg weight | highway_mpg weight |
    | :--- | ---: | ---: |
    | first | {mileage_components[0, 0]:+.4f} | {mileage_components[0, 1]:+.4f} |
    | second | {mileage_components[1, 0]:+.4f} | {mileage_components[1, 1]:+.4f} |

    Read these as directions, not as variables.

    - The **first** component weights both columns positively and almost equally, so it points up the diagonal. That is where this cloud spreads most: cars differ mainly in being economical or thirsty.
    - The **second** weights them with **opposite signs**. It points across the band, along the thin direction where almost nothing varies.

    The two vectors are perpendicular, which PCA guarantees.

    **About the signs.** Your machine may show these with every sign flipped. That would describe the same two lines, and it changes nothing else in this lesson. The weights are only meaningful relative to each other, and only their magnitudes matter for the variance.
    """)
    return


@app.cell(hide_code=True)
def _(mileage_variance, mileage_share, mo):
    mo.md(f"""
    Now the two variance attributes, side by side, because this is where students most often look at the wrong number:

    | Component | `explained_variance_` | `explained_variance_ratio_` |
    | :--- | ---: | ---: |
    | first | {mileage_variance[0]:.2f} | {mileage_share[0]:.1%} |
    | second | {mileage_variance[1]:.2f} | {mileage_share[1]:.1%} |

    The first number is in **miles-per-gallon squared**, which is why {mileage_variance[0]:.1f} looks large and meaningless on its own. The second is a share of the total, and it says something interpretable: **{mileage_share[0]:.1%} of everything this cloud varies in lies along one direction.**

    Almost the entire dataset is one-dimensional. The cars differ in how economical they are, and hardly at all in the way city and highway mileage disagree.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **How the next plot is made.** It redraws the cloud and then calls `builder.component_arrows`, a helper from `PlotBuilder.py`. For each component it draws an arrow that starts at the mean of the data (`mileage_pca.mean_`), points along the component direction, and has length `scale` × √variance. The square root turns a variance back into a standard deviation, which is in miles per gallon like the axes, and `scale=3` makes the arrows long enough to see: each one reaches three standard deviations from the mean.
    """)
    return


@app.cell
def _(builder, mileage, mileage_pca, px):
    _figure = px.scatter(
        mileage,
        x="city_mpg",
        y="highway_mpg",
        title="The two component directions",
        labels={
            "city_mpg": "City (miles per gallon)",
            "highway_mpg": "Highway (miles per gallon)",
        },
    )
    _figure.update_traces(marker=dict(size=7, opacity=0.6))
    _figure.update_yaxes(scaleanchor="x", scaleratio=1)
    _figure.update_layout(
        width=builder.width, height=builder.height, showlegend=False,
        margin=dict(t=55, b=50, l=60, r=20),
    )
    builder.component_arrows(
        _figure,
        mileage_pca.mean_,
        mileage_pca.explained_variance_,
        mileage_pca.components_,
        scale=3,
    )
    _figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The arrows make the geometry concrete.

    - Their **direction** is `components_`: the long arrow runs up the band, the short one crosses it.
    - Their **length** is set by `explained_variance_`: the long arrow is drawn longer because that direction holds far more variance.
    - They meet at the **mean** of the data, marked by their shared start point, and they are at right angles.

    A component direction alone is not enough to describe the data; you need the direction *and* how much the data spreads along it. That is exactly the pair of attributes above.

    **How the next plot is made.** `mileage_pca.transform(mileage)` applies the projection formula z = (x − μ) Wᵀ to every car, giving each one a score on Component 1 and on Component 2. The scores are put in a table named `mileage_plane` and plotted with equal axis scales, plus light grey lines at zero so the centre of the data is easy to find.
    """)
    return


@app.cell
def _(builder, mileage, mileage_pca, pd, px):
    mileage_plane = pd.DataFrame(
        mileage_pca.transform(mileage), columns=["Component 1", "Component 2"]
    )
    _figure = px.scatter(
        mileage_plane,
        x="Component 1",
        y="Component 2",
        title="The same cars, viewed from the component axes",
    )
    _figure.update_traces(marker=dict(size=7, opacity=0.6))
    _figure.update_yaxes(scaleanchor="x", scaleratio=1)
    _figure.add_hline(y=0, line_color="lightgrey", line_width=1)
    _figure.add_vline(x=0, line_color="lightgrey", line_width=1)
    _figure.update_layout(
        width=builder.width, height=builder.height, showlegend=False,
        margin=dict(t=55, b=50, l=60, r=20),
    )
    _figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This is the same cloud, rotated to sit on the new axes. Two things to check:

    - The **shape is identical**. Nothing was reduced; with two components kept out of two, `transform` is a pure rotation.
    - The **spread is now aligned with the axes**. Along Component 1 the points run far; along Component 2 they barely leave zero.

    That second observation is the whole case for dimensionality reduction. If almost all the variation is along one axis, the other axis is nearly constant — so why keep it?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Replace `None` so that `student_component_1` holds the **weights** of the first component, and `student_first_share` holds the fraction of total variance it keeps.
    """)
    return


@app.cell
def _(mileage_pca):
    student_component_1 = None
    student_first_share = None
    print("First component weights:   ", student_component_1)
    print("Its share of the variance: ", student_first_share)
    return student_component_1, student_first_share


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The next cell checks your answer. It compares your values with the fitted model using `np.allclose`, which allows for tiny rounding differences, and also accepts the weights with every sign flipped, since that describes the same direction.
    """)
    return


@app.cell
def _(mileage_pca, mo, np, student_component_1, student_first_share):
    if student_component_1 is None or student_first_share is None:
        _message = "Replace both `None` values above, then this cell will check them."
    elif (
        np.allclose(np.abs(student_component_1), np.abs(mileage_pca.components_[0]))
        and np.isclose(student_first_share, mileage_pca.explained_variance_ratio_[0])
    ):
        _message = "✓ Correct: those are the first component's weights and its share."
    else:
        _message = "✗ Not yet. Check which row of `components_` and which attribute you used."
    mo.md(_message)
    return


@app.cell(hide_code=True)
def _(mileage_pca, mo):
    mo.accordion(
        {
            "Show answer": mo.md(rf"""
```python
student_component_1 = mileage_pca.components_[0]
student_first_share = mileage_pca.explained_variance_ratio_[0]
```
The weights are approximately **({abs(mileage_pca.components_[0, 0]):.3f}, {abs(mileage_pca.components_[0, 1]):.3f})** (perhaps with both signs flipped), and the share is about **{mileage_pca.explained_variance_ratio_[0]:.3f}**, so {mileage_pca.explained_variance_ratio_[0]:.1%} of the variance.

`components_` is a 2×2 array: one row per component, one column per original variable. Row 0 is the first component, so `components_[0]` gives its two weights. `explained_variance_ratio_` is a one-dimensional array indexed the same way.

Notice the weights are almost equal and both positive, which is why the long arrow runs up the diagonal at roughly 45°.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Why can this two-column fit skip standardization, when the eight-measurement fit later cannot? Work out your answer, then reveal it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answer": mo.md(r"""
Because **both columns are already in the same unit and have similar spread.**

City mileage and highway mileage are both miles per gallon, and both range over roughly 10 to 50. Neither can dominate the other, so the components found on the raw data describe the actual relationship between the two variables.

Standardizing would divide each column by nearly the same standard deviation. That would rescale the cloud slightly and leave its **shape** essentially unchanged, so the components would point in the same directions.

With eight measurements the situation is different: curb weight runs into thousands of pounds while mileage is in tens. On the raw scale the weight column's spread is enormous, so the first component would point almost exactly at curb weight and the other seven columns would contribute almost nothing. Standardizing is what stops your units from deciding the answer.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Reducing to one component

    Keep only the first component and see what happens to the cloud. This is where reduction actually occurs: two columns become one number per car.

    **What the next cell does.** It fits a new model with `n_components=1`, so **W** holds a single direction. `transform` gives each car one score along it (205 rows, 1 column). `inverse_transform` applies x̂ = μ + z W, turning that single score back into a city and a highway value, so the rebuilt table again has two columns. Finally the cell writes the three shapes into a sentence, so you can confirm the sizes.
    """)
    return


@app.cell
def _(PCA, mileage):
    mileage_one = PCA(n_components=1).fit(mileage)
    mileage_one_share = float(mileage_one.explained_variance_ratio_[0])
    mileage_reduced = mileage_one.transform(mileage)
    mileage_restored = mileage_one.inverse_transform(mileage_reduced)
    mileage_shapes = (
        f"original {mileage.shape[0]} rows x {mileage.shape[1]} columns, "
        f"reduced {mileage_reduced.shape[0]} rows x {mileage_reduced.shape[1]} column, "
        f"restored {mileage_restored.shape[0]} rows x {mileage_restored.shape[1]} columns"
    )
    return (
        mileage_one_share,
        mileage_reduced,
        mileage_restored,
        mileage_shapes,
    )


@app.cell(hide_code=True)
def _(mileage_one_share, mileage_shapes, mo):
    mo.md(f"""
    **{mileage_shapes}.** Two numbers per car became one, and the inverse transform put it back to two.

    The kept share is **{mileage_one_share:.1%}**, identical to the first component's ratio in the two-component fit. That is expected: the first component is the same direction either way, and asking for one component does not change it.

    So roughly {1 - mileage_one_share:.1%} of the variance was discarded. The next plot shows what that looks like.

    **How the plot is made.** The original and the rebuilt coordinates are placed in one table, and `builder.scatter_with_reference` from `PlotBuilder.py` draws the originals in blue and the rebuilt points in orange on the same axes, so each car can be compared with its reconstruction.
    """)
    return


@app.cell
def _(builder, mileage, mileage_restored, pd, px):
    _combined = pd.DataFrame(
        {
            "city_mpg": mileage["city_mpg"].to_numpy(),
            "highway_mpg": mileage["highway_mpg"].to_numpy(),
            "x_reference": mileage_restored[:, 0],
            "y_reference": mileage_restored[:, 1],
        }
    )
    _figure = builder.scatter_with_reference(
        _combined,
        "city_mpg",
        "highway_mpg",
        title="One component drawn back in the original plane",
        label_points="Original",
        label_reference="Rebuilt from one component",
    )
    _figure.update_xaxes(title_text="City (miles per gallon)")
    _figure.update_yaxes(title_text="Highway (miles per gallon)")
    _figure
    return


@app.cell(hide_code=True)
def _(mileage_one_share, mo):
    mo.md(f"""
    The orange points fall on a **single straight line**. That is the clearest picture of what reduction costs.

    Each car still has two coordinates, but they are no longer independent: they were generated from one number by multiplying it along one direction. The cloud has collapsed from a band into a line.

    Look at the gap between a blue point and its orange partner. That gap is the discarded component: it points across the band, along the short arrow from earlier. Cars near the middle of the band land almost on top of their originals (little was lost); cars at the edges of the band are pushed further (more was lost).

    **The honest summary:** keeping one component here loses little, because only about {1 - mileage_one_share:.1%} of the variance was there to lose. On a cloud that is genuinely two-dimensional, the same operation would destroy most of the structure. The amount lost is exactly the variance ratio you discarded.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Why do the rebuilt points lie on a line rather than filling the band? Work out your answer, then reveal it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answer": mo.md(r"""
Because **only one direction survived the inverse transform.**

Each car was reduced to a single number: its position along Component 1. The inverse transform then reconstructs two coordinates from that one number, using only the first component's direction. A single number multiplied by a fixed direction can only produce points along that direction, so every rebuilt car lands on the same line.

The second direction was dropped, and with it the freedom to move off that line. The original blue points have an extra offset across the band, which is precisely the part that is gone.

This is a general property: **keeping k components from a d-dimensional space puts the rebuilt data on a k-dimensional surface.** With k = 1 that surface is a line; with k = 2 in a 64-dimensional space it is a plane.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    The background claimed that the loss from reduction is the variance you threw away. Check it with numbers.

    Replace `None` so that `student_reconstruction_error` holds the **average squared distance** between each original car and its rebuilt partner. The originals are in `mileage` and the rebuilt points in `mileage_restored`. Then compare your value with `mileage_pca.explained_variance_[1]`, the variance of the discarded second component.
    """)
    return


@app.cell
def _(mileage, mileage_pca, mileage_restored, np):
    student_reconstruction_error = None
    print("average squared reconstruction error:", student_reconstruction_error)
    print("variance of the discarded component: ", mileage_pca.explained_variance_[1])
    return (student_reconstruction_error,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The next cell checks your value against the exact answer, computed independently.
    """)
    return


@app.cell
def _(mileage, mileage_restored, mo, np, student_reconstruction_error):
    _expected = np.mean(np.sum((mileage.to_numpy() - mileage_restored) ** 2, axis=1))
    if student_reconstruction_error is None:
        _message = "Replace `None` above, then this cell will check it."
    elif np.isclose(student_reconstruction_error, _expected):
        _message = "✓ Correct."
    else:
        _message = "✗ Not yet. Square the differences, add them across the two columns, then average over the cars."
    mo.md(_message)
    return


@app.cell(hide_code=True)
def _(mileage, mileage_pca, mileage_restored, mo, np):
    _error = np.mean(np.sum((mileage.to_numpy() - mileage_restored) ** 2, axis=1))
    _n = len(mileage)
    mo.accordion(
        {
            "Show answer": mo.md(rf"""
```python
student_reconstruction_error = np.mean(
    np.sum((mileage.to_numpy() - mileage_restored) ** 2, axis=1)
)
```
The average squared error is **{_error:.4f}**, and the discarded component's variance is **{mileage_pca.explained_variance_[1]:.4f}**.

They differ only by the factor (n − 1)/n: {mileage_pca.explained_variance_[1]:.4f} × {_n - 1}/{_n} = **{mileage_pca.explained_variance_[1] * (_n - 1) / _n:.4f}**, which matches the error exactly. `explained_variance_` is a sample variance, divided by n − 1, while `np.mean` divides by n.

`mileage.to_numpy() - mileage_restored` gives each car's offset in both columns. Squaring and summing across `axis=1` gives the squared distance per car, and `np.mean` averages it over the cars. The offset of every car points along the discarded direction, so the squared offsets add up to exactly the variance that direction held.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Eight measurements, which must be standardized

    Now the case where standardization is required. These eight columns do not share a unit:

    - wheel base, length, width, height in **inches**;
    - curb weight in **pounds**;
    - horsepower in **horsepower**;
    - city and highway mileage in **miles per gallon**.

    Look at the ranges before doing anything else.

    **What the next cell does.** It selects the eight columns into `measures` and builds a small summary table: the minimum, the maximum, their difference (the spread), and the number of missing values per column. It then keeps only the cars with no missing value in any of the eight columns (`dropna`), because PCA cannot work with gaps, and counts how many were dropped.
    """)
    return


@app.cell
def _(cars, pd):
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
    measure_ranges = pd.DataFrame(
        {
            "min": measures.min(),
            "max": measures.max(),
            "spread": measures.max() - measures.min(),
            "missing": measures.isnull().sum(),
        }
    )
    measures_complete = measures.dropna()
    measures_dropped = len(measures) - len(measures_complete)
    return measure_columns, measure_ranges, measures_complete, measures_dropped


@app.cell
def _(measure_ranges, mo):
    mo.Html(measure_ranges.round(1).to_html())
    return


@app.cell(hide_code=True)
def _(measure_ranges, measures_complete, measures_dropped, mo):
    _widest = measure_ranges["spread"].idxmax()
    _widest_value = measure_ranges.loc[_widest, "spread"]
    _narrowest = measure_ranges["spread"].idxmin()
    _narrowest_value = measure_ranges.loc[_narrowest, "spread"]
    mo.md(f"""
    The spreads are not comparable at all. **{_widest}** ranges over **{_widest_value:,.0f}** units, while **{_narrowest}** ranges over only **{_narrowest_value:,.1f}**.

    That is a ratio of about **{_widest_value / _narrowest_value:,.0f} to 1**. Without standardizing, the first component would essentially be the curb-weight direction, and every other column would be noise in the background.

    **{measures_dropped} rows** have a missing value in one of these columns and are dropped, leaving **{len(measures_complete)}** complete cars.

    The two box plots show the difference standardization makes.

    **How the next cell works.** `StandardScaler().fit_transform` subtracts each column's mean and divides by its standard deviation, z = (x − mean) / sd, and the result is stored as `measures_z`. The figure has two panels stacked vertically (`make_subplots(rows=2, cols=1)`). A loop goes over the two tables, original and standardized, and for each one adds a box per column with `go.Box`, so the same eight columns appear in both panels.
    """)
    return


@app.cell
def _(StandardScaler, go, make_subplots, measure_columns, measures_complete, pd):
    measures_z = pd.DataFrame(
        StandardScaler().fit_transform(measures_complete), columns=measure_columns
    )
    _figure = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Original units", "After standardization"),
        vertical_spacing=0.18,
    )
    for _row, _frame, _colour in (
        (1, measures_complete, "#2E86C1"),
        (2, measures_z, "#D35400"),
    ):
        for _name in measure_columns:
            _figure.add_trace(
                go.Box(y=_frame[_name], name=_name, marker_color=_colour, showlegend=False),
                row=_row,
                col=1,
            )
    _figure.update_layout(
        width=820,
        height=620,
        margin=dict(t=70, b=80, l=60, r=20),
    )
    _figure.update_xaxes(tickangle=-25)
    _figure
    return (measures_z,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The top panel is the problem: `curb_weight` occupies a range from about 1,800 to 5,100, so on a shared axis the other seven columns are flattened into invisible slivers.

    The bottom panel is the fix: every column now has a mean of 0 and a standard deviation of 1, so the boxes sit at comparable heights and their **shapes** become readable. Height and horsepower vary differently, and you can see it once the units are gone.

    Note what standardization does not do. It does not make the variables Gaussian, does not remove outliers, and does not make them independent. It only removes the arbitrary choice of unit.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    The text has claimed twice that, without standardizing, the first component would be essentially the curb-weight direction. Test the claim instead of believing it.

    Fit a PCA with one component on the **raw** complete measurements, `measures_complete`, and replace `None` so that `student_raw_top_column` holds the **name** of the column with the largest absolute weight.
    """)
    return


@app.cell
def _(PCA, measures_complete, np):
    student_raw_top_column = None
    print("column that dominates the raw first component:", student_raw_top_column)
    return (student_raw_top_column,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The next cell checks your answer by fitting the raw PCA itself and comparing column names.
    """)
    return


@app.cell
def _(PCA, measure_columns, measures_complete, mo, np, student_raw_top_column):
    _raw = PCA(n_components=1).fit(measures_complete)
    _expected = measure_columns[int(np.argmax(np.abs(_raw.components_[0])))]
    if student_raw_top_column is None:
        _message = "Replace `None` above, then this cell will check it."
    elif student_raw_top_column == _expected:
        _message = "✓ Correct."
    else:
        _message = "✗ Not yet. Use the absolute value of the weights, and turn the position into a column name."
    mo.md(_message)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answer": mo.md(r"""
```python
_raw = PCA(n_components=1).fit(measures_complete)
student_raw_top_column = measures_complete.columns[np.argmax(np.abs(_raw.components_[0]))]
```
The answer is **`curb_weight`**. The plot below shows how completely it dominates.

`components_[0]` holds one weight per column. `np.abs` removes the arbitrary sign, `np.argmax` finds the position of the largest weight, and indexing the column names with that position turns it into a name.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Raw against standardized: the proof

    The next cell fits a one-component PCA twice, once on the raw measurements and once on the standardized ones, and draws the absolute weight of every column in each fit as grouped bars. Absolute values are used because the sign is arbitrary. It also records the share of variance the raw first component keeps.
    """)
    return


@app.cell
def _(PCA, builder, measure_columns, measures_complete, measures_z, np, pd, px):
    _raw = PCA(n_components=1).fit(measures_complete)
    _standard = PCA(n_components=1).fit(measures_z)
    raw_first_share = float(_raw.explained_variance_ratio_[0])
    raw_first_weights = pd.Series(np.abs(_raw.components_[0]), index=measure_columns)
    _weights = pd.DataFrame(
        {
            "column": measure_columns * 2,
            "absolute weight": np.concatenate(
                [np.abs(_raw.components_[0]), np.abs(_standard.components_[0])]
            ),
            "data": ["raw units"] * 8 + ["standardized"] * 8,
        }
    )
    _figure = px.bar(
        _weights,
        x="column",
        y="absolute weight",
        color="data",
        barmode="group",
        title="First-component weights, raw against standardized",
    )
    _figure.update_layout(
        width=820, height=builder.height, margin=dict(t=60, b=80, l=60, r=20)
    )
    _figure.update_xaxes(tickangle=-25)
    _figure
    return raw_first_share, raw_first_weights


@app.cell(hide_code=True)
def _(mo, raw_first_share, raw_first_weights):
    mo.md(f"""
    On the raw data, the first component is curb weight and almost nothing else: its weight is **{raw_first_weights['curb_weight']:.3f}**, and that single direction claims **{raw_first_share:.1%}** of the variance. The other seven columns get weights close to zero. The fit is not describing cars; it is describing the fact that pounds are small units.

    After standardizing, the weights are spread across all eight columns, and the component becomes a genuine mixture. This is what "otherwise the components describe your units, not your data" means in practice.

    ### Three components from the standardized measurements

    The next cell fits a PCA with three components on `measures_z`, the standardized table, and stores the variances, the shares, their total, and the weights (`components_`, here called loadings) for the tables and plots that follow.
    """)
    return


@app.cell
def _(PCA, measures_z):
    auto_pca = PCA(n_components=3).fit(measures_z)
    auto_variance = auto_pca.explained_variance_
    auto_share = auto_pca.explained_variance_ratio_
    auto_total = float(auto_share.sum())
    auto_loadings = auto_pca.components_
    return auto_loadings, auto_share, auto_total, auto_variance


@app.cell(hide_code=True)
def _(auto_share, auto_total, mileage_share, mo):
    mo.md(f"""
    | Component | share of variance | running total |
    | :--- | ---: | ---: |
    | first | {auto_share[0]:.1%} | {auto_share[0]:.1%} |
    | second | {auto_share[1]:.1%} | {auto_share[0] + auto_share[1]:.1%} |
    | third | {auto_share[2]:.1%} | {auto_total:.1%} |

    Three components keep **{auto_total:.1%}** of the variance in eight standardized columns. Compare that with the mileage fit, where one component kept {mileage_share[0]:.1%}.

    The contrast is the point. The mileage columns were nearly one-dimensional; these eight columns are not. No single direction describes a car, because "size", "power", and "economy" are genuinely different aspects of it.

    Note also that the shares **decrease** in order: {auto_share[0]:.1%}, then {auto_share[1]:.1%}, then {auto_share[2]:.1%}. That is guaranteed by construction. If you ever see them out of order, you are not looking at a standard PCA fit.
    """)
    return


@app.cell(hide_code=True)
def _(auto_loadings, measure_columns, mo, np):
    _rows = []
    for _component in range(3):
        _ranked = np.argsort(-np.abs(auto_loadings[_component]))
        _top = ", ".join(
            f"`{measure_columns[_i]}` ({auto_loadings[_component, _i]:+.2f})"
            for _i in _ranked[:3]
        )
        _rows.append(f"    - Component {_component + 1} is dominated by {_top}")
    mo.md("\n".join([
        "What each component actually describes — the three largest weights in each:",
        "",
        *_rows,
        "",
        "To build that list, the cell sorts each component's weights by absolute size (`np.argsort(-np.abs(...))` gives the positions from largest to smallest) and prints the top three with their signs.",
        "",
        "Read these as descriptions, not as rankings. A component is a weighted mixture, and naming it is an interpretation, not a fact the algorithm reports. Since signs are arbitrary, what can be interpreted is which columns share a sign and which have opposite signs, never whether a sign is positive.",
        "",
        "The pattern is interpretable, which is a good sign for a PCA fit:",
        "",
        "- **Component 1** gives weights of one sign to all the size and power columns (curb weight, length, width, wheel base, horsepower) and the opposite sign to both mileage columns. It is a **size versus economy** axis: large, heavy, powerful cars at one end, small economical cars at the other.",
        "- **Component 2** is led by height, with wheel base and the two mileage columns on the same side, and horsepower on the opposite side. It separates **tall, economical** cars from **low, powerful** ones.",
        "- **Component 3** sets height against width and the two mileage columns. Once size has been accounted for by Component 1, it separates **tall, narrow** cars from **wide, economical** ones.",
        "",
        "Those names are ours, not the algorithm's. PCA reports directions; the labelling is where domain knowledge enters, and another analyst might name them differently. What is not negotiable is the arithmetic behind the weights. The heat map below shows every weight, not only the top three.",
    ]))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **How the heat map is made.** `px.imshow` draws a matrix as a grid of coloured squares: one row per component, one column per measurement. The colour scale `RdBu` is centred at zero (`zmin=-1`, `zmax=1`), so red and blue mark opposite signs and pale squares mark weights close to zero. `text_auto=".2f"` writes each weight inside its square.
    """)
    return


@app.cell
def _(auto_loadings, builder, measure_columns, px):
    _figure = px.imshow(
        auto_loadings,
        x=measure_columns,
        y=["Component 1", "Component 2", "Component 3"],
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        text_auto=".2f",
        title="Weights of the three components",
        aspect="auto",
    )
    _figure.update_layout(
        width=820, height=360, margin=dict(t=60, b=80, l=100, r=20)
    )
    _figure.update_xaxes(tickangle=-25)
    _figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Read the heat map one row at a time. In the first row every size and power column has the same colour and both mileage columns have the other: that is the size-versus-economy axis. In the second, height is the strongest square. In the third, height and width have opposite colours.

    ### The cars in the new coordinates

    The components are not only descriptions; every car has a score on each one. The next cell computes those scores with `transform`, keeps the first two, and colours each car by its `body_style`, a text column PCA never saw. `cars.loc[measures_complete.index, "body_style"]` picks the body style of exactly the cars that survived `dropna`, in the same order.
    """)
    return


@app.cell
def _(auto_pca, builder, cars, measures_complete, measures_z, pd, px):
    _scores = auto_pca.transform(measures_z)
    _plane = pd.DataFrame(
        {
            "Component 1": _scores[:, 0],
            "Component 2": _scores[:, 1],
            "body style": cars.loc[measures_complete.index, "body_style"].to_numpy(),
        }
    )
    _figure = px.scatter(
        _plane,
        x="Component 1",
        y="Component 2",
        color="body style",
        title="Cars on the first two components, coloured by body style",
    )
    _figure.update_traces(marker=dict(size=7, opacity=0.75))
    _figure.update_layout(
        width=760, height=builder.height, margin=dict(t=55, b=50, l=60, r=20)
    )
    _figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Body style was never given to PCA, yet it is not scattered at random. Hatchbacks gather towards the small, economical end of Component 1, and wagons tend to sit higher on Component 2, the height axis. Hardtops and convertibles are few and low. The overlap is large, though: a component is a summary of measurements, not a classifier, and many sedans and hatchbacks have nearly identical dimensions.

    This is how PCA is used in practice. You reduce, plot, and then ask whether known labels line up with the new axes. When they do, the components have captured something real about the objects.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Replace `None` so that `student_components_for_80` holds the **number of components needed to keep at least 80%** of the variance in the eight standardized measurements. Fit a full PCA with no `n_components` argument to get every share.
    """)
    return


@app.cell
def _(PCA, measures_z, np):
    student_components_for_80 = None
    print("components needed for 80% of the variance:", student_components_for_80)
    return (student_components_for_80,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The next cell checks your count against a full fit.
    """)
    return


@app.cell
def _(PCA, measures_z, mo, np, student_components_for_80):
    _cumulative = np.cumsum(PCA().fit(measures_z).explained_variance_ratio_)
    _expected = int(np.searchsorted(_cumulative, 0.80) + 1)
    if student_components_for_80 is None:
        _message = "Replace `None` above, then this cell will check it."
    elif student_components_for_80 == _expected:
        _message = "✓ Correct."
    else:
        _message = "✗ Not yet. Look at the running total, not at the individual shares."
    mo.md(_message)
    return


@app.cell(hide_code=True)
def _(PCA, measures_z, mo, np):
    _shares = np.cumsum(PCA().fit(measures_z).explained_variance_ratio_)
    mo.accordion(
        {
            "Show answer": mo.md(rf"""
```python
_full = PCA().fit(measures_z)
_cumulative = np.cumsum(_full.explained_variance_ratio_)
student_components_for_80 = int(np.searchsorted(_cumulative, 0.80) + 1)
```
The answer is **{int(np.searchsorted(_shares, 0.80) + 1)}** components, not three or four. The cumulative shares run {_shares[0]:.1%}, then {_shares[1]:.1%}, so the second component is the one that crosses 80%.

That is worth pausing on, because it cuts against the expectation set by the section. Three components were used for the plots, but **two** already carry more than 80% of the variance. The first component alone carries {_shares[0]:.1%}, so it does most of the work and the remaining six add comparatively little.

`PCA()` with no arguments keeps every component, so `explained_variance_ratio_` has one entry per column. `np.cumsum` turns those shares into a running total, and `np.searchsorted` finds the first position where the total reaches 0.80. Adding 1 converts a zero-based index into a count.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Choosing the number of components

    The usual tool for choosing a count is the pair of curves drawn next. The next cell fits a PCA with every component (`PCA()` with no `n_components`) and passes the shares and their running total (`np.cumsum`) to `builder.variance_curves` from `PlotBuilder.py`. The left panel shows each component's own share; the right panel the running total, with dashed reference lines at 80% and 90%.
    """)
    return


@app.cell
def _(PCA, builder, measures_z, np):
    full_auto_pca = PCA().fit(measures_z)
    full_auto_cumulative = np.cumsum(full_auto_pca.explained_variance_ratio_)
    builder.variance_curves(
        full_auto_pca.explained_variance_ratio_,
        full_auto_cumulative,
        title="Eight standardized measurements: share per component and running total",
    )
    return (full_auto_cumulative,)


@app.cell(hide_code=True)
def _(full_auto_cumulative, mo):
    mo.md(f"""
    The left curve drops steeply after the first component and then flattens: the later components each add little. The right curve crosses 80% at two components ({full_auto_cumulative[1]:.1%}) and 90% at three ({full_auto_cumulative[2]:.1%}). A common reading is to keep the components before the "elbow", where the left curve flattens, or as many as reach a target share.

    Because the shares decrease in order, the choice is always "how many from the front", never "which ones".

    **Explore it.** Move the slider to choose how many components to keep. The next cells refit with that count, rebuild the standardized cars from it, and report the share kept and the average squared reconstruction error, measured in standardized units squared. Watch the two numbers move in opposite directions.
    """)
    return


@app.cell
def _(mo):
    k_slider = mo.ui.slider(1, 8, value=2, label="Components kept", show_value=True)
    k_slider
    return (k_slider,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The next cell fits `PCA(n_components=k)` with the slider's value, applies `transform` then `inverse_transform`, and measures the squared error for each column separately, so the bar chart shows which measurements the reduced description rebuilds worst.
    """)
    return


@app.cell
def _(PCA, builder, k_slider, measure_columns, measures_z, mo, np, pd, px):
    _model = PCA(n_components=k_slider.value).fit(measures_z)
    _rebuilt = _model.inverse_transform(_model.transform(measures_z))
    _column_error = ((measures_z.to_numpy() - _rebuilt) ** 2).mean(axis=0)
    _figure = px.bar(
        pd.DataFrame({"column": measure_columns, "squared error": _column_error}),
        x="column",
        y="squared error",
        title=f"Reconstruction error per column with {k_slider.value} component(s)",
    )
    _figure.update_yaxes(range=[0, 1])
    _figure.update_layout(
        width=820, height=380, margin=dict(t=60, b=80, l=60, r=20)
    )
    _figure.update_xaxes(tickangle=-25)
    mo.vstack(
        [
            mo.md(
                f"With **{k_slider.value}** component(s): share kept **{_model.explained_variance_ratio_.sum():.1%}**, "
                f"average squared error per car **{_column_error.sum():.3f}** (out of a total variance of 8)."
            ),
            _figure,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Each standardized column has variance 1, so an error of 1 for a column means it has been lost entirely, and 0 means it is rebuilt perfectly. With one component the height column is rebuilt worst, because height barely enters Component 1. With all eight components every error is zero: keeping everything is a rotation, not a reduction.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Why does standardizing change the components, while keeping the same two mileage columns un-standardized does not? Work out your answer, then reveal it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answer": mo.md(r"""
Because **standardizing changes the relative spreads, and PCA selects directions by spread.**

In the mileage case, both columns were already in the same unit with nearly the same standard deviation. Dividing each by a similar number rescales the cloud without reshaping it, so the directions of maximum variance stay where they were.

In the eight-measurement case, the spreads differed by a factor of hundreds. Dividing each column by its own standard deviation changes the relative influence of every column dramatically. Curb weight no longer dominates, so the first component stops pointing at curb weight and starts describing a genuine mixture.

Put more sharply: **standardizing is not a neutral housekeeping step. It changes what "maximum variance" means**, and therefore changes the answer. That is why the choice must be justified rather than applied automatically.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(
    auto_total,
    full_auto_cumulative,
    measures_complete,
    measures_dropped,
    mileage_one_share,
    mileage_share,
    mileage_variance,
    mo,
    raw_first_share,
):
    mo.md(f"""
    ## Conclusions

    1. **A principal component is a direction, not a variable.** It is a weighted mixture of the original columns, chosen so that the data spreads as much as possible along it. Each further component is the largest remaining spread, at a right angle to all earlier ones.
    2. **Two variance attributes, two units.** `explained_variance_` is measured in the squared units of your data, which is why {mileage_share[0]:.1%} of the total appeared as a bare {mileage_variance[0]:.1f} in the mileage fit. `explained_variance_ratio_` is the share, and the shares always sum to 1 across all components.
    3. **Standardize when the units differ, and not otherwise.** The two mileage columns share a unit and a similar spread, so standardizing would not change the shape of the cloud and the fit skips it. The eight measurements differ by hundreds of units, and on the raw scale the first component was indeed the curb-weight direction, claiming {raw_first_share:.1%} of the variance.
    4. **Reduction costs exactly the variance you discard.** Keeping one component left {mileage_one_share:.1%} of the variance, and the rebuilt cars landed on a single line. The offset of each rebuilt car from its original is the discarded second component, and the average squared offset equals that component's variance, up to the factor (n − 1)/n.
    5. **A share tells you whether reduction is even worth doing.** On mileage, one component was enough ({mileage_share[0]:.1%}). On the eight standardized measurements, one component kept {full_auto_cumulative[0]:.1%}, two kept {full_auto_cumulative[1]:.1%} and three {auto_total:.1%}. That is still a large reduction, eight numbers to two or three, but no single number summarises a car the way one number summarised its mileage, because a car has several distinct aspects: size, power, and economy.
    6. **Shares decrease in order by construction, so choose k from the running total.** The first component always explains at least as much as the second, and so on. That ordering is what makes "keep the first k" meaningful: read the cumulative curve, and stop where it reaches the share you need.
    7. **Signs are arbitrary.** A component vector and its negation describe the same line. Only the magnitudes and the variances carry information, so never interpret the sign of a loading without fixing it deliberately.
    8. **The scaler and the components were fit on the rows in the picture.** There is no holdout, so these components describe this table rather than a model validated on new cars. With {measures_dropped} rows dropped for missing values, they describe the {len(measures_complete)} complete cars.
    9. **A component is named from its weights, and the name is a judgement.** On the eight measurements, the first component set size and power against economy, and the second was led by height. Those readings come from which columns share a sign; the labels are ours, while the weights are the algorithm's.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check your understanding

    1. What is a principal component: a column of the original data, or a direction through the cloud?
    2. What is the difference between `explained_variance_` and `explained_variance_ratio_`, and which one has interpretable units?
    3. Why can two mileage columns skip standardization while eight mixed measurements cannot?
    4. Why do the rebuilt points from one component lie on a line?
    5. What does the sum of all the variance ratios equal, and why?
    6. Why should you not interpret the sign of a component's weights?
    7. The eight-measurement fit needed three components for the variance it kept. What does that tell you about cars that the mileage fit did not?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answer": mo.md(r"""
1. **A direction through the cloud.** It is a weighted mixture of the original columns, chosen so the data spreads most along it. Each new component is perpendicular to all the earlier ones.
2. **`explained_variance_` is a variance, in squared units of the data** (miles-per-gallon squared, for the mileage fit); `explained_variance_ratio_` is that variance divided by the total, so it is a **fraction with no units**. The ratio is the interpretable one, and the ratios sum to 1 across all components.
3. **Because of relative spread.** The mileage columns share a unit and vary over similar ranges, so rescaling both changes the cloud's size but not its shape. The eight measurements differ by hundreds of units, so without standardizing the widest column would decide the first component by itself.
4. **Because only one direction survived.** Each car was reduced to one number, and the inverse transform rebuilds two coordinates from that number along a single direction. One number times one direction can only produce points on that line.
5. **1, always.** The ratios divide each component's variance by the total variance, so across every component they partition the whole. Keeping the first k gives a partial sum, which is the share retained.
6. **Because a vector and its negation describe the same line.** The direction is defined only up to sign, and different implementations or data orders can produce either. Magnitudes and variances are unaffected.
7. **That a car has several independent aspects.** Mileage was essentially one-dimensional: economical versus thirsty. The eight measurements separate into distinguishable directions such as size, power, and economy, so no single number summarises a car.
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
    - [PCA, scikit-learn user guide](https://scikit-learn.org/stable/modules/decomposition.html#pca)
    - [Automobile dataset, UCI](https://archive.ics.uci.edu/ml/datasets/automobile)
    - Müller, A. C., & Guido, S. (2017). *Introduction to Machine Learning with Python: A Guide for Data Scientists*. O'Reilly Media.
    - Jolliffe, I. T. (2002). *Principal Component Analysis* (2nd ed.). Springer.
    - VanderPlas, J. (2017). *Python Data Science Handbook*. O'Reilly Media. Chapter 5.
    """)
    return


if __name__ == "__main__":
    app.run()
