# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "scikit-learn",
#     "seaborn",
#     "statsmodels>=0.15.0",
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
        "statsmodels": "statsmodels>=0.15.0",
    }
    _missing = [
        _package for _module, _package in _packages.items()
        if _import_util.find_spec(_module) is None
    ]
    if _missing:
        if _sys.platform == "emscripten":
            import micropip as _micropip
            await _micropip.install(_missing)
        else:
            import subprocess as _subprocess
            _subprocess.check_call([_sys.executable, "-m", "pip", "install", *_missing])

    packages_ready = True
    return (packages_ready,)


@app.cell
def _(packages_ready):

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, root_mean_squared_error
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import PolynomialFeatures
    from statsmodels.graphics.gofplots import qqplot

    plt.rcParams["figure.max_open_warning"] = 0
    sns.set_theme(style="whitegrid")

    return (
        LinearRegression,
        PolynomialFeatures,
        mo,
        np,
        pd,
        plt,
        qqplot,
        r2_score,
        root_mean_squared_error,
        sns,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Polynomial Regression

    ## Objectives

    - Fit curves of degree 2, 3, and 5 to a wavy response.
    - Compare those curves with R² and RMSE on rows they did not see.
    - Check the residuals of a low-degree fit and a high-degree fit.

    ## Background

    A polynomial model adds powers of the predictor: `x`, `x²`, `x³`, and so on. The fit is still ordinary least squares. The curve appears because the columns are powers of `x`, not because the coefficients themselves are nonlinear.

    A higher degree can follow the training points more closely. It can also follow noise. The test rows are the check.

    ## Datasets Used

    - **Synthetic curve.** One hundred values of `x` drawn uniformly from 0 to 10. The response is `sin(x) * x` plus a little normal noise. The random seed is 50.
    """)
    return


@app.cell
def _(np, pd, plt, sns, train_test_split):
    np.random.seed(50)
    curve_x = np.random.uniform(0, 10, size=100)
    curve_y = np.sin(curve_x) * curve_x + np.random.normal(0, 1, size=100)
    x_train, x_test, y_train, y_test = train_test_split(
        curve_x, curve_y, test_size=0.3, random_state=20
    )
    print(f"Train = {len(x_train)} points \t Test = {len(x_test)} points")
    _figure, _axes = plt.subplots(figsize=(6, 4))
    sns.scatterplot(x=x_train, y=y_train, s=50, color="cornflowerblue", label="Training", ax=_axes)
    sns.scatterplot(x=x_test, y=y_test, s=50, color="darkblue", label="Test", ax=_axes)
    _axes.set_title("sin(x) * x, plus noise")
    _axes.set_xlabel("x")
    _axes.set_ylabel("y")
    _figure
    return _figure, x_test, x_train, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Degrees 2, 3, and 5

    `PolynomialFeatures` builds the power columns. `include_bias=False` leaves out the column of ones, because `LinearRegression` already estimates an intercept.

    The powers are defined on the training values, then applied to the test values with `transform`. The orange curve is the prediction on a fine grid, so the line does not jump between shuffled rows.

    Each title reports the test R² for that degree.
    """)
    return


@app.cell
def _(
    LinearRegression,
    PolynomialFeatures,
    np,
    pd,
    plt,
    r2_score,
    root_mean_squared_error,
    x_test,
    x_train,
    y_test,
    y_train,
):
    poly_rows = []
    _figure, _axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    _grid = np.linspace(0, 10, 300)
    for _ax, _degree in zip(_axes, (2, 3, 5)):
        _poly = PolynomialFeatures(degree=_degree, include_bias=False)
        _train_powers = _poly.fit_transform(x_train.reshape(-1, 1))
        _test_powers = _poly.transform(x_test.reshape(-1, 1))
        _model = LinearRegression().fit(_train_powers, y_train)
        _train_hat = _model.predict(_train_powers)
        _test_hat = _model.predict(_test_powers)
        _curve = _model.predict(_poly.transform(_grid.reshape(-1, 1)))
        _ax.scatter(x_train, y_train, s=20, color="cornflowerblue")
        _ax.scatter(x_test, y_test, s=20, color="darkblue")
        _ax.plot(_grid, _curve, color="darkorange")
        _ax.set_title(f"Degree {_degree}\nTest R² = {r2_score(y_test, _test_hat):.3f}")
        poly_rows.append(
            {
                "degree": _degree,
                "train_r2": r2_score(y_train, _train_hat),
                "test_r2": r2_score(y_test, _test_hat),
                "train_rmse": root_mean_squared_error(y_train, _train_hat),
                "test_rmse": root_mean_squared_error(y_test, _test_hat),
            }
        )
    poly_scores = pd.DataFrame(poly_rows).set_index("degree")
    _figure.tight_layout()
    _figure
    return _figure


@app.cell
def _(mo, poly_scores):
    mo.Html(poly_scores.round(3).to_html())
    return mo.Html(poly_scores.round(3).to_html())


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    RMSE is the typical size of a residual, in the units of `y`. A lower test RMSE is a closer fit on the rows that were held out.

    The Q-Q plots below use the test residuals of degree 2 and degree 5. Points along the reference line are consistent with a normal residual. A bend means the residuals are not normal; it does not by itself show that the curve is missing the wave.
    """)
    return


@app.cell
def _(
    LinearRegression,
    PolynomialFeatures,
    plt,
    qqplot,
    x_test,
    x_train,
    y_test,
    y_train,
):
    _figure, _axes = plt.subplots(1, 2, figsize=(10, 4))
    for _ax, _degree in zip(_axes, (2, 5)):
        _poly = PolynomialFeatures(degree=_degree, include_bias=False)
        _model = LinearRegression().fit(
            _poly.fit_transform(x_train.reshape(-1, 1)), y_train
        )
        _resid = y_test - _model.predict(_poly.transform(x_test.reshape(-1, 1)))
        qqplot(_resid, line="s", ax=_ax)
        _ax.set_title(f"Degree {_degree} test residuals")
    _figure.tight_layout()
    _figure
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Which argument of `PolynomialFeatures` leaves the column of ones out of the design? Replace `None` with its name.
    """)
    return


@app.cell
def _():
    student_bias = None
    print(student_bias)
    return (student_bias,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r'''
```python
student_bias = "include_bias"
```
**include_bias.** Set to `False`, the power columns start at `x` and the intercept is left to `LinearRegression`.
''')
        },
        lazy=True,
    )
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Why must `PolynomialFeatures` be fit on the training `x` only, then `transform` applied to the test `x`? Answer in one sentence in `student_transform_reason`.
    """)
    return


@app.cell
def _():
    student_transform_reason = None
    print(student_transform_reason)
    return (student_transform_reason,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
The power columns are defined from the training range and statistics; the test rows must use the **same** column definitions, not refit powers on test data (which would leak information and change the feature space).
""")
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Which degree has the lowest test RMSE in `poly_scores`? Store it as an integer in `student_best_degree`.
    """)
    return


@app.cell
def _(poly_scores):
    student_best_degree = None
    print(student_best_degree)
    return (student_best_degree,)


@app.cell(hide_code=True)
def _(mo, poly_scores):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
```python
student_best_degree = int(poly_scores["test_rmse"].idxmin())
student_best_degree
```
On this split the lowest test RMSE is at degree **5** (test RMSE about 1.75).
""")
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Is the fitted model linear in its coefficients? Replace `None` with `True` or `False` in `student_linear_in_coefs`.
    """)
    return


@app.cell
def _():
    student_linear_in_coefs = None
    print(student_linear_in_coefs)
    return (student_linear_in_coefs,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
```python
student_linear_in_coefs = True
```
Ordinary least squares still estimates one coefficient per column; the nonlinearity is in the **features** (`x`, `x²`, …), not in the coefficients.
""")
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    For degree 5, how much larger is test RMSE than training RMSE on this split? Store the gap in `student_rmse_gap` (test minus train).
    """)
    return


@app.cell
def _(poly_scores):
    student_rmse_gap = None
    print(student_rmse_gap)
    return (student_rmse_gap,)


@app.cell(hide_code=True)
def _(mo, poly_scores):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
```python
student_rmse_gap = poly_scores.loc[5, "test_rmse"] - poly_scores.loc[5, "train_rmse"]
student_rmse_gap
```
The gap is about **0.538** (test RMSE 1.754 minus training RMSE 1.216), a sign the high-degree curve is fitting training noise.
""")
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(mo, poly_scores):
    _best = int(poly_scores["test_rmse"].idxmin())
    _extra = poly_scores.loc[5, "test_rmse"] - poly_scores.loc[5, "train_rmse"]
    mo.md(f"""
    ## Conclusions

    **Key takeaways:**

    - The fitted equation stays linear in its coefficients. The bend comes from the power columns.
    - On this split, the lowest test RMSE is degree {_best}, at {poly_scores.loc[_best, "test_rmse"]:.3f}.
    - Degree 5 training RMSE is {poly_scores.loc[5, "train_rmse"]:.3f}. Its test RMSE is {poly_scores.loc[5, "test_rmse"]:.3f}. The test residuals are {_extra:.3f} larger than the training residuals.
    - The degree-2 residual plot is the place to see what a low-degree curve still misses.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check your understanding

    1. What columns does a degree-3 model add to `x`?
    2. Why is the orange curve drawn on a sorted grid?
    3. Why is `transform` used on the test values, rather than a new fit?
    4. What does a lower RMSE mean?
    5. Why can a higher degree look better on the training rows and worse on the test rows?
    6. Is the polynomial model nonlinear in its coefficients?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
1. **`x²` and `x³`:** With `include_bias=False`, the columns are `x`, `x²`, and `x³`.
2. **The rows are shuffled:** A line through the training rows in their split order jumps back and forth. A grid from 0 to 10 draws the function.
3. **The powers are already defined:** The training fit fixes the degree. The test rows only need those same columns.
4. **Smaller typical residuals:** RMSE is in the units of `y`.
5. **The extra powers can fit noise:** Training error can fall while test error rises.
6. **No:** Least squares still estimates one coefficient per column. The curve is in the columns.
""")
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - [PolynomialFeatures](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)
    - [LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
    - VanderPlas, J. (2017). *Python Data Science Handbook*. O'Reilly Media. Chapter 5.
    """)
    return


if __name__ == "__main__":
    app.run()
