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
    import pandas as pd
    import seaborn as sns
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, root_mean_squared_error
    from sklearn.model_selection import train_test_split
    import statsmodels.api as sm

    plt.rcParams["figure.max_open_warning"] = 0
    sns.set_theme(style="whitegrid")

    return (
        LinearRegression,
        mo,
        pd,
        plt,
        r2_score,
        root_mean_squared_error,
        sm,
        sns,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Multiple Linear Regression

    ## Objectives

    - Predict sales from TV, radio, and newspaper budgets together.
    - Read which predictor has a coefficient distinguishable from zero.
    - Refit without a predictor that the t-test does not support, and compare the test error.

    ## Background

    Multiple linear regression uses one intercept and one slope per predictor. The slopes are partial: each one is the change in sales tied to that budget when the other budgets are held fixed.

    A large p-value on a slope means the data do not distinguish that slope from zero. Dropping the column then asks whether the test error moves.

    ## Datasets Used

    - **Advertising.** Budgets for TV, radio, and newspaper, and the sales that followed, in 200 markets.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The table

    Pandas reads the file from GitHub, which is what a browser session is allowed to download. On this computer the same file is `advertising.csv` in this folder. The browser rule that blocks some other sites is called **CORS**.
    """)
    return


@app.cell
def _():
    advertising_file = (
        "https://raw.githubusercontent.com/Goliath-Research/"
        "Machine-Learning/master/Regression/advertising.csv"
    )
    return (advertising_file,)


@app.cell
def _(advertising_file, mo, pd):
    # ads = pd.read_csv("advertising.csv")
    ads = pd.read_csv(advertising_file)
    print(f"{ads.shape[0]} markets, {ads.shape[1]} columns")
    mo.Html(ads.head().to_html())
    return mo.Html(ads.head().to_html())


@app.cell
def _(ads, plt, sns):
    _figure, _axes = plt.subplots(figsize=(5, 4))
    sns.heatmap(ads.corr(), annot=True, cmap="Blues", ax=_axes)
    _axes.set_title("Correlations")
    _figure
    return _figure


@app.cell(hide_code=True)
def _(ads, mo):
    _sales_corr = ads.corr()["Sales"].drop("Sales").sort_values(ascending=False)
    _lines = "\n".join(
        f"- **{_name}** and Sales: {_value:.3f}" for _name, _value in _sales_corr.items()
    )
    mo.md(f"""
    The last row of the heatmap is the correlation of each budget with sales.

    {_lines}

    The weakest correlation with sales is the smallest number in that list. The pair plot shows the same clouds.
    """)
    return


@app.cell
def _(ads, sns):
    _pairs = sns.pairplot(ads, corner=True)
    _pairs.figure.suptitle("Each pair of columns", y=1.02)
    pair_figure = _pairs.figure
    pair_figure
    return pair_figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A scikit-learn fit

    The holdout is 30 percent of the markets, fixed by `random_state=20`. The printed equation is the training fit. R² and RMSE are then computed on both sides of the split.
    """)
    return


@app.cell
def _(LinearRegression, ads, r2_score, root_mean_squared_error, train_test_split):
    budget_names = ["TV", "Radio", "Newspaper"]
    budgets_train, budgets_test, sales_train, sales_test = train_test_split(
        ads[budget_names], ads["Sales"], test_size=0.3, random_state=20
    )
    full_line = LinearRegression().fit(budgets_train, sales_train)
    full_train_hat = full_line.predict(budgets_train)
    full_test_hat = full_line.predict(budgets_test)
    full_train_r2 = r2_score(sales_train, full_train_hat)
    full_test_r2 = r2_score(sales_test, full_test_hat)
    full_test_rmse = root_mean_squared_error(sales_test, full_test_hat)
    _terms = " + ".join(
        f"{_coef:.4f}*{_name}" for _name, _coef in zip(budget_names, full_line.coef_)
    )
    print(f"Train = {len(budgets_train)} \t Test = {len(budgets_test)}")
    print(f"Sales = {full_line.intercept_:.4f} + {_terms}")
    print(f"Training R² = {full_train_r2:.3f}")
    print(f"Testing  R² = {full_test_r2:.3f}    RMSE = {full_test_rmse:.3f}")
    return (
        budget_names,
        budgets_test,
        budgets_train,
        full_test_r2,
        full_test_rmse,
        sales_test,
        sales_train,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The same fit in statsmodels

    `add_constant` inserts the intercept. The summary adds a t-test for each slope. `P>|t|` is the p-value for the hypothesis that the slope is zero.

    The F-test at the top asks a different question: whether all of the slopes could be zero at once.
    """)
    return


@app.cell
def _(budgets_train, mo, pd, sales_train, sm):
    full_ols = sm.OLS(sales_train, sm.add_constant(budgets_train)).fit()
    print(
        f"R² = {full_ols.rsquared:.4f}, F = {full_ols.fvalue:.2f} "
        f"(p = {full_ols.f_pvalue:.2e})"
    )
    _coef_table = pd.DataFrame(
        {
            "coef": full_ols.params,
            "std err": full_ols.bse,
            "t": full_ols.tvalues,
            "P>|t|": full_ols.pvalues,
        }
    )
    mo.Html(_coef_table.round(4).to_html())


@app.cell(hide_code=True)
def _(full_ols, mo):
    _largest = full_ols.pvalues.drop("const").idxmax()
    mo.md(f"""
    The largest slope p-value is **{_largest}**, at {full_ols.pvalues[_largest]:.3f}.

    The model F-test p-value is {full_ols.f_pvalue:.3e}. Training R² is {full_ols.rsquared:.3f}.

    The next fit drops {_largest} and keeps the other budgets.
    """)
    return


@app.cell
def _(
    budgets_test,
    budgets_train,
    full_ols,
    r2_score,
    root_mean_squared_error,
    mo,
    pd,
    sales_test,
    sales_train,
    sm,
):
    dropped_budget = full_ols.pvalues.drop("const").idxmax()
    kept = [name for name in budgets_train.columns if name != dropped_budget]
    reduced_ols = sm.OLS(sales_train, sm.add_constant(budgets_train[kept])).fit()
    reduced_test_hat = reduced_ols.predict(sm.add_constant(budgets_test[kept]))
    reduced_test_r2 = r2_score(sales_test, reduced_test_hat)
    reduced_test_rmse = root_mean_squared_error(sales_test, reduced_test_hat)
    reduced_train_r2 = r2_score(
        sales_train, reduced_ols.predict(sm.add_constant(budgets_train[kept]))
    )
    print(
        f"R² = {reduced_ols.rsquared:.4f}, F = {reduced_ols.fvalue:.2f} "
        f"(p = {reduced_ols.f_pvalue:.2e})"
    )
    _coef_table = pd.DataFrame(
        {
            "coef": reduced_ols.params,
            "std err": reduced_ols.bse,
            "t": reduced_ols.tvalues,
            "P>|t|": reduced_ols.pvalues,
        }
    )
    mo.Html(_coef_table.round(4).to_html())
    print(f"Dropped column: {dropped_budget}")
    print(f"Reduced training R² = {reduced_train_r2:.3f}")
    print(
        f"Reduced testing  R² = {reduced_test_r2:.3f}    RMSE = {reduced_test_rmse:.3f}"
    )
    return dropped_budget, kept, reduced_test_r2, reduced_test_rmse, reduced_train_r2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Which row of the summary tests all of the slopes together? Replace `None` with the name of that statistic.
    """)
    return


@app.cell
def _():
    student_joint_test = None
    print(student_joint_test)
    return (student_joint_test,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r'''
```python
student_joint_test = "F-statistic"
```
**F-statistic.** Its p-value asks whether every slope could be zero. Each `P>|t|` asks that question for one slope.
''')
        },
        lazy=True,
    )
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Which budget has the **strongest** correlation with sales in the heatmap? Store its name in `student_strongest_budget`.
    """)
    return


@app.cell
def _(ads):
    student_strongest_budget = None
    print(student_strongest_budget)
    return (student_strongest_budget,)


@app.cell(hide_code=True)
def _(ads, mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
```python
student_strongest_budget = ads.corr()["Sales"].drop("Sales").idxmax()
student_strongest_budget
```
**TV** has the largest correlation with sales (about 0.901). Correlation is not the same as a partial slope, but it matches the dominant pattern in the pair plot.
""")
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    What does a partial slope mean when several budgets are in the model? Complete: "It is the change in sales when that budget increases by one unit while ___."
    """)
    return


@app.cell
def _():
    student_partial_phrase = None
    print(student_partial_phrase)
    return (student_partial_phrase,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
**…while the other budgets in the model are held fixed.** That is what "partial" means in multiple regression.
""")
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Which predictor has the largest p-value on its slope in `full_ols` (excluding the intercept)? Store the name in `student_largest_p`.
    """)
    return


@app.cell
def _(full_ols):
    student_largest_p = None
    print(student_largest_p)
    return (student_largest_p,)


@app.cell(hide_code=True)
def _(full_ols, mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
```python
student_largest_p = full_ols.pvalues.drop("const").idxmax()
student_largest_p
```
**Newspaper** has the largest p-value (about 0.93), so this sample does not distinguish its slope from zero.
""")
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    After dropping that column, did test RMSE change much? Store `True` or `False` in `student_rmse_stable` if test RMSE moved by less than 0.05 from the full model.
    """)
    return


@app.cell
def _(full_test_rmse, reduced_test_rmse):
    student_rmse_stable = None
    print(student_rmse_stable)
    return (student_rmse_stable,)


@app.cell(hide_code=True)
def _(full_test_rmse, mo, reduced_test_rmse):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
```python
student_rmse_stable = abs(full_test_rmse - reduced_test_rmse) < 0.05
student_rmse_stable
```
**True.** Full-model test RMSE is about 2.093 and reduced-model test RMSE is about 2.093 on this split—the drop barely moves prediction error.
""")
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(
    dropped_budget,
    full_ols,
    full_test_r2,
    full_test_rmse,
    kept,
    mo,
    reduced_test_r2,
    reduced_test_rmse,
    reduced_train_r2,
):
    mo.md(f"""
    ## Conclusions

    **Key takeaways:**

    - The full training fit has R² {full_ols.rsquared:.3f}. The joint F-test p-value is {full_ols.f_pvalue:.3e}.
    - **{dropped_budget}** has the largest slope p-value, {full_ols.pvalues[dropped_budget]:.3f}. The reduced model keeps {", ".join(kept)}.
    - Full model test R² is {full_test_r2:.3f} and test RMSE is {full_test_rmse:.3f}.
    - Reduced model training R² is {reduced_train_r2:.3f}. Test R² is {reduced_test_r2:.3f} and test RMSE is {reduced_test_rmse:.3f}.
    - A slope that the t-test does not support can leave the test error almost unchanged. The comparison above is the check, on this split.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check your understanding

    1. What does one slope mean when the other budgets stay fixed?
    2. Which correlation with sales is the weakest in the heatmap?
    3. What does a large `P>|t|` say about that slope?
    4. What does the F-test ask?
    5. Why compare test RMSE after dropping a column?
    6. Which rows are used to estimate the slopes?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
1. **A partial change:** It is the change in sales tied to that budget when the other budgets in the model are held fixed.
2. **Newspaper:** Its correlation with sales is the weakest of the three (about 0.158). TV is the strongest.
3. **The data do not separate it from zero:** A large p-value is not proof that the budget never matters. It says this sample does not detect a slope.
4. **Whether every slope could be zero together:** That is separate from the test of one slope.
5. **To see whether prediction changes:** A column can fail its t-test and still be checked on the test markets.
6. **The training markets:** The test markets are scored after the slopes are set.
""")
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - [LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
    - [statsmodels OLS](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html)
    - VanderPlas, J. (2017). *Python Data Science Handbook*. O'Reilly Media. Chapter 5.
    """)
    return


if __name__ == "__main__":
    app.run()
