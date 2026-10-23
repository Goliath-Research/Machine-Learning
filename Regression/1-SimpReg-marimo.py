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
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, root_mean_squared_error
    from sklearn.model_selection import train_test_split

    return LinearRegression, r2_score, root_mean_squared_error, train_test_split


@app.cell
def _():
    import statsmodels.api as sm
    from statsmodels.graphics.gofplots import qqplot

    return qqplot, sm


@app.cell
def _(plt, sns):
    def numeric_panel(frame, title):
        _figure, _axes = plt.subplots(2, 2, figsize=(10, 7))
        sns.histplot(data=frame, kde=True, ax=_axes[0, 0])
        _axes[0, 0].set_title("Histograms")
        sns.violinplot(data=frame, ax=_axes[0, 1])
        _axes[0, 1].set_title("Violin plots")
        sns.boxplot(data=frame, ax=_axes[1, 0])
        _axes[1, 0].set_title("Box plots")
        sns.boxenplot(data=frame, ax=_axes[1, 1])
        _axes[1, 1].set_title("Boxen plots")
        _figure.suptitle(title)
        _figure.tight_layout()
        return _figure

    return (numeric_panel,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Simple Linear Regression

    ## Objectives

    - Fit a line for sales against TV advertising spend.
    - Score that line on rows it did not see, with R² and root mean squared error.
    - Read a statsmodels summary and a Q-Q plot of the residuals.

    ## Background

    Simple linear regression predicts one numeric response from one numeric predictor. The fitted line is `sales = intercept + slope * TV`. R² is the share of the variance in sales that the line accounts for. RMSE is the typical size of a prediction error, in the same units as sales.

    ## Datasets Used

    - **Advertising.** Budgets for TV, radio, and newspaper, and the sales that followed, in 200 markets. This lesson uses TV as the only predictor.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The table

    Each row is one market. The columns are the variables. The feature matrix `X` holds the predictors. The target `y` is the column to predict. For regression, `y` is numeric.

    Pandas can read the file from GitHub, which is what a browser session is allowed to download. On this computer the same file is `advertising.csv` in this folder. The browser rule that blocks some other sites is called **CORS**.
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
    features = ads[["TV", "Radio", "Newspaper"]]
    sales = ads["Sales"]
    print(f"{ads.shape[0]} markets, {ads.shape[1]} columns")
    return mo.Html(ads.head().to_html())


@app.cell
def _(features, numeric_panel):
    feature_panel = numeric_panel(features, "Advertising budgets")
    return feature_panel


@app.cell
def _(numeric_panel, sales):
    sales_panel = numeric_panel(sales.to_frame(), "Sales")
    return sales_panel


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Five steps in scikit-learn

    1. Choose the model class. Here that class is `LinearRegression`.
    2. Set the hyperparameters before the fit. `fit_intercept=True` tells the line it may cross the sales axis somewhere other than zero.
    3. Arrange `TV` as a column and `Sales` as the target. Scikit-learn expects the predictors as a table, even when there is only one column.
    4. Fit on the training rows.
    5. Predict the test rows. Those markets were not used to place the line.

    The holdout is 30 percent of the markets, fixed by `random_state=20`. Learned quantities such as `coef_` and `intercept_` end in an underscore.
    """)
    return


@app.cell
def _(LinearRegression, ads, np, pd, plt, sns, train_test_split):
    tv = ads[["TV"]]
    sales_target = ads["Sales"]
    tv_train, tv_test, sales_train, sales_test = train_test_split(
        tv, sales_target, test_size=0.3, random_state=20
    )
    line = LinearRegression(fit_intercept=True)
    line.fit(tv_train, sales_train)
    print(f"Train = {len(tv_train)} markets \t Test = {len(tv_test)} markets")
    print(f"Sales = {line.intercept_:.2f} + {line.coef_[0]:.4f} * TV")
    _figure, _axes = plt.subplots(figsize=(6, 4))
    sns.scatterplot(
        x=tv_train["TV"], y=sales_train, s=60, color="cornflowerblue", label="Training", ax=_axes
    )
    sns.scatterplot(
        x=tv_test["TV"], y=sales_test, s=60, color="darkblue", label="Test", ax=_axes
    )
    _xs = pd.DataFrame({"TV": np.linspace(tv["TV"].min(), tv["TV"].max(), 100)})
    _axes.plot(_xs["TV"], line.predict(_xs), color="firebrick", label="Fitted line")
    _axes.set_title("TV spend and sales")
    _axes.legend()
    return _figure


@app.cell
def _(line, mo, pd, r2_score, root_mean_squared_error, sales_test, sales_train, tv_test, tv_train):
    train_pred = line.predict(tv_train)
    test_pred = line.predict(tv_test)
    train_r2 = r2_score(sales_train, train_pred)
    test_r2 = r2_score(sales_test, test_pred)
    train_rmse = root_mean_squared_error(sales_train, train_pred)
    test_rmse = root_mean_squared_error(sales_test, test_pred)
    preview = pd.DataFrame(
        {
            "TV": tv_test["TV"].to_numpy(),
            "sales": sales_test.to_numpy(),
            "predicted": test_pred,
        }
    )
    print(f"Training R² = {train_r2:.3f}    RMSE = {train_rmse:.2f}")
    print(f"Testing  R² = {test_r2:.3f}    RMSE = {test_rmse:.2f}")
    return mo.Html(preview.head().round(2).to_html())


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    R² near 1 means the line accounts for most of the up-and-down movement in sales. RMSE is in sales units. A lower RMSE means the markets sit closer to the line.

    Scikit-learn stores the slope and the intercept. It does not print a t-test for either one. Statsmodels does.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Which hyperparameter allows the line to meet the sales axis away from zero? Replace `None` with its name.
    """)
    return


@app.cell
def _():
    student_intercept = None
    print(student_intercept)
    return (student_intercept,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r'''
```python
student_intercept = "fit_intercept"
```
**fit_intercept.** Set to `True`, the fit estimates an intercept. Set to `False`, the line is forced through the origin.
''')
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The same line in statsmodels

    `OLS` is ordinary least squares, the same criterion scikit-learn uses for this line. `add_constant` inserts the intercept column. The summary adds standard errors, t-tests, and a note on whether the slope is distinguishable from zero.

    A residual is `actual sales - predicted sales`. On a Q-Q plot, residuals from a normal distribution lie along the reference line. A bend in that plot means a straight line is missing some of the shape in the TV-and-sales cloud.
    """)
    return


@app.cell
def _(sales_train, sm, tv_train):
    ols_line = sm.OLS(sales_train, sm.add_constant(tv_train)).fit()
    print(ols_line.summary())


@app.cell
def _(ols_line, plt, qqplot, sales_test, sales_train, test_pred, train_pred):
    _figure, _axes = plt.subplots(1, 2, figsize=(10, 4))
    qqplot(sales_train.to_numpy() - train_pred, line="s", ax=_axes[0])
    _axes[0].set_title("Training residuals")
    qqplot(sales_test.to_numpy() - test_pred, line="s", ax=_axes[1])
    _axes[1].set_title("Test residuals")
    _figure.tight_layout()
    print(f"Statsmodels slope = {ols_line.params['TV']:.4f}")
    print(f"Statsmodels intercept = {ols_line.params['const']:.2f}")
    return _figure


@app.cell(hide_code=True)
def _(line, mo, test_r2, test_rmse, train_r2):
    mo.md(f"""
    ## Conclusions

    **Key takeaways:**

    - One extra unit of TV spend goes with a slope of {line.coef_[0]:.4f} sales units on this training fit. The intercept is {line.intercept_:.2f}.
    - Training R² is {train_r2:.3f}. Test R² is {test_r2:.3f}. The two are close, so the line is not only memorizing the training markets.
    - Test RMSE is {test_rmse:.2f} sales units.
    - The Q-Q plots are not a perfect straight line. TV spend does not capture every bend in sales.
    - Statsmodels reports the same line and adds a test of the slope. Scikit-learn is the shorter path to a prediction.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check your understanding

    1. What is stored in `X`, and what is stored in `y`?
    2. Why is TV reshaped into a column before `fit`?
    3. What does the trailing underscore on `coef_` mean?
    4. What does RMSE measure?
    5. What does a bend away from the Q-Q reference line suggest?
    6. Which rows are used to place the line?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
1. **Predictors and the response:** `X` holds TV, radio, and newspaper in the first view. The fitted line uses only TV. `y` is sales.
2. **Scikit-learn wants a table:** One predictor is still a two-dimensional table with one column.
3. **It was learned during fit:** Values the model estimates, such as the slope, are marked that way.
4. **The typical size of a residual:** It is the square root of the average squared gap between actual and predicted sales.
5. **The residuals are not normal:** A straight line is missing some pattern. That does not by itself name the missing pattern.
6. **The training markets:** The test markets are scored after the line is already placed.
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
