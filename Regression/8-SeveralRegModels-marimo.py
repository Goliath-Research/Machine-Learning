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
    import pandas as pd
    import seaborn as sns

    plt.rcParams["figure.max_open_warning"] = 0
    sns.set_theme(style="whitegrid")

    return mo, pd, plt, sns


@app.cell
def _():
    from pathlib import Path

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score, root_mean_squared_error
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor

    return (
        DecisionTreeRegressor,
        KNeighborsRegressor,
        Path,
        RandomForestRegressor,
        SVR,
        StandardScaler,
        r2_score,
        root_mean_squared_error,
        train_test_split,
    )


@app.cell
def _():
    import statsmodels.api as sm

    return (sm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Several Regression Methods

    ## Objectives

    - Fit a price model that includes a categorical body style.
    - Read the statsmodels summary, including the category that was left out.
    - Score linear regression, k-nearest neighbors, support vector regression, a tree, and a forest on the same test rows.

    ## Background

    A category such as body style cannot enter a numeric model as text. One-hot columns mark each style with 0 or 1. A linear model drops one of those columns so the intercept has a baseline style. A tree can use a column for every style.

    k-nearest neighbors and the RBF kernel measure distance, so their numeric columns are standardized from the training rows. The tree and the forest split on raw units and do not need that step. The price stays in dollars for every model, so RMSE is in dollars.

    ## Datasets Used

    - **Automobile imports.** The UCI Automobile Data Set. This lesson keeps horsepower, city mpg, highway mpg, body style, and price.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The table

    The original file is the UCI imports file, with `?` marking a missing value. A copy sits next to this lesson as `imports-85.data`. A browser session cannot read that local path, so the cell uses the GitHub copy when the local file is absent. The browser rule that blocks some other sites is called **CORS**.

    Rows with a missing value in the five columns are dropped. The models below use only complete rows.
    """)
    return


@app.cell
def _(Path, mo, pd):
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
    local_auto = Path(__file__).with_name("imports-85.data")
    # autos = pd.read_csv("imports-85.data", header=None, names=auto_headers, na_values="?")
    auto_source = (
        local_auto
        if local_auto.exists()
        else (
            "https://raw.githubusercontent.com/Goliath-Research/"
            "Machine-Learning/master/Regression/imports-85.data"
        )
    )
    cars = (
        pd.read_csv(auto_source, header=None, names=auto_headers, na_values="?")
        [["horsepower", "city_mpg", "highway_mpg", "body_style", "price"]]
        .dropna()
        .reset_index(drop=True)
    )
    print(f"{len(cars)} complete rows, from {auto_source}")
    return mo.Html(cars.head().to_html())


@app.cell
def _(cars, plt, sns):
    _figure, _axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.scatterplot(
        data=cars, x="horsepower", y="price", hue="body_style", ax=_axes[0]
    )
    _axes[0].set_title("Price and horsepower")
    sns.countplot(x=cars["body_style"], ax=_axes[1], color="steelblue")
    _axes[1].set_title("Body style")
    _axes[1].tick_params(axis="x", rotation=30)
    _figure.tight_layout()
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A linear model with a baseline style

    `drop_first=True` omits one body style. That omitted style is the baseline: its price level sits in the intercept, and the other style coefficients are differences from it.

    The numeric columns stay in their original units here. A slope for horsepower is then dollars per horsepower. The split is 30 percent, fixed by `random_state=20`.
    """)
    return


@app.cell
def _(cars, pd, sm, train_test_split):
    style_drop = pd.get_dummies(cars["body_style"], drop_first=True).astype(int)
    omitted_style = sorted(set(cars["body_style"]) - set(style_drop.columns))
    numeric_names = ["horsepower", "city_mpg", "highway_mpg"]
    design = pd.concat([style_drop, cars[numeric_names]], axis=1)
    price = cars["price"]
    design_train, design_test, price_train, price_test = train_test_split(
        design, price, test_size=0.3, random_state=20
    )
    price_ols = sm.OLS(price_train, sm.add_constant(design_train)).fit()
    print(f"Omitted body style: {omitted_style[0]}")
    print(f"Train = {len(design_train)} \t Test = {len(price_test)}")
    print(price_ols.summary())
    return (
        design_test,
        design_train,
        numeric_names,
        omitted_style,
        price_ols,
        price_test,
        price_train,
        style_drop,
    )


@app.cell(hide_code=True)
def _(mo, omitted_style, price_ols):
    _slopes = price_ols.pvalues.drop("const").sort_values()
    _lines = "\n".join(
        f"- **{_name}**: coefficient {price_ols.params[_name]:,.1f}, p-value {_pvalue:.3f}"
        for _name, _pvalue in _slopes.items()
    )
    mo.md(f"""
    Training R² is {price_ols.rsquared:.3f}. Adjusted R² is {price_ols.rsquared_adj:.3f}. The F-test p-value is {price_ols.f_pvalue:.3e}.

    The omitted style is **{omitted_style[0]}**. Each style coefficient is a difference from that baseline, in dollars, with the numeric columns held fixed.

    {_lines}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Five predictors of price

    The tree and the forest see one column per body style, plus the three numeric columns in original units. Neighbors and the RBF support-vector model see the same style columns, with the three numeric columns replaced by training z-scores.

    The neighbor model uses k = 3. The support-vector model uses an RBF kernel, `C=1000`, and `epsilon=500` dollars. The tree requires at least 15 training rows in a leaf. The forest averages 100 trees with `bootstrap=False`, so each tree sees every training row.
    """)
    return


@app.cell
def _(
    DecisionTreeRegressor,
    KNeighborsRegressor,
    RandomForestRegressor,
    SVR,
    StandardScaler,
    cars,
    numeric_names,
    pd,
    r2_score,
    root_mean_squared_error,
    train_test_split,
):
    style_full = pd.get_dummies(cars["body_style"]).astype(int)
    full_design = pd.concat([style_full, cars[numeric_names]], axis=1)
    full_train, full_test, y_train, y_test = train_test_split(
        full_design, cars["price"], test_size=0.3, random_state=20
    )
    scaler = StandardScaler().fit(full_train[numeric_names])
    scaled_train = full_train.copy()
    scaled_test = full_test.copy()
    scaled_train[numeric_names] = scaler.transform(full_train[numeric_names])
    scaled_test[numeric_names] = scaler.transform(full_test[numeric_names])

    neighbors = KNeighborsRegressor(n_neighbors=3).fit(scaled_train, y_train)
    support = SVR(kernel="rbf", C=1000, epsilon=500).fit(scaled_train, y_train)
    price_tree = DecisionTreeRegressor(random_state=0, min_samples_leaf=15).fit(
        full_train, y_train
    )
    price_forest = RandomForestRegressor(
        n_estimators=100, bootstrap=False, random_state=0
    ).fit(full_train, y_train)
    fitted_models = {
        "k-nearest neighbors": (neighbors, scaled_train, scaled_test),
        "support vectors": (support, scaled_train, scaled_test),
        "tree": (price_tree, full_train, full_test),
        "forest": (price_forest, full_train, full_test),
    }
    score_rows = []
    for _name, (_model, _train_x, _test_x) in fitted_models.items():
        score_rows.append(
            {
                "model": _name,
                "train_r2": r2_score(y_train, _model.predict(_train_x)),
                "test_r2": r2_score(y_test, _model.predict(_test_x)),
                "train_rmse": root_mean_squared_error(y_train, _model.predict(_train_x)),
                "test_rmse": root_mean_squared_error(y_test, _model.predict(_test_x)),
            }
        )
    model_scores = pd.DataFrame(score_rows).set_index("model")
    print(f"Same split size: train {len(y_train)}, test {len(y_test)}")
    print(model_scores.round(1).to_string())


@app.cell
def _(
    design_test,
    design_train,
    model_scores,
    mo,
    pd,
    price_ols,
    price_test,
    price_train,
    r2_score,
    root_mean_squared_error,
    sm,
):
    ols_train_hat = price_ols.predict(sm.add_constant(design_train))
    ols_test_hat = price_ols.predict(sm.add_constant(design_test))
    comparison = pd.concat(
        [
            pd.DataFrame(
                [
                    {
                        "model": "linear",
                        "train_r2": r2_score(price_train, ols_train_hat),
                        "test_r2": r2_score(price_test, ols_test_hat),
                        "train_rmse": root_mean_squared_error(price_train, ols_train_hat),
                        "test_rmse": root_mean_squared_error(price_test, ols_test_hat),
                    }
                ]
            ).set_index("model"),
            model_scores,
        ]
    )
    best_rmse = comparison["test_rmse"].idxmin()
    best_r2 = comparison["test_r2"].idxmax()
    print(comparison.round(1).to_string())
    return mo.Html(comparison.round(1).to_html())


@app.cell
def _(plt, price_forest, sns):
    _names = list(price_forest.feature_names_in_)
    _figure, _axes = plt.subplots(figsize=(7, 4))
    sns.barplot(
        x=_names,
        y=price_forest.feature_importances_,
        color="steelblue",
        ax=_axes,
    )
    _axes.set_title("Forest importance")
    _axes.tick_params(axis="x", rotation=30)
    _figure.tight_layout()
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Which argument of `get_dummies` omits one category so a linear model has a baseline? Replace `None` with its name.
    """)
    return


@app.cell
def _():
    student_baseline = None
    print(student_baseline)
    return (student_baseline,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r'''
```python
student_baseline = "drop_first"
```
**drop_first.** The omitted style is the baseline. Its level is inside the intercept, and the other style coefficients are differences from it.
''')
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(best_r2, best_rmse, comparison, mo, omitted_style, price_ols):
    mo.md(f"""
    ## Conclusions

    **Key takeaways:**

    - The linear model omits **{omitted_style[0]}**. Training R² is {price_ols.rsquared:.3f} and the F-test p-value is {price_ols.f_pvalue:.3e}. The condition number is {price_ols.condition_number:,.0f}: city mpg and highway mpg move together, so those two slopes are hard to separate.
    - On this test split, the smallest RMSE is **{best_rmse}**, at {comparison.loc[best_rmse, "test_rmse"]:,.0f} dollars. The largest test R² is **{best_r2}**, at {comparison.loc[best_r2, "test_r2"]:.3f}.
    - Neighbors and the RBF model see z-scored horsepower and mpg. The linear model, the tree, and the forest see the original units. Price is in dollars for every row of the table.
    - A training R² close to 1, next to a lower test R², means the fit is using the training cars more tightly than it can use a new car.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check your understanding

    1. Why does a linear model drop one body-style column?
    2. What is the meaning of a style coefficient?
    3. Why are horsepower and mpg standardized for neighbors and the RBF model?
    4. Why can the tree skip that standardization?
    5. What unit is RMSE in for every model in the table?
    6. Which score decides the model named in the conclusions?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
1. **So the intercept has a baseline:** With a column for every style, the styles add up to the intercept column and the slopes are not uniquely defined.
2. **A difference from the omitted style:** It is the dollar gap associated with that style when the numeric columns are held fixed.
3. **They measure distance:** Horsepower and miles per gallon are otherwise on different scales, so one of them would dominate the neighbors and the kernel.
4. **A split looks at one column at a time:** The units of horsepower do not have to match the units of mpg for a threshold on one of them to be chosen.
5. **Dollars:** The response was never standardized, so the root mean squared error is a typical dollar gap.
6. **The test RMSE and the test R²:** The conclusions name the row that wins each of those columns on this split.
""")
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - [Automobile Data Set](https://archive.ics.uci.edu/dataset/10/automobile), UCI Machine Learning Repository
    - [statsmodels OLS](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html)
    - VanderPlas, J. (2017). *Python Data Science Handbook*. O'Reilly Media. Chapter 5.
    """)
    return


if __name__ == "__main__":
    app.run()
