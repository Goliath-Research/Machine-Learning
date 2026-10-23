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
    from sklearn.metrics import r2_score, root_mean_squared_error
    from sklearn.model_selection import GridSearchCV, train_test_split
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.preprocessing import StandardScaler

    return (
        GridSearchCV,
        KNeighborsRegressor,
        StandardScaler,
        r2_score,
        root_mean_squared_error,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # k-Nearest Neighbors Regression

    ## Objectives

    - Predict a numeric response from the average of the nearest training rows.
    - See how the prediction changes as k grows.
    - Choose k by cross-validation on the training rows.

    ## Background

    k-nearest neighbors stores the training rows. A new row is predicted as the average response of the k closest training rows. Distance treats every predictor unit as comparable, so the budgets are standardized first.

    A small k follows local wiggles. A large k smooths them. The test rows are for a final score. They are not the place to pick k.

    ## Datasets Used

    - **Advertising.** Budgets for TV, radio, and newspaper, and the sales that followed, in 200 markets.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The table

    Pandas reads the file from GitHub, which is what a browser session is allowed to download. On this computer the same file is `advertising.csv` in this folder. The browser rule that blocks some other sites is called **CORS**.

    The scaler is fit on the training budgets only, after the split. Sales stays in its original units, so RMSE is in sales units. Fitting the scaler on the whole table, including sales, would let the test markets change the means and would put the response on a different scale.
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
    return mo.Html(ads.head().to_html())


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## One predictor: TV

    The split uses `random_state=10`. Each panel is a different k. The orange steps are predictions on a fine grid of standardized TV, so the line shows the neighborhood averages instead of jumping between shuffled markets.
    """)
    return


@app.cell
def _(
    KNeighborsRegressor,
    StandardScaler,
    ads,
    np,
    plt,
    r2_score,
    train_test_split,
):
    tv_train, tv_test, sales_tv_train, sales_tv_test = train_test_split(
        ads[["TV"]], ads["Sales"], test_size=0.3, random_state=10
    )
    tv_scaler = StandardScaler().fit(tv_train)
    tv_train_z = tv_scaler.transform(tv_train)
    tv_test_z = tv_scaler.transform(tv_test)
    _grid = np.linspace(tv_train_z.min(), tv_train_z.max(), 200).reshape(-1, 1)
    _figure, _axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for _ax, _k in zip(_axes, (3, 5, 11)):
        _model = KNeighborsRegressor(n_neighbors=_k).fit(tv_train_z, sales_tv_train)
        _hat = _model.predict(tv_test_z)
        _ax.scatter(tv_train_z, sales_tv_train, s=20, color="cornflowerblue")
        _ax.scatter(tv_test_z, sales_tv_test, s=20, color="darkblue")
        _ax.plot(_grid, _model.predict(_grid), color="darkorange")
        _ax.set_title(f"k = {_k}\nTest R² = {r2_score(sales_tv_test, _hat):.3f}")
        _ax.set_xlabel("TV (training z-score)")
    _figure.tight_layout()
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Three budgets

    TV, radio, and newspaper are standardized from the training rows. The split uses `random_state=20`. There is no useful scatter once the predictor has three columns, so the comparison is R² and RMSE for k = 3 and k = 5.
    """)
    return


@app.cell
def _(
    KNeighborsRegressor,
    StandardScaler,
    ads,
    mo,
    pd,
    r2_score,
    root_mean_squared_error,
    train_test_split,
):
    budget_names = ["TV", "Radio", "Newspaper"]
    budgets_train, budgets_test, sales_train, sales_test = train_test_split(
        ads[budget_names], ads["Sales"], test_size=0.3, random_state=20
    )
    budget_scaler = StandardScaler().fit(budgets_train)
    budgets_train_z = budget_scaler.transform(budgets_train)
    budgets_test_z = budget_scaler.transform(budgets_test)
    knn_rows = []
    for _k in (3, 5):
        _model = KNeighborsRegressor(n_neighbors=_k).fit(budgets_train_z, sales_train)
        _train_hat = _model.predict(budgets_train_z)
        _test_hat = _model.predict(budgets_test_z)
        knn_rows.append(
            {
                "k": _k,
                "train_r2": r2_score(sales_train, _train_hat),
                "test_r2": r2_score(sales_test, _test_hat),
                "train_rmse": root_mean_squared_error(sales_train, _train_hat),
                "test_rmse": root_mean_squared_error(sales_test, _test_hat),
            }
        )
    knn_scores = pd.DataFrame(knn_rows).set_index("k")
    print(f"Train = {len(budgets_train)} \t Test = {len(budgets_test)}")
    return mo.Html(knn_scores.round(3).to_html())


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Choosing k

    The left panel scores every k from 1 to 30 on the test markets. That picture uses the test rows to pick k, so the winning k is no longer an honest test.

    The right-hand search stays inside the training rows. `GridSearchCV` with `cv=10` refits on nine folds and scores the tenth, ten times. Scikit-learn maximizes the score, and RMSE should be small, so the scoring name is `neg_root_mean_squared_error`. The plotted RMSE is the positive value, `-best_score_`.
    """)
    return


@app.cell
def _(
    GridSearchCV,
    KNeighborsRegressor,
    budgets_test_z,
    budgets_train_z,
    np,
    pd,
    plt,
    r2_score,
    root_mean_squared_error,
    sales_test,
    sales_train,
):
    k_values = list(range(1, 31))
    test_rmse_by_k = []
    for _k in k_values:
        _hat = (
            KNeighborsRegressor(n_neighbors=_k)
            .fit(budgets_train_z, sales_train)
            .predict(budgets_test_z)
        )
        test_rmse_by_k.append(root_mean_squared_error(sales_test, _hat))
    test_picture_k = k_values[int(np.argmin(test_rmse_by_k))]
    rmse_search = GridSearchCV(
        KNeighborsRegressor(),
        {"n_neighbors": k_values},
        cv=10,
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
    )
    r2_search = GridSearchCV(
        KNeighborsRegressor(),
        {"n_neighbors": k_values},
        cv=10,
        scoring="r2",
        n_jobs=1,
    )
    rmse_search.fit(budgets_train_z, sales_train)
    r2_search.fit(budgets_train_z, sales_train)
    cv_rmse_k = rmse_search.best_params_["n_neighbors"]
    cv_r2_k = r2_search.best_params_["n_neighbors"]
    cv_rmse = -rmse_search.best_score_
    cv_r2 = r2_search.best_score_
    chosen = KNeighborsRegressor(n_neighbors=cv_rmse_k).fit(budgets_train_z, sales_train)
    chosen_test_r2 = r2_score(sales_test, chosen.predict(budgets_test_z))
    chosen_test_rmse = root_mean_squared_error(sales_test, chosen.predict(budgets_test_z))
    _figure, _axes = plt.subplots(1, 2, figsize=(10, 4))
    _axes[0].plot(k_values, test_rmse_by_k, color="firebrick")
    _axes[0].axvline(test_picture_k, color="gray", linestyle="--")
    _axes[0].set_title(f"Test-set picture, lowest RMSE at k = {test_picture_k}")
    _axes[0].set_xlabel("k")
    _axes[0].set_ylabel("Test RMSE")
    _axes[1].plot(k_values, -rmse_search.cv_results_["mean_test_score"], color="firebrick", label="CV RMSE")
    _axes[1].plot(k_values, r2_search.cv_results_["mean_test_score"], color="steelblue", label="CV R²")
    _axes[1].axvline(cv_rmse_k, color="gray", linestyle="--")
    _axes[1].set_title("10-fold scores on the training rows")
    _axes[1].set_xlabel("k")
    _axes[1].legend()
    _figure.tight_layout()
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Which scoring string tells `GridSearchCV` to prefer a smaller RMSE? Replace `None` with that string.
    """)
    return


@app.cell
def _():
    student_rmse_scoring = None
    print(student_rmse_scoring)
    return (student_rmse_scoring,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r'''
```python
student_rmse_scoring = "neg_root_mean_squared_error"
```
**neg_root_mean_squared_error.** The search maximizes the score. The leading `neg_` makes a smaller RMSE a larger score. The RMSE itself is `-best_score_`.
''')
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(
    chosen_test_r2,
    chosen_test_rmse,
    cv_r2,
    cv_r2_k,
    cv_rmse,
    cv_rmse_k,
    knn_scores,
    mo,
    test_picture_k,
):
    mo.md(f"""
    ## Conclusions

    **Key takeaways:**

    - With one standardized budget, a larger k replaces the steps with a smoother average.
    - On the three-budget holdout, k = 3 has test R² {knn_scores.loc[3, "test_r2"]:.3f} and test RMSE {knn_scores.loc[3, "test_rmse"]:.3f}. k = 5 has test R² {knn_scores.loc[5, "test_r2"]:.3f} and test RMSE {knn_scores.loc[5, "test_rmse"]:.3f}.
    - The test-set picture is lowest at k = {test_picture_k}. That k was chosen with the test rows, so it is not a fresh score.
    - Ten-fold search on the training rows picks k = {cv_rmse_k} for RMSE, with mean CV RMSE {cv_rmse:.3f}. It picks k = {cv_r2_k} for R², with mean CV R² {cv_r2:.3f}.
    - The RMSE winner, scored once on the test markets, has R² {chosen_test_r2:.3f} and RMSE {chosen_test_rmse:.3f}.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check your understanding

    1. What does the model predict for a new market?
    2. Why are the budgets standardized, and why is sales left alone?
    3. Why is the scaler fit after the split?
    4. What happens to the prediction as k grows?
    5. Why is the RMSE scoring name negative?
    6. Why is the left-hand curve not the way to choose k?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
1. **The average of the neighbors:** It is the mean sales of the k closest training markets.
2. **Distance needs a common unit:** A dollar of newspaper and a dollar of TV are not otherwise on the same scale. Sales is the response, so it is not a feature to be scaled.
3. **The test markets stay out of the means:** The training budgets set the mean and the standard deviation. The test budgets only use those numbers.
4. **It gets smoother:** More neighbors average away local jumps.
5. **The search maximizes a score:** `neg_root_mean_squared_error` turns a small error into a large score. Undo the sign to read the RMSE.
6. **It spends the test set:** A k chosen because it won on the test rows no longer has an untouched score.
""")
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - [KNeighborsRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)
    - [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
    - VanderPlas, J. (2017). *Python Data Science Handbook*. O'Reilly Media. Chapter 5.
    """)
    return


if __name__ == "__main__":
    app.run()
