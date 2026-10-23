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
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR

    return (
        GridSearchCV,
        SVR,
        StandardScaler,
        r2_score,
        root_mean_squared_error,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Support Vector Regression

    ## Objectives

    - Fit an RBF support-vector curve and mark the points that define it.
    - Change epsilon, C, and gamma one at a time.
    - Search a small grid on the training rows, then repeat the idea on the advertising budgets.

    ## Background

    Support vector regression looks for a function with a tube of width epsilon around it. Points inside the tube are not penalized. Points outside it are the support vectors, and they pull the function.

    `C` sets how hard those outside points are pulled. `gamma` sets how local an RBF kernel is. A large gamma lets each point affect only its neighbors.

    ## Datasets Used

    - **Synthetic curve.** One hundred values of `x` from 0 to 10. The response is `sin(x) * x` plus normal noise. The random seed is 50.
    - **Advertising.** Budgets for TV, radio, and newspaper, and the sales that followed, in 200 markets.
    """)
    return


@app.cell
def _(np, pd, plt, sns, train_test_split):
    np.random.seed(50)
    curve = pd.DataFrame(
        {
            "x": np.random.uniform(0, 10, size=100),
            "y": 0.0,
        }
    )
    curve["y"] = np.sin(curve["x"]) * curve["x"] + np.random.normal(0, 1, size=100)
    x_train, x_test, y_train, y_test = train_test_split(
        curve["x"].to_numpy(),
        curve["y"].to_numpy(),
        test_size=0.3,
        random_state=10,
    )
    print(f"Train = {len(x_train)} points \t Test = {len(x_test)} points")
    _figure, _axes = plt.subplots(figsize=(6, 4))
    sns.scatterplot(x=x_train, y=y_train, s=40, color="cornflowerblue", label="Training", ax=_axes)
    sns.scatterplot(x=x_test, y=y_test, s=40, color="darkblue", label="Test", ax=_axes)
    _axes.set_title("sin(x) * x, plus noise")
    return _figure, x_test, x_train, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Epsilon, C, and gamma

    Each panel fits on the training points and reports R² on the test points. The dashed lines are the tube, one epsilon above and below the curve. The open circles are the support vectors.

    The curve is drawn on a sorted grid. A line through the test rows in split order would zigzag.
    """)
    return


@app.cell
def _(SVR, np, plt, r2_score, x_test, x_train, y_test, y_train):
    def _draw(axes, epsilon, C, gamma):
        model = SVR(kernel="rbf", C=C, epsilon=epsilon, gamma=gamma)
        model.fit(x_train.reshape(-1, 1), y_train)
        grid = np.linspace(0, 10, 300)
        curve_hat = model.predict(grid.reshape(-1, 1))
        test_r2 = r2_score(y_test, model.predict(x_test.reshape(-1, 1)))
        support = model.support_
        axes.scatter(x_train, y_train, s=18, color="cornflowerblue")
        axes.scatter(
            x_train[support],
            y_train[support],
            s=40,
            facecolors="none",
            edgecolors="black",
        )
        axes.plot(grid, curve_hat, color="dimgray")
        axes.plot(grid, curve_hat + epsilon, color="dimgray", linestyle="--")
        axes.plot(grid, curve_hat - epsilon, color="dimgray", linestyle="--")
        axes.set_title(
            f"ε={epsilon:g}, C={C:g}, γ={gamma:g}\n"
            f"Test R²={test_r2:.3f}, support={len(support)}"
        )

    _figure, _axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    _draw(_axes[0, 0], epsilon=1, C=1, gamma=0.1)
    _draw(_axes[0, 1], epsilon=4, C=1, gamma=0.1)
    _draw(_axes[1, 0], epsilon=1, C=100, gamma=0.1)
    _draw(_axes[1, 1], epsilon=1, C=100, gamma=0.05)
    _figure.tight_layout()
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A wider epsilon swallows more points and can miss the wave. A larger C pays more attention to the points outside the tube. A smaller gamma smooths the RBF bumps.

    The next cell searches a short grid of those three values. The search uses five folds of the training points. The test points are scored once, after the search.
    """)
    return


@app.cell
def _(GridSearchCV, SVR, r2_score, root_mean_squared_error, x_test, x_train, y_test, y_train):
    curve_search = GridSearchCV(
        SVR(kernel="rbf"),
        {
            "C": [1, 10, 100],
            "epsilon": [0.5, 1],
            "gamma": [0.01, 0.05, 0.1],
        },
        cv=5,
        scoring="r2",
        n_jobs=1,
    )
    curve_search.fit(x_train.reshape(-1, 1), y_train)
    curve_test_hat = curve_search.predict(x_test.reshape(-1, 1))
    curve_test_r2 = r2_score(y_test, curve_test_hat)
    curve_test_rmse = root_mean_squared_error(y_test, curve_test_hat)
    print(f"Best training-fold parameters: {curve_search.best_params_}")
    print(f"Mean CV R² = {curve_search.best_score_:.3f}")
    print(f"Test R² = {curve_test_r2:.3f}    RMSE = {curve_test_rmse:.3f}")
    print(f"Support vectors = {len(curve_search.best_estimator_.support_)}")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Advertising

    Pandas reads the file from GitHub. On this computer the same file is `advertising.csv` in this folder. The browser rule that blocks some other sites is called **CORS**.

    TV, radio, and newspaper are standardized from the training rows. Sales is standardized from the training rows as well, so epsilon is in training standard deviations of sales. R² does not change when both the response and the predictions are shifted and scaled together. RMSE below is converted back to sales units.

    The search uses the predictor columns only. Including the sales column as a feature would hand the model the answer.
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
def _(
    GridSearchCV,
    SVR,
    StandardScaler,
    advertising_file,
    mo,
    pd,
    r2_score,
    root_mean_squared_error,
    train_test_split,
):
    # ads = pd.read_csv("advertising.csv")
    ads = pd.read_csv(advertising_file)
    budget_names = ["TV", "Radio", "Newspaper"]
    budgets_train, budgets_test, sales_train, sales_test = train_test_split(
        ads[budget_names], ads["Sales"], test_size=0.3, random_state=20
    )
    budget_scaler = StandardScaler().fit(budgets_train)
    sales_scaler = StandardScaler().fit(sales_train.to_frame())
    budgets_train_z = budget_scaler.transform(budgets_train)
    budgets_test_z = budget_scaler.transform(budgets_test)
    sales_train_z = sales_scaler.transform(sales_train.to_frame()).ravel()
    sales_scale = float(sales_scaler.scale_[0])

    def _sales_rmse(model):
        predicted_z = model.predict(budgets_test_z)
        predicted = sales_scaler.inverse_transform(predicted_z.reshape(-1, 1)).ravel()
        return root_mean_squared_error(sales_test, predicted), r2_score(sales_test, predicted)

    manual_rows = []
    for _epsilon, _C in ((1, 1), (0.5, 1), (0.5, 10)):
        _model = SVR(kernel="rbf", epsilon=_epsilon, C=_C, gamma="scale").fit(
            budgets_train_z, sales_train_z
        )
        _rmse, _r2 = _sales_rmse(_model)
        manual_rows.append(
            {
                "epsilon": _epsilon,
                "C": _C,
                "support_vectors": len(_model.support_),
                "test_r2": _r2,
                "test_rmse": _rmse,
            }
        )
    manual_scores = pd.DataFrame(manual_rows)
    ad_search = GridSearchCV(
        SVR(kernel="rbf"),
        {
            "C": [0.1, 1, 10],
            "epsilon": [0.1, 0.5, 1],
            "gamma": [0.01, 0.1, 1],
        },
        cv=5,
        scoring="r2",
        n_jobs=1,
    )
    ad_search.fit(budgets_train_z, sales_train_z)
    ad_test_rmse, ad_test_r2 = _sales_rmse(ad_search.best_estimator_)
    print(f"Train = {len(budgets_train)} \t Test = {len(budgets_test)}")
    print(f"Training standard deviation of sales = {sales_scale:.2f}")
    print(manual_scores.round(3).to_string(index=False))
    print(f"Best training-fold parameters: {ad_search.best_params_}")
    print(f"Mean CV R² = {ad_search.best_score_:.3f}")
    print(f"Test R² = {ad_test_r2:.3f}    RMSE = {ad_test_rmse:.3f} sales units")
    return mo.Html(manual_scores.round(3).to_html(index=False))


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Which hyperparameter sets the half-width of the tube? Replace `None` with its name.
    """)
    return


@app.cell
def _():
    student_tube = None
    print(student_tube)
    return (student_tube,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r'''
```python
student_tube = "epsilon"
```
**epsilon.** Points inside the tube are not support vectors. A larger epsilon makes a wider tube and usually fewer support vectors.
''')
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(ad_search, ad_test_r2, ad_test_rmse, curve_search, curve_test_r2, curve_test_rmse, mo):
    mo.md(f"""
    ## Conclusions

    **Key takeaways:**

    - Epsilon is the tube. C is the penalty on points outside it. Gamma is how local the RBF kernel is.
    - On the synthetic training folds, the short search picks `{curve_search.best_params_}`, with mean CV R² {curve_search.best_score_:.3f}. One test score of that choice is R² {curve_test_r2:.3f} and RMSE {curve_test_rmse:.3f}.
    - On the advertising training folds, the search picks `{ad_search.best_params_}`, with mean CV R² {ad_search.best_score_:.3f}. The test score is R² {ad_test_r2:.3f} and RMSE {ad_test_rmse:.3f} sales units.
    - The means and standard deviations come from the training rows. The test markets only pass through them.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check your understanding

    1. Which points are the support vectors?
    2. What happens to the tube when epsilon grows?
    3. What does a larger C ask the fit to do?
    4. What does a smaller gamma do to an RBF fit?
    5. Why is the curve drawn on a grid?
    6. Why is the grid search fit on the training rows only?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
1. **The points outside the tube:** They determine the function. Points inside the tube are not penalized.
2. **It gets wider:** More training points fall inside and drop out of the set of support vectors. The function can miss more of the wave.
3. **It penalizes outside points more:** The function bends harder to reduce those violations.
4. **It smooths the local bumps:** Each training point influences a wider neighborhood.
5. **The split shuffled the rows:** A line through unsorted x jumps across the plot.
6. **The test score has to stay fresh:** The folds choose C, epsilon, and gamma. The test rows are scored after that choice.
""")
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - [SVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)
    - [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
    - VanderPlas, J. (2017). *Python Data Science Handbook*. O'Reilly Media. Chapter 5.
    """)
    return


if __name__ == "__main__":
    app.run()
