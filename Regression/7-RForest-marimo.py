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
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score, root_mean_squared_error
    from sklearn.model_selection import cross_val_score
    from sklearn.tree import DecisionTreeRegressor, plot_tree

    return (
        DecisionTreeRegressor,
        RandomForestRegressor,
        cross_val_score,
        plot_tree,
        r2_score,
        root_mean_squared_error,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Random Forest for Regression

    ## Objectives

    - Compare one regression tree with a forest of shallow trees.
    - Check that the forest prediction is the average of its trees.
    - Read an out-of-bag score, a feature-importance bar, and a cross-validated score on the advertising data.

    ## Background

    A random forest grows many trees on bootstrap samples and averages their predictions. Each tree also sees a random subset of the predictors at each split. The average is more stable than any one tree.

    Rows left out of a tree's bootstrap sample are out-of-bag rows. Their predictions, averaged over the trees that did not train on them, are an internal check. That check exists only when `bootstrap=True`.

    ## Datasets Used

    - **Synthetic table.** One hundred rows and four numeric predictors. The response is a linear combination of those predictors plus a little noise. The random seed is 50.
    - **Advertising.** Budgets for TV, radio, and newspaper, and the sales that followed, in 200 markets.
    """)
    return


@app.cell
def _(np, pd):
    np.random.seed(50)
    synthetic = pd.DataFrame(
        {
            "x1": np.random.uniform(0, 10, size=100),
            "x2": np.random.normal(10, 4, size=100),
            "x3": np.random.normal(50, 10, size=100),
            "x4": np.random.uniform(50, 100, size=100),
        }
    )
    synthetic["y"] = (
        1.1 * synthetic["x1"]
        + 1.2 * synthetic["x2"]
        + 1.8 * synthetic["x3"]
        + 0.7 * synthetic["x4"]
        + np.random.normal(0, 1, size=100)
    )
    feature_names = ["x1", "x2", "x3", "x4"]
    synthetic_x = synthetic[feature_names].to_numpy()
    synthetic_y = synthetic["y"].to_numpy()
    print(f"{len(synthetic)} rows")
    return feature_names, synthetic_x, synthetic_y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## One tree and a forest of twenty

    Both are limited to depth 2, and both are fit on every row of this synthetic table. That makes the comparison easy to draw. It also means the R² below is a training score, not a holdout score.

    The forest prediction is the mean of the twenty tree predictions. The cell checks that the manual average matches `predict`.
    """)
    return


@app.cell
def _(
    DecisionTreeRegressor,
    RandomForestRegressor,
    feature_names,
    np,
    plt,
    plot_tree,
    r2_score,
    root_mean_squared_error,
    synthetic_x,
    synthetic_y,
):
    one_tree = DecisionTreeRegressor(random_state=0, max_depth=2).fit(synthetic_x, synthetic_y)
    shallow_forest = RandomForestRegressor(
        n_estimators=20,
        max_depth=2,
        oob_score=True,
        random_state=0,
    ).fit(synthetic_x, synthetic_y)
    tree_matrix = np.column_stack(
        [estimator.predict(synthetic_x) for estimator in shallow_forest.estimators_]
    )
    manual_average = tree_matrix.mean(axis=1)
    forest_prediction = shallow_forest.predict(synthetic_x)
    average_gap = float(np.max(np.abs(manual_average - forest_prediction)))
    print(f"Trees in the forest = {len(shallow_forest.estimators_)}")
    print(f"Largest gap between the manual average and predict = {average_gap:.2e}")
    print(f"One tree     R² = {r2_score(synthetic_y, one_tree.predict(synthetic_x)):.3f}    RMSE = {root_mean_squared_error(synthetic_y, one_tree.predict(synthetic_x)):.3f}")
    print(f"Forest       R² = {r2_score(synthetic_y, forest_prediction):.3f}    RMSE = {root_mean_squared_error(synthetic_y, forest_prediction):.3f}")
    print(f"Out-of-bag R² = {shallow_forest.oob_score_:.3f}")
    _figure, _axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_tree(one_tree, feature_names=feature_names, filled=True, rounded=True, fontsize=7, ax=_axes[0])
    _axes[0].set_title("One tree, depth 2")
    plot_tree(
        shallow_forest.estimators_[0],
        feature_names=feature_names,
        filled=True,
        rounded=True,
        fontsize=7,
        ax=_axes[1],
    )
    _axes[1].set_title("First tree inside the forest")
    _figure.tight_layout()
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The first forest tree does not use all 100 rows. Bootstrap sampling leaves some rows out, which is why an out-of-bag score exists.

    Individual trees are scored on the full table below. Those scores are training scores for rows the tree did see and a partial check for rows it did not. The forest score is higher than a typical member because the average cancels jumps that one tree makes.
    """)
    return


@app.cell
def _(mo, pd, plt, shallow_forest, sns, synthetic_x, synthetic_y, feature_names):
    member_scores = pd.Series(
        [estimator.score(synthetic_x, synthetic_y) for estimator in shallow_forest.estimators_],
        name="training R²",
    )
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": shallow_forest.feature_importances_,
        }
    )
    _figure, _axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.barplot(x=member_scores.index, y=member_scores.to_numpy(), color="steelblue", ax=_axes[0])
    _axes[0].set_title("Each tree, scored on all 100 rows")
    _axes[0].set_xlabel("tree")
    _axes[0].set_ylabel("R²")
    sns.barplot(data=importance, x="feature", y="importance", color="steelblue", ax=_axes[1])
    _axes[1].set_title("Feature importance")
    _figure.tight_layout()
    print(f"Lowest member R² = {member_scores.min():.3f}")
    print(f"Highest member R² = {member_scores.max():.3f}")
    return _figure, importance


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## More trees, and no depth limit

    Training R² climbs toward 1 as the trees grow deep and as more of them are averaged. The out-of-bag score does not have to follow it. That score is the one that uses rows the tree did not train on.
    """)
    return


@app.cell
def _(RandomForestRegressor, mo, pd, synthetic_x, synthetic_y):
    forest_rows = []
    for _trees, _depth in ((20, 2), (20, None), (50, None), (100, None), (500, None)):
        _model = RandomForestRegressor(
            n_estimators=_trees,
            max_depth=_depth,
            oob_score=True,
            random_state=0,
        ).fit(synthetic_x, synthetic_y)
        forest_rows.append(
            {
                "trees": _trees,
                "max_depth": "2" if _depth == 2 else "none",
                "train_r2": _model.score(synthetic_x, synthetic_y),
                "oob_r2": _model.oob_score_,
            }
        )
    forest_scores = pd.DataFrame(forest_rows)
    print(forest_scores.round(3).to_string(index=False))
    return mo.Html(forest_scores.round(3).to_html(index=False))


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Advertising

    Pandas reads the file from GitHub. On this computer the same file is `advertising.csv` in this folder. The browser rule that blocks some other sites is called **CORS**.

    These forests set `bootstrap=False`, so every tree sees every row in the fold and there is no out-of-bag score. `cross_val_score` builds the folds itself. The default score is R². `explained_variance` is a second name for almost the same question on this target.
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
def _(RandomForestRegressor, advertising_file, cross_val_score, mo, pd):
    # ads = pd.read_csv("advertising.csv")
    ads = pd.read_csv(advertising_file)
    budgets = ads[["TV", "Radio", "Newspaper"]]
    sales = ads["Sales"]
    depth_forest = RandomForestRegressor(
        n_estimators=100,
        max_depth=3,
        bootstrap=False,
        random_state=0,
    )
    leaf_forest = RandomForestRegressor(
        n_estimators=100,
        min_samples_leaf=5,
        bootstrap=False,
        random_state=0,
    )
    cv_rows = []
    for _label, _model, _folds, _scoring in (
        ("depth 3", depth_forest, 5, "r2"),
        ("depth 3", depth_forest, 10, "r2"),
        ("depth 3", depth_forest, 10, "explained_variance"),
        ("leaf ≥ 5", leaf_forest, 10, "r2"),
    ):
        _scores = cross_val_score(_model, budgets, sales, cv=_folds, scoring=_scoring)
        cv_rows.append(
            {
                "forest": _label,
                "folds": _folds,
                "scoring": _scoring,
                "mean": _scores.mean(),
                "std": _scores.std(),
            }
        )
    cv_table = pd.DataFrame(cv_rows)
    print(f"{len(ads)} markets")
    print(cv_table.round(3).to_string(index=False))
    return mo.Html(cv_table.round(3).to_html(index=False))


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Which attribute holds the out-of-bag score after a forest with `bootstrap=True` is fit? Replace `None` with its name.
    """)
    return


@app.cell
def _():
    student_oob = None
    print(student_oob)
    return (student_oob,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r'''
```python
student_oob = "oob_score_"
```
**oob_score_.** The trailing underscore marks a value computed during `fit`. It is present when `oob_score=True` and `bootstrap=True`.
''')
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(cv_table, forest_scores, importance, mo, shallow_forest):
    _deep = forest_scores[forest_scores["max_depth"] == "none"]
    _top = importance.loc[importance["importance"].idxmax(), "feature"]
    _depth3 = cv_table.query("forest == 'depth 3' and folds == 10 and scoring == 'r2'")["mean"].iloc[0]
    _leaf = cv_table.query("forest == 'leaf ≥ 5' and scoring == 'r2'")["mean"].iloc[0]
    mo.md(f"""
    ## Conclusions

    **Key takeaways:**

    - The forest prediction matches the average of its trees. The out-of-bag R² of the depth-2 forest is {shallow_forest.oob_score_:.3f}.
    - With no depth limit, training R² runs from {_deep["train_r2"].min():.3f} to {_deep["train_r2"].max():.3f} as the tree count goes from {int(_deep["trees"].min())} to {int(_deep["trees"].max())}. The out-of-bag R² on those same fits runs from {_deep["oob_r2"].min():.3f} to {_deep["oob_r2"].max():.3f}.
    - The largest importance in the depth-2 forest is **{_top}**. The formula that built `y` puts its largest coefficient on `x3`.
    - On the advertising budgets, with bootstrap turned off, the 10-fold R² for the depth-3 forest is {_depth3:.3f}. The leaf-limited forest's 10-fold R² is {_leaf:.3f}.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check your understanding

    1. What does the forest report as its prediction?
    2. Why can a forest score be higher than the score of every tree in it?
    3. What is an out-of-bag row?
    4. Why is there no out-of-bag score when `bootstrap=False`?
    5. What does a high training R² next to a lower out-of-bag R² say?
    6. What does `cross_val_score` do with the advertising rows?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
1. **The average of the trees:** Each tree votes with a number, and the forest returns the mean.
2. **The average cancels individual jumps:** A tree that is high on one row is offset by a tree that is low.
3. **A row left out of that tree's bootstrap sample:** The tree never trained on it, so the prediction is a check.
4. **Every tree saw every row:** Nothing was held out of the fit, so there is no internal leftover sample.
5. **The training score is memorizing:** The out-of-bag score is the closer description of new rows from the same process.
6. **It refits on folds:** Each fold is scored with a forest that did not train on that fold. The printed number is the mean of those scores.
""")
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - [RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
    - [cross_val_score](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html)
    - VanderPlas, J. (2017). *Python Data Science Handbook*. O'Reilly Media. Chapter 5.
    """)
    return


if __name__ == "__main__":
    app.run()
