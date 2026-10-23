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
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeRegressor, plot_tree

    return (
        DecisionTreeRegressor,
        plot_tree,
        r2_score,
        root_mean_squared_error,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Decision Trees for Regression

    ## Objectives

    - Read a regression tree as a set of averages on regions of x.
    - Limit a tree with depth, a minimum split size, or a minimum leaf size.
    - Compare those limits on the advertising budgets.

    ## Background

    A regression tree splits the predictor into regions and predicts the average response in each region. The split is chosen to reduce squared error. A leaf's printed value is that average.

    An unlimited tree can put one training row in each leaf. The limits keep the regions large enough to be read, and large enough to have a chance on new rows.

    ## Datasets Used

    - **Ten points.** A short x and y, small enough to draw the whole tree.
    - **Advertising.** Budgets for TV, radio, and newspaper, and the sales that followed, in 200 markets.
    """)
    return


@app.cell
def _(DecisionTreeRegressor, np, pd, plt, sns, train_test_split):
    toy = pd.DataFrame(
        {
            "x": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            "y": np.array([1, 1.3, 1.2, 1.8, 2, 4.5, 4.7, 5.2, 5.6, 5]),
        }
    )
    toy_x_train, toy_x_test, toy_y_train, toy_y_test = train_test_split(
        toy[["x"]], toy["y"], test_size=0.3, random_state=10
    )
    stump = DecisionTreeRegressor(random_state=0, max_depth=1).fit(toy_x_train, toy_y_train)
    depth_two = DecisionTreeRegressor(random_state=0, max_depth=2).fit(toy_x_train, toy_y_train)
    print(f"Train = {len(toy_x_train)} points \t Test = {len(toy_x_test)} points")
    _figure, _axes = plt.subplots(figsize=(6, 4))
    sns.scatterplot(
        x=toy_x_train["x"], y=toy_y_train, s=60, color="cornflowerblue", label="Training", ax=_axes
    )
    sns.scatterplot(
        x=toy_x_test["x"], y=toy_y_test, s=60, color="darkblue", label="Test", ax=_axes
    )
    _axes.set_title("Ten points")
    return _figure, depth_two, stump, toy_x_test, toy_x_train, toy_y_test, toy_y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A stump and a depth-2 tree

    `squared_error` on a node is the mean squared error of predicting that node's average. `samples` is the number of training rows in the node. `value` is the average y in the node, which is also the prediction if the tree stops there.

    The step charts predict on a grid from 1 to 10. Depth 1 is one split. Depth 2 splits some of those regions again.
    """)
    return


@app.cell
def _(np, plt, plot_tree, r2_score, stump, toy_x_test, toy_x_train, toy_y_test, toy_y_train):
    _tree = stump.tree_
    _root = float(np.asarray(_tree.value[0]).ravel()[0])
    print(f"Root: squared error {_tree.impurity[0]:.3f}, samples {_tree.n_node_samples[0]}, value {_root:.3f}")
    print(f"Split: x <= {_tree.threshold[0]:.1f}")
    for _node in (1, 2):
        _value = float(np.asarray(_tree.value[_node]).ravel()[0])
        print(
            f"Leaf {_node}: squared error {_tree.impurity[_node]:.3f}, "
            f"samples {_tree.n_node_samples[_node]}, value {_value:.3f}"
        )
    print(f"Test R² = {r2_score(toy_y_test, stump.predict(toy_x_test)):.3f}")
    _figure, _axes = plt.subplots(figsize=(6, 3.5))
    plot_tree(
        stump,
        feature_names=["x"],
        filled=True,
        node_ids=True,
        rounded=True,
        ax=_axes,
    )
    _axes.set_title("Maximum depth 1")
    return _figure


@app.cell
def _(depth_two, np, pd, plt, r2_score, stump, toy_x_test, toy_x_train, toy_y_test, toy_y_train):
    _grid = pd.DataFrame({"x": np.linspace(1, 10, 300)})
    print(f"Depth 1 test R² = {r2_score(toy_y_test, stump.predict(toy_x_test)):.3f}")
    print(f"Depth 2 test R² = {r2_score(toy_y_test, depth_two.predict(toy_x_test)):.3f}")
    _figure, _axes = plt.subplots(figsize=(7, 4))
    _axes.scatter(toy_x_train["x"], toy_y_train, s=50, color="cornflowerblue", label="Training")
    _axes.scatter(toy_x_test["x"], toy_y_test, s=50, color="darkblue", label="Test")
    _axes.plot(_grid["x"], stump.predict(_grid), color="firebrick", label="Depth 1")
    _axes.plot(_grid["x"], depth_two.predict(_grid), color="darkgreen", label="Depth 2")
    _axes.set_title("Leaf averages on a grid")
    _axes.legend()
    return _figure


@app.cell
def _(plt, plot_tree, depth_two):
    _figure, _axes = plt.subplots(figsize=(8, 4))
    plot_tree(
        depth_two,
        feature_names=["x"],
        filled=True,
        node_ids=True,
        rounded=True,
        ax=_axes,
    )
    _axes.set_title("Maximum depth 2")
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The depth-2 fit follows the seven training points more closely. With three test points, a high training R² can sit next to a weak test R². The sample is only large enough to see the mechanism.

    ## Advertising, one budget

    Pandas reads the file from GitHub. On this computer the same file is `advertising.csv` in this folder. The browser rule that blocks some other sites is called **CORS**.

    Three limits on the same TV split (`random_state=20`):

    - `max_depth=2`
    - `min_samples_split=70`, so a node with fewer than 70 training rows does not split
    - `min_samples_leaf=15`, so every leaf keeps at least 15 training rows
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
    DecisionTreeRegressor,
    advertising_file,
    pd,
    plt,
    plot_tree,
    r2_score,
    root_mean_squared_error,
    train_test_split,
):
    # ads = pd.read_csv("advertising.csv")
    ads = pd.read_csv(advertising_file)
    tv_train, tv_test, sales_tv_train, sales_tv_test = train_test_split(
        ads[["TV"]], ads["Sales"], test_size=0.3, random_state=20
    )
    tv_settings = {
        "max_depth=2": {"max_depth": 2},
        "min_samples_split=70": {"min_samples_split": 70},
        "min_samples_leaf=15": {"min_samples_leaf": 15},
    }
    tv_rows = []
    _figure, _axes = plt.subplots(1, 3, figsize=(14, 4))
    for _ax, (_label, _params) in zip(_axes, tv_settings.items()):
        _model = DecisionTreeRegressor(random_state=0, **_params).fit(tv_train, sales_tv_train)
        tv_rows.append(
            {
                "limit": _label,
                "leaves": _model.get_n_leaves(),
                "train_r2": r2_score(sales_tv_train, _model.predict(tv_train)),
                "test_r2": r2_score(sales_tv_test, _model.predict(tv_test)),
                "test_rmse": root_mean_squared_error(sales_tv_test, _model.predict(tv_test)),
            }
        )
        plot_tree(
            _model,
            feature_names=["TV"],
            filled=True,
            rounded=True,
            fontsize=7,
            ax=_ax,
        )
        _ax.set_title(_label)
    tv_scores = pd.DataFrame(tv_rows).set_index("limit")
    _figure.tight_layout()
    print(tv_scores.round(3).to_string())
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Three budgets

    The same markets, now with TV, radio, and newspaper. Model A keeps at least 20 training rows in every leaf. Model B refuses to split a node that has 50 or fewer training rows. Model C uses the defaults, so a leaf can hold a single market.

    The default tree is too large to read as a picture. The cell prints its depth and its number of leaves instead, and draws the two limited trees.
    """)
    return


@app.cell
def _(
    DecisionTreeRegressor,
    ads,
    pd,
    plt,
    plot_tree,
    r2_score,
    root_mean_squared_error,
    train_test_split,
):
    budgets_train, budgets_test, sales_train, sales_test = train_test_split(
        ads[["TV", "Radio", "Newspaper"]],
        ads["Sales"],
        test_size=0.3,
        random_state=20,
    )
    multi_settings = {
        "A leaf ≥ 20": {"min_samples_leaf": 20},
        "B split ≥ 50": {"min_samples_split": 50},
        "C defaults": {},
    }
    multi_rows = []
    fitted = {}
    for _label, _params in multi_settings.items():
        _model = DecisionTreeRegressor(random_state=0, **_params).fit(
            budgets_train, sales_train
        )
        fitted[_label] = _model
        multi_rows.append(
            {
                "model": _label,
                "depth": _model.get_depth(),
                "leaves": _model.get_n_leaves(),
                "train_r2": r2_score(sales_train, _model.predict(budgets_train)),
                "test_r2": r2_score(sales_test, _model.predict(budgets_test)),
                "test_rmse": root_mean_squared_error(sales_test, _model.predict(budgets_test)),
            }
        )
    multi_scores = pd.DataFrame(multi_rows).set_index("model")
    _figure, _axes = plt.subplots(1, 2, figsize=(12, 4))
    for _ax, _label in zip(_axes, ("A leaf ≥ 20", "B split ≥ 50")):
        plot_tree(
            fitted[_label],
            feature_names=["TV", "Radio", "Newspaper"],
            filled=True,
            rounded=True,
            fontsize=7,
            ax=_ax,
        )
        _ax.set_title(_label)
    _figure.tight_layout()
    print(multi_scores.round(3).to_string())
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Which parameter stops a split when the node has fewer rows than the cutoff? Replace `None` with its name.
    """)
    return


@app.cell
def _():
    student_split_limit = None
    print(student_split_limit)
    return (student_split_limit,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r'''
```python
student_split_limit = "min_samples_split"
```
**min_samples_split.** A node smaller than this number is not split. `min_samples_leaf` is the related limit on how small a child may be. `max_depth` limits how many splits can be stacked.
''')
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(mo, multi_scores, tv_scores):
    _best = multi_scores["test_rmse"].idxmin()
    mo.md(f"""
    ## Conclusions

    **Key takeaways:**

    - A leaf predicts the average y of the training rows that landed there. The diagram's `value` is that average, and `squared_error` is how far those rows sit from it.
    - On TV alone, the three limits produce these test RMSEs: {", ".join(f"{_i} {_v:.3f}" for _i, _v in tv_scores["test_rmse"].round(3).items())}.
    - With all three budgets, the lowest test RMSE on this split is **{_best}**, at {multi_scores.loc[_best, "test_rmse"]:.3f}. Its test R² is {multi_scores.loc[_best, "test_r2"]:.3f}. Its training R² is {multi_scores.loc[_best, "train_r2"]:.3f}.
    - The default tree reaches depth {int(multi_scores.loc["C defaults", "depth"])} and {int(multi_scores.loc["C defaults", "leaves"])} leaves. A training R² near {multi_scores.loc["C defaults", "train_r2"]:.3f} with that many leaves is a memorized training set, and the test RMSE is the check.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check your understanding

    1. What number does a leaf predict?
    2. What does `squared_error` on a node measure?
    3. What is the difference between `min_samples_split` and `min_samples_leaf`?
    4. Why is the default tree hard to read?
    5. Why can training R² be high while test R² is lower?
    6. Which rows choose the splits?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
1. **The average response in that region:** Every row that falls into the leaf gets the same prediction.
2. **The mean squared gap from that average:** Splits are chosen to reduce it.
3. **Where the limit is applied:** `min_samples_split` looks at the node that might split. `min_samples_leaf` looks at the children that would be created.
4. **It has one region per small group of markets:** Depth and leaf count grow until the training rows are isolated.
5. **The leaves memorized the training markets:** New markets fall into regions that were fit to a handful of rows.
6. **The training rows:** The test rows are scored after the regions exist.
""")
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - [DecisionTreeRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
    - [plot_tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.plot_tree.html)
    - VanderPlas, J. (2017). *Python Data Science Handbook*. O'Reilly Media. Chapter 5.
    """)
    return


if __name__ == "__main__":
    app.run()
