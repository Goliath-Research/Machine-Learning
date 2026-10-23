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
    from sklearn.datasets import make_blobs
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import GridSearchCV, train_test_split
    from sklearn.tree import DecisionTreeClassifier, plot_tree

    return (
        DecisionTreeClassifier,
        GridSearchCV,
        RandomForestClassifier,
        confusion_matrix,
        make_blobs,
        plot_tree,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Random Forests for Classification

    ## Objectives

    - Compare one shallow tree with a forest of shallow trees.
    - Read an out-of-bag score and a feature-importance bar.
    - Tune the number of trees, the depth, and the leaf size on the raisin data.

    ## Background

    A random forest fits many decision trees and lets them vote. Each tree sees a bootstrap sample: rows drawn with replacement, so some training rows are left out of that tree. Those left-out rows are the out-of-bag rows. Their votes, using only the trees that did not train on them, are an internal check on the forest.

    The trees also see a random subset of the features at each split. Averaging the votes keeps a single deep, brittle tree from deciding the label alone.

    ## Datasets Used

    - **Synthetic blobs.** Two features and two classes.
    - **Raisin Dataset.** Seven measurements of Kecimen and Besni raisins. Kecimen is 0 and Besni is 1. Trees and forests do not need those columns standardized.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## One tree and a forest of 20

    Both models are limited to depth 2. The forest grows 20 such trees. `oob_score=True` asks it to keep the out-of-bag check. `random_state=0` fixes the bootstrap draws.

    The three plotted trees are members 0, 6, and 19. They are not the same tree. Each one was grown on its own bootstrap sample, so the leaf counts do not add up to the full training set.
    """)
    return


@app.cell
def _(
    DecisionTreeClassifier,
    RandomForestClassifier,
    make_blobs,
    pd,
    plot_tree,
    plt,
    train_test_split,
):
    _X, _cl = make_blobs(
        n_samples=100, n_features=2, centers=2, random_state=2, cluster_std=3
    )
    dfb = pd.DataFrame({"X1": _X[:, 0], "X2": _X[:, 1], "Class": _cl})
    Xb = dfb[["X1", "X2"]]
    yb = dfb["Class"]
    Xb_train, Xb_test, yb_train, yb_test = train_test_split(
        Xb, yb, test_size=0.2, random_state=23
    )
    tree_one = DecisionTreeClassifier(random_state=0, max_depth=2).fit(Xb_train, yb_train)
    forest_blobs = RandomForestClassifier(
        n_estimators=20,
        max_depth=2,
        oob_score=True,
        random_state=0,
    ).fit(Xb_train, yb_train)
    print(f"Train = {len(Xb_train)} cases \t Test = {len(Xb_test)} cases")
    print(f"Trees in the forest: {len(forest_blobs.estimators_)}")
    print(f"One tree  train / test = {tree_one.score(Xb_train, yb_train):.3f} / {tree_one.score(Xb_test, yb_test):.3f}")
    print(f"Forest    train / test = {forest_blobs.score(Xb_train, yb_train):.3f} / {forest_blobs.score(Xb_test, yb_test):.3f}")
    print(f"Out-of-bag score = {forest_blobs.oob_score_:.3f}")
    _figure, _axes = plt.subplots(1, 3, figsize=(12, 4))
    for _axis, _index in zip(_axes, (0, 6, 19)):
        plot_tree(
            forest_blobs.estimators_[_index],
            feature_names=list(Xb_train.columns),
            filled=True,
            rounded=True,
            ax=_axis,
        )
        _axis.set_title(f"Tree {_index}")
    _figure.tight_layout()
    return _figure


@app.cell
def _(Xb_train, forest_blobs, mo, pd, yb_train):
    _x_values = Xb_train.to_numpy()
    _y_values = yb_train.to_numpy()
    _scores = pd.Series(
        {
            f"tree {index}": estimator.score(_x_values, _y_values)
            for index, estimator in enumerate(forest_blobs.estimators_)
        }
    )
    print(f"Lowest single-tree training score = {_scores.min():.3f}")
    print(f"Highest single-tree training score = {_scores.max():.3f}")
    print(f"Forest training score = {forest_blobs.score(Xb_train, yb_train):.3f}")
    return mo.Html(_scores.round(3).to_frame("training score").to_html())


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A single tree in the forest can score worse than the forest. The forest label is the majority vote, not the score of the best tree.

    The out-of-bag score is that vote restricted to rows a tree did not train on. It is not the same number as the test accuracy. The test rows were never in the training table. The out-of-bag rows were in the training table, just left out of some of the trees.
    """)
    return


@app.cell
def _(Xb_train, forest_blobs, mo, pd, plt, sns):
    importance = pd.DataFrame(
        {
            "feature": Xb_train.columns,
            "importance": forest_blobs.feature_importances_,
        }
    )
    _figure, _axes = plt.subplots(figsize=(5, 3))
    sns.barplot(data=importance, x="feature", y="importance", ax=_axes, color="steelblue")
    _axes.set_title("Feature importance in the blob forest")
    return _figure, mo.Html(importance.round(3).to_html())


@app.cell
def _(Xb_test, Xb_train, confusion_matrix, forest_blobs, mo, np, pd, plt, sns, tree_one, yb_test, yb_train):
    _rng = np.random.default_rng(0)
    _grid = pd.DataFrame(
        {
            "X1": _rng.uniform(Xb_train.X1.min(), Xb_train.X1.max(), size=800),
            "X2": _rng.uniform(Xb_train.X2.min(), Xb_train.X2.max(), size=800),
        }
    )
    _figure, _axes = plt.subplots(1, 2, figsize=(10, 4))
    for _axis, _model, _title in (
        (_axes[0], tree_one, "One tree"),
        (_axes[1], forest_blobs, "Forest of 20"),
    ):
        _painted = _grid.copy()
        _painted["Class"] = _model.predict(_grid)
        sns.scatterplot(
            data=_painted,
            x="X1",
            y="X2",
            hue="Class",
            palette={0: "cornflowerblue", 1: "indianred"},
            alpha=0.15,
            s=20,
            legend=False,
            ax=_axis,
        )
        sns.scatterplot(
            x=Xb_train.X1,
            y=Xb_train.X2,
            hue=yb_train,
            palette={0: "cornflowerblue", 1: "indianred"},
            s=40,
            legend=False,
            ax=_axis,
        )
        _axis.set_title(_title)
    _figure.tight_layout()
    _rows = []
    for _name, _model in (("tree", tree_one), ("forest", forest_blobs)):
        _matrix = confusion_matrix(yb_test, _model.predict(Xb_test))
        _rows.append(
            pd.DataFrame(_matrix, index=["true 0", "true 1"], columns=["pred 0", "pred 1"]).assign(model=_name)
        )
    return _figure, mo.Html(pd.concat(_rows).to_html())


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Raisin forests

    Model A requires at least 50 training rows in each leaf. Model B splits a node only when it has more than 100 rows. Model C limits every tree to depth 4. Each of those forests uses the default of 100 trees.

    The grid search then varies the number of trees, the leaf size, the split size, and the depth together. It uses five folds of the training rows. `n_jobs=1` keeps the search in one process.
    """)
    return


@app.cell
def _():
    raisin_file = (
        "https://raw.githubusercontent.com/Goliath-Research/"
        "Machine-Learning/master/Classification/Raisin.csv"
    )
    return (raisin_file,)


@app.cell
def _(RandomForestClassifier, pd, raisin_file, train_test_split):
    # dfr = pd.read_csv("Raisin.csv")
    dfr = pd.read_csv(raisin_file)
    _coded = dfr.copy()
    _coded["Besni"] = _coded["Class"].map({"Kecimen": 0, "Besni": 1})
    _yr = _coded["Besni"]
    Xr = _coded.drop(columns=["Class", "Besni"])
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        Xr, _yr, test_size=0.2, random_state=23
    )
    forest_leaf = RandomForestClassifier(random_state=0, min_samples_leaf=50).fit(Xr_train, yr_train)
    forest_split = RandomForestClassifier(random_state=0, min_samples_split=100).fit(Xr_train, yr_train)
    forest_depth = RandomForestClassifier(random_state=0, max_depth=4).fit(Xr_train, yr_train)
    print(dfr.shape)
    print(f"Train = {len(Xr_train)} cases \t Test = {len(Xr_test)} cases")
    for _name, _model in (
        ("min_samples_leaf=50", forest_leaf),
        ("min_samples_split=100", forest_split),
        ("max_depth=4", forest_depth),
    ):
        print(f"{_name}: {len(_model.estimators_)} trees, train {_model.score(Xr_train, yr_train):.3f}, test {_model.score(Xr_test, yr_test):.3f}")
    return


@app.cell
def _(GridSearchCV, RandomForestClassifier, Xr_test, Xr_train, yr_test, yr_train):
    _search = GridSearchCV(
        RandomForestClassifier(random_state=0),
        {
            "n_estimators": [10, 50, 100],
            "min_samples_leaf": [10, 50, 80],
            "min_samples_split": [10, 50, 100],
            "max_depth": [None, 2, 6, 8],
        },
        cv=5,
        n_jobs=1,
    )
    _search.fit(Xr_train, yr_train)
    forest_best = _search.best_estimator_
    print(f"Best parameters: {_search.best_params_}")
    print(f"Training accuracy = {forest_best.score(Xr_train, yr_train):.3f}")
    print(f"Testing accuracy  = {forest_best.score(Xr_test, yr_test):.3f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    An out-of-bag row is a training row that was left out of which trees? Replace `None` with `some` or with `every`.
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
student_oob = "some"
```
**some.** Each tree draws its own bootstrap sample. A row left out of that draw is out-of-bag for that tree, and it can still be in the sample of another tree. The out-of-bag score uses, for each row, only the trees that missed it.
''')
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(Xb_test, forest_best, forest_blobs, mo, tree_one, Xr_test, yb_test, yr_test):
    mo.md(f"""
    ## Conclusions

    **Key takeaways:**

    - The forest label is a majority vote of its trees. On the blobs, the single depth-2 tree has test accuracy {tree_one.score(Xb_test, yb_test):.3f}, and the forest of 20 has test accuracy {forest_blobs.score(Xb_test, yb_test):.3f}.
    - Individual trees in that forest can score lower than the vote. The leaf counts differ because each tree saw a bootstrap sample.
    - The out-of-bag score for the blob forest is {forest_blobs.oob_score_:.3f}. It reuses training rows that a given tree did not see. It is not a substitute for a test set that was held out of the whole fit.
    - Feature importance is the forest's summary of which column reduced impurity most often. On the blobs that bar has two columns, `X1` and `X2`.
    - Raisin columns are not standardized. The cross-validated forest uses {forest_best.get_params()["n_estimators"]} trees, depth {forest_best.get_params()["max_depth"]}, leaf size {forest_best.get_params()["min_samples_leaf"]}, and split size {forest_best.get_params()["min_samples_split"]}. Its test accuracy is {forest_best.score(Xr_test, yr_test):.3f}.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check your understanding

    1. What does the forest return as its predicted class?
    2. Why do the leaf counts of one plotted tree not equal the full training set?
    3. Which trees vote on a row in the out-of-bag score?
    4. What does a taller bar in the importance chart mean?
    5. Why can the raisin fit skip standardization?
    6. What does the grid search vary that the three single forests do not vary together?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
1. **The majority class:** Each tree votes, and the forest keeps the class with the most votes.
2. **That tree saw a bootstrap sample:** Some training rows were drawn more than once and some were left out.
3. **The trees that did not train on it:** A row can be in some samples and out of others.
4. **The column was used more to reduce impurity:** Importance is a summary across the trees, not a coefficient.
5. **A threshold does not use distance:** Area and extent can stay on their own scales.
6. **All four settings at once:** The three forests each change one setting. The grid scores combinations of `n_estimators`, depth, leaf size, and split size.
""")
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - Cinar, I., Koklu, M., and Tasdemir, S. (2020). Classification of Raisin Grains Using Machine Vision and Artificial Intelligence Methods. *Gazi Journal of Engineering Sciences*, 6(3), 200-209. [DOI](https://dergipark.org.tr/tr/download/article-file/1227592)
    - [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
    - VanderPlas, J. (2017). *Python Data Science Handbook*. O'Reilly Media. Chapter 5.
    """)
    return


if __name__ == "__main__":
    app.run()
