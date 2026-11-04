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
    from sklearn.datasets import make_blobs
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import GridSearchCV, train_test_split
    from sklearn.tree import DecisionTreeClassifier, plot_tree

    plt.rcParams["figure.max_open_warning"] = 0
    sns.set_theme(style="whitegrid")

    return (
        DecisionTreeClassifier,
        GridSearchCV,
        confusion_matrix,
        make_blobs,
        mo,
        np,
        pd,
        plot_tree,
        plt,
        sns,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Decision Trees for Classification

    ## Objectives

    - Read a classification tree, including the Gini value in a node.
    - Compare a depth limit with a leaf-size limit.
    - Tune `max_depth`, `min_samples_leaf`, and `min_samples_split` on the raisin data.

    ## Background

    A classification tree asks one yes-or-no question at a time. Each question is a threshold on one feature. The leaf at the end of the path holds a class.

    Gini impurity measures how mixed a node is. It is 0 when every row in the node has the same class. For two equally common classes it is 0.5. A split is chosen to reduce that impurity.

    Trees do not use the numeric scale of a column. A threshold on area is not disturbed by a threshold on extent, so this lesson does not standardize the raisin measurements.

    ## Datasets Used

    - **Synthetic blobs.** Two features and two classes.
    - **Raisin Dataset.** Seven measurements of Kecimen and Besni raisins. Kecimen is 0 and Besni is 1.
    """)
    return


@app.cell
def _(confusion_matrix, pd):
    def confusion_table(y_true, y_pred):
        labels = sorted(set(y_true).union(set(y_pred)))
        matrix = confusion_matrix(y_true, y_pred, labels=labels)
        return pd.DataFrame(
            matrix,
            index=[f"true {label}" for label in labels],
            columns=[f"pred {label}" for label in labels],
        )

    return (confusion_table,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Two limits on the same blobs

    The split is 80/20 with `random_state=23`.

    - Tree A stops at depth 2, so it can ask at most two questions.
    - Tree B allows a leaf only when it would contain at least 10 training rows.

    `random_state=0` fixes the tie-breaking inside the tree, so the picture stays the same on each run.
    """)
    return


@app.cell
def _(DecisionTreeClassifier, make_blobs, pd, train_test_split):
    _X, _cl = make_blobs(
        n_samples=100, n_features=2, centers=2, random_state=2, cluster_std=3
    )
    dfb = pd.DataFrame({"X1": _X[:, 0], "X2": _X[:, 1], "Class": _cl})
    Xb = dfb[["X1", "X2"]]
    yb = dfb["Class"]
    Xb_train, Xb_test, yb_train, yb_test = train_test_split(
        Xb, yb, test_size=0.2, random_state=23
    )
    tree_a = DecisionTreeClassifier(random_state=0, max_depth=2)
    tree_b = DecisionTreeClassifier(random_state=0, min_samples_leaf=10)
    tree_a.fit(Xb_train, yb_train)
    tree_b.fit(Xb_train, yb_train)
    print(f"Train = {len(Xb_train)} cases \t Test = {len(Xb_test)} cases")
    print(f"Tree A train / test = {tree_a.score(Xb_train, yb_train):.3f} / {tree_a.score(Xb_test, yb_test):.3f}")
    print(f"Tree B train / test = {tree_b.score(Xb_train, yb_train):.3f} / {tree_b.score(Xb_test, yb_test):.3f}")
    return Xb_test, tree_a, tree_b, yb_test


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    On this blob split, what is Tree A's test accuracy (three decimals)?
    """)
    return


@app.cell(hide_code=True)
def _(Xb_test, mo, tree_a, yb_test):
    _acc = tree_a.score(Xb_test, yb_test)
    mo.accordion(
        {
            "Show answers": mo.md(f"""
**{_acc:.3f}.** Tree A stops at depth 2, so it can ask at most two questions.
""")
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(Xb_train, mo, tree_a):
    _node = tree_a.tree_
    _counts = (_node.value[0, 0] * _node.n_node_samples[0]).round().astype(int)
    _feature = Xb_train.columns[_node.feature[0]]
    mo.md(f"""
    ## Reading the root of tree A

    The root holds all {int(_node.n_node_samples[0])} training rows. The class counts are {int(_counts[0])} and {int(_counts[1])}. Its Gini impurity is {_node.impurity[0]:.3f}.

    The first question is `{_feature} <= {_node.threshold[0]:.3f}`. Follow the plot for the next question.     A leaf with Gini 0 contains only one class.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    What is the Gini impurity at the root of Tree A (three decimals)?
    """)
    return


@app.cell(hide_code=True)
def _(mo, tree_a):
    _gini = tree_a.tree_.impurity[0]
    mo.accordion(
        {
            "Show answers": mo.md(f"""
**{_gini:.3f}.** Gini is 0 in a pure node and larger when the classes are mixed.
""")
        },
        lazy=True,
    )
    return


@app.cell
def _(Xb_train, plot_tree, plt, tree_a, tree_b):
    _figure, _axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_tree(
        tree_a,
        feature_names=list(Xb_train.columns),
        filled=True,
        node_ids=True,
        rounded=True,
        ax=_axes[0],
    )
    _axes[0].set_title("Tree A, max_depth = 2")
    plot_tree(
        tree_b,
        feature_names=list(Xb_train.columns),
        filled=True,
        node_ids=True,
        rounded=True,
        ax=_axes[1],
    )
    _axes[1].set_title("Tree B, min_samples_leaf = 10")
    _figure.tight_layout()
    _figure


@app.cell
def _(Xb_train, np, pd, plt, sns, tree_a, tree_b, yb_train):
    _rng = np.random.default_rng(0)
    _grid = pd.DataFrame(
        {
            "X1": _rng.uniform(Xb_train.X1.min(), Xb_train.X1.max(), size=800),
            "X2": _rng.uniform(Xb_train.X2.min(), Xb_train.X2.max(), size=800),
        }
    )
    _figure, _axes = plt.subplots(1, 2, figsize=(10, 4))
    for _axis, _model, _title in (
        (_axes[0], tree_a, "Depth limit"),
        (_axes[1], tree_b, "Leaf-size limit"),
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
            ax=_axis,
            legend=False,
        )
        _axis.set_title(_title)
    _figure.tight_layout()
    _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The regions are rectangles, because every question cuts one axis. A deeper tree cuts those rectangles into smaller ones. That can fit the training points more closely and then miss the test points.
    """)
    return


@app.cell
def _(Xb_test, confusion_table, mo, pd, tree_a, tree_b, yb_test):
    _tables = []
    for _name, _model in (("Tree A", tree_a), ("Tree B", tree_b)):
        _tables.append(confusion_table(yb_test, _model.predict(Xb_test)).assign(model=_name))
    mo.Html(pd.concat(_tables).to_html())


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Which tree has the higher test accuracy on the blobs, Tree A or Tree B?
    """)
    return


@app.cell(hide_code=True)
def _(Xb_test, mo, tree_a, tree_b, yb_test):
    _acc_a = tree_a.score(Xb_test, yb_test)
    _acc_b = tree_b.score(Xb_test, yb_test)
    _winner = "Tree A" if _acc_a >= _acc_b else "Tree B"
    mo.accordion(
        {
            "Show answers": mo.md(f"""
**{_winner}.** Tree A scores {_acc_a:.3f} on the test rows; Tree B scores {_acc_b:.3f}.
""")
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Raisin trees

    Three restrictions, each used on its own:

    - Model A: every leaf has at least 50 training rows.
    - Model B: a node is split only when it has more than 100 training rows (`min_samples_split=100`).
    - Model C: the depth is at most 4.

    The search at the end tries those ideas together. `GridSearchCV` refits the winning combination on all of the training rows.
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
def _(DecisionTreeClassifier, pd, plot_tree, plt, raisin_file, train_test_split):
    # dfr = pd.read_csv("Raisin.csv")
    dfr = pd.read_csv(raisin_file)
    _coded = dfr.copy()
    _coded["Besni"] = _coded["Class"].map({"Kecimen": 0, "Besni": 1})
    _yr = _coded["Besni"]
    Xr = _coded.drop(columns=["Class", "Besni"])
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        Xr, _yr, test_size=0.2, random_state=23
    )
    tree_leaf = DecisionTreeClassifier(random_state=0, min_samples_leaf=50).fit(Xr_train, yr_train)
    tree_split = DecisionTreeClassifier(random_state=0, min_samples_split=100).fit(Xr_train, yr_train)
    tree_depth = DecisionTreeClassifier(random_state=0, max_depth=4).fit(Xr_train, yr_train)
    print(dfr.shape)
    print(f"Train = {len(Xr_train)} cases \t Test = {len(Xr_test)} cases")
    for _name, _model in (
        ("min_samples_leaf=50", tree_leaf),
        ("min_samples_split=100", tree_split),
        ("max_depth=4", tree_depth),
    ):
        print(
            f"{_name}: train {_model.score(Xr_train, yr_train):.3f}, "
            f"test {_model.score(Xr_test, yr_test):.3f}"
        )
    _figure, _axes = plt.subplots(figsize=(12, 5))
    plot_tree(
        tree_leaf,
        feature_names=list(Xr_train.columns),
        filled=True,
        rounded=True,
        ax=_axes,
    )
    _axes.set_title("Raisin tree with at least 50 rows in each leaf")
    _figure
    return Xr_test, Xr_train, tree_depth, tree_leaf, tree_split, yr_test


@app.cell
def _(GridSearchCV, DecisionTreeClassifier, Xr_test, Xr_train, plot_tree, plt, yr_test, yr_train):
    _search = GridSearchCV(
        DecisionTreeClassifier(random_state=0),
        {
            "min_samples_leaf": [10, 50, 80],
            "min_samples_split": [10, 50, 100],
            "max_depth": [None, 2, 6, 8],
        },
        cv=5,
    )
    _search.fit(Xr_train, yr_train)
    tree_best = _search.best_estimator_
    print(f"Best parameters: {_search.best_params_}")
    print(f"Training accuracy = {tree_best.score(Xr_train, yr_train):.3f}")
    print(f"Testing accuracy  = {tree_best.score(Xr_test, yr_test):.3f}")
    _figure, _axes = plt.subplots(figsize=(12, 5))
    plot_tree(
        tree_best,
        feature_names=list(Xr_train.columns),
        filled=True,
        rounded=True,
        ax=_axes,
    )
    _axes.set_title("Tree chosen by cross-validation")
    _figure
    return (tree_best,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Among the three single-limit raisin trees, which restriction gives the highest test accuracy?
    """)
    return


@app.cell(hide_code=True)
def _(Xr_test, mo, tree_depth, tree_leaf, tree_split, yr_test):
    _scores = {
        "min_samples_leaf=50": tree_leaf.score(Xr_test, yr_test),
        "min_samples_split=100": tree_split.score(Xr_test, yr_test),
        "max_depth=4": tree_depth.score(Xr_test, yr_test),
    }
    _best = max(_scores, key=_scores.get)
    mo.accordion(
        {
            "Show answers": mo.md(f"""
**{_best}.** Test accuracies are {", ".join(f"{name} {_scores[name]:.3f}" for name in _scores)}.
""")
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    On this blob split, what is Tree B's test accuracy (three decimals)?
    """)
    return


@app.cell(hide_code=True)
def _(Xb_test, mo, tree_b, yb_test):
    _acc = tree_b.score(Xb_test, yb_test)
    mo.accordion(
        {
            "Show answers": mo.md(f"""
**{_acc:.3f}.** Tree B requires at least 10 training rows in every leaf (`min_samples_leaf=10`).
""")
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(mo, tree_a, tree_b, tree_best, Xb_test, Xr_test, yb_test, yr_test):
    mo.md(f"""
    ## Conclusions

    **Key takeaways:**

    - Each split is one threshold on one feature, so the regions on the blob charts are rectangles.
    - Gini is 0 in a pure node and larger when the classes are mixed. Tree A's root Gini is {tree_a.tree_.impurity[0]:.3f}.
    - Depth 2 on the blobs has test accuracy {tree_a.score(Xb_test, yb_test):.3f}. The leaf-size limit has test accuracy {tree_b.score(Xb_test, yb_test):.3f}.
    - Raisin columns are not standardized. A tree's thresholds do not depend on putting the columns onto one scale.
    - The cross-validated raisin tree uses {tree_best.get_params()["max_depth"]} for `max_depth`, {tree_best.get_params()["min_samples_leaf"]} for `min_samples_leaf`, and {tree_best.get_params()["min_samples_split"]} for `min_samples_split`. Its test accuracy is {tree_best.score(Xr_test, yr_test):.3f}.
    - A tree that is allowed to grow until every leaf is pure can memorize the training rows. The limits above are what keep that from happening.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check your understanding

1. What question does a split ask?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {"Show answer": mo.md(r"""
**A threshold on one feature:** The row goes left or right. Later questions cut the same axes, so the regions are rectangles.
""")},
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    2. What does a Gini of 0 say about a node?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {"Show answer": mo.md(r"""
**The node is pure:** Every training row in it has the same class.
""")},
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    3. What does `max_depth=2` stop the tree from doing?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {"Show answer": mo.md(r"""
**A third question:** The longest path has two splits.
""")},
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    4. What does `min_samples_leaf=10` require?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {"Show answer": mo.md(r"""
**At least 10 training rows in a leaf:** The tree will not peel off a smaller leaf.
""")},
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    5. Why does this lesson skip standardization for the raisins?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {"Show answer": mo.md(r"""
**Splits do not use distance:** A cutoff on area is unaffected by the size of the extent column.
""")},
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    6. What is the difference between fitting the three raisin trees and the grid search?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {"Show answer": mo.md(r"""
**The grid tries the limits together:** Each of the three trees uses one limit. Cross-validation scores combinations and then refits the winner.
""")},
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - Cinar, I., Koklu, M., and Tasdemir, S. (2020). Classification of Raisin Grains Using Machine Vision and Artificial Intelligence Methods. *Gazi Journal of Engineering Sciences*, 6(3), 200-209. [DOI](https://dergipark.org.tr/tr/download/article-file/1227592)
    - [DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
    - [plot_tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.plot_tree.html)
    - VanderPlas, J. (2017). *Python Data Science Handbook*. O'Reilly Media. Chapter 5.
    """)
    return


if __name__ == "__main__":
    app.run()