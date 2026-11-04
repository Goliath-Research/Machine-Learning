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
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )
    from sklearn.model_selection import GridSearchCV, train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler

    plt.rcParams["figure.max_open_warning"] = 0
    sns.set_theme(style="whitegrid")

    return (
        GridSearchCV,
        KNeighborsClassifier,
        StandardScaler,
        accuracy_score,
        confusion_matrix,
        f1_score,
        make_blobs,
        mo,
        np,
        pd,
        plt,
        precision_score,
        recall_score,
        sns,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # k-Nearest Neighbors Classification

    ## Objectives

    - Classify a point by a majority vote of its nearest neighbors.
    - Compare a small neighborhood with a larger one.
    - Choose `k` with cross-validation, and read accuracy, precision, recall, and F1.

    ## Background

    k-nearest neighbors stores the training rows. For a new row it finds the `k` closest training rows and lets them vote. There is no coefficient vector. The vote depends on distance, so a column with large numbers can dominate a column with small numbers unless the columns are standardized.

    ## Datasets Used

    - **Synthetic blobs.** Two features and two classes, from `make_blobs`.
    - **Raisin Dataset.** Seven measurements of Kecimen and Besni raisins. Besni is the positive class for precision, recall, and F1.
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
    ### Try it yourself

    How does k-nearest neighbors pick a class for a new row?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
**Majority vote among the nearest training rows.** The `k` closest neighbors each contribute one vote; the winning class is the label.
""")
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Two neighborhoods on the blobs

    The split is 80/20 with `random_state=23`. One model uses the 3 nearest neighbors. The other uses the 7 nearest neighbors. A smaller `k` follows local wiggles. A larger `k` takes a broader vote and usually draws a smoother region.

    Class 0 is blue and class 1 is red.
    """)
    return


@app.cell
def _(KNeighborsClassifier, make_blobs, mo, pd, train_test_split):
    _X, _cl = make_blobs(
        n_samples=100, n_features=2, centers=2, random_state=2, cluster_std=3
    )
    dfb = pd.DataFrame({"X1": _X[:, 0], "X2": _X[:, 1], "Class": _cl})
    Xb = dfb[["X1", "X2"]]
    yb = dfb["Class"]
    Xb_train, Xb_test, yb_train, yb_test = train_test_split(
        Xb, yb, test_size=0.2, random_state=23
    )
    knn_k3 = KNeighborsClassifier(n_neighbors=3)
    knn_k7 = KNeighborsClassifier(n_neighbors=7)
    knn_k3.fit(Xb_train, yb_train)
    knn_k7.fit(Xb_train, yb_train)
    mo.md(
        f"Train = {len(Xb_train)} cases, test = {len(Xb_test)} cases. "
        f"k=3 training accuracy = {knn_k3.score(Xb_train, yb_train):.3f}, "
        f"testing = {knn_k3.score(Xb_test, yb_test):.3f}. "
        f"k=7 training accuracy = {knn_k7.score(Xb_train, yb_train):.3f}, "
        f"testing = {knn_k7.score(Xb_test, yb_test):.3f}."
    )
    return knn_k3, knn_k7


@app.cell
def _(Xb_train, knn_k3, knn_k7, np, pd, plt, sns, yb_train):
    _rng = np.random.default_rng(0)
    _grid = pd.DataFrame(
        {
            "X1": _rng.uniform(Xb_train.X1.min(), Xb_train.X1.max(), size=800),
            "X2": _rng.uniform(Xb_train.X2.min(), Xb_train.X2.max(), size=800),
        }
    )
    _figure, _axes = plt.subplots(1, 2, figsize=(10, 4))
    for _axis, _model, _title in (
        (_axes[0], knn_k3, "k = 3"),
        (_axes[1], knn_k7, "k = 7"),
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
            s=50,
            ax=_axis,
        )
        _axis.set_title(_title)
    _figure.tight_layout()
    _figure


@app.cell
def _(Xb_test, confusion_table, knn_k3, knn_k7, mo, pd, yb_test):
    _side = pd.concat(
        [
            confusion_table(yb_test, knn_k3.predict(Xb_test)).assign(model="k=3"),
            confusion_table(yb_test, knn_k7.predict(Xb_test)).assign(model="k=7"),
        ]
    )
    mo.Html(_side.to_html())


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    On the blob charts, which `k` gives the smoother decision region, 3 or 7?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
**7.** More neighbors average out a single nearby point, so the painted region changes less abruptly.
""")
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Raisins, after standardization

    Distance is the whole method, so the raisin columns are standardized. The scaler is fit on the training rows. Test rows use those training means and standard deviations.

    The comparison here is `k=3` against `k=5`. The file is read from GitHub so a browser can load it. `Raisin.csv` is also in this folder. The browser restriction is called **CORS**.
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
def _(KNeighborsClassifier, StandardScaler, pd, raisin_file, train_test_split):
    # dfr = pd.read_csv("Raisin.csv")
    dfr = pd.read_csv(raisin_file)
    _coded = dfr.copy()
    _coded["Besni"] = _coded["Class"].map({"Kecimen": 0, "Besni": 1})
    _yr = _coded["Besni"]
    _Xr = _coded.drop(columns=["Class", "Besni"])
    _Xr_train, _Xr_test, yr_train, yr_test = train_test_split(
        _Xr, _yr, test_size=0.2, random_state=23
    )
    _scaler = StandardScaler()
    Xr_train = pd.DataFrame(
        _scaler.fit_transform(_Xr_train), columns=_Xr.columns, index=_Xr_train.index
    )
    Xr_test = pd.DataFrame(
        _scaler.transform(_Xr_test), columns=_Xr.columns, index=_Xr_test.index
    )
    knn_r3 = KNeighborsClassifier(n_neighbors=3)
    knn_r5 = KNeighborsClassifier(n_neighbors=5)
    knn_r3.fit(Xr_train, yr_train)
    knn_r5.fit(Xr_train, yr_train)
    mo.md(
        f"{dfr.shape[0]} rows. Train = {len(Xr_train)} cases, test = {len(Xr_test)} cases. "
        f"k=3 test accuracy = {knn_r3.score(Xr_test, yr_test):.3f}. "
        f"k=5 test accuracy = {knn_r5.score(Xr_test, yr_test):.3f}."
    )
    return Xr_test, Xr_train, yr_test, yr_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Four scores, and why the test curve is only a picture

    For Besni as the positive class:

    - **Accuracy** is the share of raisins with the right variety.
    - **Precision** is the share of predicted Besni rows that really are Besni. It matters when a false Besni is costly.
    - **Recall** is the share of true Besni rows that were found. It matters when a missed Besni is costly.
    - **F1** is the harmonic mean of precision and recall.

    The loop below scores `k` from 1 through 30 on the test rows. That curve is a picture of this particular test set. Choosing `k` from it uses the test rows twice: once to pick `k`, and again to report the score. The next section uses cross-validation on the training rows instead.
    """)
    return


@app.cell
def _(
    KNeighborsClassifier,
    Xr_test,
    Xr_train,
    accuracy_score,
    f1_score,
    mo,
    pd,
    precision_score,
    recall_score,
    yr_test,
    yr_train,
):
    k_range = list(range(1, 31))
    _rows = []
    for _k in k_range:
        _model = KNeighborsClassifier(n_neighbors=_k)
        _model.fit(Xr_train, yr_train)
        _pred = _model.predict(Xr_test)
        _rows.append(
            {
                "k": _k,
                "accuracy": accuracy_score(yr_test, _pred),
                "precision": precision_score(yr_test, _pred),
                "recall": recall_score(yr_test, _pred),
                "f1": f1_score(yr_test, _pred),
            }
        )
    test_scores = pd.DataFrame(_rows).set_index("k")
    for _name in test_scores.columns:
        _best_k = int(test_scores[_name].idxmax())
        print(
            f"On the test set, {_name} is highest at k={_best_k} "
            f"({test_scores.loc[_best_k, _name]:.3f})"
        )
    mo.Html(test_scores.head(5).round(3).to_html())
    return (test_scores,)


@app.cell
def _(plt, test_scores):
    _figure, _axes = plt.subplots(figsize=(8, 4))
    for _name, _color in (
        ("accuracy", "#007bff"),
        ("precision", "#6c757d"),
        ("recall", "#28a745"),
        ("f1", "#dc3545"),
    ):
        _axes.plot(test_scores.index, test_scores[_name], marker="o", ms=4, color=_color, label=_name)
        _best_k = int(test_scores[_name].idxmax())
        _axes.scatter(_best_k, test_scores.loc[_best_k, _name], s=80, facecolors="none", edgecolors="black")
    _axes.set_xlabel("k")
    _axes.set_title("Scores on the held-out raisins")
    _axes.legend()
    _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Why should you not pick `k` from the test-set score curve alone?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
**It spends the test rows on the choice.** The reported test score would then be for a `k` tuned to look good on those same rows.
""")
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    On the standardized raisins, what is the k=5 test accuracy (three decimals)?
    """)
    return


@app.cell(hide_code=True)
def _(KNeighborsClassifier, Xr_test, Xr_train, mo, yr_test, yr_train):
    _acc = KNeighborsClassifier(n_neighbors=5).fit(Xr_train, yr_train).score(
        Xr_test, yr_test
    )
    mo.accordion(
        {
            "Show answers": mo.md(f"""
**{_acc:.3f}.** Distance uses every column, so the rows were standardized from the training set first.
""")
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Choosing k on the training rows

    `GridSearchCV` splits the training rows into 10 folds. For each candidate `k`, it scores the model on the fold that was left out, then averages those 10 scores. The test rows are not used to pick `k`.

    The search is repeated for accuracy, precision, recall, and F1. For the raisin varieties, accuracy is a reasonable default because the two classes are about the same size. A different cost for a missed Besni would make recall or F1 the score to maximize.
    """)
    return


@app.cell
def _(GridSearchCV, KNeighborsClassifier, Xr_train, k_range, yr_train):
    _param_grid = {"n_neighbors": k_range}
    _base = KNeighborsClassifier()
    grid_acc = GridSearchCV(_base, _param_grid, cv=10, scoring="accuracy")
    grid_prc = GridSearchCV(_base, _param_grid, cv=10, scoring="precision")
    grid_rec = GridSearchCV(_base, _param_grid, cv=10, scoring="recall")
    grid_f1 = GridSearchCV(_base, _param_grid, cv=10, scoring="f1")
    grid_acc.fit(Xr_train, yr_train)
    grid_prc.fit(Xr_train, yr_train)
    grid_rec.fit(Xr_train, yr_train)
    grid_f1.fit(Xr_train, yr_train)
    mo.md(
        f"CV accuracy picks k={grid_acc.best_params_['n_neighbors']} "
        f"at {grid_acc.best_score_:.3f}. "
        f"CV precision picks k={grid_prc.best_params_['n_neighbors']} "
        f"at {grid_prc.best_score_:.3f}. "
        f"CV recall picks k={grid_rec.best_params_['n_neighbors']} "
        f"at {grid_rec.best_score_:.3f}. "
        f"CV F1 picks k={grid_f1.best_params_['n_neighbors']} "
        f"at {grid_f1.best_score_:.3f}."
    )
    return grid_acc, grid_f1, grid_prc, grid_rec


@app.cell
def _(grid_acc, grid_f1, grid_prc, grid_rec, k_range, plt):
    _figure, _axes = plt.subplots(figsize=(8, 4))
    for _grid, _color, _label in (
        (grid_acc, "#007bff", "accuracy"),
        (grid_prc, "#6c757d", "precision"),
        (grid_rec, "#28a745", "recall"),
        (grid_f1, "#dc3545", "f1"),
    ):
        _means = _grid.cv_results_["mean_test_score"]
        _axes.plot(k_range, _means, marker="o", ms=4, color=_color, label=_label)
        _best_k = _grid.best_params_["n_neighbors"]
        _axes.scatter(_best_k, _grid.best_score_, s=80, facecolors="none", edgecolors="black")
    _axes.set_xlabel("k")
    _axes.set_title("Cross-validated scores on the training raisins")
    _axes.legend()
    _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The `k` that wins on the test curve does not have to be the `k` that wins under cross-validation. The cross-validated choice is the one to keep. The test set can then be scored once, with that `k` already chosen.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    For accuracy, which `k` does 10-fold cross-validation pick on the training raisins?
    """)
    return


@app.cell(hide_code=True)
def _(grid_acc, mo):
    mo.accordion(
        {
            "Show answers": mo.md(f"""
**k={grid_acc.best_params_['n_neighbors']}.** Mean CV accuracy is {grid_acc.best_score_:.3f} at that setting.
""")
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(grid_acc, knn_k3, knn_k7, mo, Xb_test, yb_test):
    mo.md(f"""
    ## Conclusions

    **Key takeaways:**

    - The predicted class is a majority vote of the `k` nearest training rows.
    - On the blobs, test accuracy is {knn_k3.score(Xb_test, yb_test):.3f} for `k=3` and {knn_k7.score(Xb_test, yb_test):.3f} for `k=7`. The pictures show a wigglier region when `k` is small.
    - Raisin columns are standardized from the training rows so that area does not overwhelm extent.
    - Accuracy, precision, recall, and F1 answer different questions. Precision watches false Besni labels. Recall watches missed Besni labels.
    - A curve computed on the test set is only a picture. Cross-validation on the training rows picks `k`. For accuracy, that search picks k={grid_acc.best_params_["n_neighbors"]}.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - Cinar, I., Koklu, M., and Tasdemir, S. (2020). Classification of Raisin Grains Using Machine Vision and Artificial Intelligence Methods. *Gazi Journal of Engineering Sciences*, 6(3), 200-209. [DOI](https://dergipark.org.tr/tr/download/article-file/1227592)
    - Dua, D. and Graff, C. (2019). UCI Machine Learning Repository. Irvine, CA: University of California, School of Information and Computer Science.
    - [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
    - [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
    """)
    return


if __name__ == "__main__":
    app.run()