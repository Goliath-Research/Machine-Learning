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
    from sklearn.datasets import make_blobs, make_circles
    from sklearn.model_selection import GridSearchCV, train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    return GridSearchCV, SVC, StandardScaler, make_blobs, make_circles, train_test_split


@app.cell
def _(np, pd):
    def plot_svc(data, model, ax):
        import seaborn as sns

        sns.scatterplot(
            data=data,
            x="X1",
            y="X2",
            hue="Class",
            palette={0: "cornflowerblue", 1: "indianred"},
            s=70,
            ax=ax,
        )
        _x_vals = np.linspace(data.X1.min() - 0.3, data.X1.max() + 0.3, 40)
        _y_vals = np.linspace(data.X2.min() - 0.3, data.X2.max() + 0.3, 40)
        _xx, _yy = np.meshgrid(_x_vals, _y_vals)
        _grid = pd.DataFrame({"X1": _xx.ravel(), "X2": _yy.ravel()})
        _levels = model.decision_function(_grid).reshape(_xx.shape)
        ax.contour(
            _xx,
            _yy,
            _levels,
            colors="black",
            levels=[-1, 0, 1],
            alpha=0.6,
            linestyles=["--", "-", "--"],
        )
        _support = model.support_vectors_
        ax.scatter(
            _support[:, 0],
            _support[:, 1],
            s=160,
            facecolors="none",
            edgecolors="black",
        )
        ax.set_title(f"SVM ({len(_support)} support vectors)")

    return (plot_svc,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Support Vector Machines

    ## Objectives

    - Place a linear margin between two classes, and see which points hold that margin in place.
    - Switch to a radial-basis kernel when no straight line can separate the classes.
    - Soften the margin with `C`, then choose `C` and the kernel by cross-validation.

    ## Background

    A support vector machine looks for a boundary with the widest empty margin it can manage. Only the points on the margin, the support vectors, determine that boundary. Points far on the correct side of the margin can move without changing the fit.

    `C` sets how hard the margin is. A large `C` tries to keep every point on its own side. A small `C` allows some points inside the margin.

    ## Datasets Used

    - **Separated blobs.** Two tight groups, used for a linear margin.
    - **Circles.** One class inside the other, used for a radial-basis kernel.
    - **Overlapping blobs.** The same generator with a wider spread, used to compare a hard margin with a soft one.
    - **Raisin Dataset.** Seven measurements of Kecimen and Besni raisins.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## More than one separating line

    The two groups below can be separated by many straight lines. The grey lines are three of them. The black point is a new case. Which class it gets depends on which line was drawn.

    A support vector machine picks the line whose empty margin is as wide as possible. The solid contour in the later charts is that decision line. The dashed contours are the two edges of the margin. The circled points are the support vectors.
    """)
    return


@app.cell
def _(make_blobs, np, pd, plt, sns):
    _X, _cl = make_blobs(
        n_samples=100, n_features=2, centers=2, random_state=2, cluster_std=1.2
    )
    df1 = pd.DataFrame({"X1": _X[:, 0], "X2": _X[:, 1], "Class": _cl})
    _figure, _axes = plt.subplots(figsize=(6, 4))
    sns.scatterplot(
        data=df1,
        x="X1",
        y="X2",
        hue="Class",
        palette={0: "cornflowerblue", 1: "indianred"},
        s=70,
        ax=_axes,
    )
    _x_line = np.linspace(-5, 6)
    for _slope, _intercept in ((-0.5, -5), (0.4, -5.5), (-0.2, -4.5)):
        _axes.plot(_x_line, _slope * _x_line + _intercept, color="gray")
    _axes.scatter([-1], [-5], s=80, c="black", zorder=3)
    _axes.set_title("Three lines that separate the same points")
    return _figure, df1


@app.cell
def _(SVC, df1, plot_svc, plt, train_test_split):
    X1_train, X1_test, y1_train, y1_test = train_test_split(
        df1[["X1", "X2"]], df1["Class"], test_size=0.30, random_state=14
    )
    data_train1 = X1_train.copy()
    data_train1["Class"] = y1_train
    svm1 = SVC(kernel="linear", C=1e10).fit(X1_train, y1_train)
    print(f"Train = {len(X1_train)} cases \t Test = {len(X1_test)} cases")
    print(f"Support vectors in each class: {svm1.n_support_}")
    print(f"Test accuracy = {svm1.score(X1_test, y1_test):.3f}")
    _figure, _axes = plt.subplots(figsize=(6, 4))
    plot_svc(data_train1, svm1, _axes)
    return _figure, data_train1, svm1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `C=1e10` asks for a hard margin. With groups this far apart, the fit can keep the margin empty. The circled points are enough. A training point that sits farther out on the correct side is not a support vector, so deleting it would leave this line where it is.

    The next two panels fit the same hard-margin model on the first 30 training rows and on the first 60. Compare the circled points. If the extra rows are outside the margin, the line does not have to move. If a new row falls inside the margin, it becomes a support vector and the line can shift.
    """)
    return


@app.cell
def _(SVC, data_train1, plot_svc, plt):
    _figure, _axes = plt.subplots(1, 2, figsize=(10, 4))
    for _axis, _n in ((_axes[0], 30), (_axes[1], 60)):
        _part = data_train1.head(_n)
        _model = SVC(kernel="linear", C=1e10).fit(_part[["X1", "X2"]], _part["Class"])
        plot_svc(_part, _model, _axis)
        _axis.set_title(f"First {_n} training rows")
    _figure.tight_layout()
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A kernel for circles

    No straight line separates a ring from the points inside it. A radial-basis kernel (`kernel="rbf"`) measures similarity to the training points instead of cutting the plane with one line. In the picture, the decision contour closes around the inner class.

    `C=1e6` again asks the margin to be strict.
    """)
    return


@app.cell
def _(SVC, make_circles, pd, plot_svc, plt, train_test_split):
    _X, _cl = make_circles(100, factor=0.2, noise=0.1, random_state=2)
    df2 = pd.DataFrame({"X1": _X[:, 0], "X2": _X[:, 1], "Class": _cl})
    X2_train, X2_test, y2_train, y2_test = train_test_split(
        df2[["X1", "X2"]], df2["Class"], test_size=0.30, random_state=39
    )
    data_train2 = X2_train.copy()
    data_train2["Class"] = y2_train
    svm2 = SVC(kernel="rbf", C=1e6).fit(X2_train, y2_train)
    print(f"Train = {len(X2_train)} cases \t Test = {len(X2_test)} cases")
    print(f"Test accuracy = {svm2.score(X2_test, y2_test):.3f}")
    _figure, _axes = plt.subplots(figsize=(6, 4))
    plot_svc(data_train2, svm2, _axes)
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Softening the margin

    When the groups overlap, a completely empty margin may not exist. `C` trades off a wide margin against points on the wrong side.

    - Large `C` (here 10,000) keeps the margin hard. The fit spends more effort keeping training points out of it.
    - Small `C` (here 0.001) keeps the margin soft. More points are allowed inside it, and the support-vector count usually grows.

    Both models below are linear. The accuracy is computed on the held-out rows, not on a cloud of points that the model itself just labeled.
    """)
    return


@app.cell
def _(SVC, make_blobs, pd, plot_svc, plt, train_test_split):
    _X, _cl = make_blobs(
        n_samples=100, n_features=2, centers=2, random_state=2, cluster_std=3
    )
    df3 = pd.DataFrame({"X1": _X[:, 0], "X2": _X[:, 1], "Class": _cl})
    X3_train, X3_test, y3_train, y3_test = train_test_split(
        df3[["X1", "X2"]], df3["Class"], test_size=0.30, random_state=14
    )
    data_train3 = X3_train.copy()
    data_train3["Class"] = y3_train
    svm_hard = SVC(kernel="linear", C=10000).fit(X3_train, y3_train)
    svm_soft = SVC(kernel="linear", C=0.001).fit(X3_train, y3_train)
    print(f"Hard margin, C=10000, support vectors = {svm_hard.n_support_.sum()}, test accuracy = {svm_hard.score(X3_test, y3_test):.3f}")
    print(f"Soft margin, C=0.001, support vectors = {svm_soft.n_support_.sum()}, test accuracy = {svm_soft.score(X3_test, y3_test):.3f}")
    _figure, _axes = plt.subplots(1, 2, figsize=(10, 4))
    plot_svc(data_train3, svm_hard, _axes[0])
    _axes[0].set_title("C = 10000")
    plot_svc(data_train3, svm_soft, _axes[1])
    _axes[1].set_title("C = 0.001")
    _figure.tight_layout()
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Which value of `C` asks for the harder margin, `10000` or `0.001`? Replace `None` with that number.
    """)
    return


@app.cell
def _():
    student_hard_c = None
    print(student_hard_c)
    return (student_hard_c,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r'''
```python
student_hard_c = 10000
```
**10000.** A large `C` penalizes points inside the margin more heavily. `0.001` lets the margin grow and take points in.
''')
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Raisins

    Kecimen is 0 and Besni is 1. The columns are standardized from the training rows, because a linear kernel uses the numeric scale and an RBF kernel uses distances.

    The first model is a linear SVM with `C=1`. The grid then tries `linear`, `rbf`, and `sigmoid` kernels at `C` in 0.1, 1, 100, and 1000. Five-fold cross-validation on the training rows picks the pair. The test accuracy is reported after that choice.
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
def _(
    GridSearchCV,
    SVC,
    StandardScaler,
    pd,
    raisin_file,
    train_test_split,
):
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
    svm_raisin = SVC(kernel="linear", C=1).fit(Xr_train, yr_train)
    _search = GridSearchCV(
        SVC(),
        {"kernel": ("linear", "rbf", "sigmoid"), "C": [0.1, 1, 100, 1000]},
        cv=5,
    )
    _search.fit(Xr_train, yr_train)
    svm_best = _search.best_estimator_
    print(dfr.shape)
    print(f"Linear C=1 training accuracy = {svm_raisin.score(Xr_train, yr_train):.3f}")
    print(f"Linear C=1 testing accuracy  = {svm_raisin.score(Xr_test, yr_test):.3f}")
    print(f"Best parameters: {_search.best_params_}")
    print(f"Best model training accuracy = {svm_best.score(Xr_train, yr_train):.3f}")
    print(f"Best model testing accuracy  = {svm_best.score(Xr_test, yr_test):.3f}")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The GitHub address is what a browser session can read. The local name `Raisin.csv` is in a comment. The restriction that blocks some other sites is called **CORS**.

    `best_estimator_` is already refit on all of the training rows. The test score above is a single look at the rows that were not used to pick the kernel or `C`.
    """)
    return


@app.cell(hide_code=True)
def _(mo, svm1, svm_best, X1_test, y1_test):
    mo.md(f"""
    ## Conclusions

    **Key takeaways:**

    - Many lines can separate two distant groups. The support-vector fit keeps the one with the widest empty margin.
    - Only the support vectors determine that margin. On the separated blobs, the linear model uses {int(svm1.n_support_.sum())} of them, and the test accuracy is {svm1.score(X1_test, y1_test):.3f}.
    - A radial-basis kernel can enclose one class inside another. A linear kernel cannot.
    - Large `C` hardens the margin. Small `C` lets points sit inside it.
    - On the raisins, cross-validation selects kernel `{svm_best.kernel}` and C={svm_best.C}.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check your understanding

    1. What is a support vector?
    2. What do the dashed contours show?
    3. Why does the circle data need a kernel other than `linear`?
    4. What does a small `C` allow?
    5. Why are the raisin columns standardized before this fit?
    6. When is the test accuracy of the raisin search computed, relative to choosing the kernel?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
1. **A training point on the margin:** Those points are circled. Moving a point that is far outside the margin does not change the fit.
2. **The edges of the margin:** The solid contour is the decision line. The dashed contours sit one unit away in decision-function units.
3. **No straight line works:** One class sits inside the other, so the boundary has to close.
4. **Points inside the margin:** The fit accepts a wider margin and more violations.
5. **The kernel uses distances or a dot product:** Area would otherwise dominate extent. The scaler is fit on the training rows.
6. **After the choice:** Cross-validation uses the training rows. The test score is computed once the kernel and `C` are already chosen.
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
    - Dua, D. and Graff, C. (2019). UCI Machine Learning Repository. Irvine, CA: University of California, School of Information and Computer Science.
    - VanderPlas, J. (2017). *Python Data Science Handbook*. O'Reilly Media. Chapter 5.
    - [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
    - [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
    """)
    return


if __name__ == "__main__":
    app.run()
