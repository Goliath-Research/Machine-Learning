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
    from sklearn.datasets import load_wine, make_blobs
    from sklearn.discriminant_analysis import (
        LinearDiscriminantAnalysis,
        QuadraticDiscriminantAnalysis,
    )
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    return (
        LinearDiscriminantAnalysis,
        QuadraticDiscriminantAnalysis,
        confusion_matrix,
        load_wine,
        make_blobs,
        train_test_split,
        StandardScaler,
    )


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


@app.cell
def _(StandardScaler, pd):
    def scale_train_test(x_train, x_test):
        scaler = StandardScaler()
        x_train_s = pd.DataFrame(
            scaler.fit_transform(x_train),
            columns=x_train.columns,
            index=x_train.index,
        )
        x_test_s = pd.DataFrame(
            scaler.transform(x_test),
            columns=x_test.columns,
            index=x_test.index,
        )
        return x_train_s, x_test_s, scaler

    return (scale_train_test,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Discriminant Analysis

    ## Objectives

    - Separate classes with linear discriminant analysis, then with quadratic discriminant analysis.
    - Standardize a new case with the means and standard deviations learned from the training rows.
    - Read a confusion matrix and the training and test accuracy.

    ## Background

    Discriminant analysis classifies a row by how well its measurements match each class. Linear discriminant analysis (LDA) uses one shared covariance, so the boundary between two classes is a straight line. Quadratic discriminant analysis (QDA) estimates a covariance for each class, so the boundary can curve.

    ## Datasets Used

    - **Synthetic blobs.** Two features and two classes, from `make_blobs`.
    - **Raisin Dataset.** Seven measurements of Kecimen and Besni raisins.
    - **Wine Dataset.** Thirteen chemical measurements and three cultivars, loaded from scikit-learn.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A linear boundary on blobs

    The blob groups are about the same shape, which is the situation LDA is built for. The split is 80 percent training and 20 percent test, with `random_state=23`. Class 0 is blue and class 1 is red. Test rows are drawn as crosses.
    """)
    return


@app.cell
def _(make_blobs, pd, plt, sns, train_test_split):
    _X, _cl = make_blobs(
        n_samples=100, n_features=2, centers=2, random_state=2, cluster_std=3
    )
    dfb = pd.DataFrame({"X1": _X[:, 0], "X2": _X[:, 1], "Class": _cl})
    Xb = dfb[["X1", "X2"]]
    yb = dfb["Class"]
    Xb_train, Xb_test, yb_train, yb_test = train_test_split(
        Xb, yb, test_size=0.2, random_state=23
    )
    print(f"Train = {len(Xb_train)} cases \t Test = {len(Xb_test)} cases")
    _figure, _axes = plt.subplots(figsize=(6, 4))
    sns.scatterplot(
        x=Xb_train.X1,
        y=Xb_train.X2,
        hue=yb_train,
        palette={0: "cornflowerblue", 1: "indianred"},
        s=70,
        ax=_axes,
    )
    sns.scatterplot(
        x=Xb_test.X1,
        y=Xb_test.X2,
        hue=yb_test,
        palette={0: "darkblue", 1: "darkred"},
        s=90,
        marker="X",
        legend=False,
        ax=_axes,
    )
    _axes.set_title("Training circles and test crosses")
    return _figure


@app.cell
def _(LinearDiscriminantAnalysis, Xb_train, mo, pd, yb_train, Xb_test, yb_test):
    lda = LinearDiscriminantAnalysis()
    lda.fit(Xb_train, yb_train)
    _prob = lda.predict_proba(Xb_train)
    data_train = pd.concat([Xb_train, yb_train], axis=1).copy()
    data_train["prob_0"] = _prob[:, 0]
    data_train["prob_1"] = _prob[:, 1]
    data_train["Class_lda"] = lda.predict(Xb_train)
    print(f"Coefficients: {lda.coef_.round(3)}")
    print(f"Intercept:    {lda.intercept_.round(3)}")
    print(f"Training accuracy = {lda.score(Xb_train, yb_train):.3f}")
    print(f"Testing accuracy  = {lda.score(Xb_test, yb_test):.3f}")
    return mo.Html(data_train.head().round(3).to_html())


@app.cell
def _(Xb_test, confusion_table, lda, mo, yb_test):
    _table = confusion_table(yb_test, lda.predict(Xb_test))
    return mo.Html(_table.to_html())


@app.cell
def _(Xb_train, lda, np, pd, plt, sns, yb_train):
    _rng = np.random.default_rng(0)
    _grid = pd.DataFrame(
        {
            "X1": _rng.uniform(Xb_train.X1.min(), Xb_train.X1.max(), size=1000),
            "X2": _rng.uniform(Xb_train.X2.min(), Xb_train.X2.max(), size=1000),
        }
    )
    _grid["Class"] = lda.predict(_grid)
    _figure, _axes = plt.subplots(figsize=(6, 4))
    sns.scatterplot(
        data=_grid,
        x="X1",
        y="X2",
        hue="Class",
        palette={0: "cornflowerblue", 1: "indianred"},
        alpha=0.15,
        s=30,
        legend=False,
        ax=_axes,
    )
    sns.scatterplot(
        x=Xb_train.X1,
        y=Xb_train.X2,
        hue=yb_train,
        palette={0: "cornflowerblue", 1: "indianred"},
        s=70,
        ax=_axes,
    )
    _axes.set_title("Linear discriminant regions")
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The pale background is the class LDA would assign across the plot. The boundary is a straight cut. Counts on the main diagonal of the confusion matrix are the test rows that landed in the right class.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Raisins, with the scaler fit on the training rows

    Kecimen is coded as 0 and Besni as 1. Area, axis lengths, and perimeter are much larger than eccentricity and extent. LDA and QDA use the spread of each column, so this lesson standardizes them.

    The scaler is fit on the training rows only. The test rows are transformed with those training means and standard deviations. Fitting the scaler on the whole table would let the test rows change the means.

    A new raisin is handled the same way: subtract the training means, divide by the training standard deviations, then call `predict`.
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
def _(mo, pd, raisin_file, scale_train_test, train_test_split):
    # On this computer the file is also next to the notebook:
    # dfr = pd.read_csv("Raisin.csv")
    dfr = pd.read_csv(raisin_file)
    print(dfr.shape)
    _coded = dfr.copy()
    _coded["Besni"] = _coded["Class"].map({"Kecimen": 0, "Besni": 1})
    yr = _coded["Besni"]
    Xr = _coded.drop(columns=["Class", "Besni"])
    _Xr_train, _Xr_test, yr_train, yr_test = train_test_split(
        Xr, yr, test_size=0.2, random_state=23
    )
    Xr_train, Xr_test, raisin_scaler = scale_train_test(_Xr_train, _Xr_test)
    print(f"Train = {len(Xr_train)} cases \t Test = {len(Xr_test)} cases")
    return mo.Html(Xr_train.head().round(3).to_html())


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Pandas can read the file from GitHub so the notebook also runs in a browser. The browser rule that blocks some websites is called **CORS**. The local file name is in a comment.

    The first rows above are training rows after standardization. Each column is centered near 0.
    """)
    return


@app.cell
def _(LinearDiscriminantAnalysis, Xr_test, Xr_train, mo, yr_test, yr_train):
    lda_r = LinearDiscriminantAnalysis()
    lda_r.fit(Xr_train, yr_train)
    print(f"Coefficients: {lda_r.coef_.round(3)}")
    print(f"Intercept:    {lda_r.intercept_.round(3)}")
    print(f"LDA training accuracy = {lda_r.score(Xr_train, yr_train):.3f}")
    print(f"LDA testing accuracy  = {lda_r.score(Xr_test, yr_test):.3f}")
    return (lda_r,)


@app.cell
def _(QuadraticDiscriminantAnalysis, Xr_test, Xr_train, yr_test, yr_train):
    qda_r = QuadraticDiscriminantAnalysis()
    qda_r.fit(Xr_train, yr_train)
    print(f"QDA training accuracy = {qda_r.score(Xr_train, yr_train):.3f}")
    print(f"QDA testing accuracy  = {qda_r.score(Xr_test, yr_test):.3f}")
    return (qda_r,)


@app.cell
def _(Xr_test, confusion_table, lda_r, mo, pd, qda_r, yr_test):
    _side = pd.concat(
        [
            confusion_table(yr_test, lda_r.predict(Xr_test)).assign(model="LDA"),
            confusion_table(yr_test, qda_r.predict(Xr_test)).assign(model="QDA"),
        ]
    )
    return mo.Html(_side.to_html())


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Which method can draw a curved boundary, LDA or QDA? Replace `None` with the short name.
    """)
    return


@app.cell
def _():
    student_curved_boundary = None
    print(student_curved_boundary)
    return (student_curved_boundary,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r'''
```python
student_curved_boundary = "QDA"
```
**QDA.** It estimates a separate covariance for each class. LDA shares one covariance, so its two-class boundary is a line.
''')
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Three wine cultivars

    With three classes, LDA builds one linear score per class. A new wine is assigned to the class with the largest score. This is the one-versus-rest layout: each score compares one cultivar with the other two.

    `solver="svd"` is the default. `store_covariance=True` keeps the class covariances used by `predict_proba`. The chemical columns are standardized from the training rows, for the same reason as the raisin columns.
    """)
    return


@app.cell
def _(LinearDiscriminantAnalysis, load_wine, mo, pd, plt, scale_train_test, train_test_split):
    _wine = load_wine()
    dfw = pd.DataFrame(_wine.data, columns=_wine.feature_names)
    dfw["Class"] = _wine.target_names[_wine.target]
    yw = dfw["Class"]
    _Xw = dfw.drop(columns=["Class"])
    _Xw_train, _Xw_test, yw_train, yw_test = train_test_split(
        _Xw, yw, test_size=0.2, random_state=23
    )
    Xw_train, Xw_test, _wine_scaler = scale_train_test(_Xw_train, _Xw_test)
    lda_w = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
    lda_w.fit(Xw_train, yw_train)
    _prob = lda_w.predict_proba(Xw_test)
    _show = pd.DataFrame(_prob, columns=[f"prob_{name}" for name in lda_w.classes_])
    _show["Class"] = yw_test.to_numpy()
    _show["Class_pred"] = lda_w.predict(Xw_test)
    print(dfw.shape)
    print(f"Train = {len(Xw_train)} cases \t Test = {len(Xw_test)} cases")
    print(pd.DataFrame(lda_w.coef_.round(2), index=lda_w.classes_, columns=Xw_train.columns))
    print(f"Training accuracy = {lda_w.score(Xw_train, yw_train):.3f}")
    print(f"Testing accuracy  = {lda_w.score(Xw_test, yw_test):.3f}")
    _figure, _axes = plt.subplots(figsize=(6, 4))
    dfw["Class"].value_counts(sort=False).plot(
        kind="bar", rot=0, ax=_axes, title="Wine cultivars"
    )
    return _figure, mo.Html(_show.head().round(3).to_html())


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The coefficient table has three rows. Each row is the linear score for one cultivar. The probability columns in the preview are those scores after they have been turned into probabilities. The predicted class is the largest probability.
    """)
    return


@app.cell(hide_code=True)
def _(Xr_test, lda, lda_r, mo, qda_r, Xb_test, yb_test, yr_test):
    mo.md(f"""
    ## Conclusions

    **Key takeaways:**

    - LDA puts a straight boundary between two classes that share a covariance. On the blobs, the test accuracy is {lda.score(Xb_test, yb_test):.3f}.
    - QDA gives each class its own covariance, so the boundary can curve. On the standardized raisins, LDA test accuracy is {lda_r.score(Xr_test, yr_test):.3f} and QDA test accuracy is {qda_r.score(Xr_test, yr_test):.3f}.
    - The raisin and wine scalers are fit on the training rows. Test rows, and any later row, use that same scaler.
    - A confusion matrix counts correct rows on the main diagonal.
    - With three wine cultivars, LDA keeps one linear score per class and assigns the class with the largest score.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check your understanding

    1. Why is the two-class LDA boundary a straight line?
    2. What does QDA estimate that LDA does not?
    3. Why is the scaler fit on the training rows only?
    4. Which integer is Besni?
    5. What does `store_covariance=True` keep for the wine model?
    6. Where are the correct classifications in a confusion matrix?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
1. **One shared covariance:** Two classes with the same spread are separated by a linear score.
2. **A covariance per class:** That extra flexibility makes the boundary quadratic.
3. **So the test rows do not affect the means:** A new case must be scaled with the training means and standard deviations.
4. **1:** Kecimen is 0 and Besni is 1.
5. **The class covariances:** `predict_proba` uses them.
6. **On the main diagonal:** Those cells are predicted class equal to true class.
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
    - [LinearDiscriminantAnalysis](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html)
    - [QuadraticDiscriminantAnalysis](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html)
    """)
    return


if __name__ == "__main__":
    app.run()
