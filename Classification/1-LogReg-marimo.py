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
    import numpy as np
    import pandas as pd
    import seaborn as sns

    plt.rcParams["figure.max_open_warning"] = 0
    sns.set_theme(style="whitegrid")

    return mo, np, pd, plt, sns


@app.cell
def _():
    from sklearn import linear_model, metrics
    from sklearn.datasets import load_wine, make_blobs
    from sklearn.model_selection import train_test_split

    return linear_model, load_wine, make_blobs, metrics, train_test_split


@app.cell
def _():
    import statsmodels.api as sm

    return (sm,)


@app.cell
def _(metrics, pd):
    def confusion_table(y_true, y_pred):
        labels = sorted(set(y_true).union(set(y_pred)))
        matrix = metrics.confusion_matrix(y_true, y_pred, labels=labels)
        return pd.DataFrame(
            matrix,
            index=[f"true {label}" for label in labels],
            columns=[f"pred {label}" for label in labels],
        )

    return (confusion_table,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Logistic Regression

    ## Objectives

    - Fit logistic regression for two classes, then for three classes.
    - Read a predicted probability and turn it into a class at a cutoff of 0.5.
    - Compare a scikit-learn fit with a statsmodels fit on the same training rows.

    ## Background

    Logistic regression builds a linear score, then passes that score through the sigmoid function. The sigmoid output is between 0 and 1, so it can be read as a probability.

    For two classes, a probability of 0.5 or more is class 1. Below 0.5 is class 0. For three classes, the model makes one score per class and assigns the class with the largest probability.

    ## Datasets Used

    - **Synthetic blobs.** Two numeric features and two classes, from `make_blobs`.
    - **Raisin Dataset.** Seven size and shape measurements of Kecimen and Besni raisins. There are 900 rows.
    - **Wine Dataset.** Thirteen chemical measurements of wines from three cultivars in the same region of Italy. This lesson loads the copy shipped with scikit-learn. The folder also contains `wine.csv`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The sigmoid function

    The sigmoid is bounded below by 0 and above by 1.

    - A positive score gives a probability above 0.5.
    - A negative score gives a probability below 0.5.
    - A score of 0 gives a probability of exactly 0.5.

    The dashed line in the chart is that 0.5 cutoff.
    """)
    return


@app.cell
def _(np):
    def sigmoid(t):
        return 1 / (1 + np.exp(-t))

    return (sigmoid,)


@app.cell
def _(np, plt, sigmoid):
    _t = np.arange(-6, 6, 0.1)
    _figure, _axes = plt.subplots(figsize=(6, 4))
    _axes.plot(_t, sigmoid(_t), color="darkred")
    _axes.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    _axes.set_title("The sigmoid function")
    _axes.set_xlabel("linear score")
    _axes.set_ylabel("probability of class 1")
    return _figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    What does the sigmoid return when the score is 0? Replace `None` with that number.
    """)
    return


@app.cell
def _():
    student_sigmoid_at_zero = None
    print(student_sigmoid_at_zero)
    return (student_sigmoid_at_zero,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r'''
```python
student_sigmoid_at_zero = 0.5
```
**0.5.** The sigmoid of 0 is `1 / (1 + e^0)`, and `e^0` is 1, so the value is 1/2.
''')
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A two-class model on blobs

    `make_blobs` draws two round groups. Class 0 is blue and class 1 is red in the charts below.

    A holdout split estimates how the fit behaves on rows it did not train on.

    1. Keep 80 percent of the rows for training.
    2. Fit the model on those rows.
    3. Score it on the other 20 percent.

    `random_state=23` keeps the split fixed, so the counts below do not change from run to run. One hundred rows give 80 training rows and 20 test rows.
    """)
    return


@app.cell
def _(make_blobs, pd, plt, sns):
    _X, _cl = make_blobs(
        n_samples=100, n_features=2, centers=2, random_state=2, cluster_std=3
    )
    dfb = pd.DataFrame({"X1": _X[:, 0], "X2": _X[:, 1], "Class": _cl})
    _figure, _axes = plt.subplots(figsize=(6, 4))
    sns.scatterplot(
        data=dfb,
        x="X1",
        y="X2",
        hue="Class",
        palette={0: "cornflowerblue", 1: "indianred"},
        s=70,
        ax=_axes,
    )
    _axes.set_title("Make blobs data")
    return _figure


@app.cell
def _(dfb, train_test_split):
    Xb = dfb[["X1", "X2"]]
    yb = dfb["Class"]
    Xb_train, Xb_test, yb_train, yb_test = train_test_split(
        Xb, yb, test_size=0.2, random_state=23
    )
    print(f"Train = {len(Xb_train)} cases \t Test = {len(Xb_test)} cases")


@app.cell
def _(Xb_test, Xb_train, plt, sns, yb_test, yb_train):
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Scikit-learn

    `LogisticRegression` regularizes by default. Statsmodels, used in the next section, does not. A very large `C` makes the scikit-learn penalty tiny, so the two fits can be compared. `C=1e9` is that setting.
    """)
    return


@app.cell
def _(Xb_train, linear_model, yb_train):
    log_model = linear_model.LogisticRegression(C=1e9)
    log_model.fit(Xb_train, yb_train)
    print(f"Coefficients: {log_model.coef_.round(3)}")
    print(f"Intercept:    {log_model.intercept_.round(3)}")
    return (log_model,)


@app.cell
def _(Xb_train, log_model, mo, pd, yb_train):
    _prob = log_model.predict_proba(Xb_train)
    data_train = pd.concat([Xb_train, yb_train], axis=1).copy()
    data_train["prob_0_sl"] = _prob[:, 0]
    data_train["prob_1_sl"] = _prob[:, 1]
    data_train["Class_sl"] = log_model.predict(Xb_train)
    return mo.Html(data_train.head().round(3).to_html())


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `prob_0_sl` and `prob_1_sl` are the two class probabilities. They add to 1 in each row. `Class_sl` is the class whose probability is larger. That comparison is the whole decision rule.
    """)
    return


@app.cell
def _(Xb_test, log_model, mo, pd, yb_test):
    _prob = log_model.predict_proba(Xb_test)
    data_test = pd.concat([Xb_test, yb_test], axis=1).copy()
    data_test["prob_0_sl"] = _prob[:, 0]
    data_test["prob_1_sl"] = _prob[:, 1]
    data_test["Class_sl"] = log_model.predict(Xb_test)
    return mo.Html(data_test.head().round(3).to_html())


@app.cell
def _(Xb_train, log_model, np, pd, plt, sns, yb_train):
    _rng = np.random.default_rng(0)
    _grid = pd.DataFrame(
        {
            "X1": _rng.uniform(Xb_train.X1.min(), Xb_train.X1.max(), size=1000),
            "X2": _rng.uniform(Xb_train.X2.min(), Xb_train.X2.max(), size=1000),
        }
    )
    _grid["Class"] = log_model.predict(_grid)
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
    _axes.set_title("Logistic regression regions")
    return _figure


@app.cell
def _(confusion_table, data_test, mo):
    _table = confusion_table(data_test["Class"], data_test["Class_sl"])
    return mo.Html(_table.to_html())


@app.cell(hide_code=True)
def _(Xb_test, Xb_train, log_model, mo, yb_test, yb_train):
    _train_acc = log_model.score(Xb_train, yb_train)
    _test_acc = log_model.score(Xb_test, yb_test)
    mo.md(f"""
    Cases on the main diagonal of the confusion matrix were classified correctly.

    Accuracy is the share of rows with the right class. On the blobs, training accuracy is {_train_acc:.3f} and test accuracy is {_test_acc:.3f}.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Statsmodels

    `sm.Logit` fits the same two-class model by maximum likelihood. `add_constant` inserts the intercept column that scikit-learn adds on its own.

    The summary reports a pseudo R-squared, a likelihood-ratio test against a model with only an intercept, and a coefficient table. The coefficients are log-odds. A positive coefficient means that a larger feature raises the log-odds of class 1.
    """)
    return


@app.cell
def _(Xb_train, sm, yb_train):
    log_model_sm = sm.Logit(yb_train, sm.add_constant(Xb_train)).fit()
    print(log_model_sm.summary())
    return (log_model_sm,)


@app.cell
def _(log_model, log_model_sm, mo, np, pd):
    _compare = pd.DataFrame(
        {
            "sklearn": np.r_[log_model.intercept_, log_model.coef_.ravel()],
            "statsmodels": log_model_sm.params.to_numpy(),
        },
        index=["intercept", "X1", "X2"],
    )
    _gap = np.max(np.abs(_compare["sklearn"] - _compare["statsmodels"]))
    print(f"Largest absolute gap between the two fits: {_gap:.6f}")
    return mo.Html(_compare.round(4).to_html())


@app.cell(hide_code=True)
def _(log_model_sm, mo):
    mo.md(f"""
    Pseudo R-squared for this fit is {log_model_sm.prsquared:.4f}. It is not the R-squared of linear regression. Larger values mean the features improve the likelihood.

    The likelihood-ratio p-value is {log_model_sm.llr_pvalue:.3e}. It compares this fit with a model that has only an intercept. A very small value means the two features improve the fit.

    There are {int(log_model_sm.nobs)} training observations and {int(log_model_sm.df_model)} features besides the intercept.
    """)
    return


@app.cell
def _(data_test, data_train, mo, pd, sm, log_model_sm):
    _prob_train = log_model_sm.predict(sm.add_constant(data_train[["X1", "X2"]]))
    _prob_test = log_model_sm.predict(sm.add_constant(data_test[["X1", "X2"]]))
    data_test_sm = data_test.copy()
    data_test_sm["Class_sm"] = (_prob_test >= 0.5).astype(int)
    _agree = (data_test_sm["Class_sl"] == data_test_sm["Class_sm"]).mean()
    _show = pd.DataFrame(
        {
            "Class": data_test_sm["Class"].to_numpy(),
            "Class_sl": data_test_sm["Class_sl"].to_numpy(),
            "Class_sm": data_test_sm["Class_sm"].to_numpy(),
            "prob_1_sm": _prob_test.to_numpy(),
        }
    )
    print(f"Share of test rows with the same class from both libraries: {_agree:.3f}")
    print(f"Training rows scored with statsmodels: {len(_prob_train)}")
    return mo.Html(_show.head().round(3).to_html())


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Statsmodels returns the probability of class 1. The class label is 1 when that probability is at least 0.5, and 0 otherwise. On this split the two libraries assign the same test labels.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Raisin varieties

    The raisin file has two varieties grown in Turkey, Kecimen and Besni. Kecimen is coded as 0 and Besni as 1, so column 1 of `predict_proba` is the probability of Besni.

    Pandas can read a file from a web address. Whether that works depends on **where Python is running**.

    - **On your computer.** Python can read `Raisin.csv` from this folder. That line is kept in a comment.
    - **Inside a web browser.** The browser only downloads files from websites that allow it. This rule is called **CORS**. This lesson reads a GitHub copy so the browser can load the file.

    The seven predictors are not on the same scale. Area is in the tens of thousands, and extent is near 0.7. This first fit still uses the raw numbers, with `max_iter=500` so the solver can finish. Later lessons standardize these columns before methods that depend on distance.
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
def _(mo, pd, raisin_file):
    # On this computer the file is also next to the notebook:
    # dfr = pd.read_csv("Raisin.csv")
    dfr = pd.read_csv(raisin_file)
    print(dfr.shape)
    return mo.Html(dfr.head().to_html())


@app.cell
def _(dfr, plt):
    _figure, _axes = plt.subplots(figsize=(6, 4))
    dfr["Class"].value_counts().plot(kind="bar", rot=0, ax=_axes, title="Raisin varieties")
    return _figure


@app.cell
def _(dfr, mo, train_test_split):
    label_to_number = {"Kecimen": 0, "Besni": 1}
    dfr_coded = dfr.copy()
    dfr_coded["Besni"] = dfr_coded["Class"].map(label_to_number)
    yr = dfr_coded["Besni"]
    Xr = dfr_coded.drop(columns=["Class", "Besni"])
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        Xr, yr, test_size=0.2, random_state=23
    )
    print(f"Train = {len(Xr_train)} cases \t Test = {len(Xr_test)} cases")
    return mo.Html(dfr_coded.head().to_html())


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    After the recoding above, which integer means Besni? Replace `None` with that integer.
    """)
    return


@app.cell
def _():
    student_besni_code = None
    print(student_besni_code)
    return (student_besni_code,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r'''
```python
student_besni_code = 1
```
**1.** The dictionary is `{"Kecimen": 0, "Besni": 1}`. The probability in column 1 is the probability of Besni.
''')
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Scikit-learn can minimize the logistic loss with several solvers.

    - `newton-cg` uses Newton's method and works on small and medium tables.
    - `lbfgs` is a quasi-Newton method. It is the default when every column is numeric.
    - `liblinear` uses coordinate descent and is aimed at high-dimensional tables.
    - `sag` and `saga` use stochastic gradients. `saga` also supports an L1 penalty.

    `max_iter` is the iteration cap. The default is 100. This raisin fit raises it to 500. `C` is again `1e9`, so the penalty stays out of the way.
    """)
    return


@app.cell
def _(Xr_test, Xr_train, confusion_table, linear_model, mo, yr_test, yr_train):
    log_model_r = linear_model.LogisticRegression(solver="lbfgs", C=1e9, max_iter=500)
    log_model_r.fit(Xr_train, yr_train)
    _pred = log_model_r.predict(Xr_test)
    _table = confusion_table(yr_test, _pred)
    print(f"Coefficients: {log_model_r.coef_.round(3)}")
    print(f"Intercept:    {log_model_r.intercept_.round(3)}")
    print(f"Training accuracy = {log_model_r.score(Xr_train, yr_train):.3f}")
    print(f"Testing accuracy  = {log_model_r.score(Xr_test, yr_test):.3f}")
    return mo.Html(_table.to_html())


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Three wine cultivars

    Multinomial logistic regression is the same idea when the label has more than two values. The wine table has 178 samples and 13 measurements. All of the wines come from the same region in Italy, and the label is the cultivar.

    The fit uses `newton-cg`. For three classes, scikit-learn estimates one linear score per class. The printed coefficient array has one row per class.
    """)
    return


@app.cell
def _(load_wine, mo, pd, plt):
    _wine = load_wine()
    dfw = pd.DataFrame(_wine.data, columns=_wine.feature_names)
    dfw["Class"] = _wine.target_names[_wine.target]
    print(dfw.shape)
    _figure, _axes = plt.subplots(figsize=(6, 4))
    dfw["Class"].value_counts(sort=False).plot(
        kind="bar", rot=0, ax=_axes, title="Wine cultivars"
    )
    return _figure


@app.cell
def _(dfw, linear_model, mo, pd, train_test_split):
    yw = dfw["Class"]
    Xw = dfw.drop(columns=["Class"])
    Xw_train, Xw_test, yw_train, yw_test = train_test_split(
        Xw, yw, test_size=0.2, random_state=23
    )
    log_model_w = linear_model.LogisticRegression(solver="newton-cg", max_iter=200)
    log_model_w.fit(Xw_train, yw_train)
    _prob = log_model_w.predict_proba(Xw_test)
    _pred = log_model_w.predict(Xw_test)
    _show = pd.DataFrame(_prob, columns=[f"prob_{name}" for name in log_model_w.classes_])
    _show["Class"] = yw_test.to_numpy()
    _show["Class_pred"] = _pred
    print(f"Train = {len(Xw_train)} cases \t Test = {len(Xw_test)} cases")
    print("Coefficient rows, one per class:")
    print(pd.DataFrame(log_model_w.coef_.round(2), index=log_model_w.classes_, columns=Xw.columns))
    print(f"Intercepts: {log_model_w.intercept_.round(2)}")
    print(f"Training accuracy = {log_model_w.score(Xw_train, yw_train):.3f}")
    print(f"Testing accuracy  = {log_model_w.score(Xw_test, yw_test):.3f}")
    return mo.Html(_show.head().round(3).to_html())


@app.cell
def _(confusion_table, log_model_w, mo, Xw_test, yw_test):
    _table = confusion_table(yw_test, log_model_w.predict(Xw_test))
    return mo.Html(_table.to_html())


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Each test row has three probabilities, one per cultivar. The predicted class is the largest of the three. On this split the test diagonal is the whole matrix: every held-out wine is assigned to its own cultivar.
    """)
    return


@app.cell(hide_code=True)
def _(Xr_test, Xw_test, log_model, log_model_r, log_model_w, mo, Xb_test, yb_test, yr_test, yw_test):
    mo.md(f"""
    ## Conclusions

    **Key takeaways:**

    - The sigmoid turns a linear score into a probability. At a score of 0 the probability is 0.5.
    - With two classes, the predicted label is the class with the larger probability. The usual cutoff is 0.5.
    - `C=1e9` makes scikit-learn's penalty tiny, so its blob coefficients can be set next to the statsmodels coefficients.
    - Statsmodels adds the intercept through `add_constant` and reports a pseudo R-squared and a likelihood-ratio test. Scikit-learn reports coefficients, predictions, and accuracy.
    - The raisin test accuracy, with raw measurements and `C=1e9`, is {log_model_r.score(Xr_test, yr_test):.3f}.
    - The wine model has one coefficient row per cultivar. Its test accuracy on this split is {log_model_w.score(Xw_test, yw_test):.3f}.
    - The blob test accuracy is {log_model.score(Xb_test, yb_test):.3f}. A confusion matrix shows which of those errors, if any, are which class.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check your understanding

    1. What does a sigmoid value of 0.8 mean for a two-class model?
    2. Why is `C` set to `1e9` in the scikit-learn fits that are compared with statsmodels?
    3. Which integer is Besni, and which probability column matches it?
    4. How many coefficient rows does the wine model have?
    5. What do the counts on the main diagonal of a confusion matrix represent?
    6. What does `random_state=23` change about the 80/20 split?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
1. **Class 1 is more likely:** 0.8 is above 0.5, so the predicted class is class 1. The model also says the chance of class 0 is 0.2.
2. **To turn the penalty almost off:** Scikit-learn regularizes by default. Statsmodels does not. A huge `C` lets the coefficients be compared.
3. **Besni is 1:** That is column 1 of `predict_proba`. Kecimen is 0.
4. **Three:** There is one linear score for each cultivar.
5. **Correct classifications:** Each diagonal cell counts rows whose predicted class matches the true class.
6. **It fixes the split:** The same 80 training rows and 20 test rows are used every time the notebook runs.
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
    - [sklearn.linear_model.LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
    - [statsmodels Logit](https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.Logit.html)
    """)
    return


if __name__ == "__main__":
    app.run()
