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
    from sklearn.discriminant_analysis import (
        LinearDiscriminantAnalysis,
        QuadraticDiscriminantAnalysis,
    )
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, auc, f1_score, roc_curve
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    return (
        DecisionTreeClassifier,
        KNeighborsClassifier,
        LinearDiscriminantAnalysis,
        LogisticRegression,
        QuadraticDiscriminantAnalysis,
        RandomForestClassifier,
        CalibratedClassifierCV,
        SVC,
        StandardScaler,
        accuracy_score,
        auc,
        f1_score,
        roc_curve,
        train_test_split,
    )


@app.cell
def _(accuracy_score, auc, f1_score, roc_curve):
    def evaluate(model, x_test, y_test):
        _pred = model.predict(x_test)
        _prob = model.predict_proba(x_test)[:, 1]
        _fpr, _tpr, _thresholds = roc_curve(y_test, _prob)
        return {
            "fpr": _fpr,
            "tpr": _tpr,
            "auc": float(auc(_fpr, _tpr)),
            "acc": float(accuracy_score(y_test, _pred)),
            "f1": float(f1_score(y_test, _pred)),
        }

    return (evaluate,)


@app.cell
def _(plt, sns):
    def numeric_panel(frame):
        _figure, _axes = plt.subplots(2, 2, figsize=(10, 7))
        sns.histplot(data=frame, kde=True, ax=_axes[0, 0])
        _axes[0, 0].set_title("Histograms")
        sns.violinplot(data=frame, ax=_axes[0, 1])
        _axes[0, 1].set_title("Violin plots")
        sns.boxplot(data=frame, ax=_axes[1, 0])
        _axes[1, 0].set_title("Box plots")
        sns.boxenplot(data=frame, ax=_axes[1, 1])
        _axes[1, 1].set_title("Boxen plots")
        _figure.tight_layout()
        return _figure

    return (numeric_panel,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Several Classifiers on Heart Disease

    ## Objectives

    - Prepare the same patient table for models that need a common scale and for models that do not.
    - Compare logistic regression, linear and quadratic discriminant analysis, k-nearest neighbors, a support vector machine, a tree, and a forest.
    - Read accuracy, F1, and the area under the ROC curve on one shared test set.

    ## Background

    The label is whether the patient has heart disease. A false negative is a missed case. F1 uses both precision and recall, so a model that finds more of the true cases can outrank a model with a similar accuracy. The ROC curve draws the true-positive rate against the false-positive rate as the cutoff moves. Its area, the AUC, is 0.5 for a coin flip and 1 for a perfect ranking of the probabilities.

    The settings below are the ones used in this comparison. They are not a full search over every option. A different depth, or a different `k`, could change the order.

    ## Datasets Used

    - **Heart Dataset.** Clinical measurements for 918 patients. This lesson keeps age, sex, chest-pain type, resting blood pressure, cholesterol, maximum heart rate, and the heart-disease label.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The patient table

    The file is read from GitHub so a browser session can load it. On this computer the same table is `heart.csv` in this folder. The browser rule that blocks some downloads is called **CORS**.

    There are no missing cells in the columns this lesson keeps. Cholesterol is still recorded as 0 for many patients. Those zeros are real rows in the file, not blank cells. The z-score step below does not automatically delete every zero.
    """)
    return


@app.cell
def _():
    heart_file = (
        "https://raw.githubusercontent.com/Goliath-Research/"
        "Machine-Learning/master/Classification/heart.csv"
    )
    return (heart_file,)


@app.cell
def _(heart_file, mo, pd, plt, sns):
    # heart_raw = pd.read_csv("heart.csv")
    heart_raw = pd.read_csv(heart_file)
    heart = heart_raw[
        [
            "Age",
            "Sex",
            "ChestPainType",
            "RestingBP",
            "Cholesterol",
            "MaxHR",
            "HeartDisease",
        ]
    ].copy()
    print(heart_raw.shape)
    print("Missing values:")
    print(heart.isnull().sum())
    _figure, _axes = plt.subplots(1, 3, figsize=(10, 3))
    sns.countplot(x=heart["HeartDisease"], ax=_axes[0])
    _axes[0].set_title("Heart disease")
    sns.countplot(x=heart["Sex"], ax=_axes[1])
    _axes[1].set_title("Sex")
    sns.countplot(x=heart["ChestPainType"], ax=_axes[2])
    _axes[2].set_title("Chest pain")
    _figure.tight_layout()
    return _figure, mo.Html(heart.head().to_html())


@app.cell
def _(heart, numeric_panel):
    raw_panel = numeric_panel(heart[["Age", "RestingBP", "Cholesterol", "MaxHR"]])
    return (raw_panel,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Scaling, then a z-score cutoff

    Logistic regression, discriminant analysis, k-nearest neighbors, and the support vector machine use the standardized numbers. The scaler is fit on the whole selected table, before the split. A later patient would be scaled with these same means and standard deviations. A stricter split would fit the scaler on the training patients only.

    After scaling, a value farther than 3 standard deviations from the column mean is marked as an outlier. The row is removed if any of the four numeric columns crosses that line. A cholesterol of 0 looks extreme next to a typical cholesterol, but it is common in this file. A large group of zeros pulls the mean down and widens the standard deviation, so most of those zeros stay inside the cutoff.
    """)
    return


@app.cell
def _(StandardScaler, heart, mo, numeric_panel, np, pd):
    _columns = ["Age", "RestingBP", "Cholesterol", "MaxHR"]
    _scaler = StandardScaler()
    scaled = pd.DataFrame(
        _scaler.fit_transform(heart[_columns]),
        columns=_columns,
        index=heart.index,
    )
    _outlier_parts = [scaled.index[np.abs(scaled[column]) > 3] for column in _columns]
    outlier_index = set().union(*_outlier_parts)
    n_zero_cholesterol = int((heart["Cholesterol"] == 0).sum())
    n_zero_removed = int(heart.loc[list(outlier_index), "Cholesterol"].eq(0).sum())
    heart_clean = heart.drop(index=list(outlier_index))
    scaled_clean = scaled.drop(index=list(outlier_index))
    print(f"Rows before the cutoff: {len(heart)}")
    print(f"Rows removed: {len(outlier_index)}")
    print(f"Rows with cholesterol 0 before the cutoff: {n_zero_cholesterol}")
    print(f"Removed rows whose cholesterol is 0: {n_zero_removed}")
    print(f"Rows after the cutoff: {len(heart_clean)}")
    scaled_panel = numeric_panel(scaled_clean)
    return mo.Html(scaled_clean.head().round(3).to_html()), scaled_panel


@app.cell(hide_code=True)
def _(mo, n_zero_cholesterol, n_zero_removed):
    mo.md(f"""
    {n_zero_cholesterol} rows have cholesterol recorded as 0. {n_zero_removed} of the removed rows are in that group. The other zeros stay in the table.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Codes for sex and chest pain

    Logistic regression and the other numeric models use `drop_first=True`. One level is left out so the columns are not an exact copy of each other. For sex, the remaining column is the level that was not dropped. A 1 means the patient is that level.

    Chest pain has four codes: ASY, ATA, NAP, and TA. Dropping the first alphabetical level, ASY, leaves three columns. A row of zeros means ASY.

    The tree and the forest get a separate column for every chest-pain type, including ASY. A tree split is a yes-or-no question, so the extra column does not play the role it would in a linear score. Those two models also keep age, blood pressure, cholesterol, and heart rate in their original units.
    """)
    return


@app.cell
def _(heart_clean, pd, scaled_clean, train_test_split):
    sex_code = pd.get_dummies(heart_clean["Sex"], drop_first=True).astype(int)
    pain_dropped = pd.get_dummies(heart_clean["ChestPainType"], drop_first=True).astype(int)
    pain_full = pd.get_dummies(heart_clean["ChestPainType"], prefix="has").astype(int)
    features_scaled = pd.concat([scaled_clean, sex_code, pain_dropped], axis=1)
    features_tree = pd.concat(
        [
            heart_clean[["Age", "RestingBP", "Cholesterol", "MaxHR"]],
            sex_code,
            pain_full,
        ],
        axis=1,
    )
    y_heart = heart_clean["HeartDisease"]
    index_train, index_test = train_test_split(
        heart_clean.index, test_size=0.3, random_state=20
    )
    print(f"Sex column kept: {list(sex_code.columns)}")
    print(f"Chest-pain columns for the scaled models: {list(pain_dropped.columns)}")
    print(f"Chest-pain columns for the trees: {list(pain_full.columns)}")
    print(f"Train = {len(index_train)} cases \t Test = {len(index_test)} cases")
    return features_scaled, features_tree, index_test, index_train, y_heart


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Which chest-pain level is represented by a row of zeros in the scaled models? Replace `None` with that code.
    """)
    return


@app.cell
def _():
    student_dropped_pain = None
    print(student_dropped_pain)
    return (student_dropped_pain,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r'''
```python
student_dropped_pain = "ASY"
```
**ASY.** `drop_first=True` drops the first alphabetical level. The remaining columns are ATA, NAP, and TA. All three equal to 0 means the dropped level, ASY.
''')
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The seven fits

    Every model is scored on the same test patients. The positive class is heart disease, coded as 1.

    The first five models use the standardized columns. The tree and the forest use the original numeric columns and one column per chest-pain type. The tree and the forest are limited to depth 5. The neighbor model uses the default of 5 neighbors. The support vector machine uses the default radial-basis kernel. `CalibratedClassifierCV` fits that SVM and a probability scale, which the ROC curve needs.
    """)
    return


@app.cell
def _(
    LogisticRegression,
    evaluate,
    features_scaled,
    index_test,
    index_train,
    y_heart,
):
    lrm = LogisticRegression(C=1e9, max_iter=500, random_state=0)
    lrm.fit(features_scaled.loc[index_train], y_heart.loc[index_train])
    lrm_result = evaluate(lrm, features_scaled.loc[index_test], y_heart.loc[index_test])
    print(f"Logistic regression training accuracy = {lrm.score(features_scaled.loc[index_train], y_heart.loc[index_train]):.3f}")
    return (lrm_result,)


@app.cell
def _(
    LinearDiscriminantAnalysis,
    evaluate,
    features_scaled,
    index_test,
    index_train,
    y_heart,
):
    lda_heart = LinearDiscriminantAnalysis()
    lda_heart.fit(features_scaled.loc[index_train], y_heart.loc[index_train])
    lda_result = evaluate(
        lda_heart, features_scaled.loc[index_test], y_heart.loc[index_test]
    )
    print(f"LDA training accuracy = {lda_heart.score(features_scaled.loc[index_train], y_heart.loc[index_train]):.3f}")
    return (lda_result,)


@app.cell
def _(
    QuadraticDiscriminantAnalysis,
    evaluate,
    features_scaled,
    index_test,
    index_train,
    y_heart,
):
    qda_heart = QuadraticDiscriminantAnalysis()
    qda_heart.fit(features_scaled.loc[index_train], y_heart.loc[index_train])
    qda_result = evaluate(
        qda_heart, features_scaled.loc[index_test], y_heart.loc[index_test]
    )
    print(f"QDA training accuracy = {qda_heart.score(features_scaled.loc[index_train], y_heart.loc[index_train]):.3f}")
    return (qda_result,)


@app.cell
def _(
    KNeighborsClassifier,
    evaluate,
    features_scaled,
    index_test,
    index_train,
    y_heart,
):
    knn_heart = KNeighborsClassifier()
    knn_heart.fit(features_scaled.loc[index_train], y_heart.loc[index_train])
    knn_result = evaluate(
        knn_heart, features_scaled.loc[index_test], y_heart.loc[index_test]
    )
    print(f"kNN training accuracy = {knn_heart.score(features_scaled.loc[index_train], y_heart.loc[index_train]):.3f}")
    return (knn_result,)


@app.cell
def _(
    CalibratedClassifierCV,
    SVC,
    evaluate,
    features_scaled,
    index_test,
    index_train,
    y_heart,
):
    svm_heart = CalibratedClassifierCV(SVC(random_state=0), ensemble=False)
    svm_heart.fit(features_scaled.loc[index_train], y_heart.loc[index_train])
    svm_result = evaluate(
        svm_heart, features_scaled.loc[index_test], y_heart.loc[index_test]
    )
    print(f"SVM training accuracy = {svm_heart.score(features_scaled.loc[index_train], y_heart.loc[index_train]):.3f}")
    return (svm_result,)


@app.cell
def _(
    DecisionTreeClassifier,
    evaluate,
    features_tree,
    index_test,
    index_train,
    y_heart,
):
    tree_heart = DecisionTreeClassifier(max_depth=5, random_state=0)
    tree_heart.fit(features_tree.loc[index_train], y_heart.loc[index_train])
    tree_result = evaluate(
        tree_heart, features_tree.loc[index_test], y_heart.loc[index_test]
    )
    print(f"Tree training accuracy = {tree_heart.score(features_tree.loc[index_train], y_heart.loc[index_train]):.3f}")
    return (tree_result,)


@app.cell
def _(
    RandomForestClassifier,
    evaluate,
    features_tree,
    index_test,
    index_train,
    mo,
    pd,
    plt,
    sns,
    y_heart,
):
    forest_heart = RandomForestClassifier(max_depth=5, random_state=0)
    forest_heart.fit(features_tree.loc[index_train], y_heart.loc[index_train])
    forest_result = evaluate(
        forest_heart, features_tree.loc[index_test], y_heart.loc[index_test]
    )
    importance = pd.DataFrame(
        {
            "feature": features_tree.columns,
            "importance": forest_heart.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    _figure, _axes = plt.subplots(figsize=(7, 4))
    sns.barplot(data=importance, y="feature", x="importance", ax=_axes, color="steelblue")
    _axes.set_title("Random forest feature importance")
    print(f"Forest training accuracy = {forest_heart.score(features_tree.loc[index_train], y_heart.loc[index_train]):.3f}")
    return _figure, mo.Html(importance.round(3).to_html())


@app.cell
def _(
    forest_result,
    knn_result,
    lda_result,
    lrm_result,
    mo,
    pd,
    plt,
    qda_result,
    svm_result,
    tree_result,
):
    results = pd.DataFrame(
        {
            "LRM": lrm_result,
            "LDA": lda_result,
            "QDA": qda_result,
            "KNN": knn_result,
            "SVM": svm_result,
            "DTM": tree_result,
            "RFM": forest_result,
        }
    ).T
    _score_table = results[["acc", "f1", "auc"]].rename(
        columns={"acc": "accuracy", "f1": "F1", "auc": "AUC"}
    )
    _figure, _axes = plt.subplots(1, 2, figsize=(10, 4))
    _score_table["accuracy"].plot(kind="bar", rot=0, ax=_axes[0], title="Test accuracy")
    _score_table["F1"].plot(kind="bar", rot=0, ax=_axes[1], title="Test F1", color="darkred")
    _figure.tight_layout()
    return _figure, mo.Html(_score_table.round(3).to_html())


@app.cell
def _(plt, results):
    _figure, _axes = plt.subplots(figsize=(6, 5))
    _axes.plot([0, 1], [0, 1], linestyle="--", color="gray")
    for _name, _row in results.iterrows():
        _axes.plot(_row["fpr"], _row["tpr"], label=f"{_name} (AUC = {_row['auc']:.3f})")
    _axes.set_xlabel("False positive rate")
    _axes.set_ylabel("True positive rate")
    _axes.set_title("ROC curves")
    _axes.legend()
    return _figure


@app.cell(hide_code=True)
def _(mo, results):
    _best_auc = results["auc"].astype(float).idxmax()
    _best_f1 = results["f1"].astype(float).idxmax()
    mo.md(f"""
    ## Conclusions

    **Key takeaways:**

    - The comparison uses one test set of patients, after rows more than 3 standard deviations from a numeric mean were removed.
    - Logistic regression, LDA, QDA, k-nearest neighbors, and the support vector machine see standardized numbers and drop-first codes.
    - The tree and the forest see the original units and one column per chest-pain type. Both are limited to depth 5.
    - On this split, the highest test AUC is {_best_auc} ({results.loc[_best_auc, "auc"]:.3f}) and the highest test F1 is {_best_f1} ({results.loc[_best_f1, "f1"]:.3f}).
    - F1 is in the table because a missed heart-disease case and a false alarm do not have the same cost. Accuracy alone does not show that split.
    - The importance chart is the forest's ranking of the columns it was given. It is not a ranking shared by the logistic model.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check your understanding

    1. Which five models use the standardized columns?
    2. What does a row of zeros mean for chest pain in those models?
    3. What does the z-score cutoff remove?
    4. Why is F1 reported along with accuracy?
    5. What does an AUC of 0.5 mean?
    6. Which models in this lesson are limited to depth 5?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
1. **LRM, LDA, QDA, KNN, and SVM:** The tree and the forest use the original age, blood pressure, cholesterol, and heart rate.
2. **ASY:** That level was dropped. ATA, NAP, and TA are the columns that remain.
3. **Rows more than 3 standard deviations from a numeric mean:** The rule looks at age, resting blood pressure, cholesterol, and maximum heart rate. Most cholesterol zeros stay, because that value is common enough to fall inside the cutoff.
4. **Missed cases and false alarms are different:** F1 combines precision and recall. Accuracy only counts how often the label matched.
5. **No better than a coin flip:** The ROC area is the probability that a random patient with heart disease is scored above a random patient without it. 0.5 means those rankings are tied with chance.
6. **The decision tree and the random forest:** Both use `max_depth=5`.
""")
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - VanderPlas, J. (2017). *Python Data Science Handbook*. O'Reilly Media. Chapter 5.
    - [ROC and AUC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
    - [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction), the source of `heart.csv`.
    """)
    return


if __name__ == "__main__":
    app.run()
