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
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import GaussianNB, MultinomialNB
    from sklearn.preprocessing import StandardScaler

    return (
        CountVectorizer,
        GaussianNB,
        MultinomialNB,
        confusion_matrix,
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Naive Bayes Classification

    ## Objectives

    - Classify continuous measurements with Gaussian naive Bayes.
    - Classify short messages with multinomial naive Bayes and word counts.
    - Read accuracy on rows the model trained on, and on rows it did not.

    ## Background

    Naive Bayes turns class frequencies and feature frequencies into a predicted class. The "naive" part is the assumption that the features are independent once the class is known. That assumption is often false, and the method is still a fast baseline.

    Gaussian naive Bayes is for numeric features. It treats each feature, inside each class, as a normal curve. Multinomial naive Bayes is for counts, such as how often a word appears.

    ## Datasets Used

    - **Synthetic blobs.** Two numeric features and two classes. The groups are a little tighter here (`cluster_std=2.5`) than in the logistic-regression lesson.
    - **Raisin Dataset.** Seven measurements of Kecimen and Besni raisins.
    - **Six short messages.** A toy list labeled spam or not spam, used only to show word counts.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Gaussian naive Bayes on blobs

    Each class keeps its own mean and variance for `X1` and for `X2`. A new point is assigned to the class that makes those two numbers more plausible. The split is 80/20 with `random_state=23`.
    """)
    return


@app.cell
def _(GaussianNB, make_blobs, pd, plt, sns, train_test_split):
    _X, _cl = make_blobs(
        n_samples=100, n_features=2, centers=2, random_state=2, cluster_std=2.5
    )
    dfb = pd.DataFrame({"X1": _X[:, 0], "X2": _X[:, 1], "Class": _cl})
    Xb = dfb[["X1", "X2"]]
    yb = dfb["Class"]
    Xb_train, Xb_test, yb_train, yb_test = train_test_split(
        Xb, yb, test_size=0.2, random_state=23
    )
    nb_g = GaussianNB()
    nb_g.fit(Xb_train, yb_train)
    print(f"Train = {len(Xb_train)} cases \t Test = {len(Xb_test)} cases")
    print(f"Training accuracy = {nb_g.score(Xb_train, yb_train):.3f}")
    print(f"Testing accuracy  = {nb_g.score(Xb_test, yb_test):.3f}")
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
def _(Xb_train, nb_g, np, pd, plt, sns, yb_train):
    _rng = np.random.default_rng(0)
    _grid = pd.DataFrame(
        {
            "X1": _rng.uniform(Xb_train.X1.min(), Xb_train.X1.max(), size=1000),
            "X2": _rng.uniform(Xb_train.X2.min(), Xb_train.X2.max(), size=1000),
        }
    )
    _grid["Class"] = nb_g.predict(_grid)
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
    _axes.set_title("Gaussian naive Bayes regions")
    return _figure


@app.cell
def _(Xb_test, confusion_table, mo, nb_g, yb_test):
    _table = confusion_table(yb_test, nb_g.predict(Xb_test))
    return mo.Html(_table.to_html())


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The boundary can bend because each class has its own mean and variance. The confusion matrix compares the 20 test labels with the predicted labels. The main diagonal holds the correct ones.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Raisin measurements

    Kecimen is 0 and Besni is 1. Gaussian naive Bayes uses means and variances, and the raisin columns do not share a scale. Area is tens of thousands, and extent is near 0.7. The scaler is fit on the training rows only. The test rows use that same scaler.

    The file is read from GitHub so a browser session can load it. On this computer, `Raisin.csv` is also in this folder. The browser rule that blocks some downloads is called **CORS**.
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
    GaussianNB,
    StandardScaler,
    mo,
    pd,
    plt,
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
        _scaler.fit_transform(_Xr_train),
        columns=_Xr.columns,
        index=_Xr_train.index,
    )
    Xr_test = pd.DataFrame(
        _scaler.transform(_Xr_test),
        columns=_Xr.columns,
        index=_Xr_test.index,
    )
    nb_g2 = GaussianNB()
    nb_g2.fit(Xr_train, yr_train)
    print(dfr.shape)
    print(f"Train = {len(Xr_train)} cases \t Test = {len(Xr_test)} cases")
    print(f"Training accuracy = {nb_g2.score(Xr_train, yr_train):.3f}")
    print(f"Testing accuracy  = {nb_g2.score(Xr_test, yr_test):.3f}")
    _figure, _axes = plt.subplots(figsize=(6, 4))
    dfr["Class"].value_counts().plot(kind="bar", rot=0, ax=_axes, title="Raisin varieties")
    return _figure


@app.cell
def _(Xr_test, confusion_table, mo, nb_g2, yr_test):
    _table = confusion_table(yr_test, nb_g2.predict(Xr_test))
    return mo.Html(_table.to_html())


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Word counts and multinomial naive Bayes

    The next list has six messages. Two are spam. `CountVectorizer` turns each message into a row of word counts. The columns are the words that appear anywhere in the six messages. Multinomial naive Bayes learns which words show up more often in spam.

    This is a demonstration, not a spam filter. The model is scored on the same six messages it just memorized, and then it is asked to label three new sentences.
    """)
    return


@app.cell
def _(CountVectorizer, MultinomialNB, mo, pd):
    messages = [
        ("Hi, what are you doing? Let's hang out tonight!", "not spam"),
        (
            "URGENT: You have been selected for a 1 week FREE membership. Reply YES to this message to claim.",
            "spam",
        ),
        ("Hello, did you finish the project? Let's meet up later.", "not spam"),
        (
            "Congratulations, you have won a $1000 gift card! Click here to claim your prize.",
            "spam",
        ),
        ("Hi, can you send me your Math homework by the end of the day?", "not spam"),
        ("Hi, can you help me with my Physics homework?", "not spam"),
    ]
    _texts = [text for text, _label in messages]
    y_text = [label for _text, label in messages]
    vectorizer = CountVectorizer()
    X_text = vectorizer.fit_transform(_texts)
    nb_m = MultinomialNB()
    nb_m.fit(X_text, y_text)
    _counts = pd.DataFrame(X_text.toarray(), columns=vectorizer.get_feature_names_out())
    print(f"Training accuracy on the six messages = {nb_m.score(X_text, y_text):.3f}")
    print(f"Words in the vocabulary: {len(vectorizer.get_feature_names_out())}")
    return mo.Html(_counts.iloc[:, :8].to_html()), X_text, nb_m, vectorizer, y_text


@app.cell
def _(mo, nb_m, pd, vectorizer):
    new_messages = [
        "Hey, want to grab dinner tonight?",
        "You have been selected for a free cruise! Call now to claim your prize.",
        "Did you see the game last night?",
    ]
    _X_new = vectorizer.transform(new_messages)
    y_new = nb_m.predict(_X_new)
    _intended = ["not spam", "spam", "not spam"]
    _report = pd.DataFrame(
        {
            "message": new_messages,
            "intended": _intended,
            "predicted": y_new,
        }
    )
    return mo.Html(_report.to_html())


@app.cell(hide_code=True)
def _(mo, nb_m, new_messages, vectorizer):
    _predicted = list(nb_m.predict(vectorizer.transform(new_messages)))
    _intended = ["not spam", "spam", "not spam"]
    _wrong = [
        message
        for message, got, want in zip(new_messages, _predicted, _intended)
        if got != want
    ]
    _wrong_text = ", ".join(f'"{message}"' for message in _wrong) or "none"
    mo.md(f"""
    The intended labels are not spam, spam, and not spam. The predictions are {", ".join(_predicted)}.

    Messages that disagree with that intention: {_wrong_text}.

    A miss is easy to get with six training messages. The model only knows the words it has already counted. "Dinner" and "game" are not in that list, so an ordinary sentence can still be pushed toward spam by a word that happened to appear in a spam example, or by the prior count of the two classes.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Which naive Bayes model uses word counts in this lesson, Gaussian or multinomial? Replace `None` with the name.
    """)
    return


@app.cell
def _():
    student_count_model = None
    print(student_count_model)
    return (student_count_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r'''
```python
student_count_model = "multinomial"
```
**Multinomial.** `MultinomialNB` is fit on the word-count matrix from `CountVectorizer`. `GaussianNB` is the model used for the numeric blob and raisin columns.
''')
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(Xr_test, mo, nb_g, nb_g2, Xb_test, yb_test, yr_test):
    mo.md(f"""
    ## Conclusions

    **Key takeaways:**

    - Gaussian naive Bayes models each numeric feature, inside each class, as a normal curve. On the blobs, the test accuracy is {nb_g.score(Xb_test, yb_test):.3f}.
    - The raisin columns are standardized from the training rows before that Gaussian fit. The raisin test accuracy is {nb_g2.score(Xr_test, yr_test):.3f}.
    - Multinomial naive Bayes uses counts. Here the counts are words from six messages.
    - Accuracy on those six messages only says the model fit its own training list. The three new messages are the check that matters, and a short list can mislabel an ordinary sentence.
    - The independence assumption is what makes the arithmetic simple. It is also why overlapping words can produce a wrong label.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check your understanding

    1. What does the "naive" assumption say about the features?
    2. Which model is used for the raisin measurements?
    3. Why are the raisin columns standardized from the training rows?
    4. What does `CountVectorizer` put in each column?
    5. Why is a perfect score on the six messages not evidence about new mail?
    6. Which class code is Besni?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
1. **They are independent given the class:** The model multiplies per-feature probabilities. Real features are often related anyway.
2. **Gaussian naive Bayes:** The measurements are numeric, so each class keeps a mean and a variance per column.
3. **The columns do not share a scale:** The scaler is not fit on the test rows, so those rows do not change the means.
4. **A word:** Each cell is how many times that word appears in the message.
5. **Those messages were the training data:** The score does not use a held-out message. The three new sentences are the check.
6. **1:** Kecimen is 0 and Besni is 1.
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
    - [Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html)
    """)
    return


if __name__ == "__main__":
    app.run()
