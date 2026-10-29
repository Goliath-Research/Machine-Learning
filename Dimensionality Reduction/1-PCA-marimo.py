# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
#     "pandas",
#     "plotly",
#     "scikit-learn>=1.2",
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
        "plotly": "plotly",
        "sklearn": "scikit-learn>=1.2",
    }
    _missing = [
        _package for _module, _package in _packages.items()
        if _import_util.find_spec(_module) is None
    ]
    if _missing:
        if _sys.platform == "emscripten":
            # Browser Python uses micropip; it cannot launch subprocesses.
            import micropip as _micropip
            await _micropip.install(_missing)
        else:
            import subprocess as _subprocess
            _subprocess.check_call([_sys.executable, "-m", "pip", "install", *_missing])

    # The imports below wait until package installation has completed.
    packages_ready = True
    return (packages_ready,)


@app.cell
def _(packages_ready):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))

    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.io as pio
    import PlotBuilder as pb
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    pio.templates.default = "plotly_white"
    px.defaults.color_discrete_sequence = px.colors.qualitative.G10
    builder = pb.PlotBuilder(width=620, height=470)

    return PCA, Path, StandardScaler, builder, mo, np, pd, px


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Principal Component Analysis

    ## Objectives

    By the end of this lesson, you should be able to:

    1. Explain what a principal component is: a direction, not a variable.
    2. Read `explained_variance_` and `explained_variance_ratio_`, and say what each one is measured in.
    3. Explain why two mileage columns can skip standardization while eight mixed measurements cannot.
    4. Explain what a component's *sign* does and does not mean.
    5. Describe what is lost when a two-dimensional cloud is rebuilt from one component.

    ## Background

    **Principal component analysis** is a way of describing a cloud of points using new axes. It is the foundation for everything else in this folder: kernel PCA, multidimensional scaling, and the manifold methods all build on the same idea.

    ### Finding the directions

    Given a cloud of points, PCA looks for **directions** in order of how much the data spreads along them.

    1. The **first component** is the direction along which the points vary most.
    2. The **second component** is the direction of greatest remaining variation, and it must be at a right angle to the first.
    3. Each further component continues that pattern.

    Two consequences follow immediately.

    - **Components are directions, not columns.** A component is a mixture of the original variables. It is a line through the cloud, described by a vector of weights.
    - **Components are uncorrelated by construction.** Because each new one must be perpendicular to all the earlier ones, they share no linear trend.

    ### The two variance attributes

    Fit a PCA model and it reports three things that are easy to confuse:

    | Attribute | What it holds | Units |
    | :--- | :--- | :--- |
    | `components_` | the direction vectors, one row per component | weights, no units |
    | `explained_variance_` | the variance along each component | **squared units of your data** |
    | `explained_variance_ratio_` | each variance divided by the total | a fraction, 0 to 1 |

    The distinction between the last two matters. If you fit on miles per gallon, `explained_variance_` is measured in miles-per-gallon squared, which is hard to interpret. `explained_variance_ratio_` is a share, so it is comparable across datasets. When you report "this component keeps 98.6% of the variance", you are quoting the ratio.

    The ratios always sum to 1 across **all** components, including ones you did not keep.

    ### Standardization, and when to skip it

    PCA finds directions of maximum variance, so **a column with a big numerical range pulls the first component toward itself**. If you mix wheel base (around 100 inches) with horsepower (around 100), the two are comparable; if you mix curb weight (thousands of pounds) with fuel economy (tens of miles per gallon), the weight column will dominate.

    That gives the rule:

    - **Same units, similar spread → standardizing is unnecessary and arguably harmful.** Two mileage columns are both miles per gallon; converting them to z-scores would not change the geometry of the cloud, only its scale.
    - **Different units → standardize.** Otherwise the components describe your units, not your data.

    This lesson shows both cases, in that order, so the contrast is visible.

    ### Dimensionality reduction

    Keeping all the components is a rotation, not a reduction: the cloud has the same shape, viewed from a different angle. To reduce, you keep the first few and discard the rest.

    The discarded directions are the ones with least variance. The `inverse_transform` method puts the shortened description back into the original columns. What was dropped does not come back: the rebuilt points all land **on a line** (or a plane), because the missing direction no longer exists.

    That is the trade in one sentence: you keep the largest patterns and lose the small ones, and the size of what you lost is roughly the variance you threw away.

    ### One caution about signs

    A component's direction is only defined up to its sign: flipping every weight in a vector gives the same line. So `components_[0]` might read `[0.688, 0.726]` or `[-0.688, -0.726]` depending on the implementation. **Never interpret the sign of a component** unless you fix it yourself. The magnitudes, and the variance each component explains, are unaffected.

    ## Datasets used

    ### Automobile data

    **Source.** The Automobile dataset from the **UCI Machine Learning Repository**, donated in 1985 and widely used as a teaching set for mixed-type data. 205 cars, 26 columns: price, insurance risk rating, make, and technical specifications such as wheel base, curb weight, horsepower, and fuel economy. Available at the [UCI repository](https://archive.ics.uci.edu/ml/datasets/automobile).

    **What this lesson uses.** Two fits:

    - **Two mileage columns**, `city_mpg` and `highway_mpg`, both in miles per gallon. No missing values.
    - **Eight measurements**: wheel base, curb weight, horsepower, length, width, height, city mpg, and highway mpg.

    **A caution.** The file stores missing values as the string `?`, which must be converted to NaN or the affected statistics are silently wrong. It also encodes categorical columns as text (`alfa-romero`, `two`, `rwd`), so those columns cannot enter a distance computation without encoding. This lesson uses only the numeric columns.

    A copy, `auto_imports85.csv`, sits beside this notebook, with the column names attached and `?` already converted.
    """)
    return


@app.cell
def _(Path, pd):
    # Automobile data, UCI Machine Learning Repository (1985), 205 cars, 26 columns.
    # https://archive.ics.uci.edu/ml/datasets/automobile
    # The original file has no header row and stores missing values as "?".
    # auto_imports85.csv here already has the column names and NaN for missing values.
    cars = pd.read_csv(Path(__file__).resolve().parent / "auto_imports85.csv")
    mileage = cars[["city_mpg", "highway_mpg"]]
    car_counts = f"{len(cars)} cars, {len(mileage)} with both mileage columns"
    return car_counts, cars, mileage


@app.cell(hide_code=True)
def _(car_counts, mileage, mo):
    mo.md(f"""
    **{car_counts}.** Neither mileage column has a missing value, so all 205 cars can be used for the first fit.

    | | missing values |
    | :--- | ---: |
    | city_mpg | {mileage.city_mpg.isnull().sum()} |
    | highway_mpg | {mileage.highway_mpg.isnull().sum()} |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Two columns that share a unit

    City and highway fuel economy, both in miles per gallon. Before any components, look at the raw cloud.
    """)
    return


@app.cell
def _(builder, mileage, px):
    _figure = px.scatter(
        mileage,
        x="city_mpg",
        y="highway_mpg",
        title="City and highway miles per gallon",
        labels={
            "city_mpg": "City (miles per gallon)",
            "highway_mpg": "Highway (miles per gallon)",
        },
    )
    _figure.update_traces(marker=dict(size=7, opacity=0.75))
    _figure.update_yaxes(scaleanchor="x", scaleratio=1)
    _figure.update_layout(
        width=builder.width, height=builder.height, showlegend=False,
        margin=dict(t=55, b=50, l=60, r=20),
    )
    _figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The cloud is a narrow diagonal band. Highway mileage runs higher than city mileage for every car, and the two move together tightly.

    **This is not a regression.** A regression would ask "given city mileage, what is the expected highway mileage?". PCA asks a different, unsupervised question: *in which direction does this cloud spread most?* No variable is treated as the target.

    Notice also that both axes have similar ranges, roughly 10 to 50. Standardizing would rescale both by nearly the same factor, so it would not change the **shape** of this cloud. That is why this fit skips it, and the eight-measurement fit later does not.

    ### The fit, without standardizing
    """)
    return


@app.cell
def _(PCA, mileage):
    mileage_pca = PCA(n_components=2).fit(mileage)
    mileage_variance = mileage_pca.explained_variance_
    mileage_share = mileage_pca.explained_variance_ratio_
    mileage_components = mileage_pca.components_
    return mileage_pca, mileage_share, mileage_variance


@app.cell(hide_code=True)
def _(mo, mileage_components):
    mo.md(f"""
    The fit reports two components, each a pair of weights over the two original columns:

    | Component | city_mpg weight | highway_mpg weight |
    | :--- | ---: | ---: |
    | first | {mileage_components[0, 0]:+.4f} | {mileage_components[0, 1]:+.4f} |
    | second | {mileage_components[1, 0]:+.4f} | {mileage_components[1, 1]:+.4f} |

    Read these as directions, not as variables.

    - The **first** component weights both columns positively and almost equally, so it points up the diagonal. That is where this cloud spreads most: cars differ mainly in being economical or thirsty.
    - The **second** weights them with **opposite signs**. It points across the band, along the thin direction where almost nothing varies.

    The two vectors are perpendicular, which PCA guarantees.

    **About the signs.** Your machine may show these with every sign flipped. That would describe the same two lines, and it changes nothing else in this lesson. The weights are only meaningful relative to each other, and only their magnitudes matter for the variance.
    """)
    return


@app.cell(hide_code=True)
def _(mileage_variance, mileage_share, mo):
    mo.md(f"""
    Now the two variance attributes, side by side, because this is where students most often look at the wrong number:

    | Component | `explained_variance_` | `explained_variance_ratio_` |
    | :--- | ---: | ---: |
    | first | {mileage_variance[0]:.2f} | {mileage_share[0]:.1%} |
    | second | {mileage_variance[1]:.2f} | {mileage_share[1]:.1%} |

    The first number is in **miles-per-gallon squared**, which is why 88.9 looks large and meaningless on its own. The second is a share of the total, and it says something interpretable: **{mileage_share[0]:.1%} of everything this cloud varies in lies along one direction.**

    Almost the entire dataset is one-dimensional. The cars differ in how economical they are, and hardly at all in the way city and highway mileage disagree.
    """)
    return


@app.cell
def _(builder, mileage, mileage_pca, px):
    _figure = px.scatter(
        mileage,
        x="city_mpg",
        y="highway_mpg",
        title="The two component directions",
        labels={
            "city_mpg": "City (miles per gallon)",
            "highway_mpg": "Highway (miles per gallon)",
        },
    )
    _figure.update_traces(marker=dict(size=7, opacity=0.6))
    _figure.update_yaxes(scaleanchor="x", scaleratio=1)
    _figure.update_layout(
        width=builder.width, height=builder.height, showlegend=False,
        margin=dict(t=55, b=50, l=60, r=20),
    )
    builder.component_arrows(
        _figure,
        mileage_pca.mean_,
        mileage_pca.explained_variance_,
        mileage_pca.components_,
        scale=3,
    )
    _figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The arrows make the geometry concrete.

    - Their **direction** is `components_`: the long arrow runs up the band, the short one crosses it.
    - Their **length** is set by `explained_variance_`: the long arrow is drawn longer because that direction holds far more variance.
    - They meet at the **mean** of the data, marked by their shared start point, and they are at right angles.

    A component direction alone is not enough to describe the data; you need the direction *and* how much the data spreads along it. That is exactly the pair of attributes above.
    """)
    return


@app.cell
def _(builder, mileage, mileage_pca, pd, px):
    mileage_plane = pd.DataFrame(
        mileage_pca.transform(mileage), columns=["Component 1", "Component 2"]
    )
    _figure = px.scatter(
        mileage_plane,
        x="Component 1",
        y="Component 2",
        title="The same cars, viewed from the component axes",
    )
    _figure.update_traces(marker=dict(size=7, opacity=0.6))
    _figure.update_yaxes(scaleanchor="x", scaleratio=1)
    _figure.add_hline(y=0, line_color="lightgrey", line_width=1)
    _figure.add_vline(x=0, line_color="lightgrey", line_width=1)
    _figure.update_layout(
        width=builder.width, height=builder.height, showlegend=False,
        margin=dict(t=55, b=50, l=60, r=20),
    )
    _figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This is the same cloud, rotated to sit on the new axes. Two things to check:

    - The **shape is identical**. Nothing was reduced; with two components kept out of two, `transform` is a pure rotation.
    - The **spread is now aligned with the axes**. Along Component 1 the points run far; along Component 2 they barely leave zero.

    That second observation is the whole case for dimensionality reduction. If almost all the variation is along one axis, the other axis is nearly constant — so why keep it?

    ### Try it yourself

    Replace `None` so that `student_component_1` holds the **weights** of the first component, and `student_first_share` holds the fraction of total variance it keeps.
    """)
    return


@app.cell
def _(mileage_pca):
    student_component_1 = None
    student_first_share = None
    print("First component weights:   ", student_component_1)
    print("Its share of the variance: ", student_first_share)
    return student_component_1, student_first_share


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
```python
student_component_1 = mileage_pca.components_[0]
student_first_share = mileage_pca.explained_variance_ratio_[0]
```
The weights are approximately **(0.688, 0.726)**, and the share is about **0.986**, so 98.6% of the variance.

`components_` is a 2×2 array: one row per component, one column per original variable. Row 0 is the first component, so `components_[0]` gives its two weights. `explained_variance_ratio_` is a one-dimensional array indexed the same way.

Notice the weights are almost equal and both positive, which is why the long arrow runs up the diagonal at roughly 45°.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Why can this two-column fit skip standardization, when the eight-measurement fit later cannot? Work out your answer, then reveal it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
Because **both columns are already in the same unit and have similar spread.**

City mileage and highway mileage are both miles per gallon, and both range over roughly 10 to 50. Neither can dominate the other, so the components found on the raw data describe the actual relationship between the two variables.

Standardizing would divide each column by nearly the same standard deviation. That would rescale the cloud slightly and leave its **shape** essentially unchanged, so the components would point in the same directions.

With eight measurements the situation is different: curb weight runs into thousands of pounds while mileage is in tens. On the raw scale the weight column's spread is enormous, so the first component would point almost exactly at curb weight and the other seven columns would contribute almost nothing. Standardizing is what stops your units from deciding the answer.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Reducing to one component

    Keep only the first component and see what happens to the cloud. This is where reduction actually occurs: two columns become one number per car.
    """)
    return


@app.cell
def _(PCA, mileage):
    mileage_one = PCA(n_components=1).fit(mileage)
    mileage_one_share = float(mileage_one.explained_variance_ratio_[0])
    mileage_reduced = mileage_one.transform(mileage)
    mileage_restored = mileage_one.inverse_transform(mileage_reduced)
    mileage_shapes = (
        f"original {mileage.shape[0]} rows x {mileage.shape[1]} columns, "
        f"reduced {mileage_reduced.shape[0]} rows x {mileage_reduced.shape[1]} column, "
        f"restored {mileage_restored.shape[0]} rows x {mileage_restored.shape[1]} columns"
    )
    return (
        mileage_one_share,
        mileage_reduced,
        mileage_restored,
        mileage_shapes,
    )


@app.cell(hide_code=True)
def _(mileage_one_share, mileage_shapes, mo):
    mo.md(f"""
    **{mileage_shapes}.** Two numbers per car became one, and the inverse transform put it back to two.

    The kept share is **{mileage_one_share:.1%}**, identical to the first component's ratio in the two-component fit. That is expected: the first component is the same direction either way, and asking for one component does not change it.

    So roughly {1 - mileage_one_share:.1%} of the variance was discarded. The next plot shows what that looks like.
    """)
    return


@app.cell
def _(builder, mileage, mileage_restored, pd, px):
    _combined = pd.DataFrame(
        {
            "city_mpg": mileage["city_mpg"].to_numpy(),
            "highway_mpg": mileage["highway_mpg"].to_numpy(),
            "x_reference": mileage_restored[:, 0],
            "y_reference": mileage_restored[:, 1],
        }
    )
    _figure = builder.scatter_with_reference(
        _combined,
        "city_mpg",
        "highway_mpg",
        title="One component drawn back in the original plane",
        label_points="Original",
        label_reference="Rebuilt from one component",
    )
    _figure.update_xaxes(title_text="City (miles per gallon)")
    _figure.update_yaxes(title_text="Highway (miles per gallon)")
    _figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    The orange points fall on a **single straight line**. That is the clearest picture of what reduction costs.

    Each car still has two coordinates, but they are no longer independent: they were generated from one number by multiplying it along one direction. The cloud has collapsed from a band into a line.

    Look at the gap between a blue point and its orange partner. That gap is the discarded component: it points across the band, along the short arrow from earlier. Cars near the middle of the band land almost on top of their originals (little was lost); cars at the edges of the band are pushed further (more was lost).

    **The honest summary:** keeping one component here loses little, because only about 1.4% of the variance was there to lose. On a cloud that is genuinely two-dimensional, the same operation would destroy most of the structure. The amount lost is exactly the variance ratio you discarded.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Why do the rebuilt points lie on a line rather than filling the band? Work out your answer, then reveal it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
Because **only one direction survived the inverse transform.**

Each car was reduced to a single number: its position along Component 1. The inverse transform then reconstructs two coordinates from that one number, using only the first component's direction. A single number multiplied by a fixed direction can only produce points along that direction, so every rebuilt car lands on the same line.

The second direction was dropped, and with it the freedom to move off that line. The original blue points have an extra offset across the band, which is precisely the part that is gone.

This is a general property: **keeping k components from a d-dimensional space puts the rebuilt data on a k-dimensional surface.** With k = 1 that surface is a line; with k = 2 in a 64-dimensional space it is a plane.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Eight measurements, which must be standardized

    Now the case where standardization is required. These eight columns do not share a unit:

    - wheel base, length, width, height in **inches**;
    - curb weight in **pounds**;
    - horsepower in **horsepower**;
    - city and highway mileage in **miles per gallon**.

    Look at the ranges before doing anything else.
    """)
    return


@app.cell
def _(cars, pd):
    measure_columns = [
        "wheel_base",
        "curb_weight",
        "horsepower",
        "length",
        "width",
        "height",
        "city_mpg",
        "highway_mpg",
    ]
    measures = cars[measure_columns]
    measure_ranges = pd.DataFrame(
        {
            "min": measures.min(),
            "max": measures.max(),
            "spread": measures.max() - measures.min(),
            "missing": measures.isnull().sum(),
        }
    )
    measures_complete = measures.dropna()
    measures_dropped = len(measures) - len(measures_complete)
    return measure_columns, measure_ranges, measures_complete, measures_dropped


@app.cell
def _(measure_ranges, measures_dropped, mo):
    mo.Html(measure_ranges.round(1).to_html())
    return


@app.cell(hide_code=True)
def _(measure_ranges, measures_complete, measures_dropped, mo):
    _widest = measure_ranges["spread"].idxmax()
    _widest_value = measure_ranges.loc[_widest, "spread"]
    _narrowest = measure_ranges["spread"].idxmin()
    _narrowest_value = measure_ranges.loc[_narrowest, "spread"]
    mo.md(f"""
    The spreads are not comparable at all. **{_widest}** ranges over **{_widest_value:,.0f}** units, while **{_narrowest}** ranges over only **{_narrowest_value:,.1f}**.

    That is a ratio of about **{_widest_value / _narrowest_value:,.0f} to 1**. Without standardizing, the first component would essentially be the curb-weight direction, and every other column would be noise in the background.

    **{measures_dropped} rows** have a missing value in one of these columns and are dropped, leaving **{len(measures_complete)}** complete cars.

    The two box plots show the difference standardization makes.
    """)
    return


@app.cell
def _(StandardScaler, measure_columns, measures_complete, pd, px):
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    measures_z = pd.DataFrame(
        StandardScaler().fit_transform(measures_complete), columns=measure_columns
    )
    _figure = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Original units", "After standardization"),
        vertical_spacing=0.18,
    )
    for _row, _frame, _colour in (
        (1, measures_complete, "#2E86C1"),
        (2, measures_z, "#D35400"),
    ):
        for _name in measure_columns:
            _figure.add_trace(
                go.Box(y=_frame[_name], name=_name, marker_color=_colour, showlegend=False),
                row=_row,
                col=1,
            )
    _figure.update_layout(
        width=820,
        height=620,
        margin=dict(t=70, b=80, l=60, r=20),
    )
    _figure.update_xaxes(tickangle=-25)
    _figure
    return (measures_z,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The top panel is the problem: `curb_weight` occupies a range from about 1,800 to 5,100, so on a shared axis the other seven columns are flattened into invisible slivers.

    The bottom panel is the fix: every column now has a mean of 0 and a standard deviation of 1, so the boxes sit at comparable heights and their **shapes** become readable. Height and horsepower vary differently, and you can see it once the units are gone.

    Note what standardization does not do. It does not make the variables Gaussian, does not remove outliers, and does not make them independent. It only removes the arbitrary choice of unit.

    ### Three components from the standardized measurements
    """)
    return


@app.cell
def _(PCA, measures_z):
    auto_pca = PCA(n_components=3).fit(measures_z)
    auto_variance = auto_pca.explained_variance_
    auto_share = auto_pca.explained_variance_ratio_
    auto_total = float(auto_share.sum())
    auto_loadings = auto_pca.components_
    return auto_loadings, auto_share, auto_total, auto_variance


@app.cell(hide_code=True)
def _(auto_share, auto_total, mo):
    mo.md(f"""
    | Component | share of variance | running total |
    | :--- | ---: | ---: |
    | first | {auto_share[0]:.1%} | {auto_share[0]:.1%} |
    | second | {auto_share[1]:.1%} | {auto_share[0] + auto_share[1]:.1%} |
    | third | {auto_share[2]:.1%} | {auto_total:.1%} |

    Three components keep **{auto_total:.1%}** of the variance in eight standardized columns. Compare that with the mileage fit, where one component kept 98.6%.

    The contrast is the point. The mileage columns were nearly one-dimensional; these eight columns are not. No single direction describes a car, because "size", "power", and "economy" are genuinely different aspects of it.

    Note also that the shares **decrease** in order: {auto_share[0]:.1%}, then {auto_share[1]:.1%}, then {auto_share[2]:.1%}. That is guaranteed by construction. If you ever see them out of order, you are not looking at a standard PCA fit.
    """)
    return


@app.cell(hide_code=True)
def _(auto_loadings, measure_columns, mo, np):
    _rows = []
    for _component in range(3):
        _ranked = np.argsort(-np.abs(auto_loadings[_component]))
        _top = ", ".join(
            f"`{measure_columns[_i]}` ({auto_loadings[_component, _i]:+.2f})"
            for _i in _ranked[:3]
        )
        _rows.append(f"    - Component {_component + 1} is dominated by {_top}")
    mo.md("\n".join([
        "What each component actually describes — the three largest weights in each:",
        "",
        *_rows,
        "",
        "Read these as descriptions, not as rankings. A component is a weighted mixture, and naming it is an interpretation, not a fact the algorithm reports.",
        "",
        "The pattern is interpretable, which is a good sign for a PCA fit:",
        "",
        "- **Component 1** puts similar positive weight on weight, length, and width. It is a **size** axis: big cars at one end, small cars at the other.",
        "- **Component 2** contrasts height against horsepower. It separates **tall and less powerful** from **low and powerful**, roughly the difference between a van and a sports car.",
        "- **Component 3** contrasts height and width against fuel economy. It is a **shape versus efficiency** axis.",
        "",
        "Those names are ours, not the algorithm's. PCA reports directions; the labelling is where domain knowledge enters, and another analyst might name them differently. What is not negotiable is the arithmetic behind the weights.",
    ]))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Replace `None` so that `student_components_for_80` holds the **number of components needed to keep at least 80%** of the variance in the eight standardized measurements. Fit a full PCA with no `n_components` argument to get every share.
    """)
    return


@app.cell
def _(PCA, measures_z, np):
    student_components_for_80 = None
    print("components needed for 80% of the variance:", student_components_for_80)
    return (student_components_for_80,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
```python
_full = PCA().fit(measures_z)
_cumulative = np.cumsum(_full.explained_variance_ratio_)
student_components_for_80 = int(np.searchsorted(_cumulative, 0.80) + 1)
```
The answer is **2** components, not three or four. The cumulative shares run 68.1%, then 87.3%, so the second component is the one that crosses 80%.

That is worth pausing on, because it cuts against the expectation set by the section. Three components were used for the plots, but **two** already carry more than 80% of the variance. The first component alone carries 68.1%, so it does most of the work and the remaining six add comparatively little.

`PCA()` with no arguments keeps every component, so `explained_variance_ratio_` has one entry per column. `np.cumsum` turns those shares into a running total, and `np.searchsorted` finds the first position where the total reaches 0.80. Adding 1 converts a zero-based index into a count.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Why does standardizing change the components, while keeping the same two mileage columns un-standardized does not? Work out your answer, then reveal it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
Because **standardizing changes the relative spreads, and PCA selects directions by spread.**

In the mileage case, both columns were already in the same unit with nearly the same standard deviation. Dividing each by a similar number rescales the cloud without reshaping it, so the directions of maximum variance stay where they were.

In the eight-measurement case, the spreads differed by a factor of hundreds. Dividing each column by its own standard deviation changes the relative influence of every column dramatically. Curb weight no longer dominates, so the first component stops pointing at curb weight and starts describing a genuine mixture.

Put more sharply: **standardizing is not a neutral housekeeping step. It changes what "maximum variance" means**, and therefore changes the answer. That is why the choice must be justified rather than applied automatically.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(
    auto_share,
    auto_total,
    measure_ranges,
    measures_complete,
    measures_dropped,
    mileage_one_share,
    mileage_share,
    mo,
):
    mo.md(f"""
    ## Conclusions

    1. **A principal component is a direction, not a variable.** It is a weighted mixture of the original columns, chosen so that the data spreads as much as possible along it. Each further component is the largest remaining spread, at a right angle to all earlier ones.
    2. **Two variance attributes, two units.** `explained_variance_` is measured in the squared units of your data, which is why {mileage_share[0]:.1%} of the total appeared as a bare 88.9 in the mileage fit. `explained_variance_ratio_` is the share, and the shares always sum to 1 across all components.
    3. **Standardize when the units differ, and not otherwise.** The two mileage columns share a unit and a similar spread, so standardizing would not change the shape of the cloud and the fit skips it. The eight measurements differ by hundreds of units, so the first component would otherwise be the curb-weight direction.
    4. **Reduction costs exactly the variance you discard.** Keeping one component left {mileage_one_share:.1%} of the variance, and the rebuilt cars landed on a single line. The offset of each rebuilt car from its original is the discarded second component.
    5. **A share tells you whether reduction is even worth doing.** On mileage, one component was enough ({mileage_share[0]:.1%}). On the eight standardized measurements, three components kept only {auto_total:.1%}, because a car has several genuinely independent aspects: size, power, and economy.
    6. **Shares decrease in order by construction.** The first component always explains at least as much as the second, and so on. That ordering is what makes "keep the first k" meaningful.
    7. **Signs are arbitrary.** A component vector and its negation describe the same line. Only the magnitudes and the variances carry information, so never interpret the sign of a loading without fixing it deliberately.
    8. **The scaler and the components were fit on the rows in the picture.** There is no holdout, so these components describe this table rather than a model validated on new cars. With {measures_dropped} rows dropped for missing values, they describe the {len(measures_complete)} complete cars.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check your understanding

    1. What is a principal component: a column of the original data, or a direction through the cloud?
    2. What is the difference between `explained_variance_` and `explained_variance_ratio_`, and which one has interpretable units?
    3. Why can two mileage columns skip standardization while eight mixed measurements cannot?
    4. Why do the rebuilt points from one component lie on a line?
    5. What does the sum of all the variance ratios equal, and why?
    6. Why should you not interpret the sign of a component's weights?
    7. The eight-measurement fit needed three components for the variance it kept. What does that tell you about cars that the mileage fit did not?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
1. **A direction through the cloud.** It is a weighted mixture of the original columns, chosen so the data spreads most along it. Each new component is perpendicular to all the earlier ones.
2. **`explained_variance_` is a variance, in squared units of the data** (miles-per-gallon squared, for the mileage fit); `explained_variance_ratio_` is that variance divided by the total, so it is a **fraction with no units**. The ratio is the interpretable one, and the ratios sum to 1 across all components.
3. **Because of relative spread.** The mileage columns share a unit and vary over similar ranges, so rescaling both changes the cloud's size but not its shape. The eight measurements differ by hundreds of units, so without standardizing the widest column would decide the first component by itself.
4. **Because only one direction survived.** Each car was reduced to one number, and the inverse transform rebuilds two coordinates from that number along a single direction. One number times one direction can only produce points on that line.
5. **1, always.** The ratios divide each component's variance by the total variance, so across every component they partition the whole. Keeping the first k gives a partial sum, which is the share retained.
6. **Because a vector and its negation describe the same line.** The direction is defined only up to sign, and different implementations or data orders can produce either. Magnitudes and variances are unaffected.
7. **That a car has several independent aspects.** Mileage was essentially one-dimensional: economical versus thirsty. The eight measurements separate into distinguishable directions such as size, power, and economy, so no single number summarises a car.
""")
        },
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
    - [PCA, scikit-learn user guide](https://scikit-learn.org/stable/modules/decomposition.html#pca)
    - [Automobile dataset, UCI](https://archive.ics.uci.edu/ml/datasets/automobile)
    - Jolliffe, I. T. (2002). *Principal Component Analysis* (2nd ed.). Springer.
    - VanderPlas, J. (2017). *Python Data Science Handbook*. O'Reilly Media. Chapter 5.
    """)
    return


if __name__ == "__main__":
    app.run()
