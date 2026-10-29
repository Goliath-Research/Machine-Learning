#
"""Plotly plotting helpers for the Dimensionality Reduction notebooks.

These wrappers exist so every notebook in the folder uses one plotting library
and one visual style. Each method returns a Plotly figure, which a notebook
displays simply by ending a cell with it.

Only Plotly is used. The helpers cover the plot kinds that would otherwise need
matplotlib: grids of small images, multi-panel layouts, and shaded images.
"""
import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class PlotBuilder:
    """Build Plotly figures for the dimensionality-reduction lessons.

    Parameters
    ----------
    width : int, default 650
        Width in pixels applied to figures that set an explicit size.
    height : int, default 400
        Height in pixels applied to figures that set an explicit size.
    """

    def __init__(self, width=650, height=400):
        self.set_width_height(width, height)

    def set_width_height(self, width=650, height=400):
        """Set the width and height used by the plotting methods.

        Parameters
        ----------
        width : int, default 650
            Width in pixels.
        height : int, default 400
            Height in pixels.
        """
        self.width = width
        self.height = height

    def scatter_by_group(
        self, frame, x, y, hue, title="", hue_order=None, size=7, equal=True
    ):
        """Scatter two columns coloured by a grouping column.

        Replaces a seaborn scatter with `hue`. The legend is what lets a reader
        check whether the groups separate, so it is always shown.

        Parameters
        ----------
        frame : pandas.DataFrame
            Table holding the columns to plot.
        x, y : str
            Column names for the horizontal and vertical axes.
        hue : str
            Column whose values become the point colours.
        title : str, default ""
            Figure title.
        hue_order : list or None, default None
            Order of the legend entries. Pass it when the values are letters or
            digits, so the legend does not sort them as numbers.
        size : int, default 7
            Marker size in pixels.
        equal : bool, default True
            Force a 1:1 aspect ratio, so a unit on x equals a unit on y. Useful
            for coordinate data, misleading for embeddings with arbitrary axes.

        Returns
        -------
        plotly.graph_objects.Figure
            The scatter, with the legend shown.
        """
        figure = px.scatter(
            frame,
            x=x,
            y=y,
            color=hue,
            title=title,
            category_orders={hue: list(hue_order)} if hue_order is not None else None,
        )
        figure.update_traces(marker=dict(size=size, opacity=0.85))
        figure.update_layout(
            width=self.width,
            height=self.height,
            legend_title_text=hue,
            margin=dict(t=55, b=45, l=55, r=20),
        )
        if equal:
            figure.update_yaxes(scaleanchor="x", scaleratio=1)
        return figure

    def scatter_with_reference(
        self, frame, x, y, title="", label_points="Original", label_reference="Rebuilt"
    ):
        """Plot one set of points on top of another.

        Used where a reconstruction is compared with the original: two traces in
        one figure, so the reader can see the offset between them.

        Parameters
        ----------
        frame : pandas.DataFrame
            Must hold columns named `x` and `y` for the first set, and
            `x_reference` and `y_reference` for the second.
        x, y : str
            Column names for the first set of points.
        title : str, default ""
            Figure title.
        label_points, label_reference : str
            Legend labels for the two traces.

        Returns
        -------
        plotly.graph_objects.Figure
            Both sets in one figure, with a 1:1 aspect ratio.
        """
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=frame[x],
                y=frame[y],
                mode="markers",
                name=label_points,
                marker=dict(size=6, opacity=0.6),
            )
        )
        figure.add_trace(
            go.Scatter(
                x=frame["x_reference"],
                y=frame["y_reference"],
                mode="markers",
                name=label_reference,
                marker=dict(size=6, opacity=0.85, color="#D55E00"),
            )
        )
        figure.update_layout(
            width=self.width,
            height=self.height,
            title=title,
            xaxis_title=x,
            yaxis_title=y,
            margin=dict(t=55, b=45, l=55, r=20),
        )
        figure.update_yaxes(scaleanchor="x", scaleratio=1)
        return figure

    def box_by_feature(self, frame, title="", y_label="Value", color_by="feature"):
        """Draw one box per feature, side by side.

        Replaces a seaborn boxplot on a melted frame. Features are compared on
        one shared axis, which is the point: it shows immediately whether their
        spreads are comparable.

        Parameters
        ----------
        frame : pandas.DataFrame
            Columns to compare. Each becomes one box; column names become the
            axis tick labels.
        title : str, default ""
            Figure title.
        y_label : str, default "Value"
            Vertical axis title, so the reader knows the unit.
        color_by : str, default "feature"
            Legend title text. Boxes are coloured by column name.

        Returns
        -------
        plotly.graph_objects.Figure
            The box plot, without a legend, since the axis already names each
            feature.
        """
        long = frame.melt(var_name="feature", value_name="value")
        figure = px.box(long, x="feature", y="value", color="feature", title=title)
        figure.update_layout(
            width=self.width,
            height=self.height,
            showlegend=False,
            xaxis_title="",
            yaxis_title=y_label,
            margin=dict(t=55, b=70, l=55, r=20),
        )
        return figure

    def count_by_category(self, frame, column, title=""):
        """Bar chart of how many rows fall in each category.

        Replaces a seaborn countplot. Bars are counts, so the vertical axis is a
        number of rows rather than a measurement.

        Parameters
        ----------
        frame : pandas.DataFrame
            Table holding the column.
        column : str
            Categorical column to count.
        title : str, default ""
            Figure title.

        Returns
        -------
        plotly.graph_objects.Figure
            The bar chart.
        """
        figure = px.histogram(frame, x=column, title=title)
        figure.update_layout(
            width=self.width,
            height=self.height,
            showlegend=False,
            yaxis_title="Count",
            xaxis_title=column,
            margin=dict(t=55, b=55, l=55, r=20),
        )
        return figure

    def image(self, array, title="", width=None, height=None):
        """Show an image array, such as a PNG read from disk.

        Replaces `plt.imshow`. Pixel coordinates are meaningless to a reader, so
        the axis ticks are hidden.

        Parameters
        ----------
        array : numpy.ndarray
            Image data. Two dimensions are drawn as a heatmap; three are drawn
            as RGB or RGBA.
        title : str, default ""
            Figure title.
        width, height : int or None
            Optional size overrides.

        Returns
        -------
        plotly.graph_objects.Figure
            The image with axes hidden.
        """
        figure = px.imshow(array, title=title)
        figure.update_layout(
            width=width or self.width,
            height=height or self.height,
            margin=dict(t=55, b=20, l=20, r=20),
        )
        figure.update_xaxes(visible=False)
        figure.update_yaxes(visible=False)
        return figure

    def digit_grid(self, images, rows, cols, title=""):
        """Draw a grid of small square images, each 8 by 8 pixels.

        This is the case that would otherwise need matplotlib: many tiny panels
        in one figure. Each cell becomes a `go.Heatmap` trace, and the axis
        ticks are hidden because pixel coordinates carry no meaning.

        Parameters
        ----------
        images : array-like
            Either an (n, 8, 8) array of images, or an (n, 64) array of
            flattened images.
        rows, cols : int
            Grid shape. Only the first `rows * cols` images are drawn.
        title : str, default ""
            Figure title.

        Returns
        -------
        plotly.graph_objects.Figure
            A figure with `rows * cols` heatmap traces and no axis labels.
        """
        data = np.asarray(images)
        if data.ndim == 2:
            data = data.reshape(-1, 8, 8)
        figure = make_subplots(
            rows=rows,
            cols=cols,
            horizontal_spacing=0.01,
            vertical_spacing=0.02,
            subplot_titles=None,
        )
        shown = 0
        for row in range(1, rows + 1):
            for col in range(1, cols + 1):
                if shown >= len(data):
                    break
                figure.add_trace(
                    go.Heatmap(
                        z=data[shown],
                        showscale=False,
                        colorscale="Greys",
                        hoverinfo="skip",
                    ),
                    row=row,
                    col=col,
                )
                shown += 1
        figure.update_layout(
            width=self.width,
            height=90 + 42 * rows,
            title=title,
            margin=dict(t=55, b=10, l=10, r=10),
            plot_bgcolor="white",
        )
        figure.update_xaxes(visible=False)
        figure.update_yaxes(visible=False)
        return figure

    def panel(self, figure_specs, rows, cols, titles=None, titles_position="subplot"):
        """Lay several figures side by side in one figure.

        Replaces `plt.subplots` layouts. Each spec is a small recipe: a Plotly
        trace plus the cell it belongs in, so the caller keeps control of the
        content while this method handles the grid.

        Parameters
        ----------
        figure_specs : list of dict
            Each dict has `trace` (a Plotly trace), and optionally `row`, `col`,
            and `title` (per-cell title).
        rows, cols : int
            Grid shape.
        titles : list of str or None
            Optional subplot titles, in reading order.

        Returns
        -------
        plotly.graph_objects.Figure
            The combined figure.
        """
        figure = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=titles,
            horizontal_spacing=0.07,
            vertical_spacing=0.12,
        )
        for index, spec in enumerate(figure_specs):
            row = spec.get("row", index // cols + 1)
            col = spec.get("col", index % cols + 1)
            figure.add_trace(spec["trace"], row=row, col=col)
        figure.update_layout(
            width=self.width,
            height=self.height,
            margin=dict(t=70, b=45, l=55, r=20),
            showlegend=True,
        )
        return figure

    def component_arrows(self, figure, mean, variances, components, scale=3):
        """Add arrow annotations showing principal component directions.

        Arrows are drawn from the data mean, with a length proportional to the
        square root of each component's variance, so the longer arrow is the
        direction holding more variance.

        Parameters
        ----------
        figure : plotly.graph_objects.Figure
            An existing scatter of the same data, to annotate.
        mean : array-like
            The centre the arrows start from, usually `pca.mean_`.
        variances : array-like
            `explained_variance_`, in data units squared.
        components : array-like
            `components_`, one row per component.
        scale : float, default 3
            Multiplier on the arrow length.

        Returns
        -------
        plotly.graph_objects.Figure
            The same figure with one arrow per component added.
        """
        for length, vector in zip(variances, components):
            end = np.asarray(mean) + scale * np.asarray(vector) * np.sqrt(length)
            figure.add_annotation(
                x=float(end[0]),
                y=float(end[1]),
                ax=float(mean[0]),
                ay=float(mean[1]),
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.4,
                arrowwidth=2,
                arrowcolor="black",
            )
        return figure

    def variance_curves(self, ratio, cumulative, title=""):
        """Two-panel plot: per-component share and the running total.

        The cumulative panel carries reference lines at 80% and 90%, because
        those are the shares a reader usually has in mind when choosing a number
        of components.

        Parameters
        ----------
        ratio : array-like
            `explained_variance_ratio_`, one value per component.
        cumulative : array-like
            The running total of `ratio`, usually `np.cumsum(ratio)`.
        title : str, default ""
            Figure title.

        Returns
        -------
        plotly.graph_objects.Figure
            Two panels sharing the horizontal axis meaning.
        """
        components = np.arange(1, len(ratio) + 1)
        figure = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Share of one component", "Running total"),
            horizontal_spacing=0.10,
        )
        figure.add_trace(
            go.Scatter(
                x=components,
                y=ratio,
                mode="lines",
                name="share",
                line=dict(color="#C0392B", width=2),
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=components,
                y=cumulative,
                mode="lines",
                name="cumulative",
                line=dict(color="#2E86C1", width=2),
            ),
            row=1,
            col=2,
        )
        for _level in (0.80, 0.90):
            figure.add_hline(
                y=_level, line_dash="dash", line_color="grey", row=1, col=2
            )
        figure.update_xaxes(title_text="Number of components", row=1, col=1)
        figure.update_xaxes(title_text="Number of components", row=1, col=2)
        figure.update_yaxes(title_text="Share of variance", row=1, col=1)
        figure.update_yaxes(title_text="Cumulative share", row=1, col=2)
        figure.update_layout(
            width=900,
            height=self.height,
            title=title,
            showlegend=False,
            margin=dict(t=70, b=50, l=55, r=20),
        )
        return figure
