#
import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

from sklearn.cluster import cluster_optics_dbscan
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.metrics import normalized_mutual_info_score, fowlkes_mallows_score

class ClusterVisualizer:
    '''
    Class for visualizing clustering results.
    '''

    def __init__(self, data: pd.DataFrame, true_labels: np.array = None):
        '''
        Constructor for the ClusterVisualizer class.
        data: pandas DataFrame with the data to visualize. x is the first column, y is the second.
        true_labels: numpy array with the cluster assignment for each point in data, if known.        
        '''
        self.data = data
        self.true_labels = true_labels
        self.set_width_height()

    def set_width_height(self, width=600, height=400):
        '''
        Set the width and height for the plots.
        '''
        self.width = width
        self.height = height

    def plot_data(self, title='Data Points'):
        '''
        It plots data in a scatter plot. x is the first column, y is the second.
        '''
        fig = px.scatter(
            x=self.data.iloc[:, 0],
            y=self.data.iloc[:, 1]
        )
        fig.update_traces(marker=dict(size=8, opacity=0.6))
        fig.update_layout(
            coloraxis_showscale=False,
            showlegend=False,
            xaxis_title=self.data.columns[0],
            yaxis_title=self.data.columns[1],
            width=self.width,
            height=self.height,
            title=title
        )
        return fig

    def plot_data_with_true_labels(self, title='Data Points with True Labels'):
        '''
         It plots data with the true_labels in a scatter plot. x is the first column, y is the second.        

        '''
        if self.true_labels is None:
            raise ValueError('True labels are not provided.')
        fig = px.scatter(
            x=self.data.iloc[:, 0],
            y=self.data.iloc[:, 1],
            color=self.true_labels.astype(str),
            color_continuous_scale='jet'
        )
        fig.update_traces(marker=dict(size=8, opacity=0.6))
        fig.update_layout(
            showlegend=True,
            legend_title_text='True Labels',
            coloraxis_showscale=False,
            xaxis_title=self.data.columns[0],
            yaxis_title=self.data.columns[1],
            width=self.width,
            height=self.height,
            title=title
        )
        return fig

    def plot_clusters(self, labels=None, centers=None, title=''):
        '''
        Plot clusters for a given dataset. 
        If labels are not provided, all points are assigned to the same cluster.
        '''
        if labels is None:
            labels = [0] * self.data.shape[0]
        fig = px.scatter(
            x=self.data.iloc[:, 0],
            y=self.data.iloc[:, 1],
            color=labels.astype(str),
            color_continuous_scale='portland'
        )
        fig.update_traces(marker=dict(size=8, opacity=0.6))
        fig.update_layout(
            showlegend=True,
            legend_title_text='Clusters',
            xaxis_title=self.data.columns[0],
            yaxis_title=self.data.columns[1],
            width=self.width,
            height=self.height,
            coloraxis_showscale=False,
            title=title
        )
        if centers is not None:
            fig.add_trace(
                go.Scatter(
                    x=centers[:, 0],
                    y=centers[:, 1],
                    mode='markers',
                    marker=dict(color='black', size=8, opacity=0.8),
                    name='Centers'
                )
            )
        return fig

    def plot_silhouette(self, labels, title='Silhouette Scores'):
        '''
        Computes and plots the silhouette scores (Internal Validation Index)
        data: pandas DataFrame with the data.
        labels: pandas DataFrame with the cluster assignments for each point in data.
        '''
        silh_scores = np.array(
            [silhouette_score(self.data, labels[col])
             for col in labels.columns]
        ).round(4)
        best_score = silh_scores.argmax()
        colors = [0] * len(silh_scores)
        colors[best_score] = 1
        fig = px.bar(
            x=labels.columns,
            y=silh_scores,
            text=silh_scores,
            color=colors,
            color_continuous_scale='portland',
            opacity=0.7
        )
        fig.update_layout(
            width=self.width,
            height=self.height,
            title=title,
            xaxis_title='Cluster',
            yaxis_title='Silhouette Score',
            coloraxis_showscale=False,
            showlegend=False
        )
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig.update_yaxes(range=[0, silh_scores.max() + 0.1])
        return fig

    def plot_davies_bouldin(self, labels, title='Davies-Bouldin Scores'):
        '''
        Computes and plots the Davies-Bouldin scores (Internal Validation Index)
        data: pandas DataFrame with the data.
        labels: pandas DataFrame with the cluster assignments for each point in data.
        '''
        db_scores = np.array(
            [davies_bouldin_score(self.data, labels[col])
             for col in labels.columns]
        ).round(4)
        best_score = db_scores.argmin()
        colors = [0] * len(db_scores)
        colors[best_score] = 1
        fig = px.bar(
            x=labels.columns,
            y=db_scores,
            text=db_scores,
            color=colors,
            color_continuous_scale='portland',
            opacity=0.7
        )
        fig.update_layout(
            width=self.width,
            height=self.height,
            title=title,
            xaxis_title='Cluster',
            yaxis_title='Davies-Bouldin Score',
            coloraxis_showscale=False,
            showlegend=False
        )
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig.update_yaxes(range=[0, db_scores.max() + 0.1*db_scores.max()])
        return fig

    def plot_calinski_harabasz(self, labels, title='Calinski-Harabasz Scores'):
        '''
        Computes and plots the Calinski-Harabasz scores (Internal Validation Index)
        data: pandas DataFrame with the data.
        labels: pandas DataFrame with the cluster assignments for each point in data.
        '''
        ch_scores = np.array(
            [calinski_harabasz_score(self.data, labels[col])
             for col in labels.columns]
        ).round(4)
        best_score = ch_scores.argmax()
        colors = [0] * len(ch_scores)
        colors[best_score] = 1
        fig = px.bar(
            x=labels.columns,
            y=ch_scores,
            text=ch_scores,
            color=colors,
            color_continuous_scale='portland',
            opacity=0.7
        )
        fig.update_layout(
            width=self.width,
            height=self.height,
            title=title,
            xaxis_title='Cluster',
            yaxis_title='Calinski-Harabasz Score',
            coloraxis_showscale=False,
            showlegend=False
        )
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig.update_yaxes(range=[0, ch_scores.max() + 0.1*ch_scores.max()])
        return fig

    def plot_adjusted_rand(self, labels, title='Adjusted Rand Scores'):
        '''
        Computes and plots the Adjusted Rand scores (External Validation Index)        
        labels: pandas DataFrame with the cluster assignments for each point in data.
        '''
        if self.true_labels is None:
            raise ValueError('True labels are not provided.')
        ar_scores = np.array(
            [adjusted_rand_score(self.true_labels, labels[col])
             for col in labels.columns]
        ).round(4)
        best_score = ar_scores.argmax()
        colors = [0] * len(ar_scores)
        colors[best_score] = 1
        fig = px.bar(
            x=labels.columns,
            y=ar_scores,
            text=ar_scores,
            color=colors,
            color_continuous_scale='geyser',
            opacity=0.7
        )
        fig.update_layout(
            width=self.width,
            height=self.height,
            itle=title,
            xaxis_title='Cluster',
            yaxis_title='Adjusted Rand Score',
            coloraxis_showscale=False,
            showlegend=False
        )
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig.update_yaxes(range=[0, ar_scores.max() + 0.1*ar_scores.max()])
        return fig

    def plot_adjusted_mutual_info(self, labels, title='Adjusted Mutual Info Scores'):
        '''
        Computes and plots the Adjusted Mutual Info scores (External Validation Index)        
        labels: pandas DataFrame with the cluster assignments for each point in data.
        '''
        if self.true_labels is None:
            raise ValueError('True labels are not provided.')
        am_scores = np.array(
            [adjusted_mutual_info_score(self.true_labels, labels[col])
             for col in labels.columns]
        ).round(4)
        best_score = am_scores.argmax()
        colors = [0] * len(am_scores)
        colors[best_score] = 1
        fig = px.bar(
            x=labels.columns,
            y=am_scores,
            text=am_scores,
            color=colors,
            color_continuous_scale='geyser',
            opacity=0.7
        )
        fig.update_layout(
            width=self.width,
            height=self.height,
            title=title,
            xaxis_title='Cluster',
            yaxis_title='Adjusted Mutual Info Score',
            coloraxis_showscale=False,
            showlegend=False
        )
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig.update_yaxes(range=[0, am_scores.max() + 0.1*am_scores.max()])
        return fig

    def plot_normalized_mutual_info(self, labels, title='Normalized Mutual Info Scores'):
        '''
        Computes and plots the Normalized Mutual Info scores (External Validation Index)        
        labels: pandas DataFrame with the cluster assignments for each point in data.
        '''
        if self.true_labels is None:
            raise ValueError('True labels are not provided.')
        nm_scores = np.array(
            [normalized_mutual_info_score(self.true_labels, labels[col])
             for col in labels.columns]
        ).round(4)
        best_score = nm_scores.argmax()
        colors = [0] * len(nm_scores)
        colors[best_score] = 1
        fig = px.bar(
            x=labels.columns,
            y=nm_scores,
            text=nm_scores,
            color=colors,
            color_continuous_scale='geyser',
            opacity=0.7
        )
        fig.update_layout(
            width=self.width,
            height=self.height,
            title=title,
            xaxis_title='Cluster',
            yaxis_title='Normalized Mutual Info Score',
            coloraxis_showscale=False,
            showlegend=False
        )
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig.update_yaxes(range=[0, nm_scores.max() + 0.1*nm_scores.max()])
        return fig

    def plot_fowlkes_mallows(self, labels, title='Fowlkes Mallows Scores'):
        '''
        Computes and plots the Fowlkes Mallows scores (External Validation Index)        
        labels: pandas DataFrame with the cluster assignments for each point in data.
        '''
        if self.true_labels is None:
            raise ValueError('True labels are not provided.')
        fm_scores = np.array(
            [fowlkes_mallows_score(self.true_labels, labels[col])
             for col in labels.columns]
        ).round(4)
        best_score = fm_scores.argmax()
        colors = [0] * len(fm_scores)
        colors[best_score] = 1
        fig = px.bar(
            x=labels.columns,
            y=fm_scores,
            text=fm_scores,
            color=colors,
            color_continuous_scale='geyser',
            opacity=0.7
        )
        fig.update_layout(
            width=self.width,
            height=self.height,
            title=title,
            xaxis_title='Cluster',
            yaxis_title='Fowlkes Mallows Score',
            coloraxis_showscale=False,
            showlegend=False
        )
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig.update_yaxes(range=[0, fm_scores.max() + 0.1 * fm_scores.max()])
        return fig

    def plot_density_based_clustering(self, labels, title='Density-based Clustering'):
        '''
        Plot the data and clusters obtained with density-based clustering.
        '''
        clusters = labels.astype(str)
        # Replace '-1' with a specific category name for custom coloring
        clusters[clusters == '-1'] = 'Noise'

        # Create a color map that maps 'Noise' to 'grey' and other values to colors from G10 palette
        unique_clusters = np.unique(clusters)
        colors = ['grey' if cluster == 'Noise' else color
                  for cluster, color in zip(unique_clusters, px.colors.qualitative.G10)]
        color_map = dict(zip(unique_clusters, colors))

        # Plot the scatter with colors mapped to clusters
        fig = px.scatter(
            x=self.data.iloc[:, 0],
            y=self.data.iloc[:, 1],
            color=clusters,
            color_discrete_map=color_map
        )
        fig.update_layout(
            coloraxis_showscale=False,
            width=self.width,
            height=self.height,
            title=title
        )
        # Explicitly enable and configure the legend
        fig.update_layout(
            legend_title_text='Cluster',
            xaxis_title=self.data.columns[0],
            yaxis_title=self.data.columns[1]
        )
        # Update the markers and the opacity to get a more beautiful plot.
        fig.update_traces(marker=dict(size=8, opacity=0.8))
        # Set 1:1 aspect ratio for x and y axes for proportional scaling
        fig.update_layout(
            xaxis=dict(scaleanchor="y", scaleratio=1),
            yaxis=dict(scaleanchor="x", scaleratio=1)
        )
        return fig

    def plot_DBSCAN_categories(self, labels, core_sample_indices=[], title=''):
        '''
        Plots core, border, and noise points within the DBSCAN clustering results.
        '''

        # Defining the point types (Core, Border, Noise)
        point_type = np.full_like(labels, 'Border', dtype=object)
        point_type[core_sample_indices] = 'Core'
        point_type[labels == -1] = 'Noise'

        labels = labels.astype(str)
        # Create a color map that maps '-1' (Noise) to 'grey' & other values to colors from G10 palette
        unique_clusters = np.unique(labels)
        colors = ['grey' if cluster == '-1' else color
                  for cluster, color in zip(unique_clusters, px.colors.qualitative.G10)]
        color_map = dict(zip(unique_clusters, colors))

        # Create a symbol map that specifies the symbol for noise points
        symbol_map = {
            'Noise': 'circle',
            'Core': 'circle',
            'Border': 'circle-open'
        }

        # Plot the scatter with colors mapped to clusters
        fig = px.scatter(
            x=self.data.iloc[:, 0],
            y=self.data.iloc[:, 1],
            symbol=point_type,
            symbol_map=symbol_map,
            color=labels,
            color_discrete_map=color_map
        )
        fig.update_layout(
            coloraxis_showscale=False,
            width=self.width,
            height=self.height,
            title=title,
            xaxis_title=self.data.columns[0],
            yaxis_title=self.data.columns[1],
            legend_title_text='Point Types'
        )
        # Set 1:1 aspect ratio for x and y axes for proportional scaling
        fig.update_layout(
            xaxis=dict(scaleanchor="y", scaleratio=1),
            yaxis=dict(scaleanchor="x", scaleratio=1)
        )
        # Update the markers and the opacity to get a more beautiful plot.
        fig.update_traces(marker=dict(size=8, opacity=0.8))
        return fig

    def plot_optics_dbscan(self, optics, eps=0.05):
        '''
        This function plots the results of the OPTICS clustering algorithm and its 
        DBSCAN-style clustering for a given epsilon (eps). It generates two plots: 
        a scatter plot showing the clusters in the dataset, and a reachability plot 
        indicating the reachability distances of each point, colored by cluster.
        '''
        labels = cluster_optics_dbscan(
            reachability=optics.reachability_,
            core_distances=optics.core_distances_,
            ordering=optics.ordering_,
            eps=eps
        ).astype(str)

        # Changing the label '-1' to 'Noise' for better visualization
        labels[labels == '-1'] = 'Noise'

        # Create a color map
        unique_clusters = np.unique(labels)
        colors = ['grey' if cluster == 'Noise' else color
                  for cluster, color in zip(unique_clusters, px.colors.qualitative.G10)]
        color_map = dict(zip(unique_clusters, colors))

        # Scatter plot
        fig = px.scatter(
            x=self.data.iloc[:, 0],
            y=self.data.iloc[:, 1],
            color=labels,
            color_discrete_map=color_map
        )
        fig.update_layout(
            coloraxis_showscale=False,
            width=self.width,
            height=350,
            title=f'OPTICS Clustering (eps = {eps:.2f})',
            legend_title_text='Cluster'
        )
        fig.update_traces(marker=dict(size=8, opacity=0.8))
        fig.show()

        # Reachability plot
        space = np.arange(self.data.shape[0])
        fig = go.Figure()

        # Create a line for each cluster
        for cluster in unique_clusters:
            cluster_indices = np.where(labels[optics.ordering_] == cluster)[0]
            fig.add_trace(
                go.Scatter(
                    x=space[cluster_indices],
                    y=optics.reachability_[optics.ordering_][cluster_indices],
                    mode='markers',
                    marker_size=5,
                    marker_color=color_map[cluster],
                    name=cluster
                )
            )
        fig.update_layout(
            width=self.width,
            height=350,
            title=f'Reachability Plot (eps = {eps:.2f})',
            legend_title_text='Cluster'
        )
        fig.update_yaxes(title_text='Reachability')
        fig.add_hline(y=eps, line_dash="dash", line_color="black")
        fig.show()
