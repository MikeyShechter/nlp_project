import os

import numpy as np
from matplotlib import pyplot as plt


RESULTS_FOLDER = os.path.join("experiments", "results")


class ExperimentVisualizer:
    """
    A class for visualizing experiment results using the obtained clustering statistics.

    Args:
        clustering_stats (dict): A dictionary containing clustering statistics.
        clustering_methods (list of str): A list of clustering method names.
        percentiles (list of int): A list of filtering percentiles used in the experiments.
    """

    def __init__(self, clustering_stats, clustering_methods, percentiles, save_figures):
        self.clustering_stats = clustering_stats
        self.clustering_methods = clustering_methods
        self.percentiles = percentiles
        self.save_figures = save_figures

    def plot_all(self):
        self.plot_gini()
        self.plot_layers_gini_coefficient()
        self.plot_clusters_variance()
        self.plot_clusters_variance_boxplot_by_method()
        self.plot_clusters_variance_boxplot_by_percentile()

    def plot_gini(self):
        h = len(self.percentiles)
        w = len(self.clustering_stats) // h
        fig, axarr = plt.subplots(h, w, figsize=(12, 15))

        for i, (label, cluster_stats) in enumerate(self.clustering_stats.items()):
            gini_coefs = cluster_stats.get_layers_variance()

            k = i // h
            j = i % h
            axarr[j, k].plot(range(len(gini_coefs)), gini_coefs)
            axarr[j, k].set_title(str(label))

        plt.subplots_adjust(hspace=1, wspace=0.4)
        plt.tight_layout()
        # plt.savefig(os.path.join("experiments", "gini.png"), bbox_inches='tight')
        plt.show()

    def plot_layers_gini_coefficient(self):
        """
        Create separate plots for each clustering method's Gini coefficients.

        Parameters:
        clustering_stats (dict): A dictionary mapping ClusteringLabel to an object with a 'layer_variance()' method.
        clustering_methods (list): A list of clustering method names.

        Each method will have its own plot, and Gini coefficients for different percentiles are plotted on each plot.
        """
        for clustering_method in self.clustering_methods:
            plt.figure()

            for label, clustering_statistics in self.clustering_stats.items():
                if label.clustering_method == clustering_method:
                    gini_coeff = clustering_statistics.get_layers_variance()
                    x_values = np.arange(len(gini_coeff))
                    plt.plot(x_values, gini_coeff, label=f'Percentile {label.percentile}')

            plt.xlabel('Layer')
            plt.ylabel('Gini coefficient')
            plt.legend(loc='best')
            plt.title(clustering_method)

            if self.save_figures:
                plt.savefig(os.path.join(RESULTS_FOLDER, f"layer_gini_{clustering_method.lower()}.png"), bbox_inches='tight')

            plt.show()

    def plot_clusters_variance(self):
        """
        Create separate plots for each clustering method's cluster variances.

        Parameters:
        clustering_stats (dict): A dictionary mapping ClusteringLabel to an object with a 'CLUSTER()' method.
        clustering_methods (list): A list of clustering method names.

        Each method will have its own plot, and Gini coefficients for different percentiles are plotted on each plot.
        """
        for clustering_method in self.clustering_methods:
            plt.figure(figsize=(10, 6))

            for label, clustering_statistics in self.clustering_stats.items():
                if label.clustering_method == clustering_method:
                    cluster_variances = clustering_statistics.get_clusters_variance()
                    x_values = np.arange(len(cluster_variances))
                    plt.plot(x_values, cluster_variances, label=f'Percentile {label.percentile}')

            plt.xlabel('Cluster')
            plt.ylabel('Variance')
            plt.legend(loc='best')
            plt.title(clustering_method)

            if self.save_figures:
                plt.savefig(os.path.join(RESULTS_FOLDER, f"cluster_variance_{clustering_method.lower()}.png"), bbox_inches='tight')

            plt.show()

    def plot_clusters_variance_boxplot_by_method(self):
        """
        Plots a boxplot of cluster variances for different percentiles by clustering method.

        This method creates a boxplot for each clustering method, displaying the distribution of cluster variances
        at different percentiles. It provides insights into the variation of clusters within each method.
        """
        for clustering_method in self.clustering_methods:
            plt.figure()

            data = []
            labels = []

            for label, clustering_statistics in self.clustering_stats.items():
                if label.clustering_method == clustering_method:
                    clusters_variance = clustering_statistics.get_clusters_variance()
                    labels.append(f'Percentile {label.percentile}')
                    data.append(clusters_variance)

            plt.boxplot(data, labels=labels, showmeans=True)
            plt.xlabel('Percentiles')
            plt.ylabel('Cluster Variance')
            plt.title(f'{clustering_method} Cluster Statistics')
            plt.grid(True)  # Add grid lines

            if self.save_figures:
                plt.savefig(os.path.join(RESULTS_FOLDER, f"cluster_variance_boxplot_{clustering_method.lower()}.png"), bbox_inches='tight')

            plt.show()

    def plot_clusters_variance_boxplot_by_percentile(self):
        """
        Plots a boxplot of cluster variances for different clustering methods by percentile.

        This method creates a boxplot for each percentile, displaying the distribution of cluster variances
        for different clustering methods. It provides insights into the variation of clusters at specific percentiles.
        """
        for percentile in self.percentiles:
            plt.figure()

            data = []
            labels = []

            for label, clustering_statistics in self.clustering_stats.items():
                if label.percentile == percentile:
                    cluster_variances = clustering_statistics.get_clusters_variance()
                    labels.append(label.clustering_method)
                    data.append(cluster_variances)

                    # We manually add the weighted average to the boxplot
                    cluster_variances_weighted_mean = clustering_statistics.get_weighted_mean_cluster_variances()
                    plt.scatter([len(data)], [cluster_variances_weighted_mean], marker='o', color='red', label='')
                    # print(f'Label {label} weighted mean: {cluster_variances_weighted_mean}')

            plt.boxplot(data, labels=labels, showmeans=True)
            plt.xlabel('Clustering method')
            plt.ylabel('Cluster variance')
            plt.title(f'Percentile {percentile} Cluster Statistics')
            plt.grid(True)  # Add grid lines

            if self.save_figures:
                plt.savefig(os.path.join(RESULTS_FOLDER, f"cluster_variance_boxplot_percentile_{percentile}.png"), bbox_inches='tight')

            plt.show()
