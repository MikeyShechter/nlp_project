import numpy as np
import pandas as pd
import functools
from matplotlib import pyplot as plt
from numpy import ndarray
from pandas import DataFrame
from sklearn.manifold import TSNE
import scipy.stats as stats

CLUSTER_IDX = "cluster_idx"


class ClusteringStatistics:
    def __init__(self, neurons_df: DataFrame, predictions: ndarray, label: str):
        assert len(neurons_df) == len(predictions), "Data frame and clustering result have different lengths " \
                                                         f"({len(neurons_df)} and ({predictions})"
        neurons_df[CLUSTER_IDX] = predictions
        neurons_df = neurons_df[neurons_df[CLUSTER_IDX] != -1]  # filter unclustered explanations

        if len(neurons_df) < len(predictions):
            print(f"Filtered unclustered explanations, started with {len(predictions)}, remaining {len(neurons_df)}")

        self.df = neurons_df
        self.predictions = predictions
        self.num_clusters = max(predictions) + 1
        self.label = label

    @functools.lru_cache(maxsize=None)
    def get_cluster_variances(self) -> pd.Series:
        min_var = float('inf')
        cluster_vars = []
        for i in range(self.num_clusters):
            cluster_var = self.df.loc[self.df[CLUSTER_IDX] == i]['layer'].var()
            cluster_vars.append(cluster_var)

            # cluster_var_normalized = cluster_var / self.df.loc[self.df[CLUSTER_IDX] == i]['layer'].count()
            # if cluster_var_normalized < min_var:
            #     min_i = i
            #     min_var = cluster_var_normalized

        # TODO ozzafar - show the best clusters, consider normalize the min cluster var by the cluster size
        # min_cluster = self.df.loc[self.df[CLUSTER_IDX] == min_i]["layer"]
        # print(f'{min_var=},{min_i=},{min_cluster.count()=},{min_cluster.value_counts()=}')
        #
        # count_pred_i = self.df.loc[self.df[CLUSTER_IDX] == min_i].layer.value_counts().reindex(range(48), fill_value=0).sort_index()
        # plt.plot(range(48), count_pred_i.values)
        # plt.title(f'Plot {min_i}')
        #
        # plt.tight_layout()
        # plt.show()

        return pd.Series(cluster_vars).describe()

    def get_weighted_mean_cluster_variances(self) -> float:
        """Returns a weighted average of the clusters' variances (weight is the cluster's size)"""
        cluster_vars = []
        cluster_sizes = []

        for i in range(self.num_clusters):
            cluster_df = self.df[self.df[CLUSTER_IDX] == i]
            cluster_var = cluster_df['layer'].var()
            cluster_size = len(cluster_df)
            cluster_vars.append(cluster_var)
            cluster_sizes.append(cluster_size)

        weighted_mean = sum(variance * size for variance, size in zip(cluster_vars, cluster_sizes)) / sum(cluster_sizes)

        return weighted_mean

    def plot_clusters_scores_boxplot(self):
        """Returns box-plots of the scores in the top 10 clusters, by size"""
        cluster_sizes = self.df[CLUSTER_IDX].value_counts()

        num_clusters_to_select = 10
        selected_clusters = cluster_sizes.head(num_clusters_to_select).index
        selected_clusters = selected_clusters[::-1]  # Reverse order

        filtered_df = self.df[self.df[CLUSTER_IDX].isin(selected_clusters)]
        # cluster_stats = filtered_df.groupby(CLUSTER_IDX)['score'].agg(['mean', 'median', 'std'])

        # Increase the size of the plot
        plt.figure(figsize=(10, 6))  # Adjust the width and height as needed

        # Create a box plot for the scores in each cluster
        cluster_data = [filtered_df[filtered_df[CLUSTER_IDX] == cluster]['score'].values for cluster in selected_clusters]
        plt.boxplot(cluster_data, labels=selected_clusters, vert=False)
        plt.title(self.label)
        plt.xlim(-0.2, 1.0)

        # for i, cluster in enumerate(selected_clusters):
        #     cluster_size = cluster_sizes[cluster]
        #     plt.text(-0.15, i + 1, f'N={cluster_size}', verticalalignment='center')

        custom_labels = [f'Cluster {cluster}\nSize={cluster_sizes[cluster]}' for cluster in selected_clusters]
        plt.yticks(range(1, len(selected_clusters) + 1), custom_labels)

        plt.xlabel('Score')
        plt.ylabel('Cluster')

        plt.show()

    def layer_variance(self, predictions):
        print("Layer variance statistics:")
        coefs = []
        max_coef = 0
        max_coef_layer = -1
        for i in range(self.num_clusters):
            mask = (self.df.layer == i).values
            n_points_per_cluster = pd.Series(predictions[mask]).value_counts().values
            # 0 means distributed equally, 1 means extremely concentrated
            # We want higher
            gini_coef = self._gini(n_points_per_cluster)
            if gini_coef > max_coef:
                max_coef = gini_coef
                max_coef_layer = i
            coefs.append(gini_coef)

        gini_coef_mean = np.array(coefs).mean()
        print(f"{gini_coef_mean=}")
        print(f"{max_coef=} in layer: {max_coef_layer=}")
        return gini_coef_mean, max_coef, max_coef_layer

    def get_anova_test(self):
        groups = [self.df.loc[self.df[CLUSTER_IDX] == i]['layer'].array for i in range(self.num_clusters)]
        f_statistic, p_value = stats.f_oneway(*groups)
        return f_statistic, p_value

    def plot_cluster_hists(self, nrows, ncols):
        fig, axes = plt.subplots(nrows, ncols, figsize=(18, 24))  # 8 rows, 6 columns

        for i in range(nrows):
            for j in range(ncols):
                index = i * ncols + j
                count_pred_i = self.df.iloc[self.df[CLUSTER_IDX] == index].layer.value_counts().sort_index()
                axes[i, j].plot(range(48), count_pred_i.values)
                axes[i, j].set_title(f'Plot {index}')

        plt.tight_layout()
        plt.show()

    def plot_clustered_embedding(self, embeddings):
        tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        plt.figure(figsize=(10, 8))

        unique_clusters = np.unique(self.df[CLUSTER_IDX])
        for cluster in unique_clusters:
            cluster_mask = self.df[CLUSTER_IDX] == cluster
            plt.scatter(embeddings_2d[cluster_mask, 0], embeddings_2d[cluster_mask, 1], label=f'Cluster {cluster}')

        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], marker='o', s=5)  # If you don't have cluster labels

        plt.title("t-SNE or UMAP Visualization of Clustered Embeddings")
        plt.legend()
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")

        print("Showing plot")
        plt.show()

    def _gini(self, l):
        # Mean absolute difference
        mad = np.abs(np.subtract.outer(l, l)).mean()
        # Relative mean absolute difference
        rmad = mad / np.mean(l)
        # Gini coefficient
        g = 0.5 * rmad
        return g
