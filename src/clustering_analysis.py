import numpy as np
import pandas as pd
import functools
from matplotlib import pyplot as plt
from numpy import ndarray
from pandas import DataFrame
from sklearn.manifold import TSNE

CLUSTER_PREDS_COL = "cluster_idx"


class ClusteringStatistics:
    def __init__(self, neurons_df: DataFrame, predictions: ndarray):
        assert len(neurons_df) == len(predictions), "Data frame and clustering result have different lengths " \
                                                         f"({len(neurons_df)} and ({predictions})"
        neurons_df[CLUSTER_PREDS_COL] = predictions
        neurons_df = neurons_df[neurons_df[CLUSTER_PREDS_COL] != -1]  # filter unclustered explanations

        if len(neurons_df) < len(predictions):
            print(f"Filtered unclustered explanations, started with {len(predictions)}, remaining {len(neurons_df)}")

        self.neurons_df = neurons_df
        self.predictions = predictions
        self.num_clusters = max(predictions) + 1

    @functools.lru_cache(maxsize=None)
    def get_cluster_variances(self) -> pd.Series:
        cluster_vars = []
        for i in range(self.num_clusters):
            cluster_var = self.neurons_df.loc[self.neurons_df[CLUSTER_PREDS_COL] == i]['layer'].var()
            cluster_vars.append(cluster_var)

        return pd.Series(cluster_vars).describe()

    def get_weighted_mean_cluster_variances(self) -> float:
        """Returns a weighted average of the clusters' variances (weight is the cluster's size)"""
        cluster_vars = []
        cluster_sizes = []

        for i in range(self.num_clusters):
            cluster_df = self.neurons_df[self.neurons_df[CLUSTER_PREDS_COL] == i]
            cluster_var = cluster_df['layer'].var()
            cluster_size = len(cluster_df)
            cluster_vars.append(cluster_var)
            cluster_sizes.append(cluster_size)

        weighted_mean = sum(variance * size for variance, size in zip(cluster_vars, cluster_sizes)) / sum(cluster_sizes)

        return weighted_mean

    def get_layer_variance(self):
        layer_variances = []
        for i in range(self.num_clusters):
            n_points_per_cluster = self.neurons_df[self.neurons_df.layer == i].kmeans_preds.value_counts()
            var = n_points_per_cluster.var()  # Larger is better
            layer_variances.append(var)
        layer_variance_mean = np.array(layer_variances).mean()
        print(f"{layer_variance_mean=}")
        return layer_variance_mean

    def plot_cluster_hists(self, nrows, ncols):
        fig, axes = plt.subplots(nrows, ncols, figsize=(18, 24))  # 8 rows, 6 columns

        for i in range(nrows):
            for j in range(ncols):
                index = i * ncols + j
                count_pred_i = self.neurons_df.iloc[self.neurons_df[CLUSTER_PREDS_COL] == index].layer.value_counts().sort_index()
                axes[i, j].plot(range(48), count_pred_i.values)
                axes[i, j].set_title(f'Plot {index}')

        plt.tight_layout()
        plt.show()

    def plot_clustered_embedding(self, embeddings):
        tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        plt.figure(figsize=(10, 8))

        unique_clusters = np.unique(self.neurons_df[CLUSTER_PREDS_COL])
        for cluster in unique_clusters:
            cluster_mask = self.neurons_df[CLUSTER_PREDS_COL] == cluster
            plt.scatter(embeddings_2d[cluster_mask, 0], embeddings_2d[cluster_mask, 1], label=f'Cluster {cluster}')

        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], marker='o', s=5)  # If you don't have cluster labels

        plt.title("t-SNE or UMAP Visualization of Clustered Embeddings")
        plt.legend()
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")

        print("Showing plot")
        plt.show()

