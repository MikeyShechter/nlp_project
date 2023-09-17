import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame

from src.clustering import ClusteredData


class ClusteringStatistics:
    def __init__(self, neurons_df: DataFrame, clustered_data: ClusteredData):
        self.neurons_df = neurons_df
        self.clustered_data = clustered_data

    def get_layer_variance(self):
        layer_variances = []
        for i in range(self.clustered_data.num_clusters):
            n_points_per_cluster = self.neurons_df[self.neurons_df.layer == i].kmeans_preds.value_counts()
            var = n_points_per_cluster.var()  # Larger is better
            layer_variances.append(var)
        layer_variance_mean = np.array(layer_variances).mean()
        print(f"{layer_variance_mean=}")
        return layer_variance_mean

    def get_mean_cluster_variances(self) -> pd.Series:
        cluster_vars = []
        for i in range(self.clustered_data.num_clusters):
            cluster_var = self.neurons_df.loc[self.neurons_df['cluster_preds'] == i]['layer'].var()
            cluster_vars.append(cluster_var)

        return pd.Series(cluster_vars).describe()

    def plot_cluster_hists(self, nrows, ncols):
        fig, axes = plt.subplots(nrows, ncols, figsize=(18, 24))  # 8 rows, 6 columns

        for i in range(nrows):
            for j in range(ncols):
                index = i * ncols + j
                count_pred_i = self.neurons_df.iloc[self.neurons_df['cluster_preds'] == index].layer.value_counts().sort_index()
                axes[i, j].plot(range(48), count_pred_i.values)
                axes[i, j].set_title(f'Plot {index}')

        plt.tight_layout()
        plt.show()



