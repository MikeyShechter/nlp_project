import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import ndarray
from pandas import DataFrame
from scipy.stats import chi2_contingency

from utils import gini

CLUSTER_PREDS_COL = "cluster_idx"


class ClusteringStatistics:
    def __init__(self, neurons_df: DataFrame, predictions: ndarray):
        assert len(neurons_df) == len(predictions), "Data frame and clustering result have different lengths " \
                                                         f"({len(neurons_df)} and ({predictions})"
        neurons_df[CLUSTER_PREDS_COL] = predictions
        neurons_df = neurons_df[neurons_df['cluster_idx'] != -1]  # filter unclustered explanations

        if len(neurons_df) < len(predictions):
            print(f"Filtered unclustered explanations, started with {len(predictions)}, remaining {len(neurons_df)}")

        self.neurons_df = neurons_df
        self.predictions = predictions
        self.num_clusters = max(predictions) + 1

    def get_mean_cluster_variances(self) -> pd.Series:
        cluster_vars = []
        for i in range(self.num_clusters):
            cluster_var = self.neurons_df.loc[self.neurons_df[CLUSTER_PREDS_COL] == i]['layer'].var()
            cluster_vars.append(cluster_var)

        return pd.Series(cluster_vars).describe()

    def layer_variance(self, predictions):
        print("Layer variance statistics:")
        coefs = []
        max_coef = 0
        max_coef_layer = -1
        for i in range(self.num_clusters):
            mask = (self.neurons_df.layer == i).values
            n_points_per_cluster = pd.Series(predictions[mask]).value_counts().values
            # 0 means distributed equally, 1 means extremely concentrated
            # We want higher
            gini_coef = gini(n_points_per_cluster)
            if gini_coef > max_coef:
                max_coef = gini_coef
                max_coef_layer = i
            coefs.append(gini_coef)

        gini_coef_mean = np.array(coefs).mean()
        print(f"{gini_coef_mean=}")
        print(f"{max_coef=} in layer: {max_coef_layer=}")
        return gini_coef_mean, max_coef, max_coef_layer

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



