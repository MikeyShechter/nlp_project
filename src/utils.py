import json
import os
import pickle

import pandas as pd
import matplotlib.pyplot as plt
from src.clustering import *


class ClusteringParameters:
    def __init__(self, embedding_method, clustering_method, percentile):
        self.embedding_method = embedding_method
        self.clustering_method = clustering_method
        self.percentile = percentile

    def __str__(self):
        return f"{self.embedding_method}_{self.clustering_method}_{self.percentile}"

    def __eq__(self, other):
        if isinstance(other, ClusteringParameters):
            return (self.embedding_method, self.clustering_method, self.percentile) == (other.embedding_method, other.clustering_method, other.percentile)
        return False

    def __hash__(self):
        return hash((self.embedding_method, self.clustering_method, self.percentile))


def get_percentile_samples(grouped_df, percentile):
    percentile_80_value = grouped_df['score'].quantile(percentile)
    filtered_df = grouped_df[grouped_df['score'] >= percentile_80_value]
    return filtered_df


def load_df(percentile=None):
    full_df = pd.read_csv(os.path.join("data", "scores_and_explanations.csv"), sep=',')

    if percentile is None:
        df = full_df
    else:
        df = full_df.groupby(['layer']).apply(lambda x: get_percentile_samples(x, percentile)).reset_index(level=0,
                                                                                                           drop=True)
    df['explanation'] = df['explanation'].fillna('')

    return df


# Cluster sizes:
# clusters_sizes = pd.Series(predictions).value_counts().sort_index()
# clusters_sizes.describe()

def save_predictions(predictions: ndarray, label):
    directory = "experiments/predictions"
    if not os.path.exists(directory):
        os.makedirs(directory)

    np.save(f"{directory}/{label}.npy", predictions)


def try_load_predictions(label) -> ndarray:
    try:
        directory = "experiments/predictions"
        predictions = np.load(f"{directory}/{label}.npy")
        return predictions
    except FileNotFoundError:
        print(f"Can't load '{label}', file does not exist")
    except Exception as ex:
        print(f"Can't load '{label}': {str(ex)}")
    return None


def print_statistics(clustering_stats: dict):
    clustering_stats = {str(key): value.get_cluster_variances_summary() for key, value in clustering_stats.items()}

    sorted_result = dict(sorted(clustering_stats.items(), key=lambda item: item[1]['mean']))
    for (label, var) in sorted_result.items():
        print(f"Label: {label}, mean var: {var['mean']}")
    print("----------")

    sorted_result = dict(sorted(clustering_stats.items(), key=lambda item: item[1]['50%']))
    for (label, var) in sorted_result.items():
        print(f"Label: {label}, median var: {var['50%']}")
    print("----------")

    sorted_result = dict(sorted(clustering_stats.items(), key=lambda item: item[1]['max']))
    for (label, var) in sorted_result.items():
        print(f"Label: {label}, max var: {var['max']}")
    print("----------")

    sorted_result = dict(sorted(clustering_stats.items(), key=lambda item: item[1]['min']))
    for (label, var) in sorted_result.items():
        print(f"Label: {label}, min var: {var['min']}")
    print("----------")


def save_results(clustering_stats):
    clustering_stats_str = {key: str(value.get_cluster_variances_summary()) for key, value in clustering_stats.items()}

    best_mean_var = min(clustering_stats.items(), key=lambda kvp: kvp[1].get_cluster_variances_summary()['mean'])
    clustering_stats_str["best_mean_var"] = f'{best_mean_var[0]}, {best_mean_var[1].get_cluster_variances_summary()["mean"]}'

    best_min_var = min(clustering_stats.items(), key=lambda kvp: kvp[1].get_cluster_variances_summary()['min'])
    clustering_stats_str["best_min_var"] = f'{best_min_var[0]}, {best_min_var[1].get_cluster_variances_summary()["min"]}'

    with open("experiments/results.json", "w") as fp:
        clustering_stats = {str(key): str(value.get_cluster_variances_summary()) for key, value in clustering_stats.items()}
        json.dump(clustering_stats, fp, indent=4)
