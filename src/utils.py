import json
import os
import pickle

import pandas as pd
import matplotlib.pyplot as plt
from src.clustering import *


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
    clustering_stats = {key: value.get_cluster_variances() for key, value in clustering_stats.items()}

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
    clustering_stats_str = {key: str(value.get_cluster_variances()) for key, value in clustering_stats.items()}

    best_mean_var = min(clustering_stats.items(), key=lambda kvp: kvp[1].get_cluster_variances()['mean'])
    clustering_stats_str["best_mean_var"] = f'{best_mean_var[0]}, {best_mean_var[1].get_cluster_variances()["mean"]}'

    best_min_var = min(clustering_stats.items(), key=lambda kvp: kvp[1].get_cluster_variances()['min'])
    clustering_stats_str["best_min_var"] = f'{best_min_var[0]}, {best_min_var[1].get_cluster_variances()["min"]}'

    with open("experiments/results.json", "w") as fp:
        clustering_stats = {key: str(value.get_cluster_variances()) for key, value in clustering_stats.items()}
        json.dump(clustering_stats, fp, indent=4)


def save_gini(clustering_stats, h):
    w = len(clustering_stats) // h
    fig, axarr = plt.subplots(h, w, figsize=(12, 15))

    for i, (label, cluster_stats) in enumerate(clustering_stats.items()):
        gini_coefs = cluster_stats.layer_variance()

        k = i // h
        j = i % h
        axarr[j, k].plot(range(len(gini_coefs)), gini_coefs)
        axarr[j, k].set_title(label)

    plt.subplots_adjust(hspace=1, wspace=0.4)
    plt.tight_layout()
    plt.show()
    # plt.savefig(os.path.join("experiments", "gini.png"), bbox_inches='tight')
