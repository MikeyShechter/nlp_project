import os
import pickle

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.clustering import *


def get_percentile_samples(grouped_df, percentile):
    percentile_80_value = grouped_df['score'].quantile(percentile)
    filtered_df = grouped_df[grouped_df['score'] > percentile_80_value]
    return filtered_df


def load_df(percentile):
    full_df = pd.read_csv(os.path.join("data", "scores_and_explanations.csv"), sep=',')
    df = full_df.groupby(['layer']).apply(lambda x: get_percentile_samples(x, percentile)).reset_index(level=0,
                                                                                                       drop=True)
    return df


def load_embeddings(df, create_new=False, model_name='all-mpnet-base-v2'):
    """Other model: all-MiniLM-L6-v2"""
    if create_new:
        model = SentenceTransformer(model_name)
        sentences = df['explanation'].values
        embeddings = model.encode(sentences, show_progress_bar=True)
    else:
        full_embeddings = np.load("data/neurons_explanations_embeddings_full.npy")
        embeddings = full_embeddings[df.index]

    return embeddings


# Cluster sizes:
# clusters_sizes = pd.Series(predictions).value_counts().sort_index()
# clusters_sizes.describe()

def save_predictions(predictions: ndarray, label):
    directory = "experiments/predictions"
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(f'{directory}/{label}', 'wb') as file:
        pickle.dump(predictions, file)


def try_load_predictions(label) -> ndarray | None:
    try:
        with open(f'experiments/predictions/{label}', 'rb') as file:
            predictions = pickle.load(file)
        return predictions
    except FileNotFoundError:
        print(f"Can't load '{label}', file does not exist")
    except Exception as ex:
        print(f"Can't load '{label}': {str(ex)}")
    return None


def print_statistics(results: dict):
    del results["best_mean_var"]
    del results["best_min_var"]

    sorted_result = dict(sorted(results.items(), key=lambda item: item[1].mean()))
    for (label, var) in sorted_result.items():
        print(f"Label: {label}, mean var: {var.mean()}")
    print("----------")

    sorted_result = dict(sorted(results.items(), key=lambda item: item[1].median()))
    for (label, var) in sorted_result.items():
        print(f"Label: {label}, median var: {var.median()}")
    print("----------")

    sorted_result = dict(sorted(results.items(), key=lambda item: item[1].max()))
    for (label, var) in sorted_result.items():
        print(f"Label: {label}, max var: {var.max()}")
    print("----------")

    sorted_result = dict(sorted(results.items(), key=lambda item: item[1].min()))
    for (label, var) in sorted_result.items():
        print(f"Label: {label}, min var: {var.min()}")
    print("----------")


def gini(l):
    # Mean absolute difference
    mad = np.abs(np.subtract.outer(l, l)).mean()
    # Relative mean absolute difference
    rmad = mad/np.mean(l)
    # Gini coefficient
    g = 0.5 * rmad
    return g
