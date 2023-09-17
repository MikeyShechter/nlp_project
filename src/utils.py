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
    full_df = pd.read_csv("data\scores_and_explanations.csv", sep=',')
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

def save_predictions(predictions: ClusteredData, label):
    directory = "experiments/predictions"
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(f'{directory}/{label}', 'wb') as file:
        pickle.dump(predictions, file)


def try_load_predictions(label) -> ClusteredData | None:
    try:
        with open(f'experiments/predictions/{label}', 'rb') as file:
            predictions = pickle.load(file)
        return predictions
    except FileNotFoundError:
        print(f"Can't load '{label}', file does not exist")
        return None


