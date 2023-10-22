import json
import time
from itertools import product

from utils import *
from embedding import *
from clustering_analysis import *
import matplotlib.pyplot as plt

# EMBEDDING_METHODS = [TRANSFORMER, WORD2VEC]
EMBEDDING_METHODS = [TRANSFORMER]
# CLUSTERING_METHODS = ["KMEANS", "RANDOM", "DBSCAN", "BERTOPIC"]
CLUSTERING_METHODS = ["KMEANS", "RANDOM", "DBSCAN"]
# PERCENTILES = [0, 0.5, 0.9, 0.95]
PERCENTILES = [0, 0.9]
SAVE_PREDICTIONS = True
LOAD_PERDICTIONS = True
TRIM_DF = None  # Set an integer to take first K entries in the df


def main():
    clustering_stats = dict()

    for embedding_method, percentile, clustering_method in product(EMBEDDING_METHODS, PERCENTILES, CLUSTERING_METHODS):
        start_time = time.time()
        df = load_df(percentile=percentile)
        explanations = df['explanation']
        embeddings = get_embeddings(df, embedding_method)
        if isinstance(TRIM_DF, int):
            df = df[:TRIM_DF]
            embeddings = embeddings[:TRIM_DF]

        label = f"{embedding_method}_{clustering_method}_{percentile}"
        print(f"Clustering label '{label}'")

        predictions = None
        if LOAD_PERDICTIONS and TRIM_DF is None:
            predictions = try_load_predictions(label)

        if predictions is None:
            num_clusters = 48  # This argument is not always honored, it depends on the clustering method
            predictions = get_clustering_preds(orig_explanations=explanations, embeddings=embeddings, clustering_method=clustering_method,
                                               num_clusters=num_clusters)
            if SAVE_PREDICTIONS and TRIM_DF is None:
                save_predictions(predictions, label)

        clustering_statistics = ClusteringStatistics(df, predictions, label)
        clustering_stats[label] = clustering_statistics

        f_statistic, p_value = clustering_statistics.get_anova_test()
        print(f'{f_statistic=},{p_value=}')
        clustering_statistics.plot_clusters_scores_boxplot()

        print(f"Label '{label}', elapsed {int(time.time() - start_time)} seconds")
        # print(f'Cluster's variance statistics:\n{cluster_var_summary}')
        print("--------------------------------")

    print("--------------------------------\nFinished clustering\n--------------------------------")

    save_results(clustering_stats)
    print_statistics(clustering_stats)

    draw_plots(clustering_stats, PERCENTILES)

    print("done!")


if __name__ == '__main__':
    main()
