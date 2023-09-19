import json
import time

from src.clustering_analysis import *
from src.utils import *

CLUSTERING_METHODS = ["KMEANS", "DBSCAN", "GMM", "MEANSHIFT", "RANDOM"]
# CLUSTERING_METHODS = ["DBSCAN"]
# PERCENTILES = [0, 0.5, 0.9, 0.95, 0.99]
PERCENTILES = [0.99]
SAVE_PREDICTIONS = True
LOAD_PERDICTIONS = True
TRIM_DF = None  # Set an integer to take first K entries in the df


def main():
    clustering_stats = dict()

    # TODO: run also over embedding methods?
    for clustering_method in CLUSTERING_METHODS:
        for percentile in PERCENTILES:
            start_time = time.time()
            df = load_df(percentile=percentile)
            embeddings = load_embeddings(df)
            if isinstance(TRIM_DF, int):
                df = df[:TRIM_DF]

            label = f"{clustering_method=}_{percentile=}"
            print(f"Clustering label '{label}'")

            predictions = None
            if LOAD_PERDICTIONS:
                predictions = try_load_predictions(label)

            if predictions is None:
                # TODO: Run over a few n's [24, 48, 96, 192]?
                num_clusters = 48  # This argument is not always honored, it depends on the clustering method
                predictions = get_clustering_preds(embeddings=embeddings, clustering_method=clustering_method,
                                                   num_clusters=num_clusters)
                if SAVE_PREDICTIONS:
                    save_predictions(predictions, label)

            clustering_statistics = ClusteringStatistics(df, predictions)
            clustering_stats[label] = clustering_statistics

            # clustering_statistics.plot_clustered_embedding(embeddings)

            print(f"Label '{label}', elapsed {int(time.time() - start_time)} seconds")
            # print(f'Cluster's variance statistics:\n{cluster_var_summary}')
            print("--------------------------------")

    save_results(clustering_stats)
    print_statistics(clustering_stats)

    print("done!")


if __name__ == '__main__':
    main()
