import json
import time

from src.clustering_analysis import *
from src.utils import *

CLUSTERING_METHODS = ["KMEANS", "DBSCAN", "GMM", "MEANSHIFT"]
PERCENTILES = [0, 0.2, 0.5, 0.8, 0.9]
SAVE_CLUSTERING_RESULT = True
LOAD_CLUSTERING_RESULT = True
TRIM_RESULTS = None  # Set an integer to take first K entries in the df


def main():
    results = dict()
    best_mean_var, best_min_var = float('inf'), float('inf')

    # TODO: run also over embedding methods?
    for clustering_method in CLUSTERING_METHODS:
        for percentile in PERCENTILES:
            start_time = time.time()
            df = load_df(percentile=percentile)
            embeddings = load_embeddings(df)

            label = f"{clustering_method=}_{percentile=}"
            print(f"Clustering label '{label}'")

            predictions = None
            if LOAD_CLUSTERING_RESULT:
                predictions = try_load_predictions(label)

            if predictions is None:
                predictions = get_clustering_preds(embeddings=embeddings, clustering_method=clustering_method)
                if SAVE_CLUSTERING_RESULT:
                    save_predictions(predictions, label)

            if isinstance(TRIM_RESULTS, int):
                df = df[:TRIM_RESULTS]
                predictions = predictions[:TRIM_RESULTS]

            clustering_statistics = ClusteringStatistics(df, predictions)
            mean_var = clustering_statistics.get_mean_cluster_variances()

            results[label] = mean_var
            best_mean_var = min(best_mean_var, mean_var['mean'])
            best_min_var = min(best_min_var, mean_var['min'])
            print(f"Label '{label}', elapsed {int(time.time() - start_time)} seconds")
            print(f'Mean variance statistics:\n{mean_var}')
            print("--------------------------------")

    results["best_mean_var"] = best_mean_var
    results["best_min_var"] = best_min_var

    # print_statistics(results)

    with open("experiments/results.json", "w") as fp:
        results = {key: str(value) for key, value in results.items()}
        json.dump(results, fp, indent=4)

    print("done!")


if __name__ == '__main__':
    main()
