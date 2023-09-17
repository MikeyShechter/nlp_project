import json

from src.utils import *

CLUSTERING_METHODS = ["KMEANS", "DBSCAN"]
PERCENTILES = [0, 0.2, 0.5, 0.8, 0.9]
SAVE_CLUSTERING_RESULT = True
LOAD_CLUSTERING_RESULT = True


def main():
    results = dict()
    best_mean_var, best_min_var = float('inf'), float('inf')
    # TODO: run also over embedding methods?
    for clustering_method in CLUSTERING_METHODS:
        for percentile in PERCENTILES:
            df = load_df(percentile=percentile)
            embeddings = load_embeddings(df)

            label = f"{clustering_method=}_{percentile=}"
            print(f"Clustering label '{label}'")

            predictions = None
            if LOAD_CLUSTERING_RESULT:
                predictions = try_load_predictions(label)

            if not predictions:
                predictions = get_clustering_preds(embeddings=embeddings, clustering_method=clustering_method)
                if SAVE_CLUSTERING_RESULT:
                    save_predictions(predictions, label)

            df['cluster_preds'] = predictions.result

            unique_values, counts = np.unique(predictions.result, return_counts=True)

            mean_var = mean_cluster_variances(df, predictions.num_clusters)
            results[label] = str(mean_var)
            best_mean_var = min(best_mean_var, mean_var['mean'])
            best_min_var = min(best_min_var, mean_var['min'])
            print(f"Label '{label}' statistics: {mean_var}")
            print("--------------------------------")

    results["best_mean_var"] = best_mean_var
    results["best_min_var"] = best_min_var

    with open("experiments/results.json", "w") as fp:
        json.dump(results, fp)

    print("done!")


if __name__ == '__main__':
    main()
