import json

from src.utils import *


def main():
    results = dict()
    best_mean_var, best_min_var = float('inf'), float('inf')
    # TODO run also over embedding methods?
    for clustering_method in ["KMEANS"]:
        for n_clusters in [24, 48, 96, 192]:
            for percentile in [0, 0.2, 0.5, 0.8, 0.9]:
                current = f"{clustering_method=}_{n_clusters=}_{percentile=}"
                df = load_df(percentile=percentile)
                embeddings = load_embeddings(df)
                df['cluster_preds'] = get_clustering_preds(embeddings=embeddings, clustering_method=clustering_method, n_clusters=n_clusters)
                mean_var = mean_cluster_variances(df, n_clusters)
                results[current] = str(mean_var)
                best_mean_var = min(best_mean_var, mean_var['mean'])
                best_min_var = min(best_min_var, mean_var['min'])
                print(f"{current} statistics: {mean_var}")

    results["best_mean_var"] = best_mean_var
    results["best_min_var"] = best_min_var
    with open("../experiments/results.json", "w") as fp:
        json.dump(results, fp)

    print("done!")


if __name__ == '__main__':
    main()
