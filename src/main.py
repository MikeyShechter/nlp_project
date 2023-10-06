import json
import time

from src.clustering_analysis import *
from src.utils import *
from top2vec import Top2Vec


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
            clustering_statistics.layer_variance(predictions)
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


def top2vec_exp():
    # try pretrained model: 'universal-sentence-encoder'
    # Top2Vec get's argument speed=fast-learn/learn/deep-learn
    documents = list(df['explanation'].values)
    model = Top2Vec(documents, topic_merge_delta=0.01)
    model.get_num_topics()
    topic_words, word_scores, topic_nums = model.get_topics()
    model.search_documents_by_topic(0, model.get_topic_sizes())

    docs_by_topic = [model.search_documents_by_topic(
        i, model.get_topic_sizes()[0][i]) for i in range(model.get_num_topics())]
    docs_topic0 = docs_by_topic[0][0]
    model_topic0 = Top2Vec(docs_topic0)

    filtered_df_model0 = df.iloc[docs_by_topic[0][2]].reset_index(drop=True)
    docs_by_topic0 = [model_topic0.search_documents_by_topic(
        i, model_topic0.get_topic_sizes()[0][i]) for i in range(model_topic0.get_num_topics())]

    filtered_df_model0.iloc[docs_by_topic0[0][2]]['layer'].value_counts()


if __name__ == '__main__':
    main()
