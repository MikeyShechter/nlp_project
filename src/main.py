import json
import time
from itertools import product

from utils import *
from embedding import *
from clustering_analysis import *
from top2vec import Top2Vec

# EMBEDDING_METHODS = [TRANSFORMER, WORD2VEC]
EMBEDDING_METHODS = [TRANSFORMER]
# CLUSTERING_METHODS = ["KMEANS", "DBSCAN", "GMM", "MEANSHIFT", "RANDOM"]
CLUSTERING_METHODS = ["KMEANS", "RANDOM"]
# PERCENTILES = [0, 0.5, 0.9, 0.95, 0.99]
PERCENTILES = [0, 0.9]
SAVE_PREDICTIONS = True
LOAD_PERDICTIONS = True
TRIM_DF = None  # Set an integer to take first K entries in the df


def main():
    clustering_stats = dict()

    for embedding_method, clustering_method, percentile in product(EMBEDDING_METHODS, CLUSTERING_METHODS, PERCENTILES):
        start_time = time.time()
        df = load_df(percentile=percentile)
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
            predictions = get_clustering_preds(embeddings=embeddings, clustering_method=clustering_method,
                                               num_clusters=num_clusters)
            if SAVE_PREDICTIONS and TRIM_DF is None:
                save_predictions(predictions, label)

        clustering_statistics = ClusteringStatistics(df, predictions, label)
        clustering_stats[label] = clustering_statistics

        clustering_statistics.layer_variance(predictions)
        clustering_statistics.plot_clusters_scores_boxplot()
        print(f"Label '{label}', elapsed {int(time.time() - start_time)} seconds")
        # print(f'Cluster's variance statistics:\n{cluster_var_summary}')
        print("--------------------------------")

    save_results(clustering_stats)
    print_statistics(clustering_stats)

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
