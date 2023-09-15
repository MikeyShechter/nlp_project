import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import DBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation


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


def get_clustering_preds(embeddings, clustering_method="KMEANS", n_clusters=48):
    if clustering_method == "LDA":
        # Latent Dirichlet Allocation
        lda = LatentDirichletAllocation(n_components=n_clusters, random_state=0)
        lda.fit(embeddings)
        predictions = lda.fit_transform(embeddings)
    elif clustering_method == "KMEANS":
        # Create and fit a K-means clustering model
        kmeans_model = KMeans(n_clusters=n_clusters)
        predictions = kmeans_model.fit_predict(embeddings)
    elif clustering_method == "DBSCAN":
        db_scan = DBSCAN()
        predictions = db_scan.fit_predict(embeddings)
    else:
        raise NotImplementedError("Not supported clustering method")

    return predictions


# Cluster sizes:
# clusters_sizes = pd.Series(predictions).value_counts().sort_index()
# clusters_sizes.describe()


def layer_variance(df, n_clusters=48):
    layer_variances = []
    for i in range(n_clusters):
        n_points_per_cluster = df[df.layer == i].kmeans_preds.value_counts()
        var = n_points_per_cluster.var()  # Larger is better
        layer_variances.append(var)
    layer_variance_mean = np.array(layer_variances).mean()
    print(f"{layer_variance_mean=}")


def mean_cluster_variances(df, n_clusters):
    cluster_vars = []
    for i in range(n_clusters):
        cluster_var = df.loc[df['cluster_preds'] == i]['layer'].var()
        cluster_vars.append(cluster_var)

    return pd.Series(cluster_vars).describe()


def plot_cluster_hists(df, nrows, ncols):
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 24))  # 8 rows, 6 columns

    for i in range(nrows):
        for j in range(ncols):
            index = i * ncols + j
            count_pred_i = df.iloc[df['cluster_preds'] == index].layer.value_counts().sort_index()
            axes[i, j].plot(range(48), count_pred_i.values)
            axes[i, j].set_title(f'Plot {index}')

    plt.tight_layout()
    plt.show()
