from typing import List
import numpy as np
from numpy import ndarray
from sklearn.cluster import KMeans, DBSCAN, MeanShift
from sklearn.mixture import GaussianMixture
from bertopic import BERTopic


def get_clustering_preds(orig_explanations: List[str], embeddings: ndarray, clustering_method: str, num_clusters: int) -> ndarray:
    """
    Get clustering predictions for input embeddings using the specified clustering method.

    Args:
        orig_explanations (List[str]): Raw explanations - used in topic modeling.
        embeddings (ndarray): Input data for clustering.
        clustering_method (str): The clustering method to use (e.g., "KMEANS", "DBSCAN", "GMM").
        num_clusters (int): The number of clusters to create.

    Returns:
        ndarray: Cluster assignments or predictions for each data point.
    """
    # TODO: LDA can't handle negative entries, we skip it at the moment
    # if clustering_method == "LDA":
    #     # Latent Dirichlet Allocation
    #     lda = LatentDirichletAllocation(n_components=n_clusters, random_state=0)
    #     lda.fit(embeddings)
    #     predictions = lda.fit_transform(embeddings)
    print(f"Calculating predictions, data size: {len(embeddings)}")
    if clustering_method == "KMEANS":
        predictions = kmeans_clustering(embeddings, num_clusters)
    elif clustering_method == "DBSCAN":
        predictions = dbscan_clustering(embeddings)
    elif clustering_method == "GMM":
        predictions = gmm_clustering(embeddings, num_clusters)
    elif clustering_method == "MEANSHIFT":
        predictions = mean_shift_clustering(embeddings)
    elif clustering_method == "RANDOM":
        predictions = random_clustering(embeddings, num_clusters)
    elif clustering_method == "BERTOPIC":
        topic_model = BERTopic(min_topic_size=100)
        predictions, probs = topic_model.fit_transform(orig_explanations)
    else:
        raise NotImplementedError(f"Clustering method '{clustering_method}' not supported")

    return predictions


def kmeans_clustering(embeddings: ndarray, num_clusters: int) -> ndarray:
    """
    Perform K-Means clustering on input data.

    Args:
        embeddings (ndarray): Input data for clustering.
        num_clusters (int): The number of clusters to create.

    Returns:
        ndarray: Cluster assignments or predictions for each data point.
    """
    kmeans_model = KMeans(n_clusters=num_clusters)
    predictions = kmeans_model.fit_predict(embeddings)
    return predictions


def dbscan_clustering(embeddings: ndarray) -> ndarray:
    """
    Perform DBSCAN clustering on input data.

    Args:
        embeddings (ndarray): Input data for clustering.

    Returns:
        ndarray: Cluster assignments or predictions for each data point.
    """
    db_scan = DBSCAN(eps=0.5)  # TODO: increase epsilon to avoid unclustered data
    predictions = db_scan.fit_predict(embeddings)
    return predictions


def gmm_clustering(embeddings: ndarray, num_clusters: int) -> ndarray:
    """
    Perform Gaussian Mixture Model (GMM) clustering on input data.

    Args:
        embeddings (ndarray): Input data for clustering.
        num_clusters (int): The number of clusters to create.

    Returns:
        ndarray: Cluster assignments or predictions for each data point.
    """
    gmm = GaussianMixture(n_components=num_clusters)
    gmm.fit(embeddings)
    predictions = gmm.predict(embeddings)
    return predictions


def mean_shift_clustering(embeddings: ndarray) -> ndarray:
    """
    Perform Mean Shift clustering on input data.

    Args:
        embeddings (ndarray): Input data for clustering.

    Returns:
        ndarray: Cluster assignments or labels for each data point.
    """
    meanshift = MeanShift()
    meanshift.fit(embeddings)
    labels = meanshift.labels_
    return labels


def random_clustering(embeddings: ndarray, num_clusters: int) -> ndarray:
    """
    Perform random clustering on input data.

    Args:
        embeddings (ndarray): Input data for clustering.
        num_clusters (int): The number of clusters to create.

    Returns:
        ndarray: Random cluster assignments for each data point.
    """
    size = len(embeddings)

    # Generate random cluster assignments for each element in embeddings
    cluster_assignments = np.random.randint(0, num_clusters, size=size)

    return cluster_assignments