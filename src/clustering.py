from typing import List
import numpy as np
from numpy import ndarray
from sklearn.cluster import KMeans, DBSCAN, MeanShift
from sklearn.metrics import pairwise_distances_argmin_min
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
        predictions = bertopic_clustering(orig_explanations)
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
    x = db_scan.components_
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

def bertopic_clustering(sentences: list[str]) -> list[int]:
    """
    Perform BERTopic clustering on input data.

    Args:
        sentences (ndarray): Input data for clustering.

    Returns:
        ndarray: Cluster assignments or labels for each data point.
    """
    topic_model = BERTopic(min_topic_size=100)
    predictions, probs = topic_model.fit_transform(sentences)

    return predictions

def assign_unclustered_points_to_closest_cluster(embeddings: ndarray, predictions: ndarray):
    """
    Assign unclustered points (labeled as -1) to the nearest cluster based on their embeddings.

    Parameters:
    embeddings (ndarray): The embeddings of data points.
    predictions (ndarray): The cluster labels assigned to each data point, including -1 for unclustered points.

    Returns:
    ndarray: Updated predictions with unclustered points reassigned to the nearest clusters.
    """
    # Check if there are any unclustered points (-1), and return immediately if there are none
    if -1 not in predictions:
        return predictions

    # Find the unique cluster labels (excluding noise points labeled as -1)
    unique_labels = np.unique(predictions)
    unique_labels = unique_labels[unique_labels != -1]

    # Calculate the cluster centers (centroids)
    cluster_centers = []
    for label in unique_labels:
        cluster_points = embeddings[predictions == label]
        cluster_center = np.mean(cluster_points, axis=0)
        cluster_centers.append(cluster_center)

    # Find unclustered points
    unclustered_indices = np.where(predictions == -1)[0]

    distances = pairwise_distances_argmin_min(embeddings[unclustered_indices], np.array(cluster_centers))

    # Iterate through unclustered points and assign them to the nearest cluster center
    for i, idx in enumerate(unclustered_indices):
        nearest_cluster = distances[0][i]
        predictions[idx] = unique_labels[nearest_cluster]

    return predictions
