import numpy as np
from numpy import ndarray
from sklearn.cluster import KMeans, DBSCAN, MeanShift
from sklearn.mixture import GaussianMixture


def get_clustering_preds(embeddings, clustering_method, num_clusters) -> ndarray:
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
    else:
        raise NotImplementedError(f"Clustering method '{clustering_method}' not supported")

    return predictions


def kmeans_clustering(embeddings, num_clusters) -> ndarray:
    kmeans_model = KMeans(n_clusters=num_clusters)
    predictions = kmeans_model.fit_predict(embeddings)
    return predictions


def dbscan_clustering(embeddings) -> ndarray:
    db_scan = DBSCAN(eps=0.5)  # TODO: increase epsilon to avoid unclustered data
    predictions = db_scan.fit_predict(embeddings)
    return predictions


def gmm_clustering(embeddings, num_clusters) -> ndarray:
    gmm = GaussianMixture(n_components=num_clusters)
    gmm.fit(embeddings)
    predictions = gmm.predict(embeddings)

    return predictions


def mean_shift_clustering(embeddings) -> ndarray:
    # Create a Mean Shift clustering model
    meanshift = MeanShift()

    # Fit the model to your data
    meanshift.fit(embeddings)

    # Get cluster assignments for each data point
    labels = meanshift.labels_

    return labels


def random_clustering(embeddings, num_clusters) -> ndarray:
    size = len(embeddings)

    # Generate random cluster assignments for each element in embeddings
    cluster_assignments = np.random.randint(0, num_clusters, size=size)

    return cluster_assignments

