from numpy import ndarray
from sklearn.cluster import KMeans, DBSCAN, MeanShift
from sklearn.mixture import GaussianMixture


def get_clustering_preds(embeddings, clustering_method) -> ndarray:
    # TODO: LDA can't handle negative entries, we skip it at the moment
    # if clustering_method == "LDA":
    #     # Latent Dirichlet Allocation
    #     lda = LatentDirichletAllocation(n_components=n_clusters, random_state=0)
    #     lda.fit(embeddings)
    #     predictions = lda.fit_transform(embeddings)
    print(f"Calculating predictions, data size: {len(embeddings)}")
    if clustering_method == "KMEANS":
        predictions = kmeans_clustering(embeddings)
    elif clustering_method == "DBSCAN":
        predictions = dbscan_clustering(embeddings)
    elif clustering_method == "GMM":
        predictions = gmm_clustering(embeddings)
    elif clustering_method == "MEANSHIFT":
        predictions = mean_shift_clustering(embeddings)
    else:
        raise NotImplementedError(f"Clustering method '{clustering_method}' not supported")

    return predictions


def kmeans_clustering(embeddings) -> ndarray:
    # TODO: Run over a few n's?
    for num_clusters in [48]:  # [24, 48, 96, 192]:
        kmeans_model = KMeans(n_clusters=num_clusters)
        predictions = kmeans_model.fit_predict(embeddings)
        return predictions


def dbscan_clustering(embeddings) -> ndarray:
    db_scan = DBSCAN()
    predictions = db_scan.fit_predict(embeddings)
    return predictions


def gmm_clustering(embeddings) -> ndarray:
    gmm = GaussianMixture(n_components=48)
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

