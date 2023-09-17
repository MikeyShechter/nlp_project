from numpy import ndarray
from sklearn.cluster import KMeans, DBSCAN


class ClusteredData:
    def __init__(self, method: str, result: ndarray):
        self.method = method
        self.result = result
        self.num_clusters = max(result) + 1


def get_clustering_preds(embeddings, clustering_method) -> ClusteredData:
    # TODO: LDA can't handle negative entries, we skip it at the moment
    # if clustering_method == "LDA":
    #     # Latent Dirichlet Allocation
    #     lda = LatentDirichletAllocation(n_components=n_clusters, random_state=0)
    #     lda.fit(embeddings)
    #     predictions = lda.fit_transform(embeddings)
    if clustering_method == "KMEANS":
        predictions = kmeans_clustering(embeddings)
    elif clustering_method == "DBSCAN":
        predictions = dbscan_clustering(embeddings)
    else:
        raise NotImplementedError(f"Clustering method '{clustering_method}' not supported")

    return predictions


def kmeans_clustering(embeddings) -> ClusteredData:
    # TODO: Run over a few n's?
    for num_clusters in [48]:  # [24, 48, 96, 192]:
        kmeans_model = KMeans(n_clusters=num_clusters)
        predictions = kmeans_model.fit_predict(embeddings)
        return ClusteredData("KMEANS", num_clusters, predictions)


def dbscan_clustering(embeddings) -> ClusteredData:
    db_scan = DBSCAN()
    predictions = db_scan.fit_predict(embeddings)
    num_clusters = max(predictions)
    return ClusteredData("DBSCAN", num_clusters, predictions)
