import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.datasets import make_multilabel_classification

LOAD = False
NUM_LAYERS = 48
CLUSTERING_METHOD = "KMEANS"

if LOAD:
    embeddings = np.load("neurons_explanations_embeddings.npy")
else:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    df = pd.read_csv(f"scores_and_explanations.csv", sep=',')
    # Our sentences we like to encode
    sentences = df['explanation'].values
    embeddings = model.encode(sentences, show_progress_bar=True)
    np.save("neurons_explanations_embeddings", embeddings)

if CLUSTERING_METHOD == "LDA":
    ## Latent Dirichlet Allocation
    lda = LatentDirichletAllocation(n_components=5, random_state=0)
    lda.fit(embeddings)
    clusters = lda.transform(embeddings)
if CLUSTERING_METHOD == "KMEANS":
    # Create and fit a K-means clustering model
    kmeans_model = KMeans(n_clusters=NUM_LAYERS)
    predictions = kmeans_model.fit_predict(embeddings)

print("finished")
