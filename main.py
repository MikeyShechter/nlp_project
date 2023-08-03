import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.datasets import make_multilabel_classification

LOAD = True
NUM_CLUSTERS = 48 # hypothesis: number of layers
CLUSTERING_METHOD = "KMEANS"

# region Load Data
df = pd.read_csv(f"scores_and_explanations.csv", sep=',')
df = df[df.score >= 0.5]  # filtered scores
if LOAD:
    embeddings = np.load("neurons_explanations_embeddings_0.5.npy")
else:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sentences = df['explanation'].values
    embeddings = model.encode(sentences, show_progress_bar=True)
    np.save("neurons_explanations_embeddings_0.5", embeddings)
# endregion

# region Clustering
if CLUSTERING_METHOD == "LDA":
    # Latent Dirichlet Allocation
    lda = LatentDirichletAllocation(n_components=NUM_CLUSTERS, random_state=0)
    lda.fit(embeddings)
    predictions = lda.fit_transform(embeddings)
if CLUSTERING_METHOD == "KMEANS":
    # Create and fit a K-means clustering model
    kmeans_model = KMeans(n_clusters=NUM_CLUSTERS)
    predictions = kmeans_model.fit_predict(embeddings)
# endregion

# region Show Histograms
counts = df.layer.value_counts().sort_index()
fig, axes = plt.subplots(8, 6, figsize=(18, 24)) # 8 rows, 6 columns
for i in range(8):
    for j in range(6):
        index = i * 6 + j
        count_pred_i = df.iloc[predictions == index].layer.value_counts().sort_index()
        df_counts = pd.DataFrame({"orig_count": counts})
        df_counts['preds'] = count_pred_i
        df_counts = df_counts.fillna(0)
        count_pred_i_normalized = df_counts.preds / df_counts.orig_count

        axes[i, j].plot(range(NUM_CLUSTERS), count_pred_i_normalized.values)
        axes[i, j].set_title(f'Plot {index}')

plt.tight_layout()
plt.show()

# for i in range(5): #range(NUM_CLUSTERS):
#     count_pred_i = df.iloc[predictions == i].layer.value_counts().sort_index()
#     df_counts = pd.DataFrame({"orig_count": counts})
#     df_counts['preds'] = count_pred_i
#     df_counts = df_counts.fillna(0)
#     count_pred_i_normalized = df_counts.preds / df_counts.orig_count
#
#     plt.plot(range(NUM_CLUSTERS), count_pred_i_normalized.values)
#
# endregion
# plt.show()
print("finished")


