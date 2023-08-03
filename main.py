import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

df = pd.read_csv(f"scores_and_explanations.csv", sep=',')
#Our sentences we like to encode
sentences = df['explanation'].values

embeddings = model.encode(sentences)
np.save(embeddings)

