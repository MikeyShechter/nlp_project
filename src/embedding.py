import numpy as np
import spacy
import gensim.downloader
from sentence_transformers import SentenceTransformer

from src.utils import load_df

SEQUENCE_TRANSFORMER_PATH = "data/neurons_explanations_embeddings_full.npy"
WORD2VEC_PATH = "data/neurons_explanations_embeddings_full_word2vec.npy"

TRANSFORMER = "TRANSFORMER"
WORD2VEC = "WORD2VEC"


def get_embeddings(df, embedding_method, create_new=False):
    if embedding_method == TRANSFORMER:
        embeddings = load_sequence_transformer_embeddings(df, create_new)
    elif embedding_method == WORD2VEC:
        embeddings = load_word2vec_embeddings(df, create_new)

    else:
        raise NotImplementedError(f"Embedding method '{embedding_method}' not supported")
    return embeddings


def load_sequence_transformer_embeddings(df, create_new=False, model_name='all-mpnet-base-v2'):
    """Other model: all-MiniLM-L6-v2"""
    if create_new:
        model = SentenceTransformer(model_name)
        sentences = df['explanation'].values
        print("generating embeddings")
        embeddings = model.encode(sentences, show_progress_bar=True)
        np.save(SEQUENCE_TRANSFORMER_PATH, embeddings)
    else:
        full_embeddings = np.load(SEQUENCE_TRANSFORMER_PATH)
        embeddings = full_embeddings[df.index]

    return embeddings


def load_word2vec_embeddings(df, create_new=False):
    def sentence_embedding(sentence):
        doc = nlp(sentence)
        word_embeddings = [model[word.text] for word in doc if word.text in model]
        if len(word_embeddings) == 0:
            # Handle the case where none of the words are in the Word2Vec model's vocabulary
            return np.zeros(model.vector_size)
        return np.mean(word_embeddings, axis=0)

    if create_new:
        nlp = spacy.load("en_core_web_sm")
        model = gensim.downloader.load("word2vec-google-news-300")
        sentences = df['explanation'].values

        print("generating embeddings")
        embeddings = np.array([sentence_embedding(sentence) for sentence in sentences])

        print(f"saving to '{WORD2VEC_PATH}'")
        np.save(WORD2VEC_PATH, embeddings)
    else:
        full_embeddings = np.load(WORD2VEC_PATH)
        embeddings = full_embeddings[df.index]

    return embeddings


def main():
    embedding_method = None
    while embedding_method not in [TRANSFORMER, WORD2VEC]:
        embedding_method = input(
            f"Please enter embedding type to generate ('{TRANSFORMER}' or '{WORD2VEC}')").strip().upper()
    df = load_df()
    get_embeddings(df, embedding_method=embedding_method, create_new=True)


if __name__ == '__main__':
    main()
