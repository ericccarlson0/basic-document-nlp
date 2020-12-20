import matplotlib.pyplot as plt
import numpy as np
import os

from configparser import RawConfigParser
from notebooks.util.logging import list_to_string
from os import path
from scipy import spatial
from time import time
from typing import Dict
from sklearn.manifold import TSNE

properties_dir = path.normpath(path.join(os.getcwd(), "../resources/properties.ini"))
config = RawConfigParser()
config.read(properties_dir)

glove_crawl_dir = config.get("Embeddings", "crawlEmbeddings")
glove_standard_dir = config.get("Embeddings", "standardEmbeddings")
glove_twitter_dir = config.get("Embeddings", "twitterEmbeddings")

print(os.getcwd())

def create_glove_dict(dir_: str) -> Dict:
    print("Creating GloVe embedding dict (this may take a minute).")
    embedding_dict = {}

    with open(dir_, 'r', encoding='utf-8') as embedding_file:
        for line in embedding_file:
            vals = line.split()
            word = vals[0]
            vector = np.asarray(vals[1:], "float64")

            embedding_dict[word] = vector

    return embedding_dict

def create_glove_crawl_dict():
    return create_glove_dict(dir_=glove_crawl_dir)

def create_glove_twitter_dict():
    return create_glove_dict(dir_=glove_twitter_dir)

def create_glove_standard_dict():
    return create_glove_dict(dir_=glove_standard_dir)

# Just here to evaluate embeddings.
def get_nearest_embeddings(embedding_dict: Dict, embedding):
    return sorted(embedding_dict.keys(),
                  key=lambda word: spatial.distance.euclidean(embedding_dict[word], embedding))

# Just here to evaluate embeddings.
def show_tsne(embedding_dict: Dict, max_count: int = 128):
    tsne = TSNE(n_components=2, random_state=53)

    count = 0
    words = []
    vectors = []
    for word in embedding_dict:
        words.append(word)
        vectors.append(embedding_dict[word])

        if count >= max_count:
            break
        count += 1

    X = tsne.fit_transform(vectors)

    plt.scatter(X[:, 0], X[:, 1])
    for word, d1, d2 in zip(words, X[:, 0], X[:, 1]):
        plt.annotate(word, xy=(d1, d2), xytext=(0, 0), textcoords="offset points")

    plt.show()

if __name__ == '__main__':
    t0 = time()
    e_dict = create_glove_twitter_dict()
    print(f"{time() - t0: .4f} secs")

    word = "ball"
    nearest = get_nearest_embeddings(e_dict, e_dict[word])[1:6]
    print(list_to_string(nearest))

    show_tsne(e_dict)