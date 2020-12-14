import os
import re
import numpy as np
import matplotlib.pyplot as plt

from notebooks.util.logging import list_to_string
from scipy import spatial
from time import time
from typing import Dict
from sklearn.manifold import TSNE

num_regex = re.compile('[\d-]+')
project_dir = "/Users/ericcarlson/Desktop/Personal Projects/basic-document-nlp"
glove_dir = os.path.join(project_dir, "resources", "glove")

glove_crawl_dir = os.path.join(glove_dir, "common-crawl", "glove.42B.300d.txt")
glove_twitter_dir = os.path.join(glove_dir, "twitter", "glove.27B.200d.txt")
glove_standard_dir = os.path.join(glove_dir, "standard", "glove.6B.300d.txt")

def create_embedding_dict(dir_: str = "") -> Dict:
    embedding_dict = {}
    count = 0

    with open(dir_, 'r', encoding='utf-8') as embedding_file:
        for line in embedding_file:
            vals = line.split()
            word = vals[0]

            dex = 1
            while num_regex.match(vals[dex]) is None:
                # print(vals[dex])
                dex += 1

            vector = np.asarray(vals[1:], "float64")
            embedding_dict[word] = vector

            count += 1

    return embedding_dict

def create_glove_crawl_dict():
    return create_embedding_dict(dir_=glove_crawl_dir)

def create_glove_twitter_dict():
    return create_embedding_dict(dir_=glove_twitter_dir)

def create_glove_standard_dict():
    return create_embedding_dict(dir_=glove_standard_dir)


def get_nearest_embeddings(embedding_dict: Dict, embedding):
    return sorted(embedding_dict.keys(),
                  key=lambda word: spatial.distance.euclidean(embedding_dict[word], embedding))

def show_tsne(embedding_dict: Dict, max_count: int = 128):
    tsne = TSNE(n_components=2, random_state=53)

    count = 0
    words = []
    vectors = []
    for word in embedding_dict.keys():
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

def test():
    t0 = time()
    e_dict = create_glove_twitter_dict()
    print(f"{time() - t0: .4f} secs")

    word = "ball"
    nearest = get_nearest_embeddings(e_dict, e_dict[word])[1:6]
    print(list_to_string(nearest))

    show_tsne(e_dict)
