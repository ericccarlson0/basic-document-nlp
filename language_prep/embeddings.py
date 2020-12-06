import random
import re
import sys
import numpy as np
import matplotlib.pyplot as plt

from scipy import spatial
from time import time
from typing import Dict
from sklearn.manifold import TSNE

num_regex = re.compile('[\d-]+')
glove_dir = "/Users/ericcarlson/Desktop/Personal Projects/basic-document-nlp/resources/glove.42B.300d.txt"

def create_embedding_dict(max_count: int = sys.maxsize) -> Dict:
    embedding_dict = {}
    count = 0

    with open(glove_dir, 'r', encoding="utf-8") as embedding_file:
        for line in embedding_file:
            vals = line.split()
            word = vals[0]

            dex = 1
            while num_regex.match(vals[dex]) is None:
                print(vals[dex])
                dex += 1

            # TODO: correct datatype?
            vector = np.asarray(vals[1:], "float32")
            embedding_dict[word] = vector

            count += 1
            if count >= max_count:
                break

    return embedding_dict

def get_nearest_embeddings(embedding_dict: Dict, embedding):
    return sorted(embedding_dict.keys(),
                  key=lambda word: spatial.distance.euclidean(embedding_dict[word], embedding))

def show_tsne(embedding_dict: Dict, max_count: int = 128):
    tsne = TSNE(n_components=2, random_state=47)

    words = list(embedding_dict.keys())
    random.shuffle(words)
    words = words[:max_count]
    vectors = [embedding_dict[word] for word in words]

    Y = tsne.fit_transform(vectors)

    plt.scatter(Y[:, 0], Y[:, 1])
    for word, x, y in zip(words, Y[:, 0], Y[:, 1]):
        plt.annotate(word, xy=(x, y), xytext=(0, 0), textcoords="offset points")

    plt.show()


if __name__ == '__main__':
    t0 = time()
    e_dict = create_embedding_dict()
    print(f"{time() - t0} secs")

    nearest_ball = get_nearest_embeddings(e_dict, e_dict['ball'])[1:6]
    for i in range(5):
        print(f"{i}: {nearest_ball[i]}")

    show_tsne(e_dict)
