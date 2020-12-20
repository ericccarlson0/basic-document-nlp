import csv
import os
import matplotlib.pyplot as plt
import numpy as np
import spacy
import sys

from configparser import RawConfigParser
from os import path
from language_prep import embeddings
from language_prep.preprocess import is_viable_token
from sklearn.manifold import TSNE

# A method to write GloVe embeddings which results in three saved numpy arrays:
# An array of predictors, X, which are the average embeddings of documents.
# An array of responses, Y, which are the labels of documents.
# An array of average embeddings over all documents for each label. This should be much smaller than (1).
def write_glove_embedding(ocr_csv_dir: str, embedding_dir: str, save_to_dir: str, n_labels: int, n_dims: int,
                          max_doc_count: int, vis: bool = False):
    embedding_dict = embeddings.create_glove_dict(dir_=embedding_dir)
    nlp = spacy.load('en_core_web_sm')

    predictor_matrix = np.zeros((max_doc_count, n_dims))
    response_matrix = np.zeros((max_doc_count, n_labels))
    avg_embedding_matrix = np.zeros((n_labels, n_dims))

    label_counts = [0] * n_labels
    doc_count = 0

    with open(ocr_csv_dir, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',', quotechar='*')

        for row in csv_reader:
            label = int(row[0])
            doc = nlp(row[1])

            avg_embedding = np.zeros(n_dims)

            word_count = 0
            for token in doc:
                if is_viable_token(token, embedding_dict):
                    avg_embedding += embedding_dict[token.text]
                    word_count += 1
            if word_count != 0:
                avg_embedding /= word_count

            avg_embedding_matrix[label] += avg_embedding
            predictor_matrix[doc_count] += avg_embedding
            response = np.zeros(n_labels)
            response[label] = 1
            response_matrix[doc_count] += response

            label_counts[label] += 1
            doc_count += 1

            if doc_count >= max_doc_count:
                break
            if doc_count % (2 ** 10) == 0:
                print(f"{doc_count}...")

    # Remember to normalize embeddings.
    for i in range(n_labels):
        avg_embedding_matrix[i] /= label_counts[i]

    X = predictor_matrix[:, :]
    Y = np.argmax(response_matrix, axis=1)

    if vis:
        scatter_predictors(X=X, Y=Y)

    np.save(path.join(save_to_dir, "avg-embedding-matrix.npy"), avg_embedding_matrix)
    np.save(path.join(save_to_dir, "predictor-matrix.npy"), predictor_matrix)
    np.save(path.join(save_to_dir, "response-matrix.npy"), response_matrix)

# A useful visualization to see clustering and structure in lower dimensions.
def scatter_predictors(X: np.ndarray, Y: np.ndarray):
    tsne = TSNE(n_components=2, random_state=53)
    X = tsne.fit_transform(X)

    # TODO: make this less "static"?
    label_color_pairs = (
        (0, 'k'), (1, 'r'), (2, 'm'), (3, 'b'), (4, 'c'), (5, 'g'), (6, 'y')
    )

    scatterplots = []
    for label, color in label_color_pairs:
        scatterplots.append(
            scatter_on_label(X=X, Y=Y, label=label, color=color, max_count=96)
        )

    plt.legend(scatterplots,
               [str(i) for i in range(7)],
               scatterpoints=1,
               loc='upper right')

    plt.show()

def scatter_on_label(X: np.ndarray, Y: np.ndarray, label: int, color: str = 'k', max_count: int = sys.maxsize):
    count = 0
    d1 = []
    d2 = []

    for i in range(X.shape[0]):
        if label == Y[i]:
            d1.append(X[i, 0])
            d2.append(X[i, 1])

            count += 1
            if count >= max_count:
                break

    return plt.scatter(d1, d2, color=color)

if __name__ == '__main__':
    properties_dir = path.normpath(path.join(os.getcwd(), "../../resources/properties.ini"))
    config = RawConfigParser()
    config.read(properties_dir)

    glove_dir = config.get("Embeddings", "glove.directory")

    glove_crawl_dir = path.join(glove_dir, "common-crawl/glove.42B.300d.txt")
    glove_twitter_dir = path.join(glove_dir, "twitter/glove.twitter.27B.200d.txt")
    glove_standard_dir = path.join(glove_dir, "standard/glove.6B.300d.txt")

    cdip_ocr_dir = config.get("Datasets", "ocrDataset.path")
    n_labels = int(config.get("Datasets", "dataset.nLabels"))

    for embedding_dir, subfolder, n_dims in (
        (glove_crawl_dir, "common-crawl", 300),
        (glove_standard_dir, "standard", 300),
        (glove_twitter_dir, "twitter", 200)
    ):
        write_glove_embedding(
            ocr_csv_dir=cdip_ocr_dir,
            embedding_dir=embedding_dir,
            save_to_dir=path.join(glove_dir, subfolder),
            n_labels=n_labels,
            n_dims=n_dims,
            max_doc_count=4096,
        )
