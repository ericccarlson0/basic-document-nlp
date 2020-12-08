#%% Create embedding dict.

# TODO: remove stop-words and non-words

# 1) Initialize a matrix with N rows and M columns, where N is the number of labels and M is the number of
#   word embedding dimensions.
# 2) Open the CSV file whose rows have (1) the label and (2) the cleaned-up OCR text.
# 3) For each row of the CSV file... Iterate over each word. Get that word's embedding and add it to th
#  temporary vector containing the sum of the document's word embeddings. Then, divide that vector by the
#  number of that were encountered, to produce an "average embedding".
#  Still iterating over each row, add the vector to the row of the association matrix that corresponds to
#  the document's label. For instance, if the document's label is 3, assoc_matrix[3] += assoc_row.
# 4) For each label, divide the corresponding row of the association matrix by the number of times that row
#   was encountered. This is supposed to produce an average embedding for documents with that label.
# 5) Write the association matrix to a file where it can be accessed later.

from language_prep import embeddings

print("Creating embedding dict...")
embedding_dict = embeddings.create_embedding_dict()
print("Done.")

#%% Create label-embedding associations.

import csv
import spacy
import numpy as np

csv_dir = "/Users/ericcarlson/Desktop/Datasets/csv/CDIP_OCR.csv"
nlp = spacy.load('en_core_web_sm')

n_labels = 16
n_dims = 300
total_n_docs = 4096
doc_count = 0

predictor_matrix = np.zeros((total_n_docs, n_dims))
response_matrix = np.zeros((total_n_docs, n_labels))

embedding_matrix = np.zeros((n_labels, n_dims))
label_counts = [0] * n_labels

# TODO: move?
def is_viable(token_):
    if token_.is_stop:
        return False
    elif '_' in token_.text:
        return False
    else:
        return token_.text in embedding_dict


with open(csv_dir, 'r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',', quotechar='*')

    for row in csv_reader:
        label = int(row[0])
        doc = nlp(row[1])

        # Create the predictor, which is an average of all word embeddings.
        predictor = np.zeros(n_dims)

        word_count = 0
        for token in doc:
            if is_viable(token):
                predictor += embedding_dict[token.text]
                word_count += 1

        if word_count != 0:
            predictor /= word_count

        embedding_matrix[label] += predictor
        # Assign the predictor to the correct row with numpy addition.
        predictor_matrix[doc_count] += predictor
        # Assign the response to the correct row with numpy addition.
        response = np.zeros(n_labels)
        response[label] = 1
        response_matrix[doc_count] += response

        label_counts[label] += 1
        doc_count += 1

        if doc_count >= total_n_docs:
            break
        elif doc_count % (2 ** 8) == 0:
            print(f"{doc_count}...")
            print(f"{predictor[0]: .3f}, {predictor[1]: .3f}, {predictor[2]: .3f}, ...")

for i in range(n_labels):
    embedding_matrix[i] /= label_counts[i]

#%% Visualize some predictors.

import matplotlib.pyplot as plt
import sys

from sklearn.manifold import TSNE

sample_n = 1024

X = predictor_matrix[:sample_n, :]
Y = np.argmax(
    response_matrix[:sample_n, :],
    axis=1
)

tsne = TSNE(n_components=2, random_state=53)
X = tsne.fit_transform(X)

def scatter_conditionally(label_n: int, color: str = 'k', max_count: int = sys.maxsize):
    count = 0
    d1 = []
    d2 = []

    for i in range(X.shape[0]):
        if label_n == Y[i]:
            d1.append(X[i, 0])
            d2.append(X[i, 1])

            count += 1
            if count >= max_count:
                break

    return plt.scatter(d1, d2, color=color)

label_color_pairs = (
    (0, 'k'), (1, 'r'), (2, 'm'), (3, 'b'), (4, 'c'), (5, 'g'), (6, 'y')
)

scatterplots = []
for l, c in label_color_pairs:
    scatterplots.append(
        scatter_conditionally(label_n=l, color=c, max_count=96)
    )

plt.legend(scatterplots,
           ('0', '1', '2', '3', '4', '5', '6'),
           scatterpoints=1,
           loc='upper right')

# plt.scatter(X[:, 0], X[:, 1], color='k')
# for label, d1, d2 in zip(Y, X[:, 0], X[:, 1]):
#     plt.annotate(str(label), xy=(d1, d2), xytext=(0, 0), textcoords="offset points")

plt.show()

#%% Dump matrices into the appropriate folder.

import os

resource_dir = "/Users/ericcarlson/Desktop/Personal Projects/basic-document-nlp/resources/glove"
glove_fname = os.path.join(resource_dir, "avg-embedding-matrix.npy")
predictor_fname = os.path.join(resource_dir, "predictor-matrix.npy")
response_fname = os.path.join(resource_dir, "response-matrix.npy")

np.save(glove_fname, embedding_matrix)
np.save(predictor_fname, predictor_matrix)
np.save(response_fname, response_matrix)
