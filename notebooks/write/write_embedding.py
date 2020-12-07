# %% Create embedding dict.

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

# %% Create label-embedding associations.

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

        predictor /= word_count

        # Create the response, which is just a one-hot encoding of the label.
        response = [0] * n_labels
        response[label] = 1

        embedding_matrix[label] += predictor
        # Assign the predictor to the correct row with numpy addition.
        predictor_matrix[doc_count] += predictor
        # Assign the response to the correct row with numpy addition.
        response_matrix[doc_count] += response

        label_counts[label] += 1

        # print(f"{label} -> {predictor[0]:.4f}, {predictor[1]:.4f}, {predictor[2]:.4f}...")

        doc_count += 1
        if doc_count >= total_n_docs:
            break
        elif doc_count % (2 ** 8) == 0:
            print(f"{doc_count}...")

for i in range(n_labels):
    embedding_matrix[i] /= label_counts[i]

# %% Dump matrices into the appropriate folder.

import os

resource_dir = "/Users/ericcarlson/Desktop/Personal Projects/basic-document-nlp/resources/glove"
glove_fname = os.path.join(resource_dir, "avg-embedding-matrix.npy")
predictor_fname = os.path.join(resource_dir, "predictor-matrix.npy")
response_fname = os.path.join(resource_dir, "response-matrix.npy")

np.save(glove_fname, embedding_matrix)
np.save(predictor_fname, predictor_matrix)
np.save(response_fname, response_matrix)
