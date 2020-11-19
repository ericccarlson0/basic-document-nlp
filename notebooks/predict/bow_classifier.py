import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pytesseract

from typing import List
from nlp import preprocess
from ocr import skew

# %% Get JSON map of word-label counts.

file = open("../../resources/CDIP_OCR.json")
word_label_counts = json.load(file)
print(f"Words in dict: {len(word_label_counts)}")

# %% Define functions to get label distributions and prediction distributions.

num_labels = 16

def get_label_dist(w: str) -> List:
    dist = [0] * num_labels

    if w in word_label_counts:
        label_counts = word_label_counts[w]
        for label in label_counts:
            # Convert both the label and the label counts to int in case.
            dist[int(label)] += int(label_counts[label])

        count = sum(dist)
        dist = [d / count for d in dist]

    return dist

document_word_counts = []
zero_word_counts = 0

# Predicts a document by iterating over every word found in the document. If that word is not in the
# dictionary, it is skipped. If it is present, it is looked up and a probability distribution is
# produced from the counts in the dictionary. All of these probabilities, for each word in the doc,
# must be combined somehow.
def predict_doc_dist(array: np.ndarray) -> List:
    try:
        rot_array, _ = skew.correct_skew(array)
    except ValueError:
        return [0] * num_labels

    word_set = set()
    words = pytesseract.image_to_string(rot_array).split()
    for word in words:
        word = preprocess.to_lowercase(word)
        word_set.add(word)

    document_word_counts.append(len(word_set))

    dist = [0] * num_labels
    for word in word_set:
        dist = [
            i + j for i, j in zip(dist, get_label_dist(w=word))
        ]

    return dist


# %%

test_dir = "/Users/ericcarlson/Desktop/Datasets/RVL_CDIP/labels/test.txt"
base_dir = "/Users/ericcarlson/Desktop/Datasets/RVL_CDIP/images"
max_count = 512
total_count = 0
correct_count = 0

with open(test_dir, 'r') as test_file:
    for line in test_file:
        if total_count % 32 == 0:
            print(f"{total_count}... ")
        if total_count >= max_count:
            break

        relative_fname, true_label = line.split()
        absolute_fname = os.path.join(base_dir, relative_fname)
        true_label = int(true_label)

        img = cv2.imread(absolute_fname)
        if img is None:
            continue
        pred_dist = predict_doc_dist(img)
        pred_label = np.argmax(pred_dist)

        # TODO: remove.
        # print(f"Distribution: {pred_dist}")
        # print(f"True: {true_label} -> Predicted: {pred_label}")

        correct_count += \
            1 if true_label == pred_label else 0

        total_count += 1

print(f"Accuracy: {correct_count / total_count}\n")

zeros = document_word_counts.count(0)
print(f"Zeros: {zeros / total_count}\n")

# %% Plot document word counts.

plt.hist(document_word_counts, bins=32, color='k')
plt.show()
