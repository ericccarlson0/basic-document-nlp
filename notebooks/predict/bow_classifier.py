#%% Retrieve the map containing counts of words associated with labels.

import json
import cv2

word_label_counts = json.load(open(
    "/resources/CDIP-word-label-counts.json"))
print(f"# words: {len(word_label_counts)}")

#%% Define functions to retrieve label distributions and prediction distributions.

import pytesseract
import numpy as np

from image_prep import deskew
from language_prep import preprocess
from typing import List

num_labels = 16

def word_to_dist(w: str) -> List:
    dist = [0] * num_labels

    if w in word_label_counts:
        label_counts = word_label_counts[w]
        for label in label_counts:
            dist[label] += label_counts[label]

        sum_ = sum(dist)
        dist = [d / sum_ for d in dist]

    return dist

# These are just to track some interesting stats.
total_word_counts = []
zero_word_counts = 0

# Predicts a document by iterating over every word found in the document. If that word is not in the
# dictionary, it is skipped. If it is present, it is looked up and a probability distribution is
# produced from the counts in the dictionary. All of these probabilities, for each word in the doc,
# must be combined somehow.
def image_to_dist(array: np.ndarray) -> List:
    try:
        rot_array, _ = deskew.correct_skew(array)
    except ValueError:
        return [0] * num_labels

    word_counts = {}
    words = pytesseract.image_to_string(rot_array).split()

    for word in words:
        word = preprocess.to_alpha_lower(word)

        if word not in word_counts:
            word_counts[word] = 0
        word_counts[word] += 1

    total_word_counts.append(sum(word_counts))

    dist = [0] * num_labels
    for word in word_counts:
        to_add = word_to_dist(w=word)
        for i in range(num_labels):
            dist[i] += to_add[i] * word_counts[word]

    return dist


#%% Try out BOW classifier.

import os

from notebooks.util.image_dataset import TxtImageDataset

test_dir = "/Users/ericcarlson/Desktop/Datasets/RVL_CDIP/labels/test.txt"
base_img_dir = "/Users/ericcarlson/Desktop/Datasets/RVL_CDIP/images"
max_count = 256
count = 0
correct_count = 0

with TxtImageDataset(test_dir) as test_file:
    for img_dir, true_label in test_file:
        abs_img_dir = os.path.join(base_img_dir, img_dir)

        img = cv2.imread(abs_img_dir)
        if img is None:
            print("img is None...")
            continue

        pred_dist = image_to_dist(img)
        pred_label = np.argmax(pred_dist)
        # print(f"Distribution: {pred_dist}")
        # print(f"True: {true_label} -> Predicted: {pred_label}")

        correct_count += 1 if true_label == pred_label else 0
        count += 1

zeros = total_word_counts.count(0)
print(f"% zeros: {100 * zeros / count}\n")
print(f"% accuracy: {100 * correct_count / count}\n")

#%% Plot document word counts.

import matplotlib.pyplot as plt

plt.hist(total_word_counts, bins=32, color='k')
plt.show()
