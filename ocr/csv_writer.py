import argparse
import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pytesseract

from nlp import preprocess
from ocr import skew

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--base", default="/Users/ericcarlson/Desktop/Datasets/RVL_CDIP/images",
                    help="The 'base directory/ where the images referenced in the dataset reside.")
parser.add_argument("-d", "--dataset", default="/Users/ericcarlson/Desktop/Datasets/RVL_CDIP/labels/train.txt",
                    help="The dataset associating images with labels.")
parser.add_argument("-f", "--format", default="txt",
                    help="The data format of the dataset. As of now, this could be txt or csv.")

args = vars(parser.parse_args())
base_dir = args["base"]
dataset_dir = args["dataset"]
data_format = args["format"]

csv_dir = "/Users/ericcarlson/Desktop/Datasets/csv/CDIP_OCR.csv"

rotations = []
labels = [0] * 16
# This is a dict of dicts, from word -> label -> count.
word_label_counts = {}

count = 0
max_count = 4096
def inc_word_label_counts(w: str, l: int):
    # Add empty dict when word is originally seen.
    if w not in word_label_counts:
        word_label_counts[w] = {}
    label_counts = word_label_counts[w]
    if l not in label_counts:
        label_counts[l] = 0
    label_counts[l] += 1

with open(csv_dir, "w", newline="") as csv_file:
    writer = csv.writer(csv_file)

    with open(dataset_dir, "r") as data_file:
        for line in data_file:
            if count % 32 == 0:
                print(f"{count}... ")
            if count >= max_count:
                break

            relative_fname, label = line.split()
            labels[int(label)] += 1

            absolute_fname = os.path.join(base_dir, relative_fname)
            img = cv2.imread(absolute_fname)

            rot_img, rotation = skew.correct_skew(img)
            rotations.append(rotation)

            ocr_str = pytesseract.image_to_string(rot_img)
            words = preprocess.tokenize_str(ocr_str, mode='basic')

            for word in words:
                inc_word_label_counts(w=word, l=int(label))

            # Does not write *full* filename.
            writer.writerow([relative_fname, ocr_str])
            count += 1

# %% Save to JSON.
import json

json_obj = json.dumps(word_label_counts, indent=4)
with open("../resources/CDIP_OCR_lower.json", "w") as json_file:
    json_file.write(json_obj)

# %% Visualize rotations.

plt.hist(rotations, bins=np.arange(-3.00, 3.20, .20), color='k')
plt.show()
