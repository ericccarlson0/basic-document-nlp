#%% Set up baseline information regarding dataset.

import os

from configparser import RawConfigParser
from os import path

properties_dir = path.normpath(path.join(os.getcwd(), "../../resources/properties.ini"))
config = RawConfigParser()
config.read(properties_dir)

# The base directory where the images referenced by the dataset reside.
base_img_dir = config.get("Datasets", "imageDataset.baseImageDir")
# The location of a text or csv dataset containing image locations and labels.
dataset_dir = config.get("Datasets", "imageDataset.path")
# The location of the csv to write OCR'd documents and labels to.
ocr_csv_dir = config.get("Datasets", "ocrDataset.path")

data_format = "txt"
max_count = 8192
n_labels = 16

labels = [0] * n_labels

#%% Write OCR'd docs and labels to a CSV file.

import csv
import cv2
import os
import pytesseract
import spacy

from image_prep import deskew
from language_prep import preprocess
from notebooks.util.datasets import TxtImageDataset

nlp = spacy.load('en_core_web_sm')

word_count = 0
real_word_count = 0

with open(ocr_csv_dir, 'w') as csv_file:
    writer = csv.writer(csv_file, delimiter=',', quotechar='*')

    with TxtImageDataset(dataset_dir) as data_file:
        for img_dir, label in data_file:
            abs_img_dir = path.join(base_img_dir, img_dir)
            labels[label] += 1

            # TODO: de-noise, too?
            img = cv2.imread(abs_img_dir)
            rot_img, _ = deskew.correct_skew(img)

            ocr_str = pytesseract.image_to_string(rot_img)
            ocr_doc = nlp(ocr_str)

            for token in ocr_doc:
                word = preprocess.to_alpha_lower(token.text)
                if preprocess.is_word(word):
                    real_word_count += 1
                word_count += 1

            clean_str = preprocess.doc_to_str(ocr_doc)
            writer.writerow([label, clean_str])

            # print(clean_str if len(clean_str) < 80
            #       else f"{clean_str[:77]}...")

            label_count = sum(labels)
            if label_count >= max_count:
                break
            if label_count % 16 == 0:
                print(f"{label_count}...")

print(f"{100 * real_word_count / word_count: .3f}% real words")
print(labels)
