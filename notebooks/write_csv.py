#%% Import main libraries and set up CLI args.

import argparse
import csv
import cv2
import os
import pytesseract
import spacy

from ocr import skew

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--base",
                    default="/Users/ericcarlson/Desktop/Datasets/Document Classification/RVL_CDIP/images",
                    help="The location of the base directory where the images referenced in the dataset reside.")
parser.add_argument("-d", "--dataset",
                    default="/Users/ericcarlson/Desktop/Datasets/Document Classification/RVL_CDIP/labels/train.txt",
                    help="The location of the dataset.")
parser.add_argument("-c", "--csv",
                    default="/Users/ericcarlson/Desktop/Datasets/csv/CDIP_OCR.csv",
                    help="The location of the csv to write to.")

# TODO: Support different formats (would probably be done in a different module).
parser.add_argument("-f", "--format",
                    default="txt",
                    help="The data format of the dataset. As of now, this could be txt or csv.")

args = vars(parser.parse_args())
base_dir = args["base"]
dataset_dir = args["dataset"]
csv_dir = args["csv"]
data_format = args["format"]

#%% Define how word counts are stored and processed.

# A dict of dicts, where words map to labels, which map to counts of word-label combinations.
word_label_counts = {}

def count_word(w: str, l: int):
    if w not in word_label_counts:
        word_label_counts[w] = {}

    label_counts = word_label_counts[w]
    if l not in label_counts:
        label_counts[l] = 0

    label_counts[l] += 1

#%% Write to csv.

# TODO: move to options
max_count = 16
labels = [0] * 16
nlp = spacy.load("en_core_web_sm")

# Turns a spaCy doc into a clean string without large gaps between paragraphs.
def remove_gaps(doc_):
    result = []
    for token_ in doc_:
        if not token_.is_space:
            result.append(token_.text_with_ws)
        else:
            result.append(token_.text[0])

    return "".join(result)


with open(csv_dir, 'w', newline='\n') as csv_file:
    writer = csv.writer(csv_file)

    # TODO: make this contingent on format
    with open(dataset_dir, 'r') as data_file:
        for line in data_file:
            if sum(labels) >= max_count:
                break

            if sum(labels) % 32 == 0:
                print(f"Label Counts: {labels}... ")

            rel_fname, label = line.split()
            labels[int(label)] += 1
            abs_fname = os.path.join(base_dir, rel_fname)

            img = cv2.imread(abs_fname)
            rot_img, rot_degree = skew.correct_skew(img)

            ocr_str = pytesseract.image_to_string(rot_img)
            ocr_doc = nlp(ocr_str)
            clean_ocr_str = remove_gaps(ocr_doc)

            # print("-"*80)
            # print(clean_ocr_str[:80])

            for token in ocr_doc:
                count_word(w=token.text, l=int(label))
            # Does not write *full* filename.
            writer.writerow([clean_ocr_str, label, rel_fname])

#%% Write to JSON.

import json

json_fname = "../resources/CDIP_OCR.json"
json_obj = json.dumps(word_label_counts, indent=4)

with open(json_fname, 'w') as json_file:
    json_file.write(json_obj)

print(f"Wrote {len(json_obj)} characters to {json_fname}.")
