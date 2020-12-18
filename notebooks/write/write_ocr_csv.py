#%% Set up command-line args.

import argparse

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
parser.add_argument("-f", "--format",
                    default="txt",
                    help="The data format of the dataset. As of now, this could be txt or csv.")
parser.add_argument("-l", "--labels",
                    default="16",
                    help="The number of distinct labels assigned to the images.")
parser.add_argument("--count",
                    default="8192",
                    help="The maximum number of images to be translated and written to the csv.")

args = vars(parser.parse_args())

base_img_dir = args['base']
dataset_dir = args['dataset']
csv_dir = args['csv']
data_format = args['format']
max_count = int(args['count'])
label_num = int(args['labels'])

labels = [0] * label_num

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

with open(csv_dir, 'w') as csv_file:
    writer = csv.writer(csv_file, delimiter=',', quotechar='*')

    with TxtImageDataset(dataset_dir) as data_file:
        for img_dir, label in data_file:
            abs_img_dir = os.path.join(base_img_dir, img_dir)
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
