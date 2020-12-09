#%% Count the number of documents each word appears in.

import csv
import spacy

from language_prep.preprocess import token_to_str

csv_dir = "/Users/ericcarlson/Desktop/Datasets/csv/CDIP_OCR.csv"
nlp = spacy.load('en_core_web_sm')

doc_count = 0

word_doc_counts = {}

def count_word(w: str):
    if w not in word_doc_counts:
        word_doc_counts[w] = 0

    word_doc_counts[w] += 1

with open(csv_dir, 'r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',', quotechar='*')

    for row in csv_reader:
        # label = int(row[0])
        doc = nlp(row[1])

        word_set = set()
        for token in doc:
            word = token_to_str(token)
            if word is not None:
                word_set.add(word)

        for word in word_set:
            count_word(w=word)

        doc_count += 1

        if doc_count % 256 == 0:
            print(f"{doc_count}...")

#%% Save data to JSON.

import json
import math

idf_json_dir = "/Users/ericcarlson/Desktop/Personal Projects/basic-document-nlp/resources/CDIP-word-doc-counts.json"

idf_dict = {}

for word in word_doc_counts:
    idf_dict[word] = math.log(
        word_doc_counts[word],
        math.e)

with open(idf_json_dir, 'w') as json_file:
    json.dump(idf_dict, json_file, indent=4)

#%% Sanity check number two.

check_n = 16
check_count = 0

for word in word_doc_counts:
    if check_count >= check_n:
        break

    print(f"{word} ->\n{word_doc_counts[word]}\t{idf_dict[word]: .4f}")

    check_count += 1