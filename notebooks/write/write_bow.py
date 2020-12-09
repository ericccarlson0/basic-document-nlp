# %% Count word-label associations

# This script is used to write counts of word-label associations to a file where it can later be accessed for
# classification or analysis.
# 1) Open the CSV file whose rows have (1) the label and (2) the cleaned up OCR text.
# 2) Iterate over each word in each row of the CSV file and increment the word-label association. For instance,
# If the label of the row is 3, and the current word in the row is "chess", we increment wl_counts["chess"][3].
# 3) Write the dict to a file where it can be accessed later.

import spacy

csv_dir = "/Users/ericcarlson/Desktop/Datasets/csv/CDIP_OCR.csv"
nlp = spacy.load('en_core_web_sm')

wl_counts = {}

def count_wl(w: str, l: int):
    if w not in wl_counts:
        wl_counts[w] = {}

    label_counts = wl_counts[w]
    if l not in label_counts:
        label_counts[l] = 0
    label_counts[l] += 1

with open(csv_dir, 'r') as csv_file:
    for line in csv_file:
        label, text = line.split(',')

        label = int(label)
        doc = nlp(text)

        for token in doc:
            count_wl(w=token.text, l=label)

#%% Dump word-label associations.

import json

json_dir = "/Users/ericcarlson/Desktop/Personal Projects/basic-document-nlp/resources/CDIP-word-label-counts.json"

with open(json_dir, 'w') as json_file:
    json.dump(wl_counts, json_file, indent=4)
