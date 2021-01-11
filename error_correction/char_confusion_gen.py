import json
import os

from configparser import RawConfigParser
from cv2 import imread
from os import listdir, path
from pytesseract import image_to_string
from util.encoding.szudzik import pair_ints, unpair_int

properties_dir = path.normpath(path.join(os.getcwd(), "../resources/properties.ini"))
config = RawConfigParser()
config.read(properties_dir)

wordsMNIST = config.get("Datasets", "wordsMNIST.directory")
labelsDir = path.join(wordsMNIST, "labels.json")
images_dir = path.join(wordsMNIST, "images")

with open(labelsDir, 'r') as labelsFile:
    labels_json = json.load(labelsFile)

# This returns a dict of the counts of the characters in a word.
# This is supposed to be used with get_char_diffs.
def char_counts(word):
    charCounts = {}
    for ch in word:
        if ch not in charCounts:
            charCounts[ch] = 0
        charCounts[ch] += 1

    return charCounts

# This method returns the difference between the character counts of word1 and word2.
# For example, if word1 has an 'l' where word2 has a 't', then the difference in character counts would be
# {'l': 1, 't': -1}.
def get_char_diffs(word1, word2):
    charCounts1 = char_counts(word1)
    charCounts2 = char_counts(word2)

    charCountDiffs = {}
    allChars = set(charCounts1.keys())
    allChars.update(charCounts2.keys())
    for c in allChars:
        count1 = charCounts1.pop(c, 0)
        count2 = charCounts2.pop(c, 0)
        if count1 != count2:
            charCountDiffs[c] = count1 - count2

    return charCountDiffs

# The dict char_confusions maps a combination of characters to the number of times these have been confused.
# The arbitrary choice is made to identify character combinations by the pairing function used by M. Szudzik
# (where ch1 is the predicted character and ch2 is the real character).
char_confusions = {}
def add_char_confusion(ch1, ch2):
    z = pair_ints(ord(ch1), ord(ch2))
    if z not in char_confusions:
        char_confusions[z] = 0
    char_confusions[z] += 1

count = 0
max_count = 2 ** 14
correct = 0
for fname in listdir(images_dir):
    if count > max_count:
        break
    if count % 64 == 0:
        print(f"{count}...")

    word_image = imread(path.join(images_dir, fname))
    pred_word = str.strip(image_to_string(word_image))
    real_word = str.strip(labels_json[fname])
    # print("-" * 80)
    # print(f"Actual:    {real_word}")
    # print(f"Predicted: {pred_word}")

    if len(pred_word) > 0:
        diffs = get_char_diffs(pred_word, real_word)
        if len(diffs) == 2:
            keys = list(diffs.keys())
            val1 = diffs[keys[0]]
            val2 = diffs[keys[1]]

            if val1 != val2:
                # PREDICTED char comes before ACTUAL char.
                if val1 == 1:
                    char1 = keys[0]
                    char2 = keys[1]
                else:
                    char1 = keys[1]
                    char2 = keys[0]

                add_char_confusion(char1, char2)
                print(f"{char1}, {char2}")

    if real_word == pred_word:
        correct += 1
    count += 1

print(f"{correct / count: .4f}\n"
      f"({correct} / {count})")

for z in char_confusions:
    x, y = unpair_int(z)
    if char_confusions[z] > 1:
        print(f"{char_confusions[z]} <- {chr(x)}, {chr(y)}")

with open(path.join(os.getcwd(), "char_confusions.json"), 'w') as json_file:
    json.dump(char_confusions, json_file, indent=4)

if __name__ == "__main__":
    pass
