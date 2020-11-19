import json

from nlp import preprocess

# This is just a script at the moment. It could be abstracted into methods. To be honest, it should
# not be too useful, anyways, except for ad-hoc solutions (i.e. you need to filter out certain words
# that you had not anticipated).

old_dict_dir = "../resources/CDIP_OCR.json"
new_dict_dir = "../resources/CDIP_OCR_filtered.json"

new_dict = {}
filtered = 0

with open(old_dict_dir, 'r') as dict_file:
    word_label_counts = json.load(dict_file)
    print(f"Old dict contains {len(word_label_counts)} words.")

    for word in word_label_counts:
        if word is None or len(word) == 0:
            continue

        if preprocess.should_filter_pos(word):
            filtered += 1
        else:
            new_dict[word] = word_label_counts[word]

print(f"Old dict contained {filtered} filtered words.")

print("Saving...")
new_dict_str = json.dumps(new_dict, indent=4)
with open(new_dict_dir, 'w') as new_dict_file:
    new_dict_file.write(new_dict_str)
print("Saved new dict.")

with open(new_dict_dir, 'r') as new_dict_file:
    new_dict = json.load(new_dict_file)
    print(f"New dict has been loaded. It has {len(new_dict)} words.")
