import json

from nlp import preprocess

dict_dir = "../resources/CDIP_OCR.json"
save_to_dir = "../resources/CDIP_OCR_filtered.json"

new_dict = {}
filtered = 0

with open(dict_dir, 'r') as dict_file:
    word_label_counts = json.load(dict_file)
    print(f"Loaded old dict... {len(word_label_counts)} words.")

    for word in word_label_counts:
        if word is None or len(word) == 0:
            continue

        if preprocess.should_filter_pos(word):
            filtered += 1
        else:
            new_dict[word] = word_label_counts[word]

print(f"Filtered {filtered}.")
print("Now saving...")

new_dict_str = json.dumps(new_dict, indent=4)
with open(save_to_dir, 'w') as new_dict_file:
    new_dict_file.write(new_dict_str)

print("Done saving.")

with open(save_to_dir, 'r') as new_dict_file:
    new_dict = json.load(new_dict_file)
    print(f"Loaded new dict... {len(new_dict)} words.")
