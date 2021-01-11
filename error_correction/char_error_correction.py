import json
import os

from os import path
from util.encoding.szudzik import unpair_int


# NOTE: In this case, char_confusions is a dict which maps encodings to counts. These encodings are ints which are
# decoded to produce a pair of ints. The FIRST part of the pair represents a predicted char, whereas the SECOND part
# represents an actual char. For instance, when char_confusions[Z] is 16 and decode(z) is X, Y, then char Y was
# confused for X sixteen times in the training set.
def correction_candidates(word: str, char_confusion_dict: dict):
    ret = []
    top_confusion_dict = top_confusions_per_char(char_confusion_dict, n=5)

    for i in range(len(word)):
        start = word[:i] if i > 0 else ''
        end = word[i + 1:] if i + 1 < len(word) else ''

        for ch, _ in top_confusion_dict[word[i]]:
            candidate = start + ch + end
            ret.append(candidate)

    return ret

def top_confusions_per_char(char_confusion_dict: dict, n: int):
    lowest_counts = {}
    top_confusion_dict = {}

    for z in char_confusion_dict:
        curr_count = char_confusion_dict[z]
        x, y = unpair_int(int(z))
        x = chr(x)
        y = chr(y)

        if x not in top_confusion_dict:
            top_confusion_dict[x] = set()
            lowest_counts[x] = 0

        lowest_count = lowest_counts[x]
        top_confusion_set = top_confusion_dict[x]
        add_tuple = False

        if len(top_confusion_set) < n:
            add_tuple = True
        elif curr_count > lowest_count:
            for char_count_tuple in top_confusion_set:
                if char_count_tuple[1] == lowest_count:
                    print(f"Removed ({char_count_tuple[0]}, {char_count_tuple[1]})")
                    top_confusion_set.remove(char_count_tuple)
                    break
            add_tuple = True

        if add_tuple:
            top_confusion_set.add((y, curr_count))
            lowest_counts[x] = curr_count

    return top_confusion_dict

def test_top_confusions(char_confusion_dict: dict):
    top_confusions = top_confusions_per_char(char_confusion_dict, n=3)

    for i in range(97, 123):
        pred = chr(i)
        if pred not in top_confusions:
            print(f"{pred} not in the dict")
        else:
            print(f"Confusions for {pred}:")
            for real, count in top_confusions[pred]:
                print(f"{real}, {count}")


if __name__ == '__main__':
    with open(path.join(os.getcwd(), "char_confusions.json")) as json_file:
        char_confusions = json.load(json_file)

    for w in ["worb", "tist"]:
        print(f"Correction candidates for {w}:")
        for new_word in correction_candidates(w, char_confusions):
            print(f"\t{new_word}")

    # test_top_confusions(char_confusions)
