import json
import numpy as np
import os
import seaborn as sns

from matplotlib import pyplot as plt
from os import path
from util.encoding.szudzik import unpair_int

#%% Populate confusion matrix.

with open(path.join(os.getcwd(), "char_confusions.json"), 'r') as json_file:
    char_confusions = json.load(json_file)

min_char = ord('0')
max_char = ord('z')
num_chars = max_char - min_char + 1

confusion_matrix = [
    [0] * num_chars for _ in range(num_chars)
]

for z in char_confusions:
    count = char_confusions[z]

    x, y = unpair_int(int(z))
    print(f"{chr(x), chr(y)}")

    # if x > 255 or y > 255:
    #     print(f"{x} or {y} out of range")
    if x > max_char or y > max_char:
        print(f"{x} or {y} out of range")
    else:
        confusion_matrix[x - min_char][y - min_char] = count

#%% Visualise confusion matrix.

sns.set_style('whitegrid')
# mask = np.triu(np.ones_like(confusion_matrix, dtype=np.bool))
# mask = mask[1:][:-1]

char_intervals = [
    (ord('0') - min_char, ord('9') - min_char),
    (ord('A') - min_char, ord('Z') - min_char),
    (ord('a') - min_char, ord('z') - min_char)
]

for i in range(3):
    for j in range(3):
        x1, x2 = char_intervals[i]
        y1, y2 = char_intervals[j]

        sub_array = [
            confusion_matrix[y][x1: x2]
            for y in range(y1, y2)
        ]

        ax = sns.heatmap(sub_array, cmap='YlOrRd',
                         linewidth=0.3, cbar_kws={'shrink': 0.8})
        ax.xaxis.tick_top()

        plt.xticks(np.arange(x2 - x1) + 0.5, labels=[chr(i + min_char) for i in range(x1, x2)])
        plt.yticks(np.arange(y2 - y1) + 0.5, labels=[chr(i + min_char) for i in range(y1, y2)])

        plt.show()
