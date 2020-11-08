import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
import time

from pathlib import Path
from PIL import Image
from scipy.ndimage import interpolation

def score_skew(array, skew):
    data = interpolation.rotate(array, skew, reshape=False, order=0)
    histogram = np.sum(data, axis=1)
    score = np.sum((histogram[1:] - histogram[:-1]) ** 2)
    # print(f"skew {skew} -> score {score}")

    return histogram, score

def correct_skew(original, dd=0.20, max_skew=4):
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    skews = np.arange(-max_skew, max_skew + dd, dd)
    max_score = 0
    best_skew = 0

    for skew in skews:
        histogram, score = score_skew(thresh, skew)
        if score > max_score:
            best_skew = skew
            max_score = score

    h, w = original.shape[:2]
    center = (w // 2, h // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, best_skew, 1.0)
    # print("Rotation Matrix:\n", rot_matrix)

    rotated = cv2.warpAffine(original, rot_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated, best_skew

# Determine skew.
# coords = np.column_stack(np.where(thresh_img > 0))
# skew = cv2.minAreaRect(coords)[-1]

# Pre-processing.
# image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# image = cv2.medianBlur(image, 3)
