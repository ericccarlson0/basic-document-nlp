import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
import time

from pathlib import Path
from PIL import Image
from scipy.ndimage import interpolation

# This module scores the skew of an image by taking a histogram of the number of dark pixels across the
# horizontal axis. In the case of images with rows of text, the histogram should look like a uniform
# distribution, as opposed to something approximating a triangle or a curve.
def score_skew(array, skew):
    data = interpolation.rotate(array, skew, reshape=False, order=0)
    histogram = np.sum(data, axis=1)
    score = np.sum((histogram[1:] - histogram[:-1]) ** 2)

    return histogram, score

def correct_skew(original_arr, dd=0.20, max_skew=4):
    try:
        gray_arr = cv2.cvtColor(original_arr, cv2.COLOR_BGR2GRAY)
    except cv2.error:
        print("ERROR: could not convert BGR2GRAY")
        raise Exception(f"Array of size {original_arr.size} could not be corrected.")

    thresh_arr = cv2.threshold(gray_arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    skews = np.arange(-max_skew, max_skew + dd, dd)
    max_score = 0
    best_skew = 0

    for skew in skews:
        _, score = score_skew(thresh_arr, skew)
        if score > max_score:
            best_skew = skew
            max_score = score

    h, w = original_arr.shape[:2]
    center = (w // 2, h // 2)

    rotated_arr = cv2.warpAffine(
        original_arr,
        cv2.getRotationMatrix2D(center, best_skew, 1.0),
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return rotated_arr, best_skew
