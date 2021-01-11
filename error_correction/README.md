The purpose of this module is to address OCR errors (that is, to perform post-processing). The paper that helped me the 
most with this was *OCR Error Correction Using Character Correction and Feature-Based Word Classification*, by Kissos 
and Dershowitz (see https://arxiv.org/pdf/1604.06225.pdf). This Kaggle dataset is used to create confusion matrices 
for characters and character bigrams: https://www.kaggle.com/backalla/words-mnist. This Kaggle dataset helped some, 
too: https://www.kaggle.com/dmollaaliod/correct-ocr-errors.
There are some arbitrary choices made throughout this sub-project, such as choosing to store the confusion matrix 
as a dictionary from pairwise encodings to counts. The README is not the single source of truth when it comes to 
explaining these decisions, even though it will be updated.

* **character_confusion_matrix.py**

...

* **etc.**

...