# Basic Document NLP

The purpose of this project is to benchmark basic NLP techniques when it comes to the task of document classification.
In the task at hand, the documents will have been generated through OCR, so extra work would need to be done to deal 
with OCR-specific errors.

## Basic Methods

* **Bag-of-Words, N-Gram**

These methods produce sparse features which can be used to input into many different models. The challenge is that 
the features are sparse and that there is the destruction of a lot of the sequential data. One of the modules in this 
project was used to evaluate the performance of LightGBM on these features, and produced a maximum accuracy of around 
65% on the CDIP dataset (16 classes). This seems to be near the upper bound of how well these methods can work for 
document classification.

* **Part of Speech, Chunking, Lemmatization**

These methods could be used to improve Bag-of-Words and N-Gram feature generators. Lemmatization has been used to help 
with N-Gram feature generators in the module mentioned above, and it contributed a few percentage points of accuracy.
Part-of-Speech tags could be used to filter terms or to group similar terms into the same category. Chunking could 
be used to produce N-Grams without having to use all of them. For instance, we could use it if we wanted 3-grams or 
4-grams with certain grammatical characteristics, but not all of them.

* **TF-IDF**

TF-IDF is another method that is used to improve Bag-of-Words and N-Gram feature generators. It normalizes counts of 
words and n-grams.

* **NER**

Named Entity Recognition could be useful in more advanced models that operate over sequences, but even then it seems 
to have limited use. The existence and number of named entities could be a useful data point for prediction, but it 
is difficult to evaluate alongside the other data and may be accounted for implicitly, anyways.

* **Embeddings**

The files embeddings.py (in language_prep), write_embedding.py (in notebooks/write), and embedding_classifier.py (in 
notebooks/predict) are used to assess the use of "average embeddings" for words or bigrams. The process to produce 
these embeddings is as follows:
* Initialize a matrix with N rows and M columns, where N is the number of labels and M is the number of dimensions in 
the pre-prepared space of embeddings.
* Open a csv file whose rows have (1) document labels and (2) whose columns have cleaned-up OCR text from the docs.
* For each row of the csv file...
* Iterate over each word in the doc, getting its embedding and adding it to a temporary vector containing the sum of the 
document's words' embeddings thus far. When the iteration is over, divide the vector by the number of words accounted 
for to produce the "average embedding".
* At the same time, add the vector to the row of the N x M matrix which corresponds to the document's label.
* For each label, divide the corresponding row of the association matrix by the number of times that label was found, 
to produce rows that represent average embeddings for documents containing each label.
* Write the N x M matrix, which has the average embedding for each label, and another matrix containing the average 
embedding for each doc to somewhere it can be accessed easily.

## Modules (as of 12/19/2020)

* **The notebooks/predict module**

In **glove_evaluation**, we perform various classical statistical techniques on the average embedding data for each 
document, to assess the overall viability of the "average embedding" approach. A comprehensive test would include a 
handful of different embeddings to try out, a host of classical classification and regression techniques, and some 
variation on these techniques in terms of training parameters, smoothing, basis expansion, etc.
The **random_forests** uses the embeddings too, and its purpose is to compare pruning parameters for the Random Forest 
Classifier in sklearn.The file could be named better, because others such as ngram_evaluation use random forests, as 
well. It has a method which is used to compare node counts, tree depth, and accuracy depending on these parameters.

In **ngram_evaluation**, we use the CountVectorizer in sklearn which can generate counts of ngrams as well as individual 
words (we just use bigrams currently, as trigrams seem to induce some over-fitting). Then, the word- and ngram-counts 
are normalized according to TF-IDF. The counting of words and normalization according to TF-IDF used to be the subject 
of a few files in this project, but these were deleted because libraries like sklearn are fast enough and easy enough 
to use to make that redundant.
Following this, the **ngram_evaluation** file tests out boosted trees with LightGBM. These can achieve an accuracy of 
up to 65% on a sample of the Big Tobacco dataset with 8192 images. More methods should be tested, especially those that 
are said to be good with sparse predictors (which word- and ngram-counts are).

The **scikit_comparison** and **sentiment_classifier** files are both mainly adapted from resources online. One comes 
from an introduction to sklearn for NLP and the other, an introduction to spaCy for NLP. These were possibly going to 
be used more in the future, but that never happened.

* The `notebooks/util` module

The `datasets` util file is used to provide context managers and generators for different types of datasets. It only 
supports a dataset of a specific type at the moment, but is meant to be extended whenever the convenience is needed. An 
example would be:

```python
with TxtImageDataset(test_dir) as test_file:
    for img_dir, label in test_file:
        # Process data.
```

The `generators` util file was added because I wanted a generator that stopped at a maximum value. It could be used 
to create or abstract different generators, too.

The `loaders` util file was added to abstract data loaders that were being used to create data for sklearn (these 
are used in `ngram_evaluation`). If the project is to continue, more data loaders could be added here to de-clutter 
the `notebooks/predict` module.

The `logging` util file was added because I wanted a convenient way to check a few values in a list or numpy array and
print them in a nice manner, in order to sanity-check and debug. It could just as well be called `printing` at the 
moment (maybe it should be).

* The `notebooks/write module`

The `write_ocr_csv` file is used to turn a simple image dataset into a CSV which contains, on each row, and image 
label and the OCR text generated for that image.

The `write_embedding` file is used to write turn files into average embeddings with are then saved in a specific 
location as .npy arrays.

