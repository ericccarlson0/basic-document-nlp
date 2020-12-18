import csv

from notebooks.util.generators import generator_with_max
from language_prep.vectorizers import LemmatizerCountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer

# This method loads ngrams from CSV as a generator, which can be treated as a matrix.
def load_csv_ngrams(ocr_csv_dir: str, max_docs: int):
    feature_pipeline = Pipeline([
        ('vectorizer', LemmatizerCountVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=.50)),
        ('tfidf', TfidfTransformer(
            norm='l2',
            smooth_idf=False))
    ])

    with open(ocr_csv_dir) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',', quotechar='*')

        docs = generator_with_max(
            (row[1] for row in csv_reader),
            max_docs
        )

        X = feature_pipeline.fit_transform(docs)

    return X

# This method loads labels from CSV as a list.
def load_csv_labels(ocr_csv_dir: str, max_docs: int):
    Y = [0] * max_docs

    with open(ocr_csv_dir) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',', quotechar='*')

        for i in range(max_docs):
            row = next(csv_reader)
            Y[i] = int(row[0])

    return Y
