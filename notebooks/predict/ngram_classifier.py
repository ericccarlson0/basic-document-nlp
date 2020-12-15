import csv
import lightgbm as lgb
import numpy as np
import os

from notebooks.util.generators import generator_with_max
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from time import time

# TODO: convert PoS that are equivalent!
#  (ex. 120 -> number, John -> name)

# TODO: Allow one to either perform TF-IDF and vectorization here, or to load these.

def load_predictors_from_text(ocr_csv_dir: str, max_docs: int):
    count_vectorizer = CountVectorizer(
        ngram_range=(1, 2),
        min_df=1./(2 ** 11),
        max_df=.75
    )
    tfidf_transformer = TfidfTransformer(norm='l2', smooth_idf=False)

    with open(ocr_csv_dir) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',', quotechar='*')

        docs = generator_with_max(
            (row[1] for row in csv_reader),
            max_docs
        )

        counts = count_vectorizer.fit_transform(docs)
        X = tfidf_transformer.fit_transform(counts)

    return X

def load_responses_from_text(ocr_csv_dir: str, max_docs: int):
    Y = [0] * max_docs

    with open(ocr_csv_dir) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',', quotechar='*')

        for i in range(doc_count):
            row = next(csv_reader)
            Y[i] = int(row[0])

    return Y

def evaluate_on_boosted_trees(X_train, X_test, X_val, Y_train, Y_test, Y_val, n_classes: int):
    lgb_train_set = lgb.Dataset(data=X_train, label=Y_train, free_raw_data=False)
    lgb_valid_set = lgb.Dataset(data=X_val, label=Y_val, free_raw_data=False)

    lgb_params = {
        'objective': 'multiclass',
        # 'objective' : 'multiclassova"
        'num_class': n_classes,
        'boosting': 'gbdt',
        # 'boosting': 'dart'
        'num_leaves': 31
    }

    n_rounds = 16

    t0 = time()
    print(f"Training Booster...")

    booster = lgb.train(lgb_params, lgb_train_set, n_rounds, valid_sets=lgb_valid_set)
    dt = time() - t0
    print(f"{dt: .4f} secs to train Booster")

    val_pred = np.argmax(booster.predict(X_val), axis=1)
    test_pred = np.argmax(booster.predict(X_test), axis=1)
    val_accuracy = accuracy_score(val_pred, Y_val)
    test_accuracy = accuracy_score(test_pred, Y_test)

    print(f"{100 * val_accuracy} accuracy on validation set")
    print(f"{100 * test_accuracy} accuracy on test set")

def evaluate_on_sgd(X_train, X_test, X_val, Y_train, Y_test, Y_val):
    sgd = SGDClassifier(
        loss='hinge',
        penalty='l1',
        alpha=5e-3,
        max_iter=2**10
    )

    t0 = time()
    print("Training SGD...")

    sgd.fit(X_train, Y_train)
    dt = time() - t0
    print(f"{dt: .4f} secs to train SGD")

    print(f"TRAIN score:\t{100 * sgd.score(X_train, Y_train): .4f}")
    print(f"VAL score:\t{100 * sgd.score(X_val, Y_val)}")
    print(f"TEST score:\t{100 * sgd.score(X_test, Y_test): .4f}")

if __name__ == '__main__':
    base_dir = "/Users/ericcarlson/Desktop"
    csv_dir = os.path.join(base_dir, "Datasets", "csv", "CDIP_OCR.csv")

    doc_count = 4096

    t0 = time()
    loaded_X = load_predictors_from_text(csv_dir, doc_count)
    loaded_Y = load_responses_from_text(csv_dir, doc_count)
    dt = time() - t0
    print(f"{dt: .4f} secs to generate predictors and responses")

    X_train, X_not_train, Y_train, Y_not_train = train_test_split(loaded_X, loaded_Y, test_size=0.30)
    X_val, X_test, Y_val, Y_test = train_test_split(X_not_train, Y_not_train, test_size=0.50)

    print('-' * 80)
    for i in range(8):
        print(f"X size: {loaded_X[i, :].getnnz()}, Y: {loaded_Y[i]}")
    print('-' * 80)

    evaluate_on_boosted_trees(X_train, X_val, X_test, Y_train, Y_val, Y_test, n_classes=16)
    # evaluate_on_sgd(X_train, X_val, X_test, Y_train, Y_val, Y_test)
