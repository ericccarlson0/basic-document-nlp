import lightgbm as lgb
import numpy as np
import os

from configparser import RawConfigParser
from notebooks.util.loaders import load_csv_labels, load_csv_ngrams
from os import path
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from time import time

# Evaluate predictions with LightGBM boosted trees.
def evaluate_on_boosted_trees(X_train, X_test, X_val, Y_train, Y_test, Y_val, n_classes: int):
    lgb_train_set = lgb.Dataset(data=X_train, label=Y_train, free_raw_data=False)
    lgb_valid_set = lgb.Dataset(data=X_val, label=Y_val, free_raw_data=False)

    lgb_params = {
        'objective': 'multiclass',
        # 'objective' : 'multiclassova'
        'num_class': n_classes,
        'boosting': 'gbdt',
        'learning_rate': 0.2,
        # 'tree_learner': 'feature',
        'num_leaves': 2**10 - 1,
        'force_col_wise': True,
        'bagging_fraction': 0.75,
        'bagging_freq': 1,
        'verbosity': -1
    }
    # More training parameters to consider:
    # max_depth, feature_fraction, lambda_l1, lambda_l2

    n_rounds = 16

    print(f"Training Booster...")
    t0 = time()
    booster = lgb.train(lgb_params, lgb_train_set, n_rounds, valid_sets=lgb_valid_set)
    dt = time() - t0
    print(f"{dt: .4f} secs to train Booster")

    test_pred = np.argmax(booster.predict(X_test), axis=1)
    val_pred = np.argmax(booster.predict(X_val), axis=1)
    train_pred = np.argmax(booster.predict(X_train), axis=1)

    test_accuracy = accuracy_score(test_pred, Y_test)
    val_accuracy = accuracy_score(val_pred, Y_val)
    train_accuracy = accuracy_score(train_pred, Y_train)

    print(f"{100 * test_accuracy: .4f} accuracy on test set")
    print(f"{100 * val_accuracy: .4f} accuracy on validation set")
    print(f"{100 * train_accuracy: .4f} accuracy on train set")

# Evaluate predictions with a linear classifier.
def evaluate_on_sgd(X_train, X_test, X_val, Y_train, Y_test, Y_val):
    sgd = SGDClassifier(
        loss='hinge',
        penalty='l1',
        alpha=5e-3,
        max_iter=2**10
    )

    print("Training SGD...")
    t0 = time()
    sgd.fit(X_train, Y_train)
    dt = time() - t0
    print(f"{dt: .4f} secs to train SGD")

    print(f"Accuracy on training:    {100 * sgd.score(X_train, Y_train): .4f}")
    print(f"Accuracy on validation:  {100 * sgd.score(X_val, Y_val): .4f}")
    print(f"Accuracy on test:        {100 * sgd.score(X_test, Y_test): .4f}")

if __name__ == '__main__':
    properties_dir = path.normpath(path.join(os.getcwd(), "../../resources/properties.ini"))
    config = RawConfigParser()
    config.read(properties_dir)

    ocr_csv_dir = config.get("Datasets", "ocrDataset.path")

    doc_count = 8192

    t0 = time()
    X = load_csv_ngrams(ocr_csv_dir, doc_count)
    Y = load_csv_labels(ocr_csv_dir, doc_count)
    dt = time() - t0
    print(f"{dt: .4f} secs to generate predictors and responses")

    X_train, X_not_train, Y_train, Y_not_train = train_test_split(X, Y, test_size=0.30)
    X_val, X_test, Y_val, Y_test = train_test_split(X_not_train, Y_not_train, test_size=0.50)

    # print('-' * 80)
    # for i in range(8):
    #     print(f"Number of features in X: {X[i, :].getnnz()}")
    # print('-' * 80)

    evaluate_on_boosted_trees(X_train, X_val, X_test, Y_train, Y_val, Y_test, n_classes=16)
    evaluate_on_sgd(X_train, X_val, X_test, Y_train, Y_val, Y_test)
