#%% Options parsing.

# Adapted from the following resource:
# https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html#sphx-glr-auto-examples-text-plot-document-classification-20newsgroups-py

import logging
import sys

from optparse import OptionParser
from sklearn import metrics
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.utils.extmath import density
from time import time

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

options = OptionParser()
options.add_option("--chi2_select", action="store", type=int, dest="select_chi2",
                   help="Select features with a CHI2 test.")
options.add_option("--print_report", action="store_true", dest="print_report",
                   help="Print the entire classification report.")
options.add_option("--confusion_matrix", action="store_true", dest="print_confusion",
                   help="Print the confusion matrix.")
options.add_option("--print_top10", action="store_true", dest="print_top10",
                   help="Print the 10 most useful features, per class, for each classifier.")
options.add_option("--use_hashing", action="store_true",
                   help="Use a hashing vectorizer.")
options.add_option("--n_features", action="store", type=int, default=2 ** 16,
                   help="Number of features used in the vectorizer.")
options.add_option("--filtered", action="store_true",
                   help="Remove information that is easy to overfit.")

def is_interactive():
    return not hasattr(sys.modules['__main__'], '__file__')

argv = [] if is_interactive() else sys.argv[1:]
opts, args = options.parse_args(argv)
if len(args) > 0:
    options.error("There should be no arguments.")
    sys.exit(1)

opts.use_hashing = False
# opts.n_features = 2 ** 10
opts.select_chi2 = 2 ** 10
opts.print_top10 = True
opts.print_confusion = True

#%% Fetch data, print basic statistics.

categories = None
remove = ('headers', 'footers', 'quotes')
random_state = 53

print("Data coming in...")
data_train = fetch_20newsgroups(subset='train', categories=categories,
                                shuffle=True, random_state=random_state,
                                remove=remove)
data_test = fetch_20newsgroups(subset='test', categories=categories,
                               shuffle=True, random_state=random_state,
                               remove=remove)
print("Done.\n")

target_names = data_train.target_names

def size_mb(docs):
    return sum(len(s.encode('utf-8')) for s in docs) / 1e6

train_size = size_mb(data_train)
test_size = size_mb(data_test)
print("TRAINING set:\n" + f"docs: {len(data_train.data)}, size: {train_size}")
print("TEST set:\n" + f"docs: {len(data_test.data)}, size: {test_size}")
print(f"{len(target_names)} categories.\n")

print(type(data_train.data))
print(data_train.data[0])
print(type(data_train.target))

#%% Create training and test sets.

y_train, y_test = data_train.target, data_test.target

print("Extracting features from training data...")
t0 = time()
if opts.use_hashing:
    vectorizer = HashingVectorizer(stop_words='english', n_features=opts.n_features, alternate_sign=False)
    X_train = vectorizer.transform(data_train.data)
else:
    # TODO: sublinear_tf and max_df are conscious choices, and should be evaluated as such
    vectorizer = TfidfVectorizer(stop_words='english', sublinear_tf=True, max_df=0.5)
    X_train = vectorizer.fit_transform(data_train.data)

dt = time() - t0
print(f"{X_train.shape[0]} samples\n{X_train.shape[1]} features")
print(f"{dt: .3f} secs\n{train_size / dt: .3f} MB/sec")

# Do the same vectorization with TEST set.
X_test = vectorizer.transform(data_test.data)

#%% Create features.

if opts.use_hashing:
    feature_names = None
else:
    feature_names = vectorizer.get_feature_names()

if opts.select_chi2:
    print(f"Extracting {opts.select_chi2} best features with CHI2 test...")
    t0 = time()
    k_best = SelectKBest(chi2, k=opts.select_chi2)
    # Fit according to TRAIN set.
    X_train = k_best.fit_transform(X_train, y_train)
    X_test = k_best.transform(X_test)

    if feature_names:
        # TODO: what is get_support?
        feature_names = [feature_names[i] for i in k_best.get_support(indices=True)]

    dt = time() - t0
    print(f"Done ({dt} secs).")

if feature_names:
    feature_names = np.asarray(feature_names)

def trim(s: str):
    return s if len(s) <= 80 else s[:77] + "..."

#%% Define how to benchmark a classifier.

def benchmark(classifier):
    s0 = time()
    print(f"Training {classifier}...")
    classifier.fit(X_train, y_train)
    train_time = time() - s0
    print(f"{train_time: .3f} seconds to train.")

    s0 = time()
    pred = classifier.predict(X_test)
    test_time = time() - s0
    print(f"{test_time: .3f} seconds to predict.")

    score = metrics.accuracy_score(y_test, pred)
    print(f"{score: .3f} accuracy.")

    if hasattr(classifier, 'coef_'):
        coefficients = classifier.coef_
        print(f"Dimensionality: {coefficients.shape[1]}")
        print(f"Density: {density(coefficients): .3f}")

        if feature_names is not None and opts.print_top10:
            print("Top keywords, per class: ")
            for i, label in enumerate(target_names):
                top10 = np.argsort(coefficients[i])[-10:]
                print(trim(
                    f"""{label}: {" ".join(feature_names[top10])}"""
                ))

        if opts.print_report:
            print("Classification report: ")
            print(metrics.classification_report(y_test, pred, target_names=target_names))

        if opts.print_confusion:
            print("Confusion matrix: ")
            print(metrics.confusion_matrix(y_test, pred))

        classifier_desc = str(classifier).split('(')[0]

        return classifier_desc, score, train_time, test_time

#%% Benchmark classifiers.

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Perceptron, SGDClassifier, RidgeClassifier
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

results = []
iterations = 128

# NOTE: changed Ridge solver, sag -> sparse_cg
for name, classifier in (
        ("Ridge", RidgeClassifier(tol=1e-2, solver='sparse_cg')),
        ("Perceptron", Perceptron(max_iter=iterations)),
        ("K-Nearest", KNeighborsClassifier(n_neighbors=8)),
        ("Random Forest", RandomForestClassifier())):
    print("-" * 80)
    print(name)
    results.append(benchmark(classifier))

for penalty in ("l1", "l2"):
    print("-" * 80)
    print(f"{penalty.upper()} Penalty")
    results.append(benchmark(
        LinearSVC(penalty=penalty, dual=False, tol=1e-3)
    ))
    results.append(benchmark(
        SGDClassifier(alpha=1e-4, max_iter=iterations, penalty=penalty)
    ))

print("-" * 80)
print("Elastic Net Penalty")
results.append(benchmark(
    SGDClassifier(alpha=1e-4, max_iter=iterations, penalty="elasticnet")
))

print("-" * 80)
print("Nearest Centroid (w/o threshold)")
results.append(benchmark(NearestCentroid()))

print("-" * 80)
print("Sparse Naive Bayes")
for nb in (MultinomialNB(alpha=1e-2), BernoulliNB(alpha=1e-2), ComplementNB(alpha=1e-1)):
    results.append(benchmark(nb))

print("Linear SVC with\nL1 feature selection, L2 classification")
results.append(benchmark(
    Pipeline([
        ('feature_selection', SelectFromModel(LinearSVC(penalty='l1', dual=False, tol=1e-3))),
        ('classification', LinearSVC(penalty='l2'))
    ])
))

