#%% Load numpy arrays and create models.

import numpy as np
import os

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from time import time

# Here, we perform a simple linear regression on the "average embedding" of each doc.
# The end goal is to test various standard statistical methods on this average embedding in order to assess
# the viability of the "average embedding" approach. That would mean more linear methods with various basis
# expansions, LDA/QDA, ... which are all easy to test with sklearn.

resources_dir = "/Users/ericcarlson/Desktop/Personal Projects/basic-document-nlp/resources/glove"
predictor_dir = os.path.join(resources_dir, "predictor-matrix.npy")
response_dir = os.path.join(resources_dir, "response-matrix.npy")

X = np.load(predictor_dir)
Y = np.load(response_dir)

X_train = X[:1008, :]
Y_train = Y[:1008, :]
Y_train_class = np.argmax(Y_train, axis=1)

X_test = X[1008:, :]
Y_test = Y[1008:, :]
Y_test_class = np.argmax(Y_test, axis=1)

print(f"X train: {X_train.shape}")
print(f"Y train: {Y_train.shape}")
print(f"Y train, class: {Y_train_class.shape}\n")

print(f"X test: {X_test.shape}")
print(f"Y test: {Y_test.shape}")
print(f"Y test, class: {Y_test_class.shape}\n")

regression_kwargs = {
    'normalize': True,
    'copy_X': True,
    'n_jobs': 4
}

standard_regression = LinearRegression(**regression_kwargs)

interaction_regression = Pipeline([
    ('interactions', PolynomialFeatures(interaction_only=True, degree=2)),
    ('regression', LinearRegression(normalize=True, copy_X=True, n_jobs=4))
])

lda = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')
qda = QuadraticDiscriminantAnalysis()

regressors = (
    ('Standard Linear Regression', standard_regression),
    ('Linear Regression w/ Interactions', interaction_regression),
)
classifiers = (
    ('Linear Discriminant Analysis', lda),
    ('Quadratic Discriminant Analysis', qda)
)

for name, regressor in regressors:
    t0 = time()
    regressor.fit(X_train, Y_train)
    print(f"{time() - t0: .4f} secs, {name}")

for name, classifier in classifiers:
    t0 = time()
    classifier.fit(X_train, Y_train_class)
    print(f"{time() - t0: .4f} secs, {name}")

# standard_regression.fit(X_train, Y_train)
# interaction_regression.fit(X_train, Y_train)
# lda.fit(X_train, Y_train)
# qda.fit(X_train, Y_train)

#%% Assess train and test set predictions.


def assess_predictions(X_set: np.ndarray, Y_set: np.ndarray, type_: str, n: int):
    print("-" * 80)
    print(f"Results on {type_} set: ")

    for name, regressor in regressors:
        pred = regressor.predict(X_set)

        for i in range(n):
            label = np.argmax(Y_set[i])
            print(f"{pred[i][label]: .4f} <- {name}")

    for name, classifier in classifiers:
        pred_classes = classifier.predict(X_set)

        for i in range(n):
            true_class = np.argmax(Y_set[i])
            print(f"{pred_classes[i]} / {true_class} <- {name}")


assess_predictions(X_train, Y_train, "TRAIN", 8)
assess_predictions(X_test, Y_test, "TEST", 8)

# print(regressor.coef_.shape)
# print(regressor.coef_[0])
