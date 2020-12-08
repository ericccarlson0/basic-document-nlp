#%% Generate matrices, remove NaN and infinite values.

# Here, we perform a simple linear regression on the "average embedding" of each doc.
# The end goal is to test various standard statistical methods on this average embedding in order to assess
# the viability of the "average embedding" approach. That would mean more linear methods with various basis
# expansions, LDA/QDA, ... which are all easy to test with sklearn.

import numpy as np
import os

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from time import time

resources_dir = "/Users/ericcarlson/Desktop/Personal Projects/basic-document-nlp/resources/glove"
predictor_dir = os.path.join(resources_dir, "predictor-matrix.npy")
response_dir = os.path.join(resources_dir, "response-matrix.npy")

X = np.load(predictor_dir)
Y = np.load(response_dir)

print("Is there NaN?", np.any(np.isnan(X)))
print("Is there infinite?", not np.all(np.isfinite(X)))
# for i in range(X.shape[0]):
#     print(not np.any(np.isnan(X[i])), np.all(np.isfinite(X[i])))
X[np.isinf(X)] = 0
X[np.isnan(X)] = 0

#%% Divide train and test sets.

test_size = 1024

X_train = X[:-test_size, :]
Y_train = Y[:-test_size, :]
Y_train_class = np.argmax(Y_train, axis=1)

X_test = X[-test_size:, :]
Y_test = Y[-test_size:, :]
Y_test_class = np.argmax(Y_test, axis=1)

print(f"X train: {X_train.shape}")
print(f"Y train: {Y_train.shape}")
print(f"Y train, class: {Y_train_class.shape}\n")
print(f"X test: {X_test.shape}")
print(f"Y test: {Y_test.shape}")
print(f"Y test, class: {Y_test_class.shape}\n")

#%% Create and train regressors, classifiers.

regression_kwargs = {
    'normalize': True,
    'copy_X': True,
}

standard_regression = LinearRegression(**regression_kwargs)
interaction_regression = Pipeline([
    ('interactions', PolynomialFeatures(interaction_only=True, degree=2)),
    ('regression', LinearRegression(**regression_kwargs))
])
# TODO: We could use the classifiers instead...
ridge_regression = Ridge(solver='auto', **regression_kwargs)
lasso_regression = Lasso(**regression_kwargs)
elastic_regression = ElasticNet(
    alpha=1.0, l1_ratio=0.5,
    **regression_kwargs
)


lda = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')
qda = QuadraticDiscriminantAnalysis()

regressors = (
    ('Standard Linear Regression', standard_regression),
    ('Linear Regression w/ Interactions', interaction_regression),
    ('Ridge (L2) Regression', ridge_regression),
    ('LASSO (L1) Regression', lasso_regression),
    ('ElasticNet Regression', elastic_regression)
)
classifiers = (
    ('Linear Discriminant Analysis', lda),
    ('Quadratic Discriminant Analysis', qda)
)

for name, regressor in regressors:
    t0 = time()
    regressor.fit(X_train, Y_train)
    print(f"{time() - t0: .3f} s, {name}")

for name, classifier in classifiers:
    t0 = time()
    classifier.fit(X_train, Y_train_class)
    print(f"{time() - t0: .3f} s, {name}")

#%% Assess predictions.

from sklearn.metrics import accuracy_score

def rotate(n: int, max_n: int):
    if n == max_n - 1:
        return 0
    else:
        return n + 1

def zero_if_huge(n: int):
    if not 8 > n > -8:
        return 0
    else:
        return n

def assess_predictions(X_set: np.ndarray, Y_set: np.ndarray, type_: str, pred_n: int):
    print("-" * 80)
    print(f"Results on {type_} set: ")

    for name, regressor in regressors:
        pred = regressor.predict(X_set)

        for i in range(pred_n):
            label = np.argmax(Y_set[i])
            next_ = rotate(int(label), 16)

            print(f"{zero_if_huge(pred[i][label]): .3f} | "
                  f"{zero_if_huge(pred[i][next_]): .3f} <- {name}")

        accuracy = accuracy_score(
            np.argmax(Y_set, axis=1),
            np.argmax(pred, axis=1))
        print(f"Accuracy: {100 * accuracy: .3f}")

    for name, classifier in classifiers:
        pred_classes = classifier.predict(X_set)

        for i in range(pred_n):
            true_class = np.argmax(Y_set[i])
            print(f"{pred_classes[i]} | "
                  f"{true_class} <- {name}")

        accuracy = accuracy_score(np.argmax(Y_set, axis=1), pred_classes)
        print(f"Accuracy: {100 * accuracy: .3f}")


assess_predictions(X_train, Y_train, "TRAIN", 8)
assess_predictions(X_test, Y_test, "TEST", 8)

# print(regressor.coef_.shape)
# print(regressor.coef_[0])
