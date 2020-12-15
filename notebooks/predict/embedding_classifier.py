#%% Generate matrices, remove NaN and infinite values.

import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from notebooks.util.logging import numerical_list_to_string

glove_dir = "/Users/ericcarlson/Desktop/Personal Projects/basic-document-nlp/resources/glove"

predictors = {}
responses = {}

def perform_classification(embedding_dir: str, classifier, name: str):
    if embedding_dir not in predictors:
        X = np.load(os.path.join(embedding_dir, "predictor-matrix.npy"))
        X[np.isinf(X)] = 0
        X[np.isnan(X)] = 0

        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        print("mean: ", numerical_list_to_string(scaler.mean_))
        print("var: ", numerical_list_to_string(scaler.var_))

        Y = np.load(os.path.join(embedding_dir, "response-matrix.npy"))
        Y = np.argmax(Y, axis=1)

        predictors[embedding_dir] = X
        responses[embedding_dir] = Y
    else:
        X = predictors[embedding_dir]
        Y = responses[embedding_dir]

    X_train, X_test = train_test_split(X)
    Y_train, Y_test = train_test_split(Y)
    print(f"X train, test: {X_train.shape}, {X_test.shape}")
    print(f"Y train, test: {Y_train.shape}, {Y_test.shape}")

    t0 = time()
    print(f"Training {name}...")
    classifier.fit(X_train, Y_train)
    print(f"{time() - t0: .4f} secs")

    print(f"Test Accuracy:  {100 * classifier.score(X_test, Y_test): .4f}")
    print(f"Train Accuracy: {100 * classifier.score(X_train, Y_train): .4f}")

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

#%% Perform comparisons.

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from time import time

forest_kwargs = {
    'criterion': 'entropy',
    'ccp_alpha': .0035
}

standard_regression = LinearRegression()
interaction_regression = Pipeline([
    ('interactions', PolynomialFeatures(interaction_only=True)),
    ('regression', LinearRegression())
])
ridge_regression = Ridge(solver='auto')
lasso_regression = Lasso(max_iter=2**15)
elastic_regression = ElasticNet(max_iter=2*15, alpha=1.0, l1_ratio=0.5)

# solver = 'eigen', shrinkage = 'auto'
lda = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')
qda = QuadraticDiscriminantAnalysis()

forest_default = RandomForestClassifier()
forest_128 = RandomForestClassifier(n_estimators=128, **forest_kwargs)
forest_256 = RandomForestClassifier(n_estimators=256, **forest_kwargs)

regressors = (
    ('Standard Linear Regression', standard_regression),
    ('Linear Regression w/ Interactions', interaction_regression),
    ('Ridge (L2) Regression', ridge_regression),
    ('LASSO (L1) Regression', lasso_regression),
    ('ElasticNet Regression', elastic_regression)
)
random_forests = (
    ('Random Forest (Default)', forest_default),
    ('Random Forest (128)', forest_128),
    ('Random Forest (256)', forest_256)
)
classifiers = (
    ('Linear Discriminant Analysis', lda),
    ('Quadratic Discriminant Analysis', qda),
    *random_forests
)

if __name__ == '__main__':
    for subfolder in ("standard", "twitter"):
        for name, classifier in classifiers:
            perform_classification(
                embedding_dir=os.path.join(glove_dir, subfolder),
                classifier=classifier,
                name=name
            )

    print('-' * 80)

    for name, random_forest in random_forests:
        trees = random_forest.estimators_
        print(f"Decision tree sizes for {name}")
        print(f"{trees[0].tree_.node_count}, "
              f"{trees[1].tree_.node_count}, "
              f"{trees[2].tree_.node_count}, ...")
