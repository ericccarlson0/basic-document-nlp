import numpy as np
import os

from sklearn.linear_model import LinearRegression

resources_dir = "/Users/ericcarlson/Desktop/Personal Projects/basic-document-nlp/resources/glove"
predictor_dir = os.path.join(resources_dir, "predictors.npy")
response_dir = os.path.join(resources_dir, "responses.npy")

X = np.load(predictor_dir)
Y = np.load(response_dir)

print(f"Predictor shape: {X.shape}")
print(f"Response shape: {Y.shape}")

regressor = LinearRegression(normalize=True, copy_X=True, n_jobs=4)
regressor.fit(X, Y)

predicted = regressor.predict(X)
for i in range(8):
    print(i, predicted[i])

# print(regressor.coef_.shape)
# print(regressor.coef_[0])
