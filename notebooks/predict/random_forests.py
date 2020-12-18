import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from time import time

def avg_depths(random_forest):
    depth_sum = 0
    tree_count = 0

    for tree in random_forest.estimators_:
        depth_sum += tree.tree_.max_depth
        tree_count += 1

        if tree_count >= 2**10:
            break

    return depth_sum / tree_count

def avg_node_counts(random_forest):
    nodes_sum = 0
    tree_count = 0

    for tree in random_forest.estimators_:
        nodes_sum += tree.tree_.node_count
        tree_count += 1

        if tree_count >= 2**10:
            break

    return nodes_sum / tree_count

def compare_ccp_alphas(X, y, ccp_alphas, **kwargs):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=53)

    decision_trees = {}
    for alpha in ccp_alphas:
        random_forest = RandomForestClassifier(ccp_alpha=alpha, **kwargs)

        t0 = time()
        print(f"Training RF with ALPHA={alpha: .5f}...")
        random_forest.fit(X_train, y_train)
        print(f"{time() - t0: .3f} secs.")

        decision_trees[alpha] = random_forest

    train_scores = []
    test_scores = []
    node_counts = []
    # depth_counts = []

    for alpha in ccp_alphas:
        rf = decision_trees[alpha]

        train_scores.append(rf.score(X_train, y_train))
        test_scores.append(rf.score(X_test, y_test))
        node_counts.append(avg_node_counts(rf))

    fig, ax = plt.subplots()
    ax.set_xlabel("ALPHA")
    ax.set_ylabel("Accuracy")
    ax.plot(ccp_alphas, train_scores, color='k', label='train', drawstyle='steps-post')
    ax.plot(ccp_alphas, test_scores, color='m', label='test', drawstyle='steps-post')
    ax.legend()

    plt.show()

    fig, ax = plt.subplots()
    ax.set_xlabel("ALPHA")
    ax.set_ylabel("Total nodes")
    ax.plot(ccp_alphas, node_counts, color='k', drawstyle='steps-post')

    plt.show()


if __name__ == '__main__':
    glove_dir = "/Users/ericcarlson/Desktop/Personal Projects/basic-document-nlp/resources/glove"

    X = np.load(os.path.join(glove_dir, "standard", "predictor-matrix.npy"))
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    y = np.load(os.path.join(glove_dir, "standard", "response-matrix.npy"))
    y = np.argmax(y, axis=1)

    ccp_alphas = np.arange(1, 12) * 1e-3

    compare_ccp_alphas(X, y,
                       ccp_alphas=ccp_alphas,
                       criterion='entropy', n_estimators=2**6)
