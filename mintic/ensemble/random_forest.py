import numpy as np

def boostrap_sample():
    return

def entropy(y):
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return -np.sum(probs * np.log2(probs + 1e-12))

def _information_gain(X,y, feature_idx):
    base_entropy = entropy(y)
    values, counts = np.unique(X[:, feature_idx], return_counts=True)

    weighted_entropy = 0.0

    for v, c in zip(values, counts):
        subset_y = y[X[:, feature_idx] == v]
        weighted_entropy += (c / len(y)) * entropy(subset_y)

def build_id3_tree(X, y, feature_indices=None, depth=0, max_depth=None):

    X = np.asarray(X)
    y = np.asarray(y)
