import numpy as np

# --------------------------------------------------------------------------------------------------------------------
#            utilidades internas

def _entropy(y):
    y = np.asarray(y)

    if y.size == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log2(probs)))

def _majority_class(y):
    values, counts = np.unique(y, return_counts=True)
    return values[int(np.argmax(counts))]

def _information_gain(X,y, feature):
    base = _entropy(y)
    column = X[:, feature]
    total = len(y)
    weighted = 0.0
    for value in np.unique(column):
        mask = column == value
        weighted += (mask.sum() / total) * _entropy(y[mask])

    return base - weighted

def _predict_one(tree, x):
    node = tree
    while node.get("type") == "node":
        value = x[node["feature"]]
        child = node["children"].get(value)
        if child is None:
            return node["default"]
        node = child
    return node["class"]

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

def bootstrap_sample(X, y, random_state=None):
    X = np.asarray(X)
    y = np.asarray(y)
    n = X.shape[0]
    rng = np.random.RandomState(random_state)
    indices = rng.randint(0, n, size=n)
    return X[indices], y[indices]

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------

def build_id3_tree(X, y, features=None, depth=0, max_depth=None):
    return
