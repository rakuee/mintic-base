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

    X = np.asarray(X)
    y = np.asarray(y)
 
    if features is None:
        features = list(range(X.shape[1]))
 
    # Caso 1: nodo vacío
    if y.size == 0:
        return {"type": "leaf", "class": None}
 
    # Caso 2: nodo puro
    if np.unique(y).size == 1:
        return {"type": "leaf", "class": y[0]}
 
    # Caso 3: sin atributos disponibles o profundidad máxima alcanzada
    if not features or (max_depth is not None and depth >= max_depth):
        return {"type": "leaf", "class": _majority_class(y)}
 
    # Selección del mejor atributo por ganancia de información
    gains = [_information_gain(X, y, f) for f in features]
    best_index = int(np.argmax(gains))
 
    # Caso 4: ninguna partición aporta información
    if gains[best_index] <= 0:
        return {"type": "leaf", "class": _majority_class(y)}
 
    best_feature = features[best_index]
    remaining = [f for f in features if f != best_feature]
 
    node = {
        "type": "node",
        "feature": best_feature,
        "default": _majority_class(y),
        "children": {},
    }
 
    column = X[:, best_feature]
    for value in np.unique(column):
        mask = column == value
        node["children"][value] = build_id3_tree(
            X[mask], y[mask], remaining, depth + 1, max_depth
        )
 
    return node

#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------

def build_random_forest(X, y, n_trees=10, random_state=None):
    rng = np.random.default_rng(random_state)
    forest = []

    n_samples = X.shape[0]

    for i in range(n_trees):
        indices = rng.integers(low=0, high=n_samples, size=n_samples)

        X_boot = X[indices]
        Y_boot = y[indices]

        tree = build_id3_tree(X_boot, Y_boot)

        forest.append(tree)

    return forest

