import numpy as np
from collections import Counter
from random import choices, seed

class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
    
    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)
    
    def _grow_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        if depth < self.max_depth and num_samples >= self.min_samples_split:
            best_split = self._find_best_split(X, y, num_features)
            if best_split:
                left_idxs, right_idxs = best_split["indices"]
                left_tree = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
                right_tree = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
                return {"feature_idx": best_split["feature_idx"], "threshold": best_split["threshold"],
                        "left": left_tree, "right": right_tree}
        return {"leaf": True, "value": self._most_common_label(y)}
    
    def _find_best_split(self, X, y, num_features):
        best_split = {}
        max_gain = -float('inf')
        for feature_idx in range(num_features):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                gain, indices = self._split_data(X, y, feature_idx, threshold)
                if gain > max_gain:
                    best_split = {"feature_idx": feature_idx, "threshold": threshold, "indices": indices}
                    max_gain = gain
        return best_split if max_gain > 0 else None

    def _split_data(self, X, y, feature_idx, threshold):
        left_idxs = np.where(X[:, feature_idx] <= threshold)[0]
        right_idxs = np.where(X[:, feature_idx] > threshold)[0]
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0, (left_idxs, right_idxs)
        gain = self._information_gain(y, left_idxs, right_idxs)
        return gain, (left_idxs, right_idxs)
    
    def _information_gain(self, y, left_idxs, right_idxs):
        p = float(len(left_idxs)) / len(y)
        return self._gini(y) - p * self._gini(y[left_idxs]) - (1 - p) * self._gini(y[right_idxs])
    
    def _gini(self, y):
        m = len(y)
        return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in np.unique(y))
    
    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]
    
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])
    
    def _traverse_tree(self, x, node):
        if node.get("leaf", False):
            return node["value"]
        if x[node["feature_idx"]] <= node["threshold"]:
            return self._traverse_tree(x, node["left"])
        else:
            return self._traverse_tree(x, node["right"])

class RandomForest:
    def __init__(self, n_estimators=10, max_depth=10, min_samples_split=2, max_features='sqrt', seed_value=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.seed_value = seed_value
        self.trees = []
    
    def fit(self, X, y):
        seed(self.seed_value)
        self.trees = []
        for i in range(self.n_estimators):
            print(f'Training tree {i+1}/{self.n_estimators}')
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
    
    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = choices(range(n_samples), k=n_samples)
        return X[indices], y[indices]
    
    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = [Counter(tree_pred).most_common(1)[0][0] for tree_pred in tree_preds]
        return np.array(y_pred)
    
    def predict_proba(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred_proba = [np.mean(tree_pred) for tree_pred in tree_preds]
        return np.array(y_pred_proba)