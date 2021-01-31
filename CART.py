import numpy as np


class CART():
    def __init__(self, criterion='gini', min_criterion=0.05):
        """
        Initalize CART Algorith decision tree.
        criterion: 'gini' or 'entropy'
        """
        self.feature = None
        self.label = None
        self.n_samples = None
        self.gain = None
        self.left = None
        self.right = None
        self.threshold = None
        self.depth = 0

        self.root = None
        self.criterion = criterion

        self.min_criterion = min_criterion

    def fit(self, X, y):
        self.root = CART()
        self.root._grow_tree(X, y, self.criterion)
        self.root._prune(self.min_criterion, self.root.n_samples)

    def predict(self, X):
        return np.array([self.root._predict(row) for row in X])

    def print_tree(self):
        self.root._show_tree(0, ' ')

    def _grow_tree(self, X, y, criterion):
        self.n_samples = X.shape[0]

        if len(np.unique(y)) == 1:
            self.label = y[0]
            return

        best_gain = 0.0
        best_feature = None
        best_threshold = None

        self.label = max([(c, len(y[y == c]))
                          for c in np.unique(y)], key=lambda x: x[1])[0]

        impurity_node = self._calc_impurity(criterion, y)

        for col in range(X.shape[1]):
            feature_level = np.unique(X[:, col])
            thresholds = (feature_level[:-1] + feature_level[1:]) / 2.0

            for threshold in thresholds:
                y_left = y[X[:, col] <= threshold]
                impurity_l = self._calc_impurity(criterion, y_left)
                n_l = float(y_left.shape[0]) / self.n_samples

                y_right = y[X[:, col] > threshold]
                impurity_r = self._calc_impurity(criterion, y_right)
                n_r = float(y_right.shape[0]) / self.n_samples

                impurity_gain = impurity_node - \
                    (n_l * impurity_l + n_r * impurity_r)
                if impurity_gain > best_gain:
                    best_gain = impurity_gain
                    best_feature = col
                    best_threshold = threshold

        self.feature = best_feature
        self.gain = best_gain
        self.threshold = best_threshold
        self._split_tree(X, y, criterion)

    def _split_tree(self, X, y, criterion):
        X_left = X[X[:, self.feature] <= self.threshold]
        y_left = y[X[:, self.feature] <= self.threshold]
        self.left = CART()
        self.left.depth = self.depth + 1
        self.left._grow_tree(X_left, y_left, criterion)

        X_right = X[X[:, self.feature] > self.threshold]
        y_right = y[X[:, self.feature] > self.threshold]
        self.right = CART()
        self.right.depth = self.depth + 1
        self.right._grow_tree(X_right, y_right, criterion)

    @staticmethod
    def _calc_entropy(target):
        entropy = 0.0
        for c in np.unique(target):
            p = float(len(target[target == c])) / target.shape[0]
            if p > 0.0:
                entropy -= p * np.log2(p)
        return entropy

    @staticmethod
    def _calc_gini(target):
        return 1.0 - sum([(float(len(target[target == c])) / float(target.shape[0])) ** 2.0 for c in np.unique(target)])

    def _calc_impurity(self, criterion, target):
        if criterion == 'gini':
            return self._calc_gini(target)
        elif criterion == 'entropy':
            return self._calc_entropy(target)
        else:
            raise Exception(f"Wrong criterion used! {criterion}")

    def _prune(self, min_criterion, n_samples):
        if self.feature is None:
            return

        self.left._prune(min_criterion, n_samples)
        self.right._prune(min_criterion, n_samples)

        if self.left.feature is None and self.right.feature is None:
            if (self.gain * float(self.n_samples) / n_samples) < min_criterion:
                # Prune:
                self.left = None
                self.right = None
                self.feature = None

    def _predict(self, obs):
        if self.feature != None:
            if obs[self.feature] <= self.threshold:
                return self.left._predict(obs)
            else:
                return self.right._predict(obs)
        else:
            return self.label

    def _show_tree(self, depth, cond):
        base = '    ' * depth + cond
        if self.feature != None:
            print(base + 'if X[' + str(self.feature) +
                  '] <= ' + str(self.threshold))
            self.left._show_tree(depth+1, 'then ')
            self.right._show_tree(depth+1, 'else ')
        else:
            print(base + '{value: ' + str(self.label) +
                  ', samples: ' + str(self.n_samples) + '}')

