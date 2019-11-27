from DecisionTree import DecisionTree

import numpy as np


class RandomForest(object):
    def __init__(self, n_trees=0, mode=None, criteria=None,
                 sample_rate=0.75, feature_rate=0.75, seed=1):
        """
        A random forest consists of N_TREES decision trees.
        Args:
            n_trees: Number of decision trees in the forest.
            mode: string or list, based on either
                'ig' - information gain, or,
                'gini' - gini index.
            criteria: dict or list, specify the stopping criteria.
            sample_rate: Sample data rate for each decision tree.
            feature_rate: Sample feature rate for each decision tree.
            seed: Random sample seed.
        """
        if isinstance(mode, str):
            mode = [mode for _ in range(n_trees)]
        elif isinstance(mode, list) and len(mode) == n_trees:
            pass
        else:
            raise ValueError("mode type error, it needs to be either "
                             "a string, or a list with len == n_trees.")
        if isinstance(criteria, dict):
            criteria = [criteria for _ in range(n_trees)]
        elif isinstance(criteria, list) and len(criteria) == n_trees:
            pass
        else:
            raise ValueError("criteria type error, it needs to be either "
                             "a dict, or a list with len == n_trees.")
        self.n_trees = n_trees
        self.mode = mode
        self.criteria = criteria
        self.sample_rate = sample_rate
        self.feature_rate = feature_rate
        self.seed = seed
        self.trees = []

    def set_criteria(self, criteria):
        """
        Change the criteria of current random forest.
        After changing, clear the trees for fitting again.
        """
        if isinstance(criteria, dict):
            criteria = [criteria for _ in range(self.n_trees)]
        elif isinstance(criteria, list) and len(criteria) == self.n_trees:
            pass
        else:
            raise ValueError("criteria type error, it needs to be either "
                             "a dict, or a list with len == n_trees.")
        self.criteria = criteria
        self.trees = []

    def set_mode(self, mode):
        """
        Change the mode of current random forest.
        After changing, clear the trees for fitting again.
        """
        if isinstance(mode, str):
            mode = [mode for _ in range(self.n_trees)]
        elif isinstance(mode, list) and len(mode) == self.n_trees:
            pass
        else:
            raise ValueError("mode type error, it needs to be either "
                             "a string, or a list with len == n_trees.")
        self.mode = mode
        self.trees = []

    def set_sample_rate(self, rate):
        """
        Change the sample rate of current random forest.
        After changing, clear the trees for fitting again.
        """
        self.sample_rate = rate
        self.trees = []

    def set_feature_rate(self, rate):
        """
        Change the feature rate of current random forest.
        After changing, clear the trees for fitting again.
        """
        self.feature_rate = rate
        self.trees = []

    def fit(self, X, y):
        """
        Train each tree in the forest.
        After training, report the training evaluation.
        """
        np.random.seed(self.seed)
        for i in range(self.n_trees):
            sample = int(X.shape[0] * self.sample_rate)
            sample_idx = np.random.randint(0, X.shape[0], size=sample)
            sample_X = X[sample_idx]
            sample_y = y[sample_idx]
            tree = DecisionTree(
                sample_X, sample_y,
                self.mode[i], self.criteria[i],
                seed=self.seed,
                feature_rate=self.feature_rate
            )
            tree.train(sample_X, sample_y, verbose=False)
            self.trees.append(tree)
        print("#Train\t", end="")
        self.validate(X, y)

    def predict(self, X):
        """
        Predict the label given X, each row of X represents
        a sample.
        """
        predictions = []
        for i in range(self.n_trees):
            predictions.append(self.trees[i].predict(X))
        predictions = np.array(predictions)
        return [np.bincount(predictions[:, i]).argmax()
                for i in range(predictions.shape[1])]

    def validate(self, val_X, val_y):
        """
        Validate the performance given X and y.
        """
        pre_y = self.predict(val_X)
        correct = [1 if val_y[i] == pre_y[i] else 0 for i in range(len(val_y))]
        rate = sum(correct) / val_y.shape[0]
        print("Random Forest | s_rate: %.2f | f_rate: %.2f "
              "| mode: %s | val rate: %.4f" %
              (self.sample_rate, self.feature_rate, self.mode, rate))
        return rate
