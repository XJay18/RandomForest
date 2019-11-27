import numpy as np
import random
from scipy import stats
from utils import features, class_names

from rcviz import viz

# for split between two integer values
HALF = 0.5


class DecisionTree(object):
    class Node(object):
        def __init__(self, label):
            """
            The node in a decision tree.
            Args:
                label: The class label of a node.
                depth: The depth of a node in a decision tree.
            """
            self.label = label
            self.left = None
            self.right = None
            self.idx = None
            self.thresh = None

        def set_l(self, node):
            """
            Set NODE as current left child.
            Args:
                node: The left child.
            """
            self.left = node

        def set_r(self, node):
            """
            Set NODE as current right child.
            Args:
                node: The right child.
            """
            self.right = node

        def set_idx(self, idx):
            """
            Set feature to split.
            Args:
                idx: The column index of the feature to split.
            """
            self.idx = idx

        def set_thr(self, thresh):
            """
            Set split threshold.
            Args:
                thresh: The threshold to split the data.
                    If feature <= threshold, then comes
                    to the left subtree, else, the right
                    subtree.
            """
            self.thresh = thresh

        def __str__(self):
            if self.idx is not None and self.thresh is not None:
                return str(features[self.idx]) + " thr: " + str(self.thresh)
            else:
                return "leaf"

    def __init__(self, X, y, mode, criteria, seed=1, feature_rate=1):
        """
        A decision tree.
        Args:
            X: The original features to train.
            y: The original labels.
            mode: Based on either 'ig' - information gain, or,
                                'gini' - gini index.
            criteria: dict, specify the stopping criteria.
            feature_rate: Helper argument for random forest to random
                select some features from the original features.
        """
        if mode not in ["ig", "gini"]:
            raise ValueError("mode should be either 'ig' or 'gini', "
                             "but found %s" % mode)
        self.tree = None
        self.n_features = X.shape[1]
        self.n_classes = len(set(y))
        self.mode = mode
        self.max_depth = criteria.get("max_depth", None)
        self.node_purity = criteria.get("node_purity", None)
        self.min_gain = criteria.get("min_gain", None)
        self.seed = seed
        self.feature_rate = feature_rate

    def set_criteria(self, criteria):
        """
        Change the criteria of current decision tree.
        """
        self.max_depth = criteria.get("max_depth", None)
        self.node_purity = criteria.get("node_purity", None)
        self.min_gain = criteria.get("min_gain", None)

    def feature_selector(self):
        """
        Return a list of index of features to be considered to
        split during training.
        """
        idx = list(range(self.n_features))
        if self.feature_rate == 1:
            return idx
        random.seed(self.seed)
        feature_idx = random.sample(
            idx, int(self.feature_rate * self.n_features))
        return sorted(feature_idx)

    @staticmethod
    def entropy(y):
        _, counts = np.unique(y, return_counts=True)
        return stats.entropy(counts, base=2)

    @staticmethod
    def information_gain(X, y, thresh):
        en = DecisionTree.entropy(y)
        num_d = y.shape[0]
        left_indicies = X <= thresh
        # left partition
        left_sub = y[left_indicies]
        en_left = DecisionTree.entropy(left_sub)
        en_left = (left_sub.shape[0] / num_d) * en_left
        # right partition
        right_sub = y[~left_indicies]
        en_right = DecisionTree.entropy(right_sub)
        en_right = (right_sub.shape[0] / num_d) * en_right
        # information gain
        ig = en - en_left - en_right
        return ig

    @staticmethod
    def gini_impurity(y):
        total = y.shape[0]
        _, counts = np.unique(y, return_counts=True)
        return 1 - np.sum(np.square(counts / total))

    @staticmethod
    def gini_purification(X, y, thresh):
        num_d = y.shape[0]
        left_indicies = X <= thresh
        # left partition
        left_sub = y[left_indicies]
        gini_left = DecisionTree.gini_impurity(left_sub)
        gini_left = (left_sub.shape[0] / num_d) * gini_left
        # right partition
        right_sub = y[~left_indicies]
        gini_right = DecisionTree.gini_impurity(right_sub)
        gini_right = (right_sub.shape[0] / num_d) * gini_right
        # gini index
        gini_index = gini_left + gini_right
        return gini_index

    def split(self, X, y, idx, thresh):
        """
        Split the data given the index and threshold.
        Args:
            X: Data to split.
            y: Labels corresponding to the data.
            idx: int, specify a vector of feature.
            thresh: float, used for splitting the data into
            two branches.
        """
        feature = X[:, idx]
        left_indices = feature <= thresh
        left_x = X[left_indices]
        left_y = y[left_indices]
        right_x = X[~left_indices]
        right_y = y[~left_indices]
        return (left_x, left_y), (right_x, right_y)

    def segmenter(self, X, y):
        """
        Find the best feature and threshold to split.
        """
        best_idx = 0
        best_thresh = 0
        best_criterion = -np.inf if self.mode == "ig" else np.inf
        for idx in self.feature_selector():
            feature = X[:, idx]
            values = np.unique(feature)
            for value in values:
                if self.mode == "ig":
                    c = self.information_gain(feature, y, value + HALF)
                    if c > best_criterion:
                        best_criterion = c
                        best_idx = idx
                        best_thresh = value + HALF
                else:
                    c = self.gini_purification(feature, y, value + HALF)
                    if c < best_criterion:
                        best_criterion = c
                        best_idx = idx
                        best_thresh = value + HALF
        if self.min_gain and self.mode == "ig" \
                and best_criterion < self.min_gain:
            return None, None
        return best_idx, best_thresh

    def train(self, X, y, verbose=True):
        """
        Train the decision tree given training data X and y.
        If verbose, after training, return the training evaluation.
        """
        self.tree = self.__train(X, y)
        if verbose:
            print("#Train\t", end="")
            return self.validate(X, y)

    def __train(self, X, y, depth=0):
        """
        Recursively split the data and create nodes to build the
        decision tree.
        """
        counts = [np.sum(y == i) for i in range(self.n_classes)]
        label = np.argmax(counts)
        node = self.Node(label)
        # Check if the node has no data to split
        if X.shape[0] == 0:
            return node
        if np.min(counts) == 0:
            return node
        # Check the node purity stopping criteria.
        if self.node_purity:
            proportion = [i / X.shape[0] for i in counts]
            if np.max(proportion) >= self.node_purity:
                return node
        # Check the max depth stopping criteria.
        if not self.max_depth or depth < self.max_depth:
            idx, thr = self.segmenter(X, y)
            # Check the min information gain stopping criteria.
            if idx is None and thr is None:
                return node
            # Check if X[:, idx] have the same value.
            if np.unique(X[:, idx]).shape[0] == 1:
                return node
            node.set_idx(idx)
            node.set_thr(thr)
            sub_l, sub_r = self.split(X, y, idx, thr)
            node.set_l(self.__train(*sub_l, depth=depth + 1))
            node.set_r(self.__train(*sub_r, depth=depth + 1))
        return node

    def predict(self, X):
        """
        Predict the label given X, each row of X represents
        a sample.
        """
        return [self.__predict(sample) for sample in X]

    def __predict(self, sample):
        """
        Predict the label given X, one sample.
        """
        node = self.tree
        while node.left:
            if sample[node.idx] <= node.thresh:
                node = node.left
            else:
                node = node.right
        return node.label

    def validate(self, val_X, val_y):
        """
        Validate the performance given X and y.
        """
        pre_y = self.predict(val_X)
        correct = [1 if val_y[i] == pre_y[i] else 0 for i in range(len(val_y))]
        rate = sum(correct) / val_y.shape[0]
        print("Decision Tree | MD: %s | NP: %s | MG: %s |"
              " mode: %s | val rate: %.4f" %
              (self.max_depth, self.node_purity,
               self.min_gain, self.mode, rate))
        return rate

    @staticmethod
    @viz
    def n(tree):
        """
        For visualization usage.
        """
        node = tree
        if node.left is None and node.right is None:
            return "*label: " + str(class_names[node.label])
        DecisionTree.n(node.left)
        DecisionTree.n(node.right)

    @staticmethod
    def __inorder(node, seq):
        if node:
            DecisionTree.__inorder(node.left, seq)
            seq.append(" | " + str(node) + " | ")
            DecisionTree.__inorder(node.right, seq)

    @staticmethod
    def __preorder(node, seq):
        if node:
            seq.append(" | " + str(node) + " | ")
            DecisionTree.__preorder(node.left, seq)
            DecisionTree.__preorder(node.right, seq)

    def __repr__(self):
        seq = []
        DecisionTree.__inorder(self.tree, seq)
        string_in = "".join(seq)
        seq = []
        DecisionTree.__preorder(self.tree, seq)
        string_pre = "".join(seq)
        return "Inorder: \n" + string_in + "\n\nPreorder: \n" + string_pre
