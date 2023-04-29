from abc import ABC

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# TODO v
# python main.py --name Sona
# send the image to the TA as well
# take into account that the line running it shouldn't be only for classification.csv, also be cautious of non-numerical
# data, check get dummies handler of test


# todo: keep the tree as a dict like this
# returns: {"feature": 3, "threshold": 2.5, "true": {"feature: , "threshold": ,}, "false":} (if there has been a place
# to cut & when we want to create the tree up to the end) try adding max depth & so on
# or
# None if there was no more place

# much better if we do it this way, don't make dicts, make 2 classes as node and leaf if there should be returned a
# number, return a node type object which has node.left, node.right, by default .left/.right = None, continue as
# long as no node is created if all of them are leafs
# leaf(node) has a number of which class it came and the probability of that class (second one is optional)

# without maxdepth do as long as it can go deeper

# bolor featureneri bolor hnaravor spliterov ancnum enq amenapoqr giniov@ vercnum enq
# make this work for categoricals either

# todo: write for classsification


class DecisionTree:

    CLASSES = None

    def __init__(self):
        self.classes = None
        self.X = None
        self.y = None
        self.tree = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        # creates the dictionary/tree
        self.X = X
        self.y = y
        DecisionTree.CLASSES = np.unique(self.y)
        for i in range(self.X.shape[1]):
            feature = self.X[:, i]
            self.tree = Node(feature, self.y)

    def predict(self, X):
        # should go over tree, go over true/false, returns numbers as predictions
        pass


class Node:
    def __init__(self, feature, y):
        self.left: Node
        self.right: Node
        self.y: np.ndarray = y
        self.feature: np.ndarray = feature
        self.value = None
        self.threshold: float = 0.0
        self.gini: float = 1.1

        # TODO: self.value

        threshold_candidates = np.unique(feature)

        for thr in threshold_candidates:
            y_left = self.y[self.feature > thr]
            y_right = self.y[self.feature <= thr]
            current_gini = self.gini_node(y_left, y_right)
            if current_gini < self.gini:
                self.gini = current_gini
                self.threshold = thr

        self.y_left = self.y[self.feature > self.threshold]
        self.y_right = self.y[self.feature > self.threshold]
        self.left = Node(self.feature[self.feature > self.threshold], self.y_left)
        self.right = Node(self.feature[self.feature <= self.threshold], self.y_right)

    def gini_node(self, y1, y2):
        return (
            y1.sum() * self.__gini_branch(y1) + y2.sum() * self.__gini_branch(y2)
        ) / (y1.sum() + y2.sum())

    @staticmethod
    def __gini_branch(y):
        p = 0
        for y_i in DecisionTree.CLASSES:
            p += y_i**2
        return 1 - p


class Leaf:
    def __init__(self):
        self.value = None


X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=24)
dt = DecisionTree()
dt.fit(X_train, y_train)
