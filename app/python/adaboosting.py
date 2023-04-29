import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd


class AdaBoost:
    def __init__(self, num_classifiers=7, classifier=DecisionTreeClassifier()):
        self.X = None
        self.y = None
        self.num_samples = None
        self.num_classifiers = num_classifiers
        self.classifier = classifier
        self.alphas = None
        self.estimators = None

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.num_samples = X.shape[0]

        weights = np.full(self.num_samples, 1 / self.num_samples)

        self.alphas = np.zeros(self.num_samples)
        self.estimators = np.array([])

        for i in range(self.num_samples):
            self.classifier.fit(self.X, self.y, sample_weight=weights)
            self.estimators = np.append(self.estimators, self.classifier)
            self.alphas[i] = self.__calculate_alpha(weights, self.y, self.classifier.predict(self.X))
            weights = np.exp(self.alphas[i] * (y != self. classifier.predict(self.X)))

    def predict(self, X):
        s = 0
        for i in range(len(self.alphas)):
            s += self.alphas[i] * self.estimators[i].predict(X)
        return np.sign(s)

    @staticmethod
    def __calculate_error(weights, y_true, y_pred):
        return np.sum(weights[y_true != y_pred]) / np.sum(weights)

    def __calculate_alpha(self, weights, y_true, y_pred):
        err = self.__calculate_error(weights, y_true, y_pred)
        return np.log((1 - err) / err)


df = pd.read_csv("../csv/classification.csv")
y = df["default"].values
df = df.drop(["default"], axis=1)
df = df.select_dtypes(include=["number"]).values
dt = DecisionTreeClassifier(max_depth=2)

adaboost = AdaBoost(10, dt)
adaboost.fit(df, y)
y_pred = adaboost.predict(df)
print(y_pred)
