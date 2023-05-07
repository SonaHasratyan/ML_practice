import numpy as np
from sklearn.datasets import make_blobs
from random import choice


class DBScan:
    def __init__(self, epsilon=0.001, density=5):
        self.distances = None
        self.m = 0
        self.epsilon = epsilon
        self.density = density

    def fit(self, X):
        self.m = X.shape[0]
        self.__calculate_distances(X)

        clusters = []
        tmp_X = X
        non_core_indices = []
        indices = np.arange(self.m)

        while True:
            while True:
                index = choice(indices)
                new_cluster_bool = self.distances[index] < self.epsilon
                new_cluster = tmp_X[new_cluster_bool]
                if new_cluster.shape[0] >= self.density:
                    break

            clusters.append(new_cluster)
            tmp_X = tmp_X[not new_cluster]

            # todo

            if tmp_X.shape[0] == 0:
                break

    def predict(self, X_test):
        pass

    def __calculate_distances(self, X):
        self.distances = np.zeros((self.m, self.m))

        for i in range(self.m):
            for j in range(self.m - i):
                if i != j:
                    self.distances[i][j] = np.linalg.norm(X[i] - X[j])
                    self.distances[j][i] = self.distances[i][j]

