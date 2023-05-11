import numpy as np
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt


class SpectralClustering:
    def __init__(self, n_clusters=4, n_neighbors=7):
        self.m = 0
        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors
        self.graph = None
        self.W = None  # similarity matrix
        self.D = None  # diagonal matrix
        self.L = None  # Laplacian

    def fit(self, X):
        self.m = X.shape[0]
        self.__init_matrices(X)

    def predict(self, X_test):
        pass

    def __init_matrices(self, X):
        self.W = np.zeros((self.m, self.m))
        self.D = self.W.copy()

        for i in range(self.m):
            for j in range(self.m - i):
                if i == j:
                    continue
                self.W[i][j] = np.exp(-(np.linalg.norm(X[i] - X[j])) / (0.5 * np.sqrt(self.m)))

            for _ in range(self.n_neighbors):
                max_el = max(self.W[i])
                max_ind = np.where(self.W[i] == max_el)[0][0]
                self.W[max_ind][i] = max_el
                self.W[i][max_ind] = 0

            # todo: v
            # self.W[:, i] = sum(self.W[:, i]) ** 2
            self.W[i] = self.W[:, i]

            self.D[i][i] = sum(self.W[i])

        self.L = self.D - self.W


M = 1000
X_circles, y_circles = make_circles(n_samples=M, noise=0.02, factor=0.8, random_state=78)
_X_circles, _y_circles = make_circles(n_samples=M, noise=0.02, factor=0.5, random_state=78)
_X_circles /= 2
_y_circles += 2
X_circles = np.concatenate([X_circles, _X_circles], axis=0)
y_circles = np.concatenate([y_circles, _y_circles])
plt.figure(figsize=(16, 8))
plt.scatter(X_circles[:, 0], X_circles[:, 1], c=y_circles)
plt.show()

sc = SpectralClustering()
sc.fit(X_circles)


