import numpy as np
from sklearn.datasets import make_circles, make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering


"""
    The algorithm is based on this paper: https://arxiv.org/pdf/0711.0189.pdf
"""


class _SpectralClustering:
    """
    Normalized spectral clustering according to Ng, Jordan, and Weiss (2002)
    """

    def __init__(self, n_clusters=4, n_neighbors=7):
        self.m = 0
        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors
        self.graph = None
        # similarity (affinity at first place) could have been done via kneighbors_graph of sklearn
        self.W = None  # similarity matrix
        self.D = None  # diagonal matrix
        self.L = None  # Laplacian
        self.L_sym = None
        self.clusters = []

    def fit(self, X):
        self.m = X.shape[0]
        self.__init_matrices(X)

        return self


    def fit_predict(self, X):
        self.m = X.shape[0]
        self.__init_matrices(X)
        km = KMeans(n_clusters=self.n_clusters)
        km.fit(self.T)
        km.predict(self.T)
        km = KMeans(n_clusters=self.n_clusters)
        km.fit(self.T)
        self.clusters = km.predict(self.T)

        return self.clusters

    def __init_matrices(self, X):
        self.W = np.zeros((self.m, self.m))
        self.D = self.W.copy()

        for i in range(self.m):
            for _j in range(self.m - i):
                j = i + _j
                if i == j:
                    continue
                self.W[i][j] = np.exp(-(np.linalg.norm(X[i] - X[j])))

            for _ in range(self.n_neighbors):
                max_el = max(self.W[i])
                max_ind = np.where(self.W[i] == max_el)[0][0]
                self.W[max_ind][i] = max_el
                self.W[i][max_ind] = 0

            # todo: v
            self.W[:, i] /= sum(self.W[:, i]) ** 2
            self.W[i] = self.W[:, i]

            self.D[i][i] = sum(self.W[i])

        self.L = self.D - self.W

        D_minus_sqrt = self.D.copy()
        np.fill_diagonal(D_minus_sqrt, 1 / np.sqrt(self.D).diagonal())

        self.L_sym = np.identity(self.m) - D_minus_sqrt @ self.W @ D_minus_sqrt

        self.U = np.linalg.eigh(self.L_sym)[1][:, : self.n_clusters]
        self.T = np.array(
            [
                self.U[i] / np.sqrt(sum(np.square(self.U[i])))
                for i in range(self.U.shape[0])
            ]
        )


_sc = _SpectralClustering(n_neighbors=15)
# sc = SpectralClustering(4, n_neighbors=7)


M = 1000
X_circles, y_circles = make_circles(
    n_samples=M, noise=0.02, factor=0.8, random_state=78
)
_X_circles, _y_circles = make_circles(
    n_samples=M, noise=0.02, factor=0.5, random_state=78
)
_X_circles /= 2
_y_circles += 2
X_circles = np.concatenate([X_circles, _X_circles], axis=0)
y_circles = np.concatenate([y_circles, _y_circles])

plt.figure(figsize=(16, 8))
plt.scatter(X_circles[:, 0], X_circles[:, 1], c=y_circles)
plt.show()

_sc.fit_predict(X_circles)
_y_circles_pred = _sc.clusters

plt.figure(figsize=(16, 8))
plt.scatter(X_circles[:, 0], X_circles[:, 1], c=_y_circles_pred)
plt.show()

# sc.fit_predict(X_circles)
# y_circles_pred = sc.fit_predict(X_circles)
# plt.figure(figsize=(16, 8))
# plt.scatter(X_circles[:, 0], X_circles[:, 1], c=y_circles_pred)
# plt.show()

# --------------BLOBS--------------

X_blobs, y_blobs = make_blobs(M, n_features=4, random_state=78, cluster_std=0.6, centers=4)
plt.figure(figsize=(16, 8))
plt.scatter(X_blobs[:, 0], X_blobs[:, 1], c=y_blobs)
plt.show()

_sc.fit_predict(X_blobs)
_y_blobs_pred = _sc.clusters

plt.figure(figsize=(16, 8))
plt.scatter(X_blobs[:, 0], X_blobs[:, 1], c=_y_blobs_pred)
plt.show()
