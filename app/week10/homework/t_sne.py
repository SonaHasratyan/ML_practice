import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.manifold._t_sne import TSNE


class _TSNE:
    def __init__(
        self,
        n_components=2,
        perplexity=30.0,
        learning_rate=0.03,
        momentum=1e-07,
        n_iter=100,
    ):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.n_iter = n_iter

        self.X = None
        self.y = None
        self.prev_y = None
        self.m = 0
        self.affinity = None  # p_{ij}
        self.similarity = None  # q_{ij}

        np.random.seed(78)

    def fit(self, X):
        self.X = X
        self.m = self.X.shape[0]
        self.__compute_affinity()
        self.__init_y()

    def fit_transform(self, X):
        self.X = X
        self.m = self.X.shape[0]
        self.__compute_affinity()
        self.__init_y()

        for i in range(self.n_iter):
            self.__compute_similarity()
            grad = self.__compute_grad()
            prev_y = self.y.copy()
            self.y = self.y + self.momentum * grad + self.learning_rate * (self.y - self.prev_y)
            self.prev_y = prev_y

    def __compute_affinity(self):
        self.affinity = np.zeros((self.m, self.m))
        for i in range(self.m):
            for j in range(self.m):
                if i == j:
                    continue

                self.affinity[i][j] = -(np.linalg.norm(self.X[i] - self.X[j]) ** 2)

            sigma_i = self.__binary_search_sigma_i(self.affinity[i])
            self.affinity[i] = np.exp(self.affinity[i] / (2 * sigma_i ** 2))
            self.affinity[i] /= sum(self.affinity[i])

        for i in range(self.m):
            for _j in range(self.m - i):
                j = _j + i

                self.affinity[i][j] = (
                    (self.affinity[i][j] + self.affinity[j][i]) / 2 * self.m
                )
                self.affinity[j][i] = self.affinity[i][j]

    @staticmethod
    def __perp(affinity):
        return 2 ** (-sum(affinity * np.log2(affinity)))

    def __binary_search_sigma_i(self, row_i, lower_bound=1e-10, upper_bound=1e10, eps=1e-8):

        tmp_sigma = (lower_bound + upper_bound) / 2

        affinity = np.exp(row_i / (2 * tmp_sigma ** 2))
        affinity /= sum(affinity)

        perp = self.__perp(affinity)
        if perp > self.perplexity:
            upper_bound = tmp_sigma
        else:
            lower_bound = tmp_sigma

        if np.abs(perp - self.perplexity) <= eps:
            return tmp_sigma
        
        return self.__binary_search_sigma_i(row_i, lower_bound, upper_bound, eps)

    def __init_y(self):
        # todo v: * np.eye?
        self.y = np.random.normal(loc=0.0, scale=10 ** (-4), size=self.m)
        self.prev_y = np.zeros(self.m)

    def __compute_similarity(self):
        self.similarity = np.zeros((self.m, self.m))

        for i in range(self.m):
            for _j in range(self.m - i):
                j = _j + i

                self.similarity[i][j] += np.exp(
                    -(np.linalg.norm(self.y[i] - self.y[j]) ** 2)
                )
                self.similarity[j][i] = self.similarity[i][j]

            self.similarity[i] /= sum(self.similarity[i])

    def __compute_grad(self):
        grad = np.zeros(self.m)
        for i in range(self.m):
            y_diff = np.full(self.m, self.y[i]) - self.y
            grad[i] = 4 * sum(
                (self.affinity[i] - self.similarity[i])
                * y_diff
                * ((1 + np.linalg.norm(y_diff) ** 2) ** (-1))
            )

        return grad


# --------------BLOBS--------------
M = 100
X_blobs, y_blobs = make_blobs(
    M, n_features=3, random_state=78, cluster_std=0.6, centers=3
)
plt.figure(figsize=(16, 8))
plt.scatter(X_blobs[:, 0], X_blobs[:, 1], c=y_blobs)
plt.show()

_tsne = _TSNE(n_components=3)
_tsne.fit_transform(X_blobs)
_y_blobs_pred = _tsne.y

plt.figure(figsize=(16, 8))
plt.scatter(X_blobs[:, 0], X_blobs[:, 1], c=_y_blobs_pred)
plt.show()
