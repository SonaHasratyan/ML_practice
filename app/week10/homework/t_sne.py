import numpy as np
from sklearn.datasets import load_digits

# from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE


class _TSNE:
    def __init__(
        self,
        n_components=2,
        perplexity=30.0,
        learning_rate=100,
        momentum=0.5,
        n_iter=248,
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
            self.y = (
                self.y
                - self.learning_rate * grad
                + self.momentum * (self.y - self.prev_y)
            )
            self.prev_y = prev_y

        return self.y

    def __compute_affinity(self):
        self.affinity = np.zeros((self.m, self.m))
        for i in range(self.m):
            for j in range(self.m):
                if i == j:
                    continue

                self.affinity[i][j] = -(np.linalg.norm(self.X[i] - self.X[j]) ** 2)

            sigma_i = self.__binary_search_sigma_i(self.affinity[i])
            self.affinity[i] = np.exp(self.affinity[i] / (2 * (sigma_i**2)))
            self.affinity[i] /= sum(self.affinity[i])

        for i in range(self.m):
            for _j in range(self.m - i):
                j = _j + i

                self.affinity[i][j] = (self.affinity[i][j] + self.affinity[j][i]) / (
                    2 * self.m
                )
                self.affinity[j][i] = self.affinity[i][j]

    @staticmethod
    def __perp(affinity):
        return 2 ** (-sum(affinity * np.log2(affinity)))

    def __binary_search_sigma_i(
        self, row_i, lower_bound=1e-5, upper_bound=1e5, eps=1e-8
    ):
        tmp_sigma = (lower_bound + upper_bound) / 2
        affinity = np.exp(row_i / (2 * (tmp_sigma**2)))
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
        self.y = 1e-4 * np.random.randn(self.m, self.n_components)
        self.prev_y = np.zeros(self.y.shape)

    def __compute_similarity(self):
        self.similarity = np.zeros((self.m, self.m))

        for i in range(self.m):
            for _j in range(self.m - i):
                j = _j + i

                self.similarity[i][j] = (
                    1 + (np.linalg.norm(self.y[i] - self.y[j]) ** 2)
                ) ** (-1)
                self.similarity[j][i] = self.similarity[i][j]

            self.similarity[i] /= sum(self.similarity[i])

    def __compute_grad(self):
        grad = np.zeros(self.y.shape)
        for i in range(self.m):
            y_diff = self.y[i] - self.y
            grad[i] = 4 * sum(
                (self.affinity[i] - self.similarity[i])[:, np.newaxis]
                * y_diff
                * ((1 + np.linalg.norm(y_diff, axis=1) ** 2) ** (-1))[:, np.newaxis]
            )
        return grad


M = 100
X_digits, y_digits = load_digits(return_X_y=True)

# tsne = TSNE(n_components=10, init="pca", method="exact")
# X_new = tsne.fit_transform(X_digits[:M])

_tsne = _TSNE(n_components=10)
X_new = _tsne.fit_transform(X_digits[:M])
print(X_new)
