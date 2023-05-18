import numpy as np


class TSNE:
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
        self.m = 0
        self.variances = None
        self.affinity = None  # p_{ij}
        self.similarity = None  # q_{ij}

        np.random.seed(78)

    def fit(self, X):
        self.X = X
        self.m = self.X.shape[0]
        self.__compute_variances()
        self.__fill_affinity()
        self.__init_y()

    def fit_transform(self, X):
        self.X = X
        self.m = self.X.shape[0]
        self.__compute_variances()
        self.__fill_affinity()
        self.__init_y()

        for i in range(self.n_iter):
            self.__compute_similarity()
            grad = self.__compute_grad()
            # todo v
            self.y = self.y + self.momentum * grad + self.learning_rate * (self.y)

    def __fill_affinity(self):
        self.affinity = np.zeros((self.m, self.m))
        for i in range(self.m):
            for j in range(self.m):
                if i == j:
                    continue

                self.affinity[i][j] += np.exp(
                    -(np.linalg.norm(self.X[i] - self.X[j]) ** 2)
                    / (2 * self.variances[i][j] ** 2)
                )

            self.affinity[i] /= sum(self.affinity[i])

        for i in range(self.m):
            for _j in range(self.m - i):
                j = _j + i

                self.affinity[i][j] = (
                    (self.affinity[i][j] + self.affinity[j][i]) / 2 * self.m
                )
                self.affinity[j][i] = self.affinity[i][j]

    def __compute_variances(self):
        # todo v
        self.variances = np.full((self.m, self.m), fill_value=self.perplexity)

    def __init_y(self):
        self.y = np.random.normal(loc=0.0, scale=(10 ** (-4)) * np.eye(self.m))

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
        grad = np.array(self.m)
        for i in range(self.m):
            y_diff = np.full(self.m, self.y[i]) - self.y
            grad[i] = 4 * sum(
                (self.affinity[i] - self.similarity[i])
                * y_diff
                * ((1 + np.linalg.norm(y_diff) ** 2) ** (-1))
            )

        return grad
