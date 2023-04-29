import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


class GMM:
    def __init__(self, k, method='k-means', max_iter=300, tol=1e-6):
        self.loss = None
        self.pi_arr = None
        self.cov_arr = None
        self.mean_arr = None
        self.k = k
        self.method = method
        self.max_iter = max_iter

    # todo: sigmas should be symetric
    def init_centers(self, X):
        # generating K random means and std-s
        if self.method == 'random_mean_std':
            np.random.seed(78)
            for i in range(self.k):
                normal = np.random.normal(size=X)
                self.mean_arr[i] = normal.mean()
                self.cov_arr[i] = normal.cov()
        if self.method == 'random_mean':
            pass  # generate K random means
        if self.method == 'k-means':
            pi_arr = []
            kmeans = KMeans(n_clusters=self.k)
            kmeans.fit(X)
            y_kmeans = kmeans.predict(X)
            cov_mtrx = np.zeros((self.k, X.shape[1], X.shape[1]))
            for i in range(self.k):
                cov_mtrx = np.cov(X[y_kmeans == i])

            pi_arr = np.unique(y_kmeans, return_counts=True)[1] / len(y_kmeans)

            return kmeans.cluster_centers_, cov_mtrx, pi_arr

            # pass  # generate initial points by KMeans algo
        if self.method == 'random_divide':
            pass  # divide data into K clusters randomly
        if self.method == 'random_gammas':
            pass  # generate random gamma matrix

    def fit(self, X):
        self.mean_arr, self.cov_arr, self.pi_arr = self.init_centers(X)
        # self.loss = self.loss(...)
        for _ in range(self.max_iter):
            gamma_mtrx = self.expectation(X)
            mean_arr, cov_arr, pi_arr = self.maximization(X, gamma_mtrx)

            # todo: v
            # loss = self.loss(...)
            # if loss==self.loss: # add tolerance comparison
            #     break
            # self.loss=loss
            self.mean_arr = mean_arr
            self.cov_arr = cov_arr
            self.pi_arr = pi_arr

    def loss(self, X, mean, cov, pi):
        pass

    def pdf(self, x, mean, cov):
        # function to calculate pdf for given params
        return proba

    def expectation(self, X):
        gamma_mtrx = np.zeros((X.shape[0], self.k))
        for i, x in enumerate(X):
            for j in range(self.k):
                gamma_mtrx[i][j] = self.pi_arr[j] * self.pdf(x, self.mean_arr[i], self.cov_arr[j])

            # gamma_mtrx[i][j] /= gamma_mtrx[i].sum(axis=1)
        gamma_mtrx /= gamma_mtrx.sum(axis=0)

        return gamma_mtrx

    def maximization(self, X, gamma_mtrx):
        # your code here
        return mean_arr, cov_arr, pi_arr

    def predict(self, X):
        return

    def predict_proba(self, X):
        # return predictions using expectation function
        return


from sklearn.datasets import make_blobs
X, y_true = make_blobs(n_samples=300, centers=4,
                       cluster_std=0.60, random_state=0)
plt.scatter(X[:, 0], X[:, 1], s=50)
gmm = GMM(k=4)
gmm.fit(X)
