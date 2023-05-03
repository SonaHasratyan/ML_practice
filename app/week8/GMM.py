import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from scipy.stats import multivariate_normal


class GMM:
    def __init__(self, k, method="random_mean_std", max_iter=300, tol=1e-6):
        self.pi_arr = None
        self.cov_arr = None
        self.mean_arr = None
        self.k = k
        self.method = method
        self.max_iter = max_iter

    def init_centers(self, X):
        if self.method == "random_mean_std":
            pass  # generating K random means and std-s
        if self.method == "random_mean":
            pass  # generate K random means
        if self.method == "k-means":
            # n - number of datapoints
            # m - number of features
            # k - number of clusters

            # mean_arr.shape == k x m
            # cov_arr.shape == k x m x m
            # pi_arr.shape == 1 x k

            kmeans = KMeans(n_clusters=self.k)
            kmeans.fit(X)
            clusters = kmeans.predict(X)
            mean_arr = kmeans.cluster_centers_
            cov_arr = []
            pi_arr = []
            for i in range(self.k):
                X_i = X[clusters == i]
                cov_arr.append(np.cov(X_i.T))
                pi_arr.append(X_i.shape[0] / X.shape[0])
            return mean_arr, np.array(cov_arr), np.array(pi_arr)

        if self.method == "random_divide":
            pass  # divide data into K clusters randomly
        if self.method == "random_gammas":
            pass  # generate random gamma matrix

    def fit(self, X):
        self.mean_arr, self.cov_arr, self.pi_arr = self.init_centers(X)
        # self.loss = self.loss(...)
        for _ in range(self.max_iter):
            mean_arr, cov_arr, pi_arr = self.maximization(X, self.expectation(X))
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

    @staticmethod
    def pdf(x, mean, cov):
        # function to calculate pdf for given params
        proba = multivariate_normal.pdf(x, mean, cov, allow_singular=True)

        return proba

    def expectation(self, X):
        gamma_mtrx = np.zeros((X.shape[0], self.k))  # gamma_mtrx.shape == n x k
        for i, x in enumerate(X):
            for k in range(self.k):
                gamma_mtrx[i][k] = self.pi_arr[k] * self.pdf(
                    x, self.mean_arr[k], self.cov_arr[k]
                )

            # gamma_mtrx[i] /= gamma_mtrx[i].sum()

        gamma_mtrx = gamma_mtrx / gamma_mtrx.sum(axis=1).reshape(-1, 1)

        return gamma_mtrx

    def maximization(self, X, gamma_mtrx):
        # mean_arr.shape == k x m
        # cov_arr.shape == k x m x m
        # pi_arr.shape == 1 x k
        # gamma_mtrx.shape == n x k

        mean_arr = np.zeros((self.k, X.shape[1]))
        cov_arr = np.zeros((self.k, X.shape[1], X.shape[1]))
        pi_arr = np.zeros(self.k)
        for k in range(self.k):
            N_k = gamma_mtrx[:, k].shape[0]
            mean_arr[k] = (gamma_mtrx[:, k].T @ X) / N_k
            cov_arr[k] = (
                ((X - mean_arr[k].T) * gamma_mtrx[:, k].reshape(-1, 1)).T
                @ (X - mean_arr[k].T)
                / N_k
            )
            pi_arr[k] = X.shape[1] / N_k

        return mean_arr, cov_arr, pi_arr

    def predict(self, X):
        return

    def predict_proba(self, X):
        # return predictions using expectation function
        return


X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
plt.scatter(X[:, 0], X[:, 1], s=50)
gmm = GMM(k=4, method="k-means", max_iter=30)
gmm.fit(X)
