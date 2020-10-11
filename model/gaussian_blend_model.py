import numpy as np
import math


class gaussian_model:
    def __init__(self, mean, cov_matrix):
        self.mean = mean
        self.cov_matrix = cov_matrix
        self.dim = len(mean)

    def gaussian_prob(self, x):
        denominator = np.sqrt(np.power(2 * math.pi, self.dim) * np.linalg.det(self.cov_matrix))
        inverse_matrix = np.linalg.inv(self.cov_matrix)
        exp_component = np.exp(-0.5 * (x - self.mean).T.dot(inverse_matrix).dot(x - self.mean))

        return exp_component / denominator

    def update(self, mean_new, cov_matrix_new):
        self.mean = mean_new
        self.cov_matrix = cov_matrix_new


class gaussian_EM:
    def __init__(self, K=3):
        self.K = K
        self.weights = [1 / K] * K

    def fit(self, X, epochs=100):
        if type(X) == list:
            X = np.array(X)
        N, dim = X.shape
        mean_arr = np.random.randn(self.K, dim)
        cov_matrix_arr = [np.diag(dim * [1]) for _ in range(self.K)]

        self.classifier_list = [gaussian_model(mean_arr[k], cov_matrix_arr[k]) for k in range(self.K)]
        res_arr = []
        for e in range(epochs):
            for i in range(N):
                gamma_nk_arr = []
                total = 0
                for k in range(self.K):
                    val = self.weights[k] * self.classifier_list[k].gaussian_prob(X[i])
                    gamma_nk_arr.append(val)
                    total += val

                gamma_nk_arr = [gamma_nk_arr[k] / total for k in range(self.K)]
                res_arr.append(gamma_nk_arr)

            Nk_list = [sum(list(zip(*res_arr))[k]) for k in range(self.K)]
            mean_new = []
            cov_matrix_arr_new = []
            for k in range(self.K):
                Nk = Nk_list[k]
                self.weights[k] = Nk / N
                mean_new.append(sum([X[i] * res_arr[i][k] for i in range(N)]) / Nk)

            mean_new = np.array(mean_new)
            for k in range(self.K):
                Nk = Nk_list[k]
                cov_matrix_arr_new.append(sum([(X[i]-mean_new[k]).dot((X[i]-mean_new[k]).T)*res_arr[i][k]
                                               for i in range(N)])/Nk)

            for k in range(self.K):
                self.classifier_list[k].update(mean_new, cov_matrix_arr_new)

    def predict(self, X):
        if type(X)==list:
            X = np.array(X)
        length = X.shape[0]
        y_pred = [0]*length

        for i in range(length):
            y_pred[i] = sum([self.weights[k]*self.classifier_list[k].gaussian_prob(X[i])
                             for k in range(self.K)])

        return y_pred


