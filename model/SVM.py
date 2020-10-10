import numpy as np
from sklearn.svm import SVC


class SVC:
    def __init__(self, kernel='rbf', c=0.1, **kwargs):
        self.kernel = kernel
        self.C = c
        self.b = 0
        self.alpha_list = None
        self.predict_prob = None
        if self.kernel == 'poly':
            self.d = kwargs.get('d', 3)
            self.bias = kwargs.get('bias', 0)
            self.gamma = kwargs.get('gamma', 0.5)
        elif self.kernel == 'rbf':
            self.gamma = kwargs.get('gamma', 0.5)

    def calc_kernel_value(self, x1, x2):
        if self.kernel == 'rbf':
            tmp = self.gamma * np.linalg.norm(x1-x2)
            return np.exp(-tmp)
        elif self.kernel == 'poly':
            (self.gamma*x1.T.dot(x2) + self.bias) ** self.d
        elif self.kernel == 'linear':
            return x1.T.dot(x2)

    def update(self, index1, index2, X, y, N):
        alpha1 = self.alpha_list[index1]
        alpha2 = self.alpha_list[index2]
        y1 = y[index1]
        y2 = y[index2]

        x1 = X[index1]
        x2 = X[index2]

        if y1!=y2:
            low_limit = max(0, alpha2 - alpha1)
            high_limit = min(self.C, self.C+alpha2-alpha1)

        else:
            low_limit = max(0, alpha1+alpha2-self.C)
            high_limit = min(self.C, alpha2+alpha1)

        K11 = self.calc_kernel_value(x1,x1)
        K22 = self.calc_kernel_value(x2,x2)
        K12 = self.calc_kernel_value(x1,x2)

        eta = K11+K22-2*K12
        E1 = self.predict_prob[index1] - y1
        E2 = self.predict_prob[index2] - y2
        alpha2_new = alpha2 + (y2*(E1 - E2)) / eta

        if alpha2_new > high_limit: alpha2_new = high_limit
        elif alpha2_new < low_limit: alpha2_new = low_limit
        alpha1_new = alpha1 + y1*y2*(alpha2 - alpha2_new)
        self.alpha_list[index2] = alpha2_new
        self.alpha_list[index1] = alpha1_new

        tmp_list = [self.alpha_list[index]*y[index]*self.calc_kernel_value(X[index],x1) for index in range(N) if index!=index1 and index!=index2]

        b1 = y1 - sum(tmp_list) - alpha1_new*y1*K11 - alpha2_new*y2*K12
        b2 = -E2-y1*K12*(alpha1_new - alpha1) -y2*K22(alpha2_new-alpha2)+self.b
        self.b = (b1 + b2) / 2

    def fit(self, X, y):
        N = X.shape[0]
        self.alpha_list = [0] * N

        for i in range(0, N-1, 2):
            self.update(i, i+1, X, y, N)

    def predict(self, X):
        N = X.shape[0]
        for i in range(N):
            self.predict_prob[i] = sum([self.alpha_list[i]*])










