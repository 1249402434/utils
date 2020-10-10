import numpy as np

class kmeans:
    def __init__(self, k=3):
        self.k = k
        self.class_list = [[] for _ in range(k)]
        #self.mean_arr = [[0.403, 0.237],[0.343, 0.099],[0.478, 0.437]]
        
        self.mean_arr = None        
    def classify(self, X):
        for i in range(self.data_size):
            data = X[i]
            
            dist = np.linalg.norm(data - self.mean_arr, axis=1)
            label = np.argmin(dist)
                    
            self.class_list[label].append(data)
        
    def compute_mean(self):
        for i in range(self.k):
            mean_new = sum(self.class_list[i]) / len(self.class_list[i]) # avoid divide by zero
            self.mean_arr[i] = mean_new 
        
    def compute_loss(self):
        loss = 0
        for i in range(self.k):
            loss += np.linalg.norm(self.class_list[i] - self.mean_arr[i])
            
        return loss / self.data_size
    
    def fit(self, X, epochs=100, tol=1e-3):
        self.data_size, self.dim = X.shape
        indices = np.random.randint(low=0, high=self.data_size, size=self.k)

        self.mean_arr = X[indices]
        for e in range(epochs):
            for j in range(self.k):
                self.class_list[j].clear()
                
            self.classify(X)
            self.compute_mean()
            loss = self.compute_loss()
            
            if loss < tol:
                break
            if (e+1) % 10 == 0:
                print("epoch %d loss %f" % ((e+1), loss))
