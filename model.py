import numpy as np

class LinearRegressionScratch:

    def __init__(self,lr = 0.005, epochs = 5000):
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = None

    def initialize_weights(self,n_features):
        self.w = np.zeros(n_features)
        self.b = 0

    def predict(self,X):
        return np.dot(X,self.w) + self.b

    def compute_cost(self,X,y):
        n = len(X)
        y_pred = self.predict(X)
        cost = (1/n) * np.sum((y_pred - y) ** 2)
        return cost    
    def fit(self,X,y):
        n_samples, n_features = X.shape

        self.initialize_weights(n_features)
        self.losses = []
        for _ in range(self.epochs):
            y_pred = self.predict(X)
            error = y_pred - y

            cost = (1 / (2 * n_samples)) * np.sum(error ** 2)
            self.losses.append(cost)

            dw = (1/n_samples) * np.dot(X.T, error)
            db = (1/n_samples) * np.sum(error)

            self.w -= self.lr * dw
            self.b -= self.lr * db