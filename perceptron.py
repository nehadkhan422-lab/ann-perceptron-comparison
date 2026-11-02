import numpy as np


class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=20):
        self.lr = learning_rate      # how big each nudge is
        self.epochs = epochs         # how many passes over the data
        self.weights = None
        self.bias = 0.0

    def predict(self, X):
        total = np.dot(X, self.weights) + self.bias
        return np.where(total >= 0, 1, 0)

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y)
        self.weights = np.zeros(X.shape[1])
        self.bias = 0.0

        for _ in range(self.epochs):
            for xi, target in zip(X, y):
                prediction = self.predict(xi)
                error = target - prediction
                self.weights += self.lr * error * xi
                self.bias += self.lr * error
        return self