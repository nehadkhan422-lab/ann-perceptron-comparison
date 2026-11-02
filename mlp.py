import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class MLP:
    def __init__(self, hidden_size=4, learning_rate=0.5, epochs=10000, seed=1):
        self.hidden_size = hidden_size
        self.lr = learning_rate
        self.epochs = epochs
        self.rng = np.random.default_rng(seed)

    def _forward(self, X):
        self.hidden = sigmoid(X @ self.W1 + self.b1)
        self.output = sigmoid(self.hidden @ self.W2 + self.b2)
        return self.output

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float).reshape(-1, 1)
        n_features = X.shape[1]

        # random starting weights
        self.W1 = self.rng.normal(0, 1, (n_features, self.hidden_size))
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = self.rng.normal(0, 1, (self.hidden_size, 1))
        self.b2 = np.zeros((1, 1))

        for _ in range(self.epochs):
            out = self._forward(X)

            # backpropagation: find how wrong each layer was
            d_out = (out - y) * out * (1 - out)
            d_hidden = (d_out @ self.W2.T) * self.hidden * (1 - self.hidden)

            # nudge the weights
            self.W2 -= self.lr * self.hidden.T @ d_out
            self.b2 -= self.lr * d_out.sum(axis=0, keepdims=True)
            self.W1 -= self.lr * X.T @ d_hidden
            self.b1 -= self.lr * d_hidden.sum(axis=0, keepdims=True)
        return self

    def predict(self, X):
        X = np.array(X, dtype=float)
        return (self._forward(X) >= 0.5).astype(int).flatten()