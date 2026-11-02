import numpy as np
from perceptron import Perceptron
from mlp import MLP

# AND problem
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

model = Perceptron(learning_rate=0.1, epochs=20)
model.fit(X, y)

predictions = model.predict(X)
print("Inputs:      ", X.tolist())
print("Correct:     ", y.tolist())
print("Perceptron:  ", predictions.tolist())
print("Accuracy:    ", np.mean(predictions == y) * 100, "%")
mlp_model = MLP(hidden_size=4, learning_rate=0.5, epochs=10000)
mlp_model.fit(X, y)
mlp_predictions = mlp_model.predict(X)
print("MLP:         ", mlp_predictions.tolist())
print("MLP Accuracy:", np.mean(mlp_predictions == y) * 100, "%")