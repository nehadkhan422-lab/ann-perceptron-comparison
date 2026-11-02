import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from perceptron import Perceptron
from mlp import MLP

# load Iris and keep only 2 flower types (versicolor vs virginica)
iris = load_iris()
mask = iris.target != 0
X = iris.data[mask]
y = (iris.target[mask] == 2).astype(int)

# split into training data and test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# scale the numbers so they are on a similar size
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# train both models
p = Perceptron(learning_rate=0.1, epochs=50).fit(X_train, y_train)
m = MLP(hidden_size=8, learning_rate=0.1, epochs=2000).fit(X_train, y_train)

# compare on the test data
print("Perceptron test accuracy:", round(np.mean(p.predict(X_test) == y_test) * 100, 1), "%")
print("MLP test accuracy:       ", round(np.mean(m.predict(X_test) == y_test) * 100, 1), "%")