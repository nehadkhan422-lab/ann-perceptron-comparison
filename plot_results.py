import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from perceptron import Perceptron
from mlp import MLP

# ---- XOR ----
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])
p_xor = Perceptron(learning_rate=0.1, epochs=20).fit(X_xor, y_xor)
m_xor = MLP(hidden_size=4, learning_rate=0.5, epochs=10000).fit(X_xor, y_xor)
xor_perc = np.mean(p_xor.predict(X_xor) == y_xor) * 100
xor_mlp = np.mean(m_xor.predict(X_xor) == y_xor) * 100

# ---- Iris (versicolor vs virginica) ----
iris = load_iris()
mask = iris.target != 0
X = iris.data[mask]
y = (iris.target[mask] == 2).astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
p = Perceptron(learning_rate=0.1, epochs=50).fit(X_train, y_train)
m = MLP(hidden_size=8, learning_rate=0.1, epochs=2000).fit(X_train, y_train)
iris_perc = np.mean(p.predict(X_test) == y_test) * 100
iris_mlp = np.mean(m.predict(X_test) == y_test) * 100

# ---- Chart ----
labels = ["XOR", "Iris"]
perc_scores = [xor_perc, iris_perc]
mlp_scores = [xor_mlp, iris_mlp]
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(6, 4))
bars1 = ax.bar(x - width / 2, perc_scores, width, label="Single perceptron")
bars2 = ax.bar(x + width / 2, mlp_scores, width, label="MLP")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Single perceptron vs MLP")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 110)
ax.legend()
for bar in list(bars1) + list(bars2):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{bar.get_height():.1f}", ha="center")

plt.tight_layout()
plt.savefig("results.png", dpi=150)
print("Saved results.png")
plt.show()