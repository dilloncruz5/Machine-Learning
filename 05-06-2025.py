import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = load_wine()
X = data.data
y = data.target

# Manual K-Fold Split
K = 5
fold_size = len(X) // K
indices = np.arange(len(X))
np.random.seed(42)
np.random.shuffle(indices)

accuracies = []

for k in range(K):
    test_indices = indices[k * fold_size: (k + 1) * fold_size]
    train_indices = np.concatenate([indices[:k * fold_size], indices[(k + 1) * fold_size:]])

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

# Print Accuracies
for i, acc in enumerate(accuracies, 1):
    print(f"Fold {i} Accuracy: {acc:.4f}")
print(f"\nAverage Accuracy: {np.mean(accuracies):.4f}")

# Visualization
plt.figure(figsize=(8, 5))
plt.bar(range(1, K + 1), accuracies, color='skyblue', edgecolor='black')
plt.axhline(np.mean(accuracies), color='red', linestyle='--', label=f'Avg Accuracy = {np.mean(accuracies):.4f}')
plt.xticks(range(1, K + 1))
plt.xlabel("Fold Number")
plt.ylabel("Accuracy")
plt.title("Accuracy per Fold (Logistic Regression on Wine Dataset)")
plt.legend()
plt.tight_layout()

# Save the figure
plt.savefig("fold_accuracies_05-06-2025.png", dpi=300)

# Show the figure
plt.show()
