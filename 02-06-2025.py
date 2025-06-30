import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, roc_auc_score
)

# Load Digits dataset
digits = load_digits()
X = digits.data
y = digits.target
classes = digits.target_names

# Binarize labels for ROC-AUC
y_bin = label_binarize(y, classes=range(10))

# Split data
X_train, X_test, y_train, y_test, y_train_bin, y_test_bin = train_test_split(
    X, y, y_bin, test_size=0.2, random_state=42, stratify=y
)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train SVM model
svm = SVC(kernel='rbf', probability=True, random_state=42)
svm.fit(X_train, y_train)

# Predictions
y_pred = svm.predict(X_test)
y_proba = svm.predict_proba(X_test)

# Evaluation
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

acc = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test_bin, y_proba, multi_class='ovr')

print(f"Accuracy:  {acc:.4f}")
print(f"ROC-AUC (OvR): {roc_auc:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title("Confusion Matrix for SVM on Digits Dataset")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()
