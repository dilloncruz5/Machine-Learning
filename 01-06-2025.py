import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define models
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
}

# Store performance
performance = {
    'Model': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1 Score': []
}

# Evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    performance['Model'].append(name)
    performance['Accuracy'].append(accuracy_score(y_test, y_pred))
    performance['Precision'].append(precision_score(y_test, y_pred))
    performance['Recall'].append(recall_score(y_test, y_pred))
    performance['F1 Score'].append(f1_score(y_test, y_pred))

# Convert to DataFrame
df = pd.DataFrame(performance)

# Print metrics
print(df)

# Visualization
plt.figure(figsize=(12, 6))
sns.set_style("whitegrid")

# Melt DataFrame for Seaborn
df_melted = df.melt(id_vars="Model", var_name="Metric", value_name="Score")

# Barplot
sns.barplot(data=df_melted, x="Model", y="Score", hue="Metric")
plt.title("Model Comparison on Breast Cancer Dataset")
plt.ylim(0.85, 1.01)
plt.legend(loc='lower right')
plt.xticks(rotation=10)
plt.tight_layout()
plt.show()
