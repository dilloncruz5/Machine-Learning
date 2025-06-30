import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

def rank_models(models, X_test, y_test, metrics=['accuracy']):
    all_results = {}

    for metric in metrics:
        scores = []
        for name, model in models:
            y_pred = model.predict(X_test)
            if metric == 'accuracy':
                score = accuracy_score(y_test, y_pred)
            elif metric == 'f1':
                score = f1_score(y_test, y_pred)
            elif metric == 'roc_auc':
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X_test)[:, 1]
                else:
                    y_prob = model.decision_function(X_test)
                score = roc_auc_score(y_test, y_prob)
            else:
                raise ValueError("Unsupported metric.")
            scores.append((name, score))
        
        result_df = pd.DataFrame(scores, columns=['Model', metric.capitalize()])
        result_df = result_df.sort_values(by=metric.capitalize(), ascending=False).reset_index(drop=True)
        all_results[metric] = result_df

    # Visualization with subplots
    plt.figure(figsize=(15, 4))
    for idx, metric in enumerate(metrics):
        df = all_results[metric]
        plt.subplot(1, len(metrics), idx + 1)
        plt.barh(df['Model'], df[metric.capitalize()], color='skyblue')
        plt.xlabel(metric.capitalize())
        plt.title(f'Ranking by {metric.capitalize()}')
        plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.savefig('ranked_models_subplots_09-06-2025.png')
    plt.show()

    return all_results

# Example usage:
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

lr = LogisticRegression(max_iter=1000)
dt = DecisionTreeClassifier()

lr.fit(X_train, y_train)
dt.fit(X_train, y_train)

models = [('Logistic Regression', lr), ('Decision Tree', dt)]

results = rank_models(models, X_test, y_test, metrics=['accuracy', 'f1', 'roc_auc'])

for metric, df in results.items():
    print(f"\nRanking by {metric.capitalize()}:")
    print(df)
