"""Evaluate logistic regression on a deterministic synthetic classification task."""
import json

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def evaluate():
    features, labels = make_classification(
        n_samples=500, n_features=8, n_informative=5,
        n_redundant=0, random_state=42,
    )
    train_x, test_x, train_y, test_y = train_test_split(
        features, labels, test_size=0.25, stratify=labels, random_state=42,
    )
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
    model.fit(train_x, train_y)
    predictions = model.predict(test_x)
    return {
        "train_rows": len(train_y),
        "test_rows": len(test_y),
        "accuracy": accuracy_score(test_y, predictions),
        "confusion_matrix": confusion_matrix(test_y, predictions).tolist(),
    }


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
