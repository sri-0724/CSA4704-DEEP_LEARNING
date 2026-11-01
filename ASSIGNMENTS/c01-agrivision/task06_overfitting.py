
import numpy as np
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

project = Path(__file__).resolve().parent.parent
data = np.load(project / "results" / "dataset_size_1000.npz")

X_train = data["X_train"].astype(np.float32) / 255.0
y_train = data["y_train"]
X_test = data["X_test"].astype(np.float32) / 255.0
y_test = data["y_test"]

models = {
    "Shallow Tree": DecisionTreeClassifier(
        max_depth=3, random_state=42
    ),
    "Deeper Tree": DecisionTreeClassifier(
        max_depth=None, min_samples_leaf=1, random_state=42
    )
}

print("TASK 6: UNDERFITTING AND OVERFITTING")

for name, model in models.items():
    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))

    print(f"\n{name}")
    print("Training accuracy:", round(train_acc, 4))
    print("Test accuracy:", round(test_acc, 4))
    print("Accuracy gap:", round(train_acc - test_acc, 4))

print("\nTask 6 completed.")