
import numpy as np
import time
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

project = Path(__file__).resolve().parent.parent
data = np.load(project / "results" / "dataset_size_1000.npz")

X_train = data["X_train"].astype(np.float32) / 255.0
y_train = data["y_train"]
X_test = data["X_test"].astype(np.float32) / 255.0
y_test = data["y_test"]

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=300, random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=20, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=50, max_depth=20,
        random_state=42, n_jobs=-1
    )
}

print("TASK 5: CLASSIFICATION BASELINES")
print("Training images:", len(y_train))
print("Test images:", len(y_test))

for name, model in models.items():
    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test, predictions, average="macro", zero_division=0
    )

    print(f"\n{name}")
    print("Accuracy:", round(accuracy, 4))
    print("Macro F1:", round(macro_f1, 4))
    print("Training time (seconds):", round(train_time, 3))

print("\nTask 5 completed.")