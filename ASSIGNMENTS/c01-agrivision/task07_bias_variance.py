
import numpy as np
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

project = Path(__file__).resolve().parent.parent
data = np.load(project / "results" / "dataset_size_1000.npz")

X = data["X_train"].astype(np.float32) / 255.0
y = data["y_train"]
X_test = data["X_test"].astype(np.float32) / 255.0
y_test = data["y_test"]

rng = np.random.default_rng(42)
n = len(y)
predictions = []

print("TASK 7: BOOTSTRAP AND BIAS-VARIANCE")

# Train models on bootstrap samples
for i in range(10):
    indices = rng.choice(n, size=n, replace=True)
    model = DecisionTreeClassifier(
        max_depth=5, random_state=42 + i
    )
    model.fit(X[indices], y[indices])
    predictions.append(model.predict(X_test))

predictions = np.array(predictions)

# Prediction variability across bootstrap models
prediction_variance = np.mean(
    np.var(predictions.astype(float), axis=0)
)

# Accuracy of the majority-vote prediction
majority_prediction = np.apply_along_axis(
    lambda column: np.bincount(
        column.astype(int), minlength=len(np.unique(y))
    ).argmax(),
    axis=0,
    arr=predictions
)

accuracy = accuracy_score(y_test, majority_prediction)

print("Bootstrap models:", len(predictions))
print("Mean prediction variance:", round(prediction_variance, 6))
print("Majority-vote test accuracy:", round(accuracy, 4))

# MLE versus MAP: simple Bernoulli example
successes = 8
trials = 10
alpha = 2
beta = 2

mle = successes / trials
map_estimate = (successes + alpha - 1) / (
    trials + alpha + beta - 2
)

print("\nMLE VS MAP EXAMPLE")
print("Successes:", successes, "out of", trials)
print("MLE estimate:", round(mle, 4))
print("MAP estimate with Beta(2,2) prior:", round(map_estimate, 4))

print("\nTask 7 completed.")