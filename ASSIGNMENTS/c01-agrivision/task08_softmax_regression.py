
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score

project = Path(__file__).resolve().parent.parent
data = np.load(project / "results" / "dataset_size_1000.npz")

X_train = data["X_train"].astype(np.float32) / 255.0
y_train = data["y_train"].astype(int)
X_test = data["X_test"].astype(np.float32) / 255.0
y_test = data["y_test"].astype(int)

# Use a small subset of pixel features to keep computation manageable
X_train = X_train[:, :100]
X_test = X_test[:, :100]

num_classes = len(data["classes"])
rng = np.random.default_rng(42)

W = rng.normal(0, 0.01, (X_train.shape[1], num_classes))
b = np.zeros(num_classes)

def softmax(scores):
    scores = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

def loss_and_gradients(X, y, W, b):
    n = len(y)
    probabilities = softmax(X @ W + b)

    loss = -np.mean(np.log(
        probabilities[np.arange(n), y] + 1e-12
    ))

    probabilities[np.arange(n), y] -= 1
    probabilities /= n

    grad_W = X.T @ probabilities
    grad_b = np.sum(probabilities, axis=0)

    return loss, grad_W, grad_b

print("TASK 8: SOFTMAX REGRESSION FROM SCRATCH")

learning_rate = 0.05
epochs = 30

for epoch in range(epochs):
    loss, grad_W, grad_b = loss_and_gradients(
        X_train, y_train, W, b
    )
    W -= learning_rate * grad_W
    b -= learning_rate * grad_b

    if epoch in [0, 9, 19, 29]:
        print(f"Epoch {epoch+1}, loss: {loss:.4f}")

predictions = np.argmax(softmax(X_test @ W + b), axis=1)
accuracy = accuracy_score(y_test, predictions)

print("Test accuracy:", round(accuracy, 4))
print("Features used:", X_train.shape[1])
print("Task 8 completed.")