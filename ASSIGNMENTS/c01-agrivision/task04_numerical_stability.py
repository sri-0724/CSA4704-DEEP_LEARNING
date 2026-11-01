
import numpy as np
from pathlib import Path

project = Path(__file__).resolve().parent.parent
data = np.load(project / "results" / "dataset_size_1000.npz")

X = data["X_train"].astype(np.float64) / 255.0

print("TASK 4: NUMERICAL STABILITY")

# Stable softmax
z = np.array([1000.0, 999.0, 995.0])
shifted = z - np.max(z)
exp_z = np.exp(shifted)
probabilities = exp_z / exp_z.sum()

print("\nSTABLE SOFTMAX")
print("Original logits:", z)
print("Shifted logits:", shifted)
print("Probabilities:", probabilities)
print("Probability sum:", probabilities.sum())

# Covariance condition number using 100 features
X_small = X[:, :100]
X_centered = X_small - X_small.mean(axis=0)
covariance = np.cov(X_centered, rowvar=False)
condition_number = np.linalg.cond(covariance)

print("\nCOVARIANCE CONDITION NUMBER")
print("Covariance shape:", covariance.shape)
print("Condition number:", f"{condition_number:.6e}")

print("Task 4 completed.")