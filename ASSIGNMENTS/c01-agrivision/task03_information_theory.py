
import numpy as np
from pathlib import Path

project = Path(__file__).resolve().parent.parent
data = np.load(project / "results" / "dataset_size_1000.npz")

y = data["y_train"]
classes = data["classes"]

print("TASK 3: INFORMATION THEORY")

# Class entropy
counts = np.bincount(y, minlength=len(classes))
p_class = counts / counts.sum()
p_nonzero = p_class[p_class > 0]

entropy = -np.sum(p_nonzero * np.log2(p_nonzero))
maximum = np.log2(len(classes))

print("\nCLASS ENTROPY")
print("Images:", len(y))
print("Classes:", len(classes))
print("Entropy (bits):", round(entropy, 6))
print("Maximum entropy (bits):", round(maximum, 6))

# Example probability distributions
p = np.array([0.7, 0.2, 0.1])
q = np.array([0.6, 0.3, 0.1])

cross_entropy = -np.sum(p * np.log(q))
kl_divergence = np.sum(p * np.log(p / q))

print("\nCROSS-ENTROPY AND KL DIVERGENCE")
print("Cross-entropy (nats):", round(cross_entropy, 6))
print("KL divergence P||Q (nats):", round(kl_divergence, 6))
print("Task 3 completed.")