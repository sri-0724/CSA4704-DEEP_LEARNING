
import numpy as np
import pandas as pd
import time
from pathlib import Path
from sklearn.decomposition import PCA

project = Path(__file__).resolve().parent.parent
results = project / "results"

data = np.load(results / "dataset_size_14528.npz")
X_train = data["X_train"].astype(np.float64) / 255.0

# Fit PCA using the training data only
print("TASK 2: PCA ANALYSIS")
print("Training data shape:", X_train.shape)

start = time.perf_counter()
pca = PCA(n_components=100, svd_solver="randomized", random_state=42)
pca.fit(X_train)
elapsed = time.perf_counter() - start

variance = np.cumsum(pca.explained_variance_ratio_)

print("\nEXPLAINED VARIANCE")
for k in [10, 20, 50, 100]:
    print(f"Components: {k}, variance: {variance[k-1]*100:.4f}%")

# Reconstruction error on a sample of training images
sample = X_train[:500]
print("\nRECONSTRUCTION MSE")
for k in [10, 20, 50, 100]:
    reduced = pca.transform(sample)[:, :k]
    reconstructed = (
        reduced @ pca.components_[:k]
        + pca.mean_
    )
    mse = np.mean((sample - reconstructed) ** 2)
    print(f"Components: {k}, MSE: {mse:.6f}")

print("\nPCA FIT TIME:", round(elapsed, 4), "seconds")
print("Original features:", X_train.shape[1])
print("Reduced features at 50 components:", 50)
print("TASK 2 COMPLETED")