
import numpy as np
from pathlib import Path

project = Path(__file__).resolve().parent.parent
file = project / "data" / "agrivision_tomato_10class.npz"

data = np.load(file)
X = data["X"]
y = data["y"]

print("TASK 1: DATA REPRESENTATION")
print("Number of images:", X.shape[0])
print("Features per image:", X.shape[1])
print("Labels:", y.shape)
print("Pixel data type:", X.dtype)
print("Pixel range:", X.min(), "to", X.max())

# Vector norms for the first image
x = X[0].astype(float)

print("\nVECTOR NORMS")
print("L1 norm:", np.sum(np.abs(x)))
print("L2 norm:", np.sqrt(np.sum(x ** 2)))
print("L-infinity norm:", np.max(np.abs(x)))

# Class distribution
names, counts = np.unique(y, return_counts=True)

print("\nCLASS DISTRIBUTION")
for name, count in zip(names, counts):
    print(f"{name}: {count}")

print("Total images:", counts.sum())