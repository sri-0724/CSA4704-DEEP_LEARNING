import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# -----------------------------------------
# TASK 6: MODEL CAPACITY, UNDERFITTING
# AND OVERFITTING
# -----------------------------------------

# 1. Find the dataset file
possible_files = [
    "agrivision_tomato_10class.npz",
    "agrivision_tomato.npz"
]

dataset_path = next(
    (Path(name) for name in possible_files if Path(name).exists()),
    None
)

if dataset_path is None:
    raise FileNotFoundError(
        "Dataset not found. Put agrivision_tomato_10class.npz "
        "or agrivision_tomato.npz in the same folder as this Python file."
    )

# 2. Load the dataset
with np.load(dataset_path, allow_pickle=False) as data:
    print("Dataset arrays:", data.files)

    arrays = {key: data[key] for key in data.files}

# Find the label array: a 1D array with fewer than 100 unique classes
label_candidates = [
    (key, value)
    for key, value in arrays.items()
    if value.ndim == 1
    and len(value) > 0
    and len(np.unique(value)) <= 100
]

if not label_candidates:
    raise ValueError(
        "Could not identify the labels automatically. "
        "Check the printed dataset array names."
    )

# Prefer common label names if available
label_names = ["y", "labels", "label", "target", "targets"]
label_candidates.sort(
    key=lambda item: (
        item[0].lower() not in label_names,
        item[1].ndim
    )
)

label_name, y = label_candidates[0]

# Find a feature array with the same number of samples
feature_candidates = [
    (key, value)
    for key, value in arrays.items()
    if key != label_name
    and value.ndim >= 2
    and value.shape[0] == len(y)
]

if not feature_candidates:
    raise ValueError(
        "Could not identify the image/features array matching the labels."
    )

feature_name, X = feature_candidates[0]

# Flatten images if they have more than two dimensions
X = X.reshape(X.shape[0], -1)

# Convert pixel values to numbers suitable for the tree model
X = X.astype(np.float32)

print("\nDataset:", dataset_path.name)
print("Features array:", feature_name)
print("Labels array:", label_name)
print("Feature shape:", X.shape)
print("Label shape:", y.shape)

# 3. Create training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 4. Test five different model capacities
# C1 is the simplest; C5 is the most flexible.
capacities = [
    ("C1", 1),
    ("C2", 2),
    ("C3", 3),
    ("C4", 5),
    ("C5", None)
]

training_errors = []
validation_errors = []
models = []

for name, depth in capacities:
    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    train_accuracy = accuracy_score(
        y_train, model.predict(X_train)
    )
    val_accuracy = accuracy_score(
        y_val, model.predict(X_val)
    )

    train_error = 1 - train_accuracy
    val_error = 1 - val_accuracy

    training_errors.append(train_error)
    validation_errors.append(val_error)
    models.append(model)

    depth_text = "None (unrestricted)" if depth is None else str(depth)

    print(
        f"{name} | max_depth={depth_text} | "
        f"Training error={train_error:.4f} | "
        f"Validation error={val_error:.4f}"
    )

training_errors = np.array(training_errors)
validation_errors = np.array(validation_errors)

# 5. Find the capacity with the lowest validation error
optimal_index = int(np.argmin(validation_errors))
optimal_capacity = capacities[optimal_index][0]
generalisation_gap = (
    validation_errors[optimal_index]
    - training_errors[optimal_index]
)

print("\n--- RESULTS ---")
print("Optimal capacity:", optimal_capacity)
print(f"Training error at optimal capacity: "
      f"{training_errors[optimal_index]:.4f}")
print(f"Validation error at optimal capacity: "
      f"{validation_errors[optimal_index]:.4f}")
print(f"Generalisation gap: {generalisation_gap:.4f}")

# 6. Plot training and validation errors
x = np.arange(len(capacities))
labels = [item[0] for item in capacities]

plt.figure(figsize=(10, 6))

# Illustrate the low-capacity and high-capacity regions
plt.axvspan(
    -0.4, 1.5,
    color="lightblue",
    alpha=0.25,
    label="Possible underfitting region"
)

plt.axvspan(
    3.5, 4.4,
    color="mistyrose",
    alpha=0.5,
    label="Possible overfitting region"
)

plt.plot(
    x, training_errors,
    marker="o",
    linewidth=2,
    label="Training error"
)

plt.plot(
    x, validation_errors,
    marker="s",
    linewidth=2,
    label="Validation error"
)

# Mark the best validation result
plt.scatter(
    optimal_index,
    validation_errors[optimal_index],
    s=150,
    marker="*",
    color="green",
    zorder=5,
    label=f"Optimal capacity: {optimal_capacity}"
)

plt.annotate(
    f"Optimal: {optimal_capacity}\n"
    f"Validation error = {validation_errors[optimal_index]:.4f}",
    xy=(optimal_index, validation_errors[optimal_index]),
    xytext=(10, 20),
    textcoords="offset points",
    arrowprops={"arrowstyle": "->"}
)

# Show the generalisation gap at the optimal capacity
plt.annotate(
    f"Gap = {generalisation_gap:.4f}",
    xy=(
        optimal_index,
        (training_errors[optimal_index]
         + validation_errors[optimal_index]) / 2
    ),
    xytext=(15, -25),
    textcoords="offset points",
    arrowprops={"arrowstyle": "->"}
)

plt.xticks(x, labels)
plt.xlabel("Model capacity (Decision Tree maximum depth)")
plt.ylabel("Error rate (1 - accuracy)")
plt.title("AgriVision: Training and Validation Error vs Model Capacity")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# Save the graph and numerical results
plt.savefig("task6_capacity_graph.png", dpi=300, bbox_inches="tight")

np.savetxt(
    "task6_capacity_results.csv",
    np.column_stack(
        [training_errors, validation_errors]
    ),
    delimiter=",",
    header="training_error,validation_error",
    comments="",
    fmt="%.6f"
)

print("\nGraph saved as task6_capacity_graph.png")
print("Results saved as task6_capacity_results.csv")

plt.show()