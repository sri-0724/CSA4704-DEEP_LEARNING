
import numpy as np
import pandas as pd
import time
import pickle
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

project = Path(__file__).resolve().parent.parent
dataset_sizes = [1000, 5000, 10000, 14528]

results = []
final_confusion_matrix = None

print("TASK 9: MODEL EVALUATION")
print("Using the first 100 pixel features for a fair comparison.\n")

for size in dataset_sizes:
    file_path = project / "results" / f"dataset_size_{size}.npz"
    data = np.load(file_path)

    # Normalize pixel values and select 100 features
    X_train = data["X_train"].astype(np.float32) / 255.0
    X_test = data["X_test"].astype(np.float32) / 255.0

    X_train = X_train[:, :100]
    X_test = X_test[:, :100]

    y_train = data["y_train"].astype(int)
    y_test = data["y_test"].astype(int)

    model = RandomForestClassifier(
        n_estimators=30,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )

    start_train = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start_train

    start_predict = time.perf_counter()
    predictions = model.predict(X_test)
    predict_time = time.perf_counter() - start_predict

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, average="macro", zero_division=0
    )
    recall = recall_score(
        y_test, predictions, average="macro", zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, average="macro", zero_division=0
    )

    # Approximate serialized model size, not total runtime memory
    model_size_kb = len(pickle.dumps(model)) / 1024

    results.append({
        "Training Images": size,
        "Accuracy": round(accuracy, 4),
        "Macro Precision": round(precision, 4),
        "Macro Recall": round(recall, 4),
        "Macro F1": round(f1, 4),
        "Training Time (s)": round(train_time, 3),
        "Prediction Time (s)": round(predict_time, 3),
        "Model Size (KB)": round(model_size_kb, 2)
    })

    print(f"Training images: {size}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro precision: {precision:.4f}")
    print(f"Macro recall: {recall:.4f}")
    print(f"Macro F1: {f1:.4f}")
    print(f"Training time: {train_time:.3f} seconds")
    print(f"Prediction time: {predict_time:.3f} seconds")
    print(f"Serialized model size: {model_size_kb:.2f} KB")
    print("-" * 40)

    if size == 14528:
        final_confusion_matrix = confusion_matrix(
            y_test, predictions,
            labels=np.arange(len(data["classes"]))
        )

# Save dataset-size comparison
results_df = pd.DataFrame(results)
results_df.to_csv(
    project / "results" / "task09_dataset_comparison.csv",
    index=False
)

# Save confusion matrix for the largest training dataset
if final_confusion_matrix is not None:
    pd.DataFrame(final_confusion_matrix).to_csv(
        project / "results" / "task09_confusion_matrix.csv",
        index=False
    )

print("\nDATASET SIZE COMPARISON")
print(results_df.to_string(index=False))
print("\nConfusion matrix saved to results/task09_confusion_matrix.csv")
print("Comparison saved to results/task09_dataset_comparison.csv")
print("\nTask 9 completed.")