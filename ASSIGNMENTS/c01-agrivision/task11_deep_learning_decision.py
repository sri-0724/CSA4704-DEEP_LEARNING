
from pathlib import Path
import pandas as pd

project = Path(__file__).resolve().parent.parent

print("TASK 11: IS DEEP LEARNING JUSTIFIED?\n")

# Results from Task 5
print("1. Traditional machine-learning baselines (Task 5)")
print("Logistic Regression: accuracy 0.4182, macro F1 0.3018")
print("Decision Tree:       accuracy 0.2963, macro F1 0.2257")
print("Random Forest:       accuracy 0.5091, macro F1 0.3725")

# Result from Task 8
print("\n2. Softmax regression from scratch (Task 8)")
print("Test accuracy: 0.2949")
print("Features used: 100 out of 4096")

# Results from Task 9
csv_path = project / "results" / "task09_dataset_comparison.csv"
df = pd.read_csv(csv_path)
best = df.loc[df["Accuracy"].idxmax()]

print("\n3. Dataset-size experiment (Task 9)")
print("Best observed accuracy:", round(best["Accuracy"], 4))
print("Training images:", int(best["Training Images"]))
print("Macro F1:", round(best["Macro F1"], 4))

print("\nDECISION")
print("Deep learning is worth testing, but is not yet proven necessary.")
print("The current experiments use limited pixel features and traditional")
print("models. A small CNN could learn spatial patterns in leaf images.")
print("Compare it with Random Forest using the same test set and metrics.")
print("Also measure inference time and model size before deployment.")

print("\nTask 11 completed.")