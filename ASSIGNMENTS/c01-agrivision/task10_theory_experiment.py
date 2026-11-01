
from pathlib import Path
import pandas as pd

project = Path(__file__).resolve().parent.parent

print("TASK 10: THEORY VS EXPERIMENT\n")

print("1. PCA")
print("Theory: More principal components generally retain more variance.")
print("Experiment: 10 components retained 60.62%;")
print("            100 components retained 82.91%.")
print("Conclusion: The experiment supports the theoretical expectation.\n")

print("2. Learning rate")
print("Theory: A learning rate that is too high can destabilize training.")
print("Experiment: At 0.5, loss increased after initially decreasing.")
print("            At 0.05, loss decreased to 2.1180.")
print("Conclusion: The smaller learning rate was more stable in this run.\n")

print("3. Dataset size")
csv_path = project / "results" / "task09_dataset_comparison.csv"
df = pd.read_csv(csv_path)

print(df[["Training Images", "Accuracy", "Macro F1",
          "Training Time (s)"]].to_string(index=False))

best_row = df.loc[df["Accuracy"].idxmax()]
print("\nHighest observed accuracy:",
      f'{best_row["Accuracy"]:.4f}',
      "with", int(best_row["Training Images"]), "training images.")

print("\nConclusion:")
print("More data improved performance initially, but the largest")
print("dataset did not improve accuracy beyond the 10,000-image run.")
print("Training time increased with dataset size in these experiments.")
print("\nTask 10 completed.")