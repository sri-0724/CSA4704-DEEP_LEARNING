import matplotlib.pyplot as plt
from pathlib import Path

# Project folders
project = Path(__file__).resolve().parent.parent
results = project / "results"
results.mkdir(exist_ok=True)

# Actual Task 9 results
dataset_sizes = [1000, 5000, 10000, 14528]

accuracy = [33.23, 38.46, 40.89, 40.89]
macro_f1 = [21.51, 25.41, 28.16, 27.66]

training_time = [0.165, 0.563, 1.234, 1.338]
model_size_kb = [2544.90, 10093.07, 17261.29, 23236.73]

# GRAPH 1: Accuracy and Macro F1
plt.figure(figsize=(9, 5))
plt.plot(
    dataset_sizes, accuracy,
    marker="o", label="Accuracy (%)"
)
plt.plot(
    dataset_sizes, macro_f1,
    marker="s", label="Macro F1 (%)"
)

plt.xlabel("Number of Training Images")
plt.ylabel("Score (%)")
plt.title("Task 9: Performance vs Dataset Size")
plt.xticks(dataset_sizes, [str(n) for n in dataset_sizes])
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()

plt.savefig(
    results / "task09_performance_graph.png",
    dpi=300
)
plt.show()
plt.close()

# GRAPH 2: Training Time
plt.figure(figsize=(9, 5))
plt.plot(
    dataset_sizes, training_time,
    marker="o", label="Training Time"
)

plt.xlabel("Number of Training Images")
plt.ylabel("Training Time (seconds)")
plt.title("Task 9: Training Time vs Dataset Size")
plt.xticks(dataset_sizes, [str(n) for n in dataset_sizes])
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()

plt.savefig(
    results / "task09_training_time_graph.png",
    dpi=300
)
plt.show()
plt.close()

# GRAPH 3: Serialized Model Size
plt.figure(figsize=(9, 5))
plt.plot(
    dataset_sizes, model_size_kb,
    marker="o", label="Model Size"
)

plt.xlabel("Number of Training Images")
plt.ylabel("Serialized Model Size (KB)")
plt.title("Task 9: Model Size vs Dataset Size")
plt.xticks(dataset_sizes, [str(n) for n in dataset_sizes])
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()

plt.savefig(
    results / "task09_model_size_graph.png",
    dpi=300
)
plt.show()
plt.close()

print("All Task 9 graphs generated successfully!")
print("Check the results folder.")