import matplotlib.pyplot as plt

# Actual values from Task 8 terminal output
epochs = [1, 10, 20, 30]
loss = [2.2927, 2.1307, 2.1209, 2.1180]

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    loss,
    marker="o",
    linewidth=2,
    label="Training Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Task 8: Softmax Regression Training Loss")
plt.xticks(epochs)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()

plt.savefig("results/task08_loss_graph.png", dpi=300)
plt.show()

print("Graph saved to results/task08_loss_graph.png")