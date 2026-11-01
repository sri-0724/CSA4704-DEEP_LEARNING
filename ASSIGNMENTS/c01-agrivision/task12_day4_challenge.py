
print("TASK 12: DAY-4 CHALLENGE")
print("AgriVision smartphone deployment plan\n")

print("Challenge requirements:")
print("- 38 disease/plant classes")
print("- Approximately 20% noisy labels")
print("- Inference below 50 milliseconds")
print("- Model size below 5 MB")

print("\nProposed solutions:")

print("1. Noisy labels")
print("Use label smoothing or a robust loss function.")
print("Check suspicious labels and use a clean validation set.")

print("\n2. Feature reduction")
print("Test PCA as a preprocessing baseline.")
print("For image-based deep learning, compare PCA with a compact CNN.")

print("\n3. Overfitting control")
print("Use data augmentation, weight regularization, and early stopping.")

print("\n4. Smartphone deployment")
print("Try a compact CNN, then apply quantization if supported.")
print("Measure actual inference latency and model file size.")

print("\n5. Evaluation")
print("Use a stratified test set with all 38 classes.")
print("Measure macro F1, per-class recall, latency, and model size.")
print("Compare performance before and after compression.")

print("\nDECISION")
print("MODIFY the current system and evaluate a compact model.")
print("Do not claim deployment readiness until all constraints are tested.")

print("\nTask 12 completed.")