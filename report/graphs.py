import matplotlib.pyplot as plt
import os


def save_accuracy_comparison(names, accuracies, task_name, output_dir="plots"):
    """Generates a bar chart comparing final accuracies."""
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, accuracies, color="skyblue", edgecolor="navy", alpha=0.7)

    plt.title(f"Final Test Accuracy: {task_name}", fontsize=14, fontweight="bold")
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.ylim(0, 105)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    # Add text labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{task_name.lower()}_accuracy.png")
    plt.close()


def save_loss_curves(experiment_data, task_name, output_dir="plots"):
    """
    Generates learning curves from a dictionary of experiments.
    experiment_data format: { 'ModelName': {'iters': [], 'train': [], 'val': []} }
    """
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(12, 7))

    for model_name, metrics in experiment_data.items():
        plt.plot(
            metrics["iters"],
            metrics["train"],
            label=f"{model_name} (Train)",
            linestyle="--",
        )
        plt.plot(
            metrics["iters"], metrics["val"], label=f"{model_name} (Val)", linewidth=2
        )

    plt.title(
        f"Training vs Validation Loss: {task_name}", fontsize=14, fontweight="bold"
    )
    plt.xlabel("Iterations", fontsize=12)
    plt.ylabel("Cross-Entropy Loss", fontsize=12)
    plt.legend(loc="upper right", frameon=True, shadow=True)
    plt.grid(True, which="both", linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{task_name.lower()}_loss_curves.png")
    plt.close()


# ==========================================
# EXAMPLE USAGE (Use this for Math & Bool)
# ==========================================
if __name__ == "__main__":

    # --- 1. MATHEMATICAL GPT DATA ---
    # Based on your logs for Task 1.4
    math_names = [
        "Addition\n(0.11M)",
        "Simple (+/-)\n(0.11M)",
        "All Ops\n(0.81M)",
        "Complex\n(1.21M)",
    ]
    math_accs = [77.0, 70.9, 97.0, 77.1]

    # Learning curve data for the best and most complex math models
    math_loss_data = {
        "Math All Ops (Best)": {
            "iters": [0, 500, 1000, 1500, 1999],
            "train": [3.0013, 1.2315, 0.9500, 0.9368, 0.9342],
            "val": [3.0048, 1.2335, 0.9539, 0.9409, 0.9402],
        },
        "Math Complex (Overfit)": {
            "iters": [0, 500, 1000, 1500, 1999],
            "train": [3.0305, 1.3462, 1.1268, 0.9995, 0.9737],
            "val": [3.0298, 1.3526, 1.1472, 1.0239, 1.0084],
        },
    }

    # --- 2. BOOLEAN GPT DATA ---
    # Based on your logs for Task 2.4
    bool_names = [
        "Baseline Small\n(0.11M)",
        "Baseline Med\n(0.80M)",
        "Long Context\n(0.81M)",
    ]
    bool_accs = [48.7, 46.3, 51.1]

    # Learning curves for Boolean runs
    bool_loss_data = {
        "Bool Small": {
            "iters": [0, 500, 1000, 1499],
            "train": [2.7392, 0.6774, 0.6156, 0.5928],
            "val": [2.7395, 0.6805, 0.6186, 0.5985],
        },
        "Bool LongContext": {
            "iters": [0, 500, 1000, 1500, 1999],
            "train": [2.7355, 0.6163, 0.5325, 0.5217, 0.5145],
            "val": [2.7369, 0.6187, 0.5348, 0.5245, 0.5185],
        },
    }

    # --- 3. EXECUTION ---
    print("Generating Math plots...")
    save_accuracy_comparison(math_names, math_accs, "Math_GPT_Operations")
    save_loss_curves(math_loss_data, "Math_GPT_Learning_Dynamics")

    print("Generating Boolean plots...")
    save_accuracy_comparison(bool_names, bool_accs, "Boolean_GPT_Architectures")
    save_loss_curves(bool_loss_data, "Boolean_GPT_Learning_Dynamics")

    print("\nSuccess! Check the 'plots/' directory for your report images.")
