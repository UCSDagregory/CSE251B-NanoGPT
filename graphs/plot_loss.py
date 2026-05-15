import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib.ticker as ticker

# 1. Define your files and the labels for the legend
data_sources = {
    #"10M Baseline": "../../training_data/10m_baseline.txt",
    #"10M MoE Control": "../../training_data/10m_moe_control.txt",
    #"10M MoE Shallow": "../../training_data/10m_moe_shallow.txt",
    #"10M MoE Sparse": "../../training_data/10m_moe_sparse.txt",
    #"10M MoE Dense": "../../training_data/10m_moe_dense.txt",
    "90M MoE Control": "../../training_data/90m_moe_control.txt",
    "90M MoE Dense": "../../training_data/90m_moe_dense.txt"
}

# Set up the figure size (Standard academic ratio)
plt.figure(figsize=(10, 6))

# 2. Loop through each file and plot it
for label, filepath in data_sources.items():
    if os.path.exists(filepath):
        # Read the CSV (Columns: Tokens, Iterations, ValLoss)
        df = pd.read_csv(filepath, header=None, names=["Tokens", "Iterations", "ValLoss"])

        # Filter out rows where ValLoss is 0.0 (initialization artifacts)
        # and ignore massive outlier spikes
        df = df[(df["ValLoss"] > 0) & (df["ValLoss"] < 200)]

        # --- TRICK 2: The Academic Smoothing ---
        # Smooth the line by averaging every 5 data points
        # min_periods=1 ensures the line doesn't disappear at the very beginning
        df["ValLoss"] = df["ValLoss"].rolling(window=100, min_periods=1).mean()

        # Plot the line!
        plt.plot(df["Tokens"], df["ValLoss"], label=label, linewidth=2)
    else:
        print(f"Warning: File not found -> {filepath}")

# 3. Format the graph for your LaTeX report
plt.title("Validation Loss vs. Tokens Consumed", fontsize=14, fontweight='bold')
plt.xlabel("Tokens Consumed", fontsize=12)
plt.ylabel("Validation Loss", fontsize=12)

# This cuts off the massive initial drop and focuses on the 3.5 to 5.0 range
plt.ylim(2.5, 5.5)
# plt.xlim(150000000, 1050000000)

# Format the X-axis to show millions (e.g., "10M" instead of "10000000")
formatter = ticker.FuncFormatter(lambda x, pos: f"{x/1e6:g}M")
plt.gca().xaxis.set_major_formatter(formatter)

# Add grid lines and a legend
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=11)
plt.tight_layout()

# 4. Save as a high-quality PDF for LaTeX!
save_path = "30m_moe_loss_curve.pdf"
plt.savefig(save_path, format="pdf", bbox_inches="tight")
print(f"Graph successfully saved to {save_path}")

plt.show()
