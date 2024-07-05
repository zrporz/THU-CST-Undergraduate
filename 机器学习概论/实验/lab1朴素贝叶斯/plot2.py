# Re-importing libraries in case the environment has been reset
import numpy as np
import matplotlib.pyplot as plt

# Define alpha values and corresponding metrics
alpha_values = np.array([0,1e-50,1e-30, 1e-20, 1e-10, 1e-5, 1e-3, 1e-1, 1])
accuracy = np.array([0.957,0.973,0.973,0.972,0.971,0.967,0.962,0.949,0.931])
precision = np.array([0.983,0.983,0.983,0.983,0.984,0.984,0.983,0.983,0.983])
recall = np.array([0.949,0.974,0.974,0.974,0.972,0.965,0.958,0.937,0.909])
f1 = np.array([0.966,0.979,0.979,0.978,0.978,0.974,0.970, 0.959,0.944])
auc = np.array([0.732,0.679,0.662,0.651,0.636,0.627,0.623,0.619,0.617])

# Create subplots
fig, axs = plt.subplots(3, 2, figsize=(30, 20))

# Set the x-axis to a logarithmic scale
for axs_ in axs:
    for ax in axs_:
        ax.set_xscale('log')

# Plot each metric on a separate subplot
axs[0][0].plot(alpha_values, accuracy, marker='o', linestyle='-')
axs[0][0].set_title('Accuracy')
axs[0][0].set_ylabel('Accuracy')

axs[0][1].plot(alpha_values, precision, marker='o', linestyle='-', color='orange')
axs[0][1].set_title('Precision')
axs[0][1].set_ylabel('Precision')

axs[1][0].plot(alpha_values, recall, marker='o', linestyle='-', color='green')
axs[1][0].set_title('Recall')
axs[1][0].set_ylabel('Recall')

axs[1][1].plot(alpha_values, f1, marker='o', linestyle='-', color='red')
axs[1][1].set_title('F1 Score')
axs[1][1].set_ylabel('F1 Score')

axs[2][0].plot(alpha_values, auc, marker='o', linestyle='-', color='purple')
axs[2][0].set_title('AUC')
axs[2][0].set_ylabel('AUC')
axs[2][0].set_xlabel('Alpha')

# Improve layout
plt.tight_layout()

plt.savefig("./plot2.png")