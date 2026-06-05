import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn


def show_training(epochs, head_acc_list, avg_loss_list, val_correct_list, val_loss_list):
    print("Showing training diagnostics.")
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))

    x = np.arange(len(avg_loss_list))

    ax[0].plot(x, avg_loss_list, color='orange')
    ax[0].set_title("Average Loss (Train)"); ax[0].set_xlabel("Epochs")

    ax[1].plot(x, head_acc_list, color='blue')
    ax[1].set_title("Head Accuracy (Train)"); ax[1].set_xlabel("Epochs")

    ax[2].plot(x, val_loss_list, color='green')
    ax[2].set_title("Average Loss (Validation)"); ax[2].set_xlabel("Epochs")


    ax[3].plot(x, val_correct_list, color='purple')
    ax[3].set_title("Macro Recall (Validation)"); ax[3].set_xlabel("Epochs")

    plt.tight_layout()
    plt.savefig("Graphs/Training_Stats.png")
    plt.show()


def confusion_matrices(cms, MATERIALS, CLASSES):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax = axes.ravel()
    fig.suptitle("Confusion Matrices of Materials\n"
                 "(rows are labels, columns are predicted)\n"
                 "[row normalized colors]")
    for m, mat in enumerate(MATERIALS):
        cm = cms[m]
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = cm / np.maximum(row_sums, 1)
        sn.heatmap(cm_norm, annot=cm, fmt="d", vmin=0, vmax=1, ax=ax[m],
                   xticklabels=CLASSES, yticklabels=CLASSES, cbar=True)
        ax[m].set_title(f"Confusion Matrix of {mat.capitalize()}")
        ax[m].set_xlabel("Predicted"); ax[m].set_ylabel("True")
    plt.tight_layout()
    plt.savefig("Graphs/Confusion_Matrices.png")
    plt.show()