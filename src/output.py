import matplotlib.pyplot as plt
import numpy as np

def show_training(epochs, head_acc_list, avg_loss_list):
    print("Showing training diagnostics.")
    fig, ax = plt.subplots(1,2)
    
    x = np.arange(0, epochs)
    y1 = avg_loss_list
    y2 = head_acc_list
    ax[0].plot(x, y1, color='orange'); ax[0].set_title("Average Loss"); ax[0].set_xlabel("Epochs")
    ax[1].plot(x, y2, color='blue'); ax[1].set_title("Head Accuracy"); ax[1].set_xlabel("Epochs")

    plt.show()

