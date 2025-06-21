import numpy as np
import matplotlib.pyplot as plt

def plot_losses(train_loss, val_loss):
    plt.plot(np.arange(len(train_loss)), train_loss, color='b', label='train')
    plt.plot(np.arange(len(val_loss)), val_loss, color='r', label='val')
    plt.legend()