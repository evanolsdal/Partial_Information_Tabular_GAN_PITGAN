import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

"""
This file contains plots for visualizing the losses of the different network parts as training 
progressed
"""

# function that plots the losses on the same graph
def plot_losses(losses, subgraphs):

    # select color palette for color blindness
    colors = plt.cm.get_cmap('viridis', len(losses))

    # Plotting the combined graph
    plt.figure(figsize=(10, 6))
    for idx, (key, values) in enumerate(losses.items()):
        plt.plot(values, label=key, color=colors(idx))
    plt.title('All Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


    # Check if there is more than one loss type, if so we plot individual subgraph of each loss
    if len(losses) > 1 and subgraphs:
        fig, axs = plt.subplots(len(losses), 1, figsize=(12, 5 * len(losses)), squeeze=False)
        axs = axs.flatten()  # This makes sure axs is always an array, even if there's only one subplot
        for i, (key, values) in enumerate(losses.items()):
            axs[i].plot(values, color=colors(i))
            axs[i].set_title(key)
            axs[i].set_xlabel('Epoch')
            axs[i].set_ylabel('Loss')
            axs[i].set_ylim(bottom=0)  # Ensure the y-axis starts at 0 
        plt.tight_layout()
        plt.show()

