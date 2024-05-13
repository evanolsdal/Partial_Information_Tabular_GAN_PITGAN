import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

"""
This file contains some functions used for visualization
"""

# Plot for basic synthetic example
def plot_2D_data(df, title):
    """
    Plot 2D data points with categories represented by different colors.

    Parameters:
    - df (pd.DataFrame): DataFrame with columns 'Dimension_1', 'Dimension_2', and 'Category'.
    """
    # Create a color map based on unique category names
    categories = df['Category'].unique()
    colors = plt.cm.get_cmap('viridis', len(categories))
    color_map = {category: colors(i) for i, category in enumerate(categories)}

    # Plot each category with its corresponding color
    plt.figure(figsize=(10, 6))
    for category, color in color_map.items():
        subset = df[df['Category'] == category]
        plt.scatter(subset['Dimension_1'], subset['Dimension_2'], s=50, color=color, label=category)
    
    # Add legend, grid, title, and labels
    plt.legend(title='Category')
    plt.grid(True)
    plt.title(title)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.show()


# Plot function for privacy utility
def plot_priv_utility(evaluations, take_avg, privacy_type, plot_cols):
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Identify utility columns (all columns except 'Latent_dim' and the privacy_type)
    utility_columns = [col for col in evaluations.columns if col not in ['Latent_dim'] and '-keys' not in col]

    # If take_avg then take the average of the utility columns for each row and let this be the only
    # utility column
    if take_avg:
        plot_cols = ['Utility']
    
    # Generate colors from the 'viridis' colormap
    colors = plt.cm.get_cmap('viridis', len(utility_columns))

    # Plot each utility measure as a separate line
    for idx, utility in enumerate(plot_cols):
        # Get the color for the current utility measure
        color = colors(idx / len(utility_columns))

        # Plotting the data
        ax.plot(evaluations[privacy_type], evaluations[utility], marker='o', label=utility, color=color)

        # Annotate points with Latent_dim values
        for idx, row in evaluations.iterrows():
            ax.annotate(f'{row["Latent_dim"]:.0f}', (row[privacy_type], row[utility]),
                         textcoords="offset points", xytext=(0, 10), ha='center')

    # Set fixed axes limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Adding labels and title
    ax.set_xlabel('Privacy')
    ax.set_ylabel('Utility')
    ax.set_title('Privacy vs Utility Plot')
    ax.legend(title='Utility Measures')
    ax.grid(True)

    # Show plot
    plt.show()