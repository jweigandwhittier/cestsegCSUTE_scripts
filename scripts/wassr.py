#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 16:21:48 2024

@author: jonah
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_wassr_aha(pixelwise, proc_data, mask, labeled_segments, savedir, save_as):
    # Save B0 map
    np.save(savedir + '/wassr.npy', pixelwise)
    # Create B0 map
    b0_image = np.zeros_like(mask, dtype=float)
    for i in range(len(mask)):
        for j in range(len(mask[0])):
            if mask[i][j]:
                b0_image[i][j] = pixelwise.pop(0)
    # Set zero values to be fully transparent
    transparent_b0 = np.ma.masked_where(b0_image == 0, b0_image)
    # Zoom into the region based on the mask with a margin of ±20 pixels
    y_indices, x_indices = np.where(mask)
    x_min, x_max = max(np.min(x_indices) - 20, 0), min(np.max(x_indices) + 20, mask.shape[1])
    y_min, y_max = max(np.min(y_indices) - 20, 0), min(np.max(y_indices) + 20, mask.shape[0])

    # Plot 1: B0 Map
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(proc_data['M0'][y_min:y_max,x_min:x_max], cmap='gray')
    im = ax.imshow(transparent_b0[y_min:y_max,x_min:x_max], cmap='plasma', alpha=0.7)

    # Title and colorbar
    ax.set_title('$B_0$ Map', fontsize=22, fontname='Arial', weight='bold')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label('$B_0$ Shift (ppm)', fontsize=18)

    ax.axis('off')

    # Save and show the B0 map
    plt.tight_layout()
    plt.show()
    plt.savefig(f'{savedir}/B0_map.tif', dpi=300)

    # Extract B0 shift values for each segment
    segment_b0_values = {}
    for segment, pixel_indices in labeled_segments.items():
        b0_values = []
        for (i, j) in pixel_indices:
            if mask[i][j]:  # Ensure the pixel is within the mask
                b0_values.append(b0_image[i][j])
        segment_b0_values[segment] = b0_values

    # Plot 2: Boxplot for B0 shifts by AHA segment
    # Convert segment B0 values into a DataFrame for easier plotting with Seaborn
    segment_data = []
    for segment, b0_values in segment_b0_values.items():
        for b0_value in b0_values:
            segment_data.append({'Segment': segment, 'B0 Shift (ppm)': b0_value})

    df = pd.DataFrame(segment_data)

    # Set plot style and palette
    sns.set(style="whitegrid")
    palette = sns.color_palette("husl", len(segment_b0_values))  # Unique colors for each segment

    # Create the Seaborn boxplot
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(x='Segment', y='B0 Shift (ppm)', data=df, palette=palette, width=0.4)

    # Customize plot appearance
    ax.set_title('B0 Shift by AHA Segment', fontsize=22)
    ax.set_xlabel('AHA Segment', fontsize=18)
    ax.set_ylabel('$B_0$ Shift (ppm)', fontsize=18)

    # Adjust box width and spacing
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Show the plot
    plt.tight_layout()
    plt.show()

    # Save the plot
    plt.savefig(f'{savedir}/B0_boxplot_seaborn.tif', dpi=300)
    
def plot_wassr(pixelwise, proc_data, mask, savedir, save_as):
    # Save B0 map
    np.save(savedir + '/wassr.npy', pixelwise)
    # Create B0 map
    b0_image = np.zeros_like(mask, dtype=float)
    for i in range(len(mask)):
        for j in range(len(mask[0])):
            if mask[i][j]:
                b0_image[i][j] = pixelwise.pop(0)
    # Set zero values to be fully transparent
    transparent_b0 = np.ma.masked_where(b0_image == 0, b0_image)
    # Zoom into the region based on the mask with a margin of ±20 pixels
    y_indices, x_indices = np.where(mask)
    x_min, x_max = max(np.min(x_indices) - 20, 0), min(np.max(x_indices) + 20, mask.shape[1])
    y_min, y_max = max(np.min(y_indices) - 20, 0), min(np.max(y_indices) + 20, mask.shape[0])

    # Plot 1: B0 Map
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(proc_data['M0'][y_min:y_max,x_min:x_max], cmap='gray')
    im = ax.imshow(transparent_b0[y_min:y_max,x_min:x_max], cmap='plasma', alpha=0.7)

    # Title and colorbar
    ax.set_title('$B_0$ Map', fontsize=22, fontname='Arial', weight='bold')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label('$B_0$ Shift (ppm)', fontsize=18)

    ax.axis('off')

    # Save and show the B0 map
    plt.tight_layout()
    plt.show()
    plt.savefig(f'{savedir}/B0_map.tif', dpi=300)