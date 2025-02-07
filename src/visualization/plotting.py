"""Plotting functions for visualization of results using GridSpec for precise layout control."""

from typing import Union, Tuple, List
from pathlib import Path
import logging

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

def plot_orthogonal_views(
    raw_data: Union[np.ndarray, cp.ndarray],
    binary_mask: Union[np.ndarray, cp.ndarray],
    labels: Union[np.ndarray, cp.ndarray],
    output_path: Union[str, Path],
    dpi: int = 300
) -> Path:
    """Plot orthogonal views of the data processing stages using GridSpec for layout control.

    Args:
        raw_data: Original image data
        binary_mask: Binary mask after processing
        labels: Final watershed segmentation labels
        output_path: Path to save the plot (including filename)
        dpi: DPI for the output image

    Returns:
        Path to saved figure

    Raises:
        ValueError: If input arrays have different shapes
    """
    logging.info("Generating orthogonal views visualization...")
    
    # Move data to CPU if needed
    if isinstance(raw_data, cp.ndarray):
        logging.info("Moving raw data to CPU...")
        raw_data = cp.asnumpy(raw_data)
    if isinstance(binary_mask, cp.ndarray):
        logging.info("Moving binary mask to CPU...")
        binary_mask = cp.asnumpy(binary_mask)
    if isinstance(labels, cp.ndarray):
        logging.info("Moving labels to CPU...")
        labels = cp.asnumpy(labels)
    
    # Validate shapes
    if not (raw_data.shape == binary_mask.shape == labels.shape):
        raise ValueError("Input arrays must have the same shape")
    
    # Calculate aspect ratios for proper scaling
    z_size, y_size, x_size = raw_data.shape
    xy_aspect = x_size / y_size
    xz_aspect = x_size / z_size
    yz_aspect = y_size / z_size
    
    # Create randomized colormap for labels
    logging.info("Creating randomized colormap...")
    n_labels = len(np.unique(labels))
    
    # Get tab20 colors (20 distinct colors)
    tab20_cmap = plt.cm.tab20(np.linspace(0, 1, 20))  # tab20 has 20 colors
    
    # Create random color mapping, repeating tab20 colors if needed
    # but shuffling each set of 20
    colors = []
    for i in range(0, n_labels, 20):
        batch = tab20_cmap.copy()
        np.random.shuffle(batch)
        colors.extend(batch)
    colors = colors[:n_labels]
    
    # Ensure background (label 0) is black
    label_colors = np.zeros((n_labels, 4))  # RGBA array
    label_colors[1:] = colors[:(n_labels-1)]  # Skip first color (index 0)
    
    # Create custom colormap
    random_cmap = plt.matplotlib.colors.ListedColormap(label_colors)
    # Calculate optimal figure size based on data dimensions
    # Let's set a base width in inches for the figure
    base_width = 15  # inches
    
    # Calculate the total height needed based on the data dimensions
    xy_height = y_size  # Height of top row (XY view)
    xz_height = z_size  # Height of middle row (XZ view)
    yz_height = z_size  # Height of bottom row (YZ view)
    total_relative_height = xy_height + xz_height + yz_height
    
    # Scale the figure height to maintain proper data proportions
    # The 1.1 factor adds a small buffer for titles
    fig_height = base_width * (total_relative_height / (3 * x_size)) * 1.1
    
    # Create figure with calculated dimensions
    fig = plt.figure(figsize=(base_width, fig_height))
    
    # Create GridSpec with custom spacing between rows
    # Calculate ratios to maintain proper rectangular proportions
    # For XZ and YZ views, we want to show the full width (x_size or y_size) 
    # against the z_size while maintaining true proportions
    xy_height = y_size  # Height of top row (XY view)
    xz_height = z_size  # Height of middle row (XZ view)
    yz_height = z_size  # Height of bottom row (YZ view)
    
    # Normalize heights to the XY view height
    height_ratios = [1, xz_height/xy_height, yz_height/xy_height]
    
    gs = GridSpec(3, 3, figure=fig,
                 height_ratios=height_ratios,
                 width_ratios=[1, 1, 1],
                 hspace=0.04,  # Small uniform space between rows
                 wspace=0)
    
    # Get middle indices for each dimension
    z, y, x = [s // 2 for s in raw_data.shape]
    
    logging.info("Plotting XY slices...")
    # Plot XY slices (top row)
    ax_xy_raw = fig.add_subplot(gs[0, 0])
    ax_xy_mask = fig.add_subplot(gs[0, 1])
    ax_xy_labels = fig.add_subplot(gs[0, 2])
    
    ax_xy_raw.imshow(raw_data[z], cmap='gray', aspect='equal')
    ax_xy_mask.imshow(binary_mask[z], cmap='binary', aspect='equal')
    ax_xy_labels.imshow(labels[z], cmap=random_cmap, aspect='equal')
    
    logging.info("Plotting XZ slices...")
    # Plot XZ slices (middle row)
    ax_xz_raw = fig.add_subplot(gs[1, 0])
    ax_xz_mask = fig.add_subplot(gs[1, 1])
    ax_xz_labels = fig.add_subplot(gs[1, 2])
    
    # For XZ view, we want width=x_size and height=z_size
    ax_xz_raw.imshow(raw_data[:, y, :], cmap='gray', aspect='auto')
    ax_xz_mask.imshow(binary_mask[:, y, :], cmap='binary', aspect='auto')
    ax_xz_labels.imshow(labels[:, y, :], cmap=random_cmap, aspect='auto')
    
    logging.info("Plotting YZ slices...")
    # Plot YZ slices (bottom row)
    ax_yz_raw = fig.add_subplot(gs[2, 0])
    ax_yz_mask = fig.add_subplot(gs[2, 1])
    ax_yz_labels = fig.add_subplot(gs[2, 2])
    
    # For YZ view, we want width=y_size and height=z_size
    ax_yz_raw.imshow(raw_data[:, :, x], cmap='gray', aspect='auto')
    ax_yz_mask.imshow(binary_mask[:, :, x], cmap='binary', aspect='auto')
    ax_yz_labels.imshow(labels[:, :, x], cmap=random_cmap, aspect='auto')
    
    # Set titles
    titles = ['Raw Data', 'Binary Mask', 'Segmented Particles']
    views = ['XY View', 'XZ View', 'YZ View']
    
    for ax, title in zip([ax_xy_raw, ax_xy_mask, ax_xy_labels], titles):
        ax.set_title(title)
    
    for ax, view in zip([ax_xy_raw, ax_xz_raw, ax_yz_raw], views):
        ax.set_ylabel(view)
    
    # Set black background for label plots
    ax_xy_labels.set_facecolor('black')
    ax_xz_labels.set_facecolor('black')
    ax_yz_labels.set_facecolor('black')
    
    # Remove ticks from all axes
    all_axes = [ax_xy_raw, ax_xy_mask, ax_xy_labels,
                ax_xz_raw, ax_xz_mask, ax_xz_labels,
                ax_yz_raw, ax_yz_mask, ax_yz_labels]
    
    for ax in all_axes:
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Save figure
    logging.info("Saving figure...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Save with small white border
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1, 
                facecolor='white', edgecolor='none')
    plt.close()
    
    logging.info(f"Saved orthogonal views to {output_path}")
    return output_path


def plot_size_distribution(
    diameters: np.ndarray,
    pixel_size: float,
    output_path: Union[str, Path],
    dpi: int = 300
) -> Path:
    """Plot particle size distribution histogram.

    Args:
        diameters: Array of equivalent spherical diameters in pixels
        pixel_size: Size of each pixel in microns
        output_path: Path to save the plot (including filename)
        dpi: DPI for the output image

    Returns:
        Path to saved figure
    """
    logging.info("Generating size distribution plot...")
    
    # Convert diameters to microns
    diameters_microns = diameters * pixel_size
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot histogram
    logging.info("Creating histogram...")
    plt.hist(diameters_microns, bins=50, edgecolor='black')
    plt.xlabel('Particle Diameter (μm)')
    plt.ylabel('Count')
    plt.title('Particle Size Distribution')
    
    # Add statistics
    logging.info("Adding statistical markers...")
    mean = np.mean(diameters_microns)
    std = np.std(diameters_microns)
    plt.axvline(mean, color='r', linestyle='--', label=f'Mean: {mean:.1f}μm')
    plt.axvline(mean + std, color='g', linestyle=':', label=f'±1σ: {std:.1f}μm')
    plt.axvline(mean - std, color='g', linestyle=':')
    plt.legend()
    
    # Save figure
    logging.info("Saving figure...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Saved size distribution plot to {output_path}")
    return output_path