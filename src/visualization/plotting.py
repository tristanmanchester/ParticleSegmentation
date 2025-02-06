"""Plotting functions for visualization of results."""

from typing import Union, Tuple, List
from pathlib import Path
import logging

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def plot_orthogonal_views(
    raw_data: Union[np.ndarray, cp.ndarray],
    binary_mask: Union[np.ndarray, cp.ndarray],
    labels: Union[np.ndarray, cp.ndarray],
    output_path: Union[str, Path],
    dpi: int = 300
) -> Path:
    """Plot orthogonal views of the data processing stages.

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
    
    logging.info("Creating figure with 3x3 grid...")
    # Create figure with 3x3 grid
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    # Get middle indices for each dimension
    z, y, x = [s // 2 for s in raw_data.shape]
    
    logging.info("Plotting XY slices...")
    # Plot XY slices (top row)
    axes[0, 0].imshow(raw_data[z], cmap='gray')
    axes[0, 1].imshow(binary_mask[z], cmap='binary')
    axes[0, 2].imshow(labels[z], cmap='nipy_spectral')
    
    logging.info("Plotting XZ slices...")
    # Plot XZ slices (middle row)
    axes[1, 0].imshow(raw_data[:, y, :], cmap='gray')
    axes[1, 1].imshow(binary_mask[:, y, :], cmap='binary')
    axes[1, 2].imshow(labels[:, y, :], cmap='nipy_spectral')
    
    logging.info("Plotting YZ slices...")
    # Plot YZ slices (bottom row)
    axes[2, 0].imshow(raw_data[:, :, x], cmap='gray')
    axes[2, 1].imshow(binary_mask[:, :, x], cmap='binary')
    axes[2, 2].imshow(labels[:, :, x], cmap='nipy_spectral')
    
    # Set titles
    titles = ['Raw Data', 'Binary Mask', 'Segmented Particles']
    views = ['XY View', 'XZ View', 'YZ View']
    
    for i, title in enumerate(titles):
        axes[0, i].set_title(title)
    for i, view in enumerate(views):
        axes[i, 0].set_ylabel(view)
    
    # Remove ticks
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Adjust layout and save
    logging.info("Adjusting layout and saving figure...")
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
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
