"""Statistical analysis of segmented particles."""

from typing import Union, Dict, Tuple
from pathlib import Path

import numpy as np
import cupy as cp
from scipy import ndimage as ndi


def calculate_particle_statistics(
    labels: Union[np.ndarray, cp.ndarray],
    pixel_size: float
) -> Dict[str, Union[np.ndarray, float]]:
    """Calculate statistics for segmented particles.

    Args:
        labels: Label map from watershed segmentation
        pixel_size: Size of each pixel in microns

    Returns:
        Dictionary containing:
            - equivalent_diameters: Array of equivalent spherical diameters (pixels)
            - volumes: Array of particle volumes (pixels^3)
            - count: Total number of particles
            - mean_diameter: Mean particle diameter (microns)
            - std_diameter: Standard deviation of particle diameters (microns)
            - min_diameter: Minimum particle diameter (microns)
            - max_diameter: Maximum particle diameter (microns)

    Raises:
        ValueError: If label map is invalid
    """
    # Move data to CPU if needed
    if isinstance(labels, cp.ndarray):
        labels = cp.asnumpy(labels)
    
    # Validate input
    if labels.dtype != np.uint16:
        raise ValueError("Label map must be uint16")
    
    # Get unique labels (excluding background)
    unique_labels = np.unique(labels)
    if unique_labels[0] == 0:
        unique_labels = unique_labels[1:]
    
    if len(unique_labels) == 0:
        raise ValueError("No particles found in label map")
    
    # Calculate volumes for each particle
    volumes = np.array([np.sum(labels == label) for label in unique_labels])
    
    # Calculate equivalent spherical diameters
    equivalent_diameters = 2 * np.power(3 * volumes / (4 * np.pi), 1/3)
    
    # Convert to microns for statistics
    diameters_microns = equivalent_diameters * pixel_size
    
    stats = {
        'equivalent_diameters': equivalent_diameters,  # in pixels
        'volumes': volumes,  # in pixels^3
        'count': len(unique_labels),
        'mean_diameter': np.mean(diameters_microns),  # in microns
        'std_diameter': np.std(diameters_microns),  # in microns
        'min_diameter': np.min(diameters_microns),  # in microns
        'max_diameter': np.max(diameters_microns)  # in microns
    }
    
    return stats


def save_statistics(
    stats: Dict[str, Union[np.ndarray, float]],
    output_path: Union[str, Path]
) -> Path:
    """Save particle statistics to a text file.

    Args:
        stats: Dictionary of particle statistics
        output_path: Directory to save the statistics file

    Returns:
        Path to saved statistics file
    """
    output_path = Path(output_path)
    output_file = output_path / 'particle_statistics.txt'
    
    with open(output_file, 'w') as f:
        f.write("Particle Segmentation Statistics\n")
        f.write("==============================\n\n")
        
        f.write(f"Total number of particles: {stats['count']}\n\n")
        
        f.write("Particle Diameters (μm):\n")
        f.write(f"  Mean: {stats['mean_diameter']:.2f}\n")
        f.write(f"  Standard Deviation: {stats['std_diameter']:.2f}\n")
        f.write(f"  Minimum: {stats['min_diameter']:.2f}\n")
        f.write(f"  Maximum: {stats['max_diameter']:.2f}\n")
    
    return output_file
