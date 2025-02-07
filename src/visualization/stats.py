"""Statistical analysis of segmented particles."""

from typing import Union, Dict, Tuple
from pathlib import Path
import logging
import time
import numpy as np
import cupy as cp
from tqdm import tqdm


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
    logging.info("Starting particle statistics calculation...")
    start_time = time.time()
    
    try:
        # Move data to CPU if needed
        if isinstance(labels, cp.ndarray):
            logging.info("Moving data to CPU...")
            labels = cp.asnumpy(labels)
        
        # Validate input
        if labels.dtype != np.int32:
            raise ValueError("Label map must be int32")
        
        # Get unique labels and their counts efficiently using np.unique
        logging.info("Calculating particle volumes...")
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        # Remove background (label 0) if present
        if unique_labels[0] == 0:
            unique_labels = unique_labels[1:]
            counts = counts[1:]
        
        if len(unique_labels) == 0:
            raise ValueError("No particles found in label map")
        
        n_particles = len(unique_labels)
        logging.info(f"Found {n_particles} particles")
        
        # Counts array already contains the volumes
        volumes = counts
        
        processing_time = time.time() - start_time
        logging.info(f"Volume calculation completed in {processing_time:.2f}s")
        
        # Calculate equivalent spherical diameters (in pixels)
        logging.info("Calculating equivalent diameters...")
        equivalent_diameters = 2 * np.power(3 * volumes / (4 * np.pi), 1/3)
        
        # Convert diameters to microns for statistics
        diameters_microns = equivalent_diameters * pixel_size
        
        # Calculate summary statistics
        logging.info("Computing summary statistics...")
        stats = {
            'equivalent_diameters': equivalent_diameters,
            'volumes': volumes,
            'count': n_particles,
            'mean_diameter': np.mean(diameters_microns),
            'std_diameter': np.std(diameters_microns),
            'min_diameter': np.min(diameters_microns),
            'max_diameter': np.max(diameters_microns)
        }
        
        logging.info(f"Statistics summary:")
        logging.info(f"  Total particles: {stats['count']}")
        logging.info(f"  Mean diameter: {stats['mean_diameter']:.2f} microns")
        logging.info(f"  Diameter range: {stats['min_diameter']:.2f} - {stats['max_diameter']:.2f} microns")
        
        return stats
        
    except Exception as e:
        logging.error(f"Error calculating particle statistics: {str(e)}")
        raise


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
    output_file = output_path / "particle_statistics.txt"
    
    with open(output_file, 'w') as f:
        f.write("Particle Statistics Summary\n")
        f.write("=========================\n\n")
        f.write(f"Total particles: {stats['count']}\n")
        f.write(f"Mean diameter: {stats['mean_diameter']:.2f} microns\n")
        f.write(f"Standard deviation: {stats['std_diameter']:.2f} microns\n")
        f.write(f"Size range: {stats['min_diameter']:.2f} - {stats['max_diameter']:.2f} microns\n")
    
    return output_file
