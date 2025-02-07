"""Statistical analysis of segmented particles."""

from typing import Union, Dict, Tuple
from pathlib import Path
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import cupy as cp
from scipy import ndimage as ndi
from tqdm import tqdm


def _calculate_volume_chunk(args):
    """Helper function for parallel volume calculation.
    
    Args:
        args: Tuple of (labels, label_range)
        
    Returns:
        List of (label, volume) tuples
    """
    labels, label_range = args
    volumes = []
    for label in label_range:
        # Process one label at a time to minimize memory usage
        volume = np.sum(labels == label)
        volumes.append((label, volume))
    return volumes

def calculate_particle_statistics(
    labels: Union[np.ndarray, cp.ndarray],
    pixel_size: float,
    n_jobs: int = -1,
    chunk_size: int = 100  # Process 100 labels at a time by default
) -> Dict[str, Union[np.ndarray, float]]:
    """Calculate statistics for segmented particles.

    Args:
        labels: Label map from watershed segmentation
        pixel_size: Size of each pixel in microns
        n_jobs: Number of processes for parallel computation. -1 means use all cores.
        chunk_size: Number of labels to process in each parallel chunk

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
        
        # Get unique labels (excluding background)
        logging.info("Finding unique particle labels...")
        unique_labels = np.unique(labels)
        if unique_labels[0] == 0:
            unique_labels = unique_labels[1:]
        
        if len(unique_labels) == 0:
            raise ValueError("No particles found in label map")
        
        n_particles = len(unique_labels)
        logging.info(f"Found {n_particles} particles")
        
        # For very small datasets, use simple loop
        if n_particles < 100:
            logging.info("Using simple loop for small dataset...")
            volumes = np.array([np.sum(labels == label) for label in tqdm(unique_labels, desc="Processing particles")])
        else:
            # For larger datasets, use parallel processing with smaller chunks
            logging.info("Using parallel volume calculation...")
            import multiprocessing
            if n_jobs == -1:
                n_jobs = multiprocessing.cpu_count()
            
            # Adjust chunk size based on number of particles
            chunk_size = min(chunk_size, max(1, n_particles // (n_jobs * 4)))
            logging.info(f"Using chunk size of {chunk_size} labels")
            
            # Create chunks using array slicing
            chunks = []
            for i in range(0, len(unique_labels), chunk_size):
                chunk_labels = unique_labels[i:i + chunk_size]
                chunks.append((labels, chunk_labels))
            
            volumes_dict = {}
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                futures = []
                for chunk in chunks:
                    futures.append(executor.submit(_calculate_volume_chunk, chunk))
                
                # Process results as they complete
                failed_chunks = 0
                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing chunks"):
                    try:
                        for label, volume in future.result():
                            volumes_dict[label] = volume
                    except Exception as e:
                        failed_chunks += 1
                        logging.error(f"Error processing chunk: {str(e)}")
                        if failed_chunks > len(futures) // 3:  # If more than 1/3 of chunks fail
                            raise RuntimeError("Too many chunk processing failures")
            
            if failed_chunks > 0:
                logging.warning(f"{failed_chunks} chunks failed to process")
            
            # Convert dictionary to array in the correct order
            volumes = np.array([volumes_dict[label] for label in unique_labels])
        
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
            'count': len(unique_labels),
            'mean_diameter': np.mean(diameters_microns),
            'std_diameter': np.std(diameters_microns),
            'min_diameter': np.min(diameters_microns),
            'max_diameter': np.max(diameters_microns)
        }
        
        logging.info(f"Statistics summary:")
        logging.info(f"  - Particle count: {stats['count']}")
        logging.info(f"  - Mean diameter: {stats['mean_diameter']:.2f} µm")
        logging.info(f"  - Size range: {stats['min_diameter']:.2f} - {stats['max_diameter']:.2f} µm")
        
        return stats
        
    except Exception as e:
        logging.error(f"Error in particle statistics calculation: {str(e)}")
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
