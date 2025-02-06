"""Watershed segmentation for particle separation."""

from typing import Union, Tuple

import numpy as np
import cupy as cp
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max


def perform_watershed_segmentation(
    binary_mask: Union[np.ndarray, cp.ndarray],
    min_distance: int,
    pixel_size: float,
    use_gpu: bool = True
) -> np.ndarray:
    """Perform watershed segmentation on the binary mask.

    Args:
        binary_mask: Processed binary mask
        min_distance: Minimum distance between particles in pixels
        pixel_size: Size of each pixel in microns
        use_gpu: Whether to use GPU acceleration (Note: watershed is CPU-only)

    Returns:
        Label map of segmented particles (uint16)

    Raises:
        ValueError: If input parameters are invalid
    """
    # Move data to CPU if needed (watershed is CPU-only)
    if isinstance(binary_mask, cp.ndarray):
        binary_mask = cp.asnumpy(binary_mask)
    
    # Compute distance transform
    distance = ndi.distance_transform_edt(binary_mask)
    
    # Find local maxima (particle centers)
    coordinates = peak_local_max(
        distance,
        min_distance=min_distance,
        labels=binary_mask
    )
    
    # Create markers for watershed
    markers = np.zeros_like(binary_mask, dtype=np.uint16)
    markers[tuple(coordinates.T)] = np.arange(1, len(coordinates) + 1)
    
    # Perform watershed segmentation
    labels = watershed(-distance, markers, mask=binary_mask)
    
    # Move result back to GPU if needed
    if use_gpu:
        labels = cp.asarray(labels)
    
    return labels


def calculate_min_distance(
    particle_size_range: Tuple[float, float],
    pixel_size: float
) -> int:
    """Calculate minimum distance between particles based on physical parameters.

    Args:
        particle_size_range: (min_size, max_size) in microns
        pixel_size: Size of each pixel in microns

    Returns:
        Minimum distance in pixels

    Raises:
        ValueError: If input parameters are invalid
    """
    min_size, max_size = particle_size_range
    
    if min_size <= 0 or max_size <= min_size:
        raise ValueError("Invalid particle size range")
    
    if pixel_size <= 0:
        raise ValueError("Pixel size must be positive")
    
    # Convert minimum particle size to pixels
    min_size_pixels = int(min_size / pixel_size)
    
    # Use 1/4 of minimum particle size as minimum distance
    return max(3, min_size_pixels // 4)
