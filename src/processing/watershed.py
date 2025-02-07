"""Watershed segmentation for particle separation."""

from typing import Union, Tuple
import logging

import numpy as np
import cupy as cp
from scipy import ndimage as ndi
from cupyx.scipy import ndimage as cundi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from tqdm import tqdm
from contextlib import nullcontext

from ..utils.gpu_utils import to_gpu, to_cpu, report_gpu_memory, clear_gpu_memory
from ..utils.timing import log_execution_time, Timer, timed_stage


@log_execution_time
def compute_distance_transform(
    binary_mask: Union[np.ndarray, cp.ndarray],
    use_gpu: bool = True,
    timer: Timer = None
) -> Union[np.ndarray, cp.ndarray]:
    """Compute Euclidean distance transform with GPU optimization.

    Args:
        binary_mask: Input binary mask
        use_gpu: Whether to use GPU acceleration
        timer: Optional Timer instance for performance tracking

    Returns:
        Distance transform array
    """
    try:
        if use_gpu and isinstance(binary_mask, cp.ndarray):
            # Use CuPy's implementation
            with timed_stage(timer, "GPU Distance Transform") if timer else nullcontext():
                report_gpu_memory("Before distance transform")
                distance = cundi.distance_transform_edt(binary_mask)
                report_gpu_memory("After distance transform")
                return distance
    except Exception as e:
        logging.warning(f"GPU distance transform failed: {str(e)}")
        logging.warning("Falling back to CPU implementation")
        use_gpu = False
    
    # Fallback to CPU implementation
    if isinstance(binary_mask, cp.ndarray):
        binary_mask = to_cpu(binary_mask)
    
    with timed_stage(timer, "CPU Distance Transform") if timer else nullcontext():
        distance = ndi.distance_transform_edt(binary_mask)
    return to_gpu(distance) if use_gpu else distance


@log_execution_time
def validate_binary_mask(
    mask: Union[np.ndarray, cp.ndarray],
    timer: Timer = None
) -> None:
    """Validate that the input is a proper binary mask.

    Args:
        mask: Input mask to validate
        timer: Optional Timer instance for performance tracking

    Raises:
        ValueError: If mask is invalid
    """
    with timed_stage(timer, "Mask Validation") if timer else nullcontext():
        # Use GPU if available
        if isinstance(mask, np.ndarray):
            xp = np
        else:
            xp = cp
        
        # Check if binary using unique values
        unique_values = xp.unique(mask)
        if not set(unique_values.get() if isinstance(unique_values, cp.ndarray) else unique_values) <= {0, 1}:
            raise ValueError(
                "Input mask must be binary (0s and 1s only). "
                f"Found values: {sorted(unique_values)}"
            )
        
        # Quick check for empty or full mask
        mask_sum = xp.sum(mask)
        if isinstance(mask_sum, cp.ndarray):
            mask_sum = mask_sum.item()
        
        if mask_sum == 0:
            raise ValueError("Binary mask is empty (all zeros)")
        
        if mask_sum == mask.size:
            raise ValueError("Binary mask is full (all ones)")
        
        # Basic sanity check on mask size
        if mask_sum < 27:  # Minimum size of 3x3x3
            logging.warning("Binary mask contains very few positive pixels")


@log_execution_time
def perform_watershed_segmentation(
    binary_mask: Union[np.ndarray, cp.ndarray],
    min_distance: int,
    pixel_size: float,
    use_gpu: bool = False,  # Deprecated, watershed is CPU-only
    timer: Timer = None
) -> np.ndarray:
    """Perform watershed segmentation on the binary mask.

    Args:
        binary_mask: Processed binary mask
        min_distance: Minimum distance between particles in pixels
        pixel_size: Size of each pixel in microns
        use_gpu: Deprecated, watershed is CPU-only
        timer: Optional Timer instance for performance tracking

    Returns:
        Label map of segmented particles (uint16)

    Raises:
        ValueError: If input parameters are invalid
    """
    
    try:
        # Validate binary mask
        validate_binary_mask(binary_mask, timer)
        
        # Compute distance transform with GPU optimization
        distance = compute_distance_transform(binary_mask, use_gpu, timer)
        
        # Move to CPU for peak detection and watershed
        # (these operations are not available on GPU)
        with timed_stage(timer, "CPU Transfer") if timer else nullcontext():
            distance_cpu = to_cpu(distance)
            binary_mask_cpu = to_cpu(binary_mask)
        
        if use_gpu:
            # Clear GPU memory after transfer
            clear_gpu_memory()
            report_gpu_memory("After CPU transfer")
        
        # Watershed segmentation
        with timed_stage(timer, "Watershed Transform") if timer else nullcontext():
            # Find local maxima for markers
            local_max = peak_local_max(
                distance_cpu,
                min_distance=min_distance,
                labels=binary_mask_cpu
            )
            
            # Create markers for watershed
            markers = np.zeros_like(binary_mask_cpu, dtype=np.int32)
            markers[tuple(local_max.T)] = np.arange(1, len(local_max) + 1)
            
            # Perform watershed
            labels = watershed(-distance_cpu, markers, mask=binary_mask_cpu)
            
            # Ensure int32 dtype
            labels = labels.astype(np.int32)
            
            if use_gpu:
                # Move back to GPU if needed
                labels = to_gpu(labels)
        
        # Validate results
        final_labels = np.unique(labels)
        if len(final_labels) <= 1:
            raise ValueError("No particles were segmented")
        
        n_particles = len(final_labels) - 1  # Subtract 1 for background
        logging.info(f"Successfully segmented {n_particles} particles")
        
        # Check if we're approaching int32 limit
        if n_particles > 0.9 * np.iinfo(np.int32).max:
            logging.warning(
                f"Number of particles ({n_particles}) is approaching int32 limit. "
                "Consider processing the data in smaller chunks."
            )
        
        print(f"Successfully segmented {n_particles} particles")
        
        return labels
    
    except Exception as e:
        logging.error(f"Watershed segmentation failed: {str(e)}")
        if use_gpu:
            logging.warning("Attempting to fall back to CPU-only processing")
            return perform_watershed_segmentation(
                to_cpu(binary_mask),
                min_distance,
                pixel_size,
                use_gpu=False,
                timer=timer
            )
        raise


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
        raise ValueError(
            f"Invalid particle size range: ({min_size}, {max_size}) microns"
        )
    
    if pixel_size <= 0:
        raise ValueError(f"Invalid pixel size: {pixel_size} microns")
    
    # Convert minimum particle size to pixels
    min_size_pixels = int(min_size / pixel_size)
    
    # Use 1/4 of minimum particle size as minimum distance
    min_distance = max(3, min_size_pixels // 4)
    print(f"\nUsing minimum distance of {min_distance} pixels between particle centers")
    
    return min_distance
