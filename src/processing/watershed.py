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
        # Convert to numpy if needed
        if isinstance(mask, cp.ndarray):
            mask = to_cpu(mask)
        
        # Check if binary
        unique_values = np.unique(mask)
        if not set(unique_values) <= {0, 1}:
            raise ValueError(
                "Input mask must be binary (0s and 1s only). "
                f"Found values: {sorted(unique_values)}"
            )
        
        # Check if empty
        if not np.any(mask):
            raise ValueError("Binary mask is empty (all zeros)")
        
        # Check if all ones
        if np.all(mask):
            raise ValueError("Binary mask is full (all ones)")
        
        # Check for isolated pixels
        structure = np.ones((3, 3, 3))
        labeled, num_features = ndi.label(mask, structure=structure)
        if num_features == 0:
            raise ValueError("No connected components found in binary mask")
        
        # Warn about potentially problematic features
        volumes = np.array([np.sum(labeled == i) for i in range(1, num_features + 1)])
        small_particles = np.sum(volumes < 27)  # 3x3x3 cube
        if small_particles > 0:
            print(f"\nWarning: Found {small_particles} particles smaller than 3x3x3 voxels")


@log_execution_time
def perform_watershed_segmentation(
    binary_mask: Union[np.ndarray, cp.ndarray],
    min_distance: int,
    pixel_size: float,
    use_gpu: bool = True,
    timer: Timer = None
) -> np.ndarray:
    """Perform watershed segmentation on the binary mask.

    Args:
        binary_mask: Processed binary mask
        min_distance: Minimum distance between particles in pixels
        pixel_size: Size of each pixel in microns
        use_gpu: Whether to use GPU acceleration (Note: watershed is CPU-only)
        timer: Optional Timer instance for performance tracking

    Returns:
        Label map of segmented particles (uint16)

    Raises:
        ValueError: If input parameters are invalid
    """
    try:
        # Report initial GPU memory usage
        if use_gpu:
            report_gpu_memory("Start watershed")
        
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
        
        # Find local maxima (particle centers)
        with timed_stage(timer, "Peak Detection") if timer else nullcontext():
            coordinates = peak_local_max(
                distance_cpu,
                min_distance=min_distance,
                labels=binary_mask_cpu
            )
        
        if len(coordinates) == 0:
            raise ValueError(
                f"No particle centers found. Try reducing min_distance "
                f"(currently {min_distance} pixels)"
            )
        
        print(f"\nFound {len(coordinates)} potential particle centers")
        
        # Create markers for watershed
        with timed_stage(timer, "Marker Creation") if timer else nullcontext():
            markers = np.zeros_like(binary_mask_cpu, dtype=np.uint16)
            markers[tuple(coordinates.T)] = np.arange(1, len(coordinates) + 1)
        
        # Perform watershed segmentation with progress bar
        with timed_stage(timer, "Watershed") if timer else nullcontext():
            print("\nPerforming watershed segmentation...")
            
            # Initialize progress bar
            total_slices = binary_mask_cpu.shape[0]
            progress_bar = tqdm(
                total=total_slices,
                desc="Processing slices",
                unit="slice"
            )
            
            # Process each slice
            labels = watershed(
                -distance_cpu,
                markers,
                mask=binary_mask_cpu
            )
            
            # Update progress bar for each slice
            for _ in range(total_slices):
                progress_bar.update(1)
            
            progress_bar.close()
        
        # Validate results
        final_labels = np.unique(labels)
        if len(final_labels) <= 1:
            raise ValueError("Watershed segmentation failed: no particles found")
        
        print(f"Successfully segmented {len(final_labels) - 1} particles")
        
        # Move result back to GPU if needed
        if use_gpu:
            with timed_stage(timer, "GPU Transfer") if timer else nullcontext():
                labels = to_gpu(labels)
                report_gpu_memory("End watershed")
        
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
