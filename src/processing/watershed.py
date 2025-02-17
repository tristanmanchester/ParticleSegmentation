"""Watershed segmentation for particle separation with robust memory handling."""

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


def handle_gpu_memory_error(func):
    """Decorator to handle GPU memory errors with CPU fallback."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (cp.cuda.memory.OutOfMemoryError, 
                cp.cuda.memory.CUDARuntimeError,
                cp.cuda.memory.CUDADriverError) as e:
            logging.warning(f"GPU memory error in {func.__name__}: {str(e)}")
            logging.info("Falling back to CPU processing")
            
            # Convert any GPU arrays to CPU
            new_args = []
            for arg in args:
                if isinstance(arg, cp.ndarray):
                    new_args.append(to_cpu(arg))
                else:
                    new_args.append(arg)
            
            # Ensure we use CPU processing
            if 'use_gpu' in kwargs:
                kwargs['use_gpu'] = False
            
            # Clear GPU memory
            clear_gpu_memory()
            
            # Retry with CPU
            return func(*new_args, **kwargs)
    return wrapper


@handle_gpu_memory_error
@log_execution_time
def compute_distance_transform(
    binary_mask: Union[np.ndarray, cp.ndarray],
    use_gpu: bool = True,
    timer: Timer = None
) -> Union[np.ndarray, cp.ndarray]:
    """Compute Euclidean distance transform with GPU optimization and CPU fallback."""
    try:
        if use_gpu and isinstance(binary_mask, cp.ndarray):
            with timed_stage(timer, "GPU Distance Transform") if timer else nullcontext():
                report_gpu_memory("Before distance transform")
                distance = cundi.distance_transform_edt(binary_mask, float64_distances=False)
                report_gpu_memory("After distance transform")
                return distance
    except Exception as e:
        logging.warning(f"GPU distance transform failed: {str(e)}")
        logging.warning("Falling back to CPU implementation")
        use_gpu = False
    
    # CPU implementation
    if isinstance(binary_mask, cp.ndarray):
        binary_mask = to_cpu(binary_mask)
    
    with timed_stage(timer, "CPU Distance Transform") if timer else nullcontext():
        distance = ndi.distance_transform_edt(binary_mask)
    return to_gpu(distance) if use_gpu else distance


@handle_gpu_memory_error
@log_execution_time
def perform_watershed_segmentation(
    binary_mask: Union[np.ndarray, cp.ndarray],
    min_distance: int,
    pixel_size: float,
    use_gpu: bool = True,
    timer: Timer = None
) -> np.ndarray:
    """Perform watershed segmentation with automatic CPU fallback for large datasets."""
    try:
        # Compute distance transform
        distance = compute_distance_transform(binary_mask, use_gpu, timer)
        
        # Move to CPU for watershed (not available on GPU)
        with timed_stage(timer, "CPU Transfer") if timer else nullcontext():
            distance_cpu = to_cpu(distance)
            binary_mask_cpu = to_cpu(binary_mask)
            del distance  # Free GPU memory
            if use_gpu:
                clear_gpu_memory()
        
        # Find local maxima for markers
        with timed_stage(timer, "Peak Detection") if timer else nullcontext():
            local_max = peak_local_max(
                distance_cpu,
                min_distance=min_distance,
                labels=binary_mask_cpu
            )
            
            markers = np.zeros_like(binary_mask_cpu, dtype=np.int32)
            markers[tuple(local_max.T)] = np.arange(1, len(local_max) + 1)
        
        # Perform watershed segmentation
        with timed_stage(timer, "Watershed Transform") if timer else nullcontext():
            labels = watershed(-distance_cpu, markers, mask=binary_mask_cpu)
            labels = labels.astype(np.int32)
        
        # Validate results
        n_particles = len(np.unique(labels)) - 1  # Subtract 1 for background
        if n_particles <= 0:
            raise ValueError("No particles were segmented")
        
        logging.info(f"Successfully segmented {n_particles} particles")
        
        return labels if not use_gpu else to_gpu(labels)
    
    except Exception as e:
        if isinstance(e, (cp.cuda.memory.OutOfMemoryError,
                         cp.cuda.memory.CUDARuntimeError,
                         cp.cuda.memory.CUDADriverError)):
            raise  # Let the decorator handle GPU memory errors
        
        logging.error(f"Watershed segmentation failed: {str(e)}")
        if use_gpu:
            logging.warning("Attempting CPU-only processing")
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
    """Calculate minimum distance between particles."""
    min_size, max_size = particle_size_range
    
    if min_size <= 0 or max_size <= min_size:
        raise ValueError(f"Invalid particle size range: ({min_size}, {max_size}) microns")
    
    if pixel_size <= 0:
        raise ValueError(f"Invalid pixel size: {pixel_size} microns")
    
    min_size_pixels = int(min_size / pixel_size)
    min_distance = max(3, min_size_pixels // 4)
    
    logging.info(f"Using minimum distance of {min_distance} pixels between particle centers")
    return min_distance