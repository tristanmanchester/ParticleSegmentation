"""Morphological operations for processing binary masks."""

from typing import Union, Tuple
import logging
import numpy as np
import cupy as cp
from scipy import ndimage as ndi
from cupyx.scipy import ndimage as cundi


def get_optimal_chunk_size(shape: Tuple[int, ...], kernel_size: int, use_gpu: bool) -> int:
    """Calculate optimal chunk size based on available GPU memory.
    
    We need to account for:
    1. Input chunk with padding
    2. GPU arrays for erosion and dilation
    3. Safety margin for other operations
    """
    if not use_gpu:
        return 100  # Default chunk size for CPU
        
    try:
        # Get available GPU memory (in bytes)
        meminfo = cp.cuda.runtime.memGetInfo()
        free_memory = meminfo[0]
        total_memory = meminfo[1]
        
        # Calculate memory needed per Z slice
        y_size, x_size = shape[1:]
        bytes_per_pixel = 1  # Binary mask uses 1 byte per pixel
        memory_per_slice = y_size * x_size * bytes_per_pixel
        
        # Account for:
        # - Input chunk with padding (1x)
        # - GPU arrays for processing (2x for erosion and dilation)
        # - Safety margin (30% of available memory)
        safety_margin = 0.7  # Use 70% of available memory
        available_memory = free_memory * safety_margin
        
        # Memory needed per chunk = (1 + 2) * (chunk_size + 2*padding) * memory_per_slice
        # Solve for chunk_size
        padding = kernel_size
        memory_overhead = 3  # 1 input + 2 for processing
        chunk_size = int(
            (available_memory / (memory_overhead * memory_per_slice)) - (2 * padding)
        )
        
        # Ensure chunk size is reasonable
        chunk_size = max(10, min(chunk_size, shape[0]))
        
        logging.info(f"GPU memory: {free_memory/1e9:.1f}GB free of {total_memory/1e9:.1f}GB total")
        logging.info(f"Calculated chunk size: {chunk_size} slices")
        
        return chunk_size
        
    except Exception as e:
        logging.warning(f"Error calculating chunk size: {e}. Using default.")
        return 100


def process_chunk(chunk: np.ndarray, structure: np.ndarray, use_gpu: bool) -> np.ndarray:
    """Process a single chunk with morphological operations."""
    if use_gpu:
        chunk = cp.asarray(chunk)
        structure = cp.asarray(structure)
        eroded = cundi.binary_erosion(chunk, structure=structure)
        processed = cundi.binary_dilation(eroded, structure=structure)
        return cp.asnumpy(processed)
    else:
        eroded = ndi.binary_erosion(chunk, structure=structure)
        processed = ndi.binary_dilation(eroded, structure=structure)
        return processed


def apply_morphological_operations(
    mask: Union[np.ndarray, cp.ndarray],
    kernel_size: int,
    use_gpu: bool = True
) -> Union[np.ndarray, cp.ndarray]:
    """Apply morphological operations (erosion followed by dilation) to the binary mask.

    Args:
        mask: Binary mask from K-means clustering
        kernel_size: Size of the structuring element
        use_gpu: Whether to use GPU acceleration

    Returns:
        Processed binary mask

    Raises:
        ValueError: If mask is not binary or kernel size is invalid
    """
    # Convert to CPU numpy array for initial processing
    if isinstance(mask, cp.ndarray):
        mask = cp.asnumpy(mask)
    
    if kernel_size < 1:
        raise ValueError("Kernel size must be positive")
    
    # Create spherical structuring element
    radius = kernel_size // 2
    grid = np.ogrid[tuple(slice(-radius, radius + 1) for _ in range(3))]
    structure = sum(x*x for x in grid) <= radius*radius
    
    try:
        # Calculate optimal chunk size
        chunk_size = get_optimal_chunk_size(mask.shape, kernel_size, use_gpu)
        
        # Process in chunks along Z axis
        z_size = mask.shape[0]
        processed_mask = np.zeros_like(mask)
        
        # Add padding to handle particles at chunk boundaries
        pad_size = kernel_size
        padded_mask = np.pad(mask, ((pad_size, pad_size), (0, 0), (0, 0)), mode='reflect')
        
        for start_idx in range(0, z_size, chunk_size):
            # Calculate chunk indices with padding
            chunk_start = start_idx
            chunk_end = min(start_idx + chunk_size, z_size)
            pad_start = chunk_start
            pad_end = chunk_end + 2 * pad_size
            
            # Process padded chunk
            chunk = padded_mask[pad_start:pad_end]
            processed_chunk = process_chunk(chunk, structure, use_gpu)
            
            # Remove padding and store result
            processed_mask[chunk_start:chunk_end] = processed_chunk[pad_size:-pad_size]
            
            logging.info(f"Processed slices {chunk_start} to {chunk_end} of {z_size}")
        
        # Convert back to GPU if needed
        if use_gpu:
            processed_mask = cp.asarray(processed_mask)
        
        return processed_mask
    
    except Exception as e:
        raise RuntimeError(f"Morphological operations failed: {str(e)}")
