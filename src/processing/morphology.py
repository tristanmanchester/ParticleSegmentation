"""Morphological operations for processing binary masks."""

from typing import Union, Tuple

import numpy as np
import cupy as cp
from scipy import ndimage as ndi
from cupyx.scipy import ndimage as cundi


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
    # Validate input
    if isinstance(mask, cp.ndarray):
        unique_values = cp.unique(mask)
    else:
        unique_values = np.unique(mask)
    
    if not all(x in [0, 1] for x in unique_values):
        raise ValueError("Input mask must be binary (0s and 1s only)")
    
    if kernel_size < 1:
        raise ValueError("Kernel size must be positive")
    
    # Create spherical structuring element
    radius = kernel_size // 2
    grid = np.ogrid[tuple(slice(-radius, radius + 1) for _ in range(3))]
    structure = sum(x*x for x in grid) <= radius*radius
    
    try:
        if use_gpu and isinstance(mask, cp.ndarray):
            # Use CuPy for GPU acceleration
            structure = cp.asarray(structure)
            eroded = cundi.binary_erosion(mask, structure=structure)
            processed = cundi.binary_dilation(eroded, structure=structure)
        else:
            # Use SciPy for CPU
            if isinstance(mask, cp.ndarray):
                mask = cp.asnumpy(mask)
            eroded = ndi.binary_erosion(mask, structure=structure)
            processed = ndi.binary_dilation(eroded, structure=structure)
            
            if use_gpu:
                processed = cp.asarray(processed)
        
        return processed
    
    except Exception as e:
        raise RuntimeError(f"Morphological operations failed: {str(e)}")
