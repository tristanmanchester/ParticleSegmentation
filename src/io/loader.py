"""Functions for loading TIFF stacks."""

from pathlib import Path
from typing import Union, Tuple
import os
import psutil

import numpy as np
import tifffile
import cupy as cp
from tqdm import tqdm
from scipy.ndimage import zoom


def bin_volume(data: np.ndarray, binning_factor: int) -> np.ndarray:
    """Bin a 3D volume by averaging neighboring voxels.
    
    Args:
        data: Input 3D array
        binning_factor: Factor by which to bin the data (e.g., 2 means 2x2x2 voxels become 1)
    
    Returns:
        Binned array with reduced dimensions
    """
    if binning_factor <= 1:
        return data
        
    zoom_factor = 1.0 / binning_factor
    return zoom(data, (zoom_factor, zoom_factor, zoom_factor), order=1)


def estimate_memory_usage(
    shape: Tuple[int, ...],
    dtype: np.dtype,
    use_gpu: bool = True
) -> Tuple[float, float, float]:
    """Estimate memory usage for loading a TIFF stack.

    Args:
        shape: Shape of the array (z, y, x)
        dtype: Data type of the array
        use_gpu: Whether GPU will be used

    Returns:
        Tuple of (required_memory_gb, available_memory_gb, gpu_memory_gb)
        gpu_memory_gb will be None if use_gpu is False

    Raises:
        MemoryError: If insufficient memory is available
    """
    bytes_per_element = np.dtype(dtype).itemsize
    total_bytes = np.prod(shape) * bytes_per_element
    required_gb = total_bytes / (1024**3)
    
    # Get available system memory
    system_memory = psutil.virtual_memory()
    available_gb = system_memory.available / (1024**3)
    
    # Check GPU memory if needed
    gpu_memory_gb = None
    if use_gpu:
        try:
            gpu_memory = cp.cuda.runtime.memGetInfo()
            gpu_memory_gb = gpu_memory[0] / (1024**3)  # Free memory in GB
        except Exception as e:
            raise RuntimeError(f"Failed to get GPU memory info: {str(e)}")
    
    return required_gb, available_gb, gpu_memory_gb


def check_memory_requirements(
    required_gb: float,
    available_gb: float,
    gpu_memory_gb: float = None,
    threshold: float = 0.8
) -> None:
    """Check if there is sufficient memory available.

    Args:
        required_gb: Required memory in GB
        available_gb: Available system memory in GB
        gpu_memory_gb: Available GPU memory in GB (if using GPU)
        threshold: Fraction of available memory to allow using

    Raises:
        MemoryError: If insufficient memory is available
    """
    if required_gb > available_gb * threshold:
        raise MemoryError(
            f"Insufficient system memory. Required: {required_gb:.1f}GB, "
            f"Available: {available_gb:.1f}GB"
        )
    
    if gpu_memory_gb is not None and required_gb > gpu_memory_gb * threshold:
        raise MemoryError(
            f"Insufficient GPU memory. Required: {required_gb:.1f}GB, "
            f"Available: {gpu_memory_gb:.1f}GB"
        )


def is_binary_mask(data: np.ndarray) -> bool:
    """Check if the input data is a binary mask.

    Args:
        data: Input array

    Returns:
        True if data is binary (contains only 0s and 1s)
    """
    unique_values = np.unique(data)
    return len(unique_values) <= 2 and all(v in [0, 1] for v in unique_values)


def load_tiff_stack(
    path: Union[str, Path],
    use_gpu: bool = True,
    validate: bool = True,
    binning_factor: int = 1
) -> Union[np.ndarray, cp.ndarray]:
    """Load a stack of TIFF images into memory.

    Args:
        path: Path to directory containing TIFF files or path to a single multi-page TIFF
        use_gpu: If True, load data directly to GPU memory
        validate: If True, perform validation checks on the data
        binning_factor: Factor by which to bin the data (e.g., 2 means 2x2x2 voxels become 1)

    Returns:
        3D array containing the image stack

    Raises:
        ValueError: If validation fails or input is invalid
        MemoryError: If insufficient memory to load stack
    """
    path = Path(path)
    
    try:
        # Get data shape and type before loading
        if path.is_file():
            with tifffile.TiffFile(path) as tif:
                shape = tif.series[0].shape
                dtype = tif.series[0].dtype
        else:
            tiff_files = sorted(path.glob('*.tif*'))
            if not tiff_files:
                raise ValueError(f"No TIFF files found in {path}")
            
            first_img = tifffile.imread(tiff_files[0])
            shape = (len(tiff_files), *first_img.shape)
            dtype = first_img.dtype
        
        # Estimate and check memory requirements
        required_gb, available_gb, gpu_memory_gb = estimate_memory_usage(
            shape, dtype, use_gpu
        )
        check_memory_requirements(required_gb, available_gb, gpu_memory_gb)
        
        print(f"\nMemory requirements:")
        print(f"Required: {required_gb:.1f}GB")
        print(f"Available System Memory: {available_gb:.1f}GB")
        if gpu_memory_gb is not None:
            print(f"Available GPU Memory: {gpu_memory_gb:.1f}GB")
        
        # Load the data
        if path.is_file():
            with tifffile.TiffFile(path) as tif:
                data = tif.asarray()
        else:
            data = np.empty(shape, dtype=dtype)
            data[0] = first_img
            
            for i, file in enumerate(tqdm(tiff_files[1:], desc="Loading TIFF stack")):
                data[i+1] = tifffile.imread(file)
        
        # Apply binning if requested
        if binning_factor > 1:
            print(f"\nApplying {binning_factor}x binning...")
            data = bin_volume(data, binning_factor)
            print(f"New shape after binning: {data.shape}")
    
    except MemoryError:
        raise MemoryError("Insufficient memory to load TIFF stack")
    except Exception as e:
        raise ValueError(f"Error loading TIFF stack: {str(e)}")

    if validate:
        _validate_stack(data)

    if use_gpu:
        try:
            return cp.asarray(data)
        except Exception as e:
            raise RuntimeError(f"Failed to transfer data to GPU: {str(e)}")
    
    return data


def _validate_stack(data: np.ndarray) -> None:
    """Validate the loaded TIFF stack.

    Args:
        data: 3D numpy array containing the image stack

    Raises:
        ValueError: If validation fails
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("Data must be a numpy array")
    
    if data.ndim != 3:
        raise ValueError(f"Expected 3D array, got {data.ndim}D")
    
    if not np.issubdtype(data.dtype, np.number):
        raise ValueError(f"Expected numeric data type, got {data.dtype}")
    
    if np.isnan(data).any():
        raise ValueError("Data contains NaN values")
    
    if np.isinf(data).any():
        raise ValueError("Data contains infinite values")
