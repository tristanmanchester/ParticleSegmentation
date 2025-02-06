"""Functions for loading TIFF stacks."""

from pathlib import Path
from typing import Union, Tuple

import numpy as np
import tifffile
import cupy as cp
from tqdm import tqdm


def load_tiff_stack(
    path: Union[str, Path],
    use_gpu: bool = True,
    validate: bool = True
) -> Union[np.ndarray, cp.ndarray]:
    """Load a stack of TIFF images into memory.

    Args:
        path: Path to directory containing TIFF files or path to a single multi-page TIFF
        use_gpu: If True, load data directly to GPU memory
        validate: If True, perform validation checks on the data

    Returns:
        3D array containing the image stack

    Raises:
        ValueError: If validation fails or input is invalid
        MemoryError: If insufficient memory to load stack
    """
    path = Path(path)
    
    try:
        if path.is_file():
            # Single multi-page TIFF
            with tifffile.TiffFile(path) as tif:
                data = tif.asarray()
        else:
            # Directory of TIFF files
            tiff_files = sorted(path.glob('*.tif*'))
            if not tiff_files:
                raise ValueError(f"No TIFF files found in {path}")
            
            # Read first image to get dimensions
            first_img = tifffile.imread(tiff_files[0])
            shape = (len(tiff_files), *first_img.shape)
            dtype = first_img.dtype
            
            # Pre-allocate array
            data = np.empty(shape, dtype=dtype)
            data[0] = first_img
            
            # Load remaining images with progress bar
            for i, file in enumerate(tqdm(tiff_files[1:], desc="Loading TIFF stack")):
                data[i+1] = tifffile.imread(file)
    
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
    # Check dimensions
    if data.ndim != 3:
        raise ValueError(f"Expected 3D array, got {data.ndim}D")
    
    # Check if grayscale (no color channels)
    if len(data.shape) > 3:
        raise ValueError("Color images not supported, expected grayscale")
    
    # Check if binary mask (optional)
    unique_values = np.unique(data)
    is_binary = len(unique_values) == 2 and set(unique_values) <= {0, 1}
    
    if is_binary:
        print("Binary mask detected")
    
    # Check bit depth
    if data.dtype not in [np.uint8, np.uint16, np.float32]:
        raise ValueError(f"Unsupported data type: {data.dtype}")
    
    print(f"Stack dimensions: {data.shape}, dtype: {data.dtype}")
