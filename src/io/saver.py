"""Functions for saving processed TIFF stacks and results."""

from pathlib import Path
from typing import Union

import numpy as np
import cupy as cp
import tifffile
from tqdm import tqdm


def save_tiff_stack(
    data: Union[np.ndarray, cp.ndarray],
    output_path: Union[str, Path],
    filename: str,
    compress: bool = True
) -> Path:
    """Save a 3D array as a TIFF stack.

    Args:
        data: 3D array to save
        output_path: Directory to save the TIFF stack
        filename: Name of the output file (without extension)
        compress: If True, use LZW compression

    Returns:
        Path to the saved file

    Raises:
        ValueError: If the data or path is invalid
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Move data to CPU if needed
    if isinstance(data, cp.ndarray):
        data = cp.asnumpy(data)
    
    # Validate data
    if not isinstance(data, np.ndarray) or data.ndim != 3:
        raise ValueError("Data must be a 3D numpy array")
    
    # Ensure uint16 for label maps
    if 'label' in filename.lower() and data.dtype != np.uint16:
        data = data.astype(np.uint16)
    
    # Construct output path
    output_file = output_path / f"{filename}.tiff"
    
    # Save with optional compression
    tifffile.imwrite(
        output_file,
        data,
        compression='lzw' if compress else None,
        metadata={'axes': 'ZYX'}
    )
    
    print(f"Saved TIFF stack to {output_file}")
    return output_file
