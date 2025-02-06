"""GPU memory management utilities."""

from typing import Optional, Tuple
import logging

import cupy as cp
import numpy as np


def get_gpu_memory_info() -> Tuple[float, float]:
    """Get current GPU memory usage.

    Returns:
        Tuple of (free_memory_gb, total_memory_gb)
    """
    try:
        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
        return free_bytes / (1024**3), total_bytes / (1024**3)
    except Exception as e:
        logging.warning(f"Failed to get GPU memory info: {str(e)}")
        return 0.0, 0.0


def report_gpu_memory(stage: str = "") -> None:
    """Report current GPU memory usage.

    Args:
        stage: Current processing stage for logging
    """
    free_gb, total_gb = get_gpu_memory_info()
    used_gb = total_gb - free_gb
    percent_used = (used_gb / total_gb) * 100 if total_gb > 0 else 0
    
    print(f"\nGPU Memory Usage{f' ({stage})' if stage else ''}:")
    print(f"  Used: {used_gb:.1f}GB / {total_gb:.1f}GB ({percent_used:.1f}%)")
    print(f"  Free: {free_gb:.1f}GB")


def clear_gpu_memory() -> None:
    """Clear all unused memory on GPU."""
    try:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception as e:
        logging.warning(f"Failed to clear GPU memory: {str(e)}")


def to_gpu(
    data: np.ndarray,
    force_cpu: bool = False,
    clear_after: bool = True
) -> cp.ndarray:
    """Transfer data to GPU with memory management.

    Args:
        data: Input numpy array
        force_cpu: If True, keep data on CPU
        clear_after: If True, clear GPU memory after transfer

    Returns:
        CuPy array on GPU or numpy array if force_cpu is True

    Raises:
        RuntimeError: If GPU transfer fails
    """
    if force_cpu:
        return data
    
    try:
        gpu_data = cp.asarray(data)
        if clear_after:
            clear_gpu_memory()
        return gpu_data
    except Exception as e:
        logging.warning(f"Failed to transfer data to GPU: {str(e)}")
        return data


def to_cpu(
    data: cp.ndarray,
    clear_after: bool = True
) -> np.ndarray:
    """Transfer data to CPU with memory management.

    Args:
        data: Input CuPy array
        clear_after: If True, clear GPU memory after transfer

    Returns:
        Numpy array on CPU
    """
    if isinstance(data, np.ndarray):
        return data
    
    try:
        cpu_data = cp.asnumpy(data)
        if clear_after:
            clear_gpu_memory()
        return cpu_data
    except Exception as e:
        logging.warning(f"Failed to transfer data to CPU: {str(e)}")
        return data
