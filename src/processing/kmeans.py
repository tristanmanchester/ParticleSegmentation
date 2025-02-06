"""K-means clustering for initial particle segmentation."""

from typing import Union, Tuple, List
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import os
import platform

import numpy as np
import cupy as cp
from sklearn.cluster import KMeans, MiniBatchKMeans
from tqdm import tqdm


def _process_chunk(args: Tuple) -> Tuple[np.ndarray, np.ndarray, slice]:
    """Process a single chunk of data with K-means.
    
    Args:
        args: Tuple containing:
            - chunk: 3D array of image data
            - n_clusters: Number of clusters
            - random_seed: Random seed for reproducibility
            - chunk_slice: Slice object for this chunk
    
    Returns:
        Tuple of (binary mask, cluster centers, chunk slice)
    """
    # Limit threads for BLAS and OpenMP within each process
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    
    chunk, n_clusters, random_seed, chunk_slice = args
    
    # Reshape to 2D
    chunk_flat = chunk.reshape(-1, 1)
    
    # Calculate safe batch size (minimum 6144 on Windows with MKL to prevent memory leak)
    min_batch_size = 6144 if platform.system() == 'Windows' else 1000
    batch_size = max(min_batch_size, min(len(chunk_flat) // 10, 10000))
    
    # Use MiniBatchKMeans for better memory efficiency
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_seed,
        batch_size=batch_size,
        n_init=3
    )
    
    # Fit and get labels
    labels = kmeans.fit_predict(chunk_flat)
    centers = kmeans.cluster_centers_.flatten()
    
    # Sort clusters by intensity (ascending)
    sorted_indices = np.argsort(centers)
    binary_mask = (labels == sorted_indices[0]).reshape(chunk.shape)
    
    return binary_mask.astype(np.uint8), centers, chunk_slice


def split_volume(shape: Tuple[int, ...], n_chunks: int) -> List[slice]:
    """Split a 3D volume into roughly equal chunks along the Z axis.
    
    Args:
        shape: Shape of the volume (z, y, x)
        n_chunks: Number of chunks to create
    
    Returns:
        List of slice objects for each chunk
    """
    z_size = shape[0]
    chunk_size = max(1, z_size // n_chunks)
    
    slices = []
    for start in range(0, z_size, chunk_size):
        end = min(start + chunk_size, z_size)
        slices.append(slice(start, end))
    
    return slices


def perform_kmeans_clustering(
    data: Union[np.ndarray, cp.ndarray],
    n_clusters: int = 3,
    target_cluster: int = 0,
    random_seed: int = None,
    n_jobs: int = -1,
    use_gpu: bool = False  # Deprecated
) -> np.ndarray:
    """Perform parallel K-means clustering on the input data.

    Args:
        data: 3D array of image data
        n_clusters: Number of clusters for K-means
        target_cluster: Index of cluster to select (0 = darkest)
        random_seed: Random seed for reproducibility
        n_jobs: Number of parallel processes (-1 for all cores)
        use_gpu: Deprecated parameter

    Returns:
        Binary mask of selected cluster (numpy array)
    """
    if use_gpu:
        logging.warning("GPU flag is deprecated for K-means, using CPU implementation")
    
    # Ensure data is on CPU
    if isinstance(data, cp.ndarray):
        data = cp.asnumpy(data)
    
    # Determine optimal number of processes based on system
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    
    # Limit maximum number of processes to avoid resource exhaustion
    max_processes = 16  # Reasonable limit for most systems
    n_jobs = min(n_jobs, max_processes)
    
    # Calculate number of chunks based on data size and available processes
    # Aim for chunks of reasonable size (about 100MB each)
    chunk_size_target = 100 * 1024 * 1024  # 100MB in bytes
    total_size = data.nbytes
    n_chunks = min(max(4, total_size // chunk_size_target), n_jobs * 2)
    
    chunk_slices = split_volume(data.shape, n_chunks)
    logging.info(f"Processing data in {len(chunk_slices)} chunks using {n_jobs} processes...")
    
    # Process chunks in parallel
    binary_mask = np.zeros_like(data, dtype=np.uint8)
    all_centers = []
    
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        # Submit all chunks for processing
        futures = []
        for chunk_slice in chunk_slices:
            chunk = data[chunk_slice]
            futures.append(
                executor.submit(
                    _process_chunk,
                    (chunk, n_clusters, random_seed, chunk_slice)
                )
            )
        
        # Process results as they complete
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing chunks"):
            chunk_mask, centers, chunk_slice = future.result()
            binary_mask[chunk_slice] = chunk_mask
            all_centers.append(centers)
    
    # Verify consistency of cluster assignments across chunks
    mean_centers = np.mean(all_centers, axis=0)
    if np.std(all_centers, axis=0).mean() > 0.1 * np.mean(mean_centers):
        logging.warning(
            "High variance in cluster centers across chunks. "
            "Consider reducing the number of chunks or using different clustering parameters."
        )
    
    logging.info(f"Created binary mask with {np.sum(binary_mask)} positive pixels")
    return binary_mask
