"""K-means clustering for initial particle segmentation."""

from typing import Union, Tuple, List
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import platform
import numpy as np
import cupy as cp
from sklearn.cluster import KMeans, MiniBatchKMeans
from tqdm import tqdm

# Pre-initialize process-level variables
_kmeans = None
_batch_size = None

def init_worker(n_clusters: int, random_seed: int):
    """Initialize worker process with a MiniBatchKMeans instance."""
    global _kmeans, _batch_size
    
    # Calculate safe batch size for Windows to prevent memory leak
    if platform.system() == 'Windows':
        _batch_size = 4096  # Reduced from 6144 to be safer
    else:
        _batch_size = 1000
    
    # Initialize MiniBatchKMeans in the worker process
    _kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_seed,
        batch_size=_batch_size,
        n_init=3
    )

def _process_chunk(chunk_with_slice: Tuple[np.ndarray, slice]) -> Tuple[np.ndarray, np.ndarray, slice]:
    """Process a single chunk of data with K-means.
    
    Args:
        chunk_with_slice: Tuple containing:
            - chunk: 3D array of image data
            - chunk_slice: Slice object for this chunk
    
    Returns:
        Tuple of (binary mask, cluster centers, chunk slice)
    """
    global _kmeans
    try:
        chunk, chunk_slice = chunk_with_slice
        
        # Reshape to 2D
        chunk_flat = chunk.reshape(-1, 1)
        
        # Process in sub-batches if chunk is very large
        if chunk_flat.shape[0] > 1_000_000:  # 1M points
            n_sub_batches = chunk_flat.shape[0] // 500_000 + 1
            labels = np.zeros(chunk_flat.shape[0], dtype=np.int32)
            for i in range(n_sub_batches):
                start = i * 500_000
                end = min((i + 1) * 500_000, chunk_flat.shape[0])
                labels[start:end] = _kmeans.fit_predict(chunk_flat[start:end])
        else:
            # Fit and get labels using pre-initialized kmeans
            labels = _kmeans.fit_predict(chunk_flat)
        
        centers = _kmeans.cluster_centers_.flatten()
        
        # Sort clusters by intensity (ascending)
        sorted_indices = np.argsort(centers)
        binary_mask = (labels == sorted_indices[0]).reshape(chunk.shape)
        
        return binary_mask.astype(np.uint8), centers, chunk_slice
    
    except Exception as e:
        logging.error(f"Error processing chunk: {str(e)}")
        raise

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
    try:
        if use_gpu:
            logging.warning("GPU flag is deprecated for K-means, using CPU implementation")
        
        # Ensure data is on CPU
        if isinstance(data, cp.ndarray):
            data = cp.asnumpy(data)
        
        # Determine number of processes
        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()
        
        # Calculate optimal number of chunks based on CPU cores and data size
        # Reduced target chunk size for Windows
        if platform.system() == 'Windows':
            chunk_size_target = 50 * 1024 * 1024  # 50MB target chunk size for Windows
        else:
            chunk_size_target = 100 * 1024 * 1024  # 100MB for other platforms
        
        n_chunks = min(max(n_jobs * 2, data.nbytes // chunk_size_target), data.shape[0])
        chunk_slices = split_volume(data.shape, n_chunks)
        
        logging.info(f"Processing {data.nbytes / (1024*1024):.1f}MB of data")
        logging.info(f"Processing in {len(chunk_slices)} chunks using {n_jobs} processes...")
        logging.info(f"Average chunk size: {data.nbytes / len(chunk_slices) / (1024*1024):.1f}MB")
        
        # Process chunks in parallel with pre-initialized workers
        binary_mask = np.zeros_like(data, dtype=np.uint8)
        all_centers = []
        
        # Prepare chunks with their slices
        chunks_with_slices = [(data[chunk_slice], chunk_slice) for chunk_slice in chunk_slices]
        
        with ProcessPoolExecutor(
            max_workers=n_jobs,
            initializer=init_worker,
            initargs=(n_clusters, random_seed)
        ) as executor:
            # Submit all chunks for processing
            futures = []
            for chunk_with_slice in chunks_with_slices:
                future = executor.submit(_process_chunk, chunk_with_slice)
                futures.append(future)
            
            # Process results as they complete
            failed_chunks = 0
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing chunks"):
                try:
                    chunk_mask, centers, chunk_slice = future.result()
                    binary_mask[chunk_slice] = chunk_mask
                    all_centers.append(centers)
                except Exception as e:
                    failed_chunks += 1
                    logging.error(f"Chunk processing failed: {str(e)}")
                    if failed_chunks > len(futures) // 3:  # If more than 1/3 of chunks fail
                        raise RuntimeError("Too many chunk processing failures")
        
        if failed_chunks > 0:
            logging.warning(f"{failed_chunks} chunks failed to process")
        
        # Verify consistency of cluster assignments across chunks
        mean_centers = np.mean(all_centers, axis=0)
        if np.std(all_centers, axis=0).mean() > 0.1 * np.mean(mean_centers):
            logging.warning(
                "High variance in cluster centers across chunks. "
                "Consider reducing the number of chunks or using different clustering parameters."
            )
        
        logging.info(f"Created binary mask with {np.sum(binary_mask)} positive pixels")
        return binary_mask
        
    except Exception as e:
        logging.error(f"K-means clustering failed: {str(e)}")
        raise
