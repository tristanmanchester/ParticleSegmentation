"""K-means clustering for initial particle segmentation."""

from typing import Union, Tuple
import logging

import numpy as np
import cupy as cp
from sklearn.cluster import KMeans
from tqdm import tqdm

from ..utils.gpu_utils import to_gpu, to_cpu, report_gpu_memory, clear_gpu_memory


def perform_kmeans_clustering(
    data: Union[np.ndarray, cp.ndarray],
    n_clusters: int = 3,
    target_cluster: int = 0,
    random_seed: int = None,
    use_gpu: bool = False  # Deprecated, K-means is CPU-only
) -> np.ndarray:
    """Perform K-means clustering on the input data.

    Args:
        data: 3D array of image data
        n_clusters: Number of clusters for K-means
        target_cluster: Index of cluster to select (0 = darkest)
        random_seed: Random seed for reproducibility
        use_gpu: Deprecated, K-means is CPU-only

    Returns:
        Binary mask of selected cluster (numpy array)
    """
    if use_gpu:
        logging.warning("GPU flag is deprecated for K-means, using CPU implementation")
    
    # Ensure data is on CPU and correct shape
    original_shape = data.shape
    if isinstance(data, cp.ndarray):
        data = cp.asnumpy(data)
    flattened = data.reshape(-1, 1)
    
    # Initialize and fit KMeans
    logging.info(f"Fitting K-means with {n_clusters} clusters...")
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_seed,
        n_init=10
    )
    
    # Fit on data
    kmeans.fit(flattened)
    
    # Get cluster assignments
    labels = kmeans.labels_
    
    # Sort clusters by intensity (ascending)
    cluster_centers = kmeans.cluster_centers_.flatten()
    sorted_indices = np.argsort(cluster_centers)
    
    # Create binary mask for target cluster
    binary_mask = (labels == sorted_indices[target_cluster]).reshape(original_shape)
    
    logging.info(f"Created binary mask with {np.sum(binary_mask)} positive pixels")
    return binary_mask.astype(np.uint8)
