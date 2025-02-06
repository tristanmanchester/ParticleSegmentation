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
    use_gpu: bool = True
) -> Union[np.ndarray, cp.ndarray]:
    """Perform K-means clustering on the input data.

    Args:
        data: 3D array of image data
        n_clusters: Number of clusters for K-means
        target_cluster: Index of cluster to select (0 = darkest)
        random_seed: Random seed for reproducibility
        use_gpu: Whether to use GPU acceleration

    Returns:
        Binary mask of selected cluster
    """
    try:
        # Report initial memory usage
        if use_gpu:
            report_gpu_memory("Start K-means")
        
        # Reshape to 2D array for clustering
        original_shape = data.shape
        if isinstance(data, cp.ndarray):
            flattened = to_cpu(data.reshape(-1, 1))
        else:
            flattened = data.reshape(-1, 1)

        # Initialize and fit KMeans
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_seed,
            n_init=10
        )
        
        logging.info("Fitting K-means clustering model...")
        cluster_labels = kmeans.fit_predict(flattened)
        
        # Sort clusters by intensity
        cluster_means = [np.mean(flattened[cluster_labels == i]) for i in range(n_clusters)]
        sorted_indices = np.argsort(cluster_means)
        
        # Create binary mask for target cluster
        mask = (cluster_labels == sorted_indices[target_cluster]).reshape(original_shape)
        
        # Convert to GPU if needed
        if use_gpu:
            mask = to_gpu(mask)
            report_gpu_memory("K-means complete")
        
        return mask
        
    except Exception as e:
        logging.error(f"Error during K-means clustering: {str(e)}")
        raise
    
    finally:
        if use_gpu:
            clear_gpu_memory()
