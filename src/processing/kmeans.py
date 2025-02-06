"""K-means clustering for initial particle segmentation."""

from typing import Union, Tuple

import numpy as np
import cupy as cp
from sklearn.cluster import KMeans
from cuml.cluster import KMeans as cuKMeans
from tqdm import tqdm


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

    Raises:
        RuntimeError: If GPU clustering fails
    """
    # Reshape to 2D array for clustering
    original_shape = data.shape
    flattened = data.reshape(-1, 1)
    
    try:
        if use_gpu and isinstance(data, cp.ndarray):
            # Use cuML for GPU acceleration
            kmeans = cuKMeans(
                n_clusters=n_clusters,
                random_state=random_seed,
                output_type='numpy'
            )
            labels = kmeans.fit_predict(flattened)
            centers = kmeans.cluster_centers_
        else:
            # Use scikit-learn for CPU
            if isinstance(data, cp.ndarray):
                flattened = cp.asnumpy(flattened)
            
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=random_seed,
                n_init=10
            )
            labels = kmeans.fit_predict(flattened)
            centers = kmeans.cluster_centers_
        
        # Sort clusters by intensity
        sorted_indices = np.argsort(centers.flatten())
        label_map = np.zeros_like(sorted_indices)
        label_map[sorted_indices] = np.arange(len(sorted_indices))
        
        # Map labels to sorted indices
        labels = label_map[labels]
        
        # Create binary mask of target cluster
        mask = (labels == target_cluster).reshape(original_shape)
        
        if use_gpu and not isinstance(mask, cp.ndarray):
            mask = cp.asarray(mask)
        
        return mask
    
    except Exception as e:
        raise RuntimeError(f"K-means clustering failed: {str(e)}")
