"""K-means clustering for initial particle segmentation."""

from typing import Union
import logging
import numpy as np
import cupy as cp
from sklearn.cluster import MiniBatchKMeans

def perform_kmeans_clustering(
    data: Union[np.ndarray, cp.ndarray],
    n_clusters: int = 3,
    target_cluster: int = 0,
    random_seed: int = None,
    batch_size: int = 4096
) -> np.ndarray:
    """Perform K-means clustering on the input data.
    
    Args:
        data: 3D array of image data
        n_clusters: Number of clusters for K-means
        target_cluster: Index of cluster to select (0 = darkest)
        random_seed: Random seed for reproducibility
        batch_size: Batch size for MiniBatchKMeans
    
    Returns:
        Binary mask of selected cluster (numpy array)
    """
    try:
        # Ensure data is on CPU
        if isinstance(data, cp.ndarray):
            data = cp.asnumpy(data)
        
        logging.info(f"Processing {data.nbytes / (1024*1024):.1f}MB of data")
        
        # Reshape to 2D for clustering
        data_flat = data.reshape(-1, 1)
        
        # Initialize and run MiniBatchKMeans
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=random_seed,
            batch_size=batch_size,
            n_init=3
        )
        
        # Fit and get labels
        labels = kmeans.fit_predict(data_flat)
        
        # Sort clusters by intensity (ascending)
        centers = kmeans.cluster_centers_.flatten()
        sorted_indices = np.argsort(centers)
        binary_mask = (labels == sorted_indices[target_cluster]).reshape(data.shape)
        
        logging.info(f"Created binary mask with {np.sum(binary_mask)} positive pixels")
        return binary_mask.astype(np.uint8)
        
    except Exception as e:
        logging.error(f"K-means clustering failed: {str(e)}")
        raise
