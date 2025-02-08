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
    batch_size: int = 4096,
    max_samples: int = 1_000_000,  # Maximum number of samples to use for initialization
    z_subsample_factor: int = None  # If set, use every nth slice for finding clusters
) -> np.ndarray:
    """Perform K-means clustering on the input data.
    
    Args:
        data: 3D array of image data
        n_clusters: Number of clusters for K-means
        target_cluster: Index of cluster to select (0 = darkest)
        random_seed: Random seed for reproducibility
        batch_size: Batch size for MiniBatchKMeans
        max_samples: Maximum number of samples to use for initialization
        z_subsample_factor: If set, use every nth slice for finding clusters
    
    Returns:
        Binary mask of selected cluster (numpy array)
    """
    try:
        # Ensure data is on CPU
        if isinstance(data, cp.ndarray):
            data = cp.asnumpy(data)
        
        logging.info(f"Processing {data.nbytes / (1024*1024):.1f}MB of data")
        
        # Set environment variable OMP_NUM_THREADS=16
        import os
        os.environ["OMP_NUM_THREADS"] = "16"
        
        # Initialize MiniBatchKMeans
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=random_seed,
            batch_size=batch_size,
            n_init=3,
            max_iter=100,
            max_no_improvement=10
        )
        
        # Determine if we should use Z subsampling
        if z_subsample_factor is None and data.shape[0] > 100:
            # Automatically determine subsample factor for large volumes
            z_subsample_factor = max(2, data.shape[0] // 100)
            logging.info(f"Using auto Z subsampling factor of {z_subsample_factor}")
        
        # First find cluster centers using subsampled data
        if z_subsample_factor and z_subsample_factor > 1:
            logging.info(f"Finding cluster centers using every {z_subsample_factor}th slice...")
            subsampled_data = data[::z_subsample_factor]
            subsampled_flat = subsampled_data.reshape(-1, 1)
            n_samples = subsampled_flat.shape[0]
            
            # Use subset sampling if still too large
            if n_samples > max_samples:
                rng = np.random.RandomState(random_seed)
                subset_idx = rng.choice(n_samples, size=max_samples, replace=False).astype(np.int64)
                subset_data = subsampled_flat[subset_idx]
                kmeans.fit(subset_data)
            else:
                kmeans.fit(subsampled_flat)
            
            logging.info("Cluster centers found. Applying to full dataset...")
        
        # Apply clustering to full dataset
        data_flat = data.reshape(-1, 1)
        n_samples = data_flat.shape[0]
        
        # Process full dataset in batches
        labels = np.empty(n_samples, dtype=np.int32)
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            labels[i:end_idx] = kmeans.predict(data_flat[i:end_idx])
            if i % (100 * batch_size) == 0:  # Reduced logging frequency
                logging.info(f"Processed {i:,} / {n_samples:,} samples")
        
        # Sort clusters by intensity (ascending)
        centers = kmeans.cluster_centers_.flatten()
        sorted_indices = np.argsort(centers)
        binary_mask = (labels == sorted_indices[target_cluster]).reshape(data.shape)
        
        logging.info(f"Created binary mask with {np.sum(binary_mask)} positive pixels")
        return binary_mask.astype(np.uint8)
        
    except Exception as e:
        logging.error(f"K-means clustering failed: {str(e)}")
        raise
