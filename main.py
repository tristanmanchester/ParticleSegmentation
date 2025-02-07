"""Main entry point for the particle segmentation pipeline."""

import logging
from pathlib import Path

import numpy as np
import cupy as cp
from tqdm import tqdm

from src.config import Config
from src.io import load_tiff_stack, save_tiff_stack
from src.io.loader import is_binary_mask
from src.processing import (
    perform_kmeans_clustering,
    apply_morphological_operations,
    perform_watershed_segmentation,
    calculate_min_distance
)
from src.visualization import (
    plot_orthogonal_views,
    plot_size_distribution,
    calculate_particle_statistics
)
from src.utils.gpu_utils import report_gpu_memory, clear_gpu_memory, to_gpu, to_cpu
from src.utils.timing import Timer, timed_stage


def setup_logging() -> None:
    """Configure logging settings."""
    format_str = '%(asctime)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=logging.INFO,
        format=format_str,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('particle_segmentation.log')
        ]
    )


def main():
    """Main execution function."""
    # Configure logging
    setup_logging()
    
    # Configuration
    input_path = Path("data/subvolumes")
    output_path = Path("output")
    use_gpu = True
    
    config = Config(
        input_path=input_path,
        output_path=output_path,
        pixel_size=0.54,  # microns
        particle_size_range=(5.0, 30.0),  # microns
        binning_factor=2,
        n_clusters=3,
        target_cluster=0,  # 0 = darkest
        use_gpu=use_gpu,
        kernel_size=1
    )
    
    # Create output directory
    config.output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize timer for performance tracking
    timer = Timer()
    
    try:
        timer.start("Total Pipeline")
        
        # Load data
        with timed_stage(timer, "Data Loading"):
            logging.info(f"Loading data from {config.input_path}")
            data = load_tiff_stack(
                config.input_path,
                use_gpu=config.use_gpu,
                validate=True,
                binning_factor=config.binning_factor
            )
            if data is None:
                raise ValueError("No data loaded")
            logging.info(f"Loaded data shape: {data.shape}")
        
        # K-means clustering
        with timed_stage(timer, "K-means Clustering"):
            logging.info("Performing K-means clustering...")
            binary_mask = perform_kmeans_clustering(
                data,
                n_clusters=config.n_clusters,
                target_cluster=config.target_cluster
            )
        
        # Morphological operations
        with timed_stage(timer, "Morphological Operations"):
            logging.info("Applying morphological operations")
            if config.use_gpu:
                binary_mask = to_gpu(binary_mask)
            # Calculate kernel size based on binning factor if not provided
            kernel_size = config.kernel_size if config.kernel_size is not None else 3 * config.binning_factor
            binary_mask = apply_morphological_operations(
                binary_mask,
                kernel_size=kernel_size,
                use_gpu=config.use_gpu
            )
        
        # Calculate minimum distance for watershed
        min_distance = calculate_min_distance(
            config.particle_size_range,
            config.pixel_size
        )
        
        # Watershed segmentation
        with timed_stage(timer, "Watershed Segmentation"):
            logging.info("Performing watershed segmentation")
            labels = perform_watershed_segmentation(
                binary_mask,
                min_distance=min_distance,
                pixel_size=config.pixel_size,
                use_gpu=config.use_gpu,
                timer=timer
            )
            
            # Move data back to CPU after watershed
            if config.use_gpu:
                binary_mask = to_cpu(binary_mask)
                labels = to_cpu(labels)
                clear_gpu_memory()
        
        # Save results
        with timed_stage(timer, "Saving Results"):
            logging.info("Saving results")
            # Save segmented particles
            save_tiff_stack(
                data=labels,
                output_path=config.output_path,
                filename="segmented_particles"
            )
            
            # Generate and save visualizations
            plot_orthogonal_views(
                raw_data=data,
                binary_mask=binary_mask,
                labels=labels,
                output_path=str(config.output_path / "orthogonal_views.png")
            )
            
            # Plot size distribution using particle statistics
            stats = calculate_particle_statistics(labels, config.pixel_size)
            plot_size_distribution(
                diameters=stats['equivalent_diameters'],
                pixel_size=config.pixel_size,
                output_path=str(config.output_path / "size_distribution.png")
            )
            
            # Save statistics
            import pandas as pd
            # Convert arrays to lists for better CSV formatting
            stats_df = pd.DataFrame({
                'equivalent_diameters': stats['equivalent_diameters'].tolist(),
                'volumes': stats['volumes'].tolist()
            })
            # Add summary statistics as a separate CSV
            summary_stats = {
                'count': len(stats['equivalent_diameters']),
                'mean_diameter': stats['mean_diameter'],
                'std_diameter': stats['std_diameter'],
                'min_diameter': stats['min_diameter'],
                'max_diameter': stats['max_diameter']
            }
            pd.DataFrame([summary_stats]).to_csv(
                str(config.output_path / "summary_statistics.csv"),
                index=False
            )
            # Save detailed particle measurements
            stats_df.to_csv(
                str(config.output_path / "particle_measurements.csv"),
                index=False
            )
        
        timer.stop("Total Pipeline")
        logging.info("Pipeline completed successfully")
        logging.info(f"\nPerformance Summary:\n{timer.get_summary()}")
    
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise
    
    finally:
        if config.use_gpu:
            clear_gpu_memory()


if __name__ == "__main__":
    main()
