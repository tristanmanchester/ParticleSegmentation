"""Main entry point for the particle segmentation pipeline."""

import argparse
from pathlib import Path

import numpy as np
import cupy as cp
from tqdm import tqdm

from src.config import Config
from src.io import load_tiff_stack, save_tiff_stack
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


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Segment NMC cathode particles from X-ray tomography data"
    )
    
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to input TIFF stack or directory"
    )
    
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to output directory"
    )
    
    parser.add_argument(
        "--pixel_size",
        type=float,
        required=True,
        help="Pixel size in microns"
    )
    
    parser.add_argument(
        "--min_particle_size",
        type=float,
        default=1.0,
        help="Minimum particle size in microns"
    )
    
    parser.add_argument(
        "--max_particle_size",
        type=float,
        default=50.0,
        help="Maximum particle size in microns"
    )
    
    parser.add_argument(
        "--binning_factor",
        type=int,
        default=1,
        help="Binning factor for downsampling"
    )
    
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=3,
        help="Number of clusters for K-means"
    )
    
    parser.add_argument(
        "--target_cluster",
        type=int,
        default=0,
        help="Target cluster index (0 = darkest)"
    )
    
    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--no_gpu",
        action="store_true",
        help="Disable GPU acceleration"
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_args()
    
    # Create configuration
    config = Config(
        input_path=args.input_path,
        output_path=args.output_path,
        pixel_size=args.pixel_size,
        particle_size_range=(args.min_particle_size, args.max_particle_size),
        binning_factor=args.binning_factor,
        n_clusters=args.n_clusters,
        target_cluster=args.target_cluster,
        random_seed=args.random_seed,
        use_gpu=not args.no_gpu
    )
    
    try:
        # Load data
        print("\nLoading TIFF stack...")
        data = load_tiff_stack(
            config.input_path,
            use_gpu=config.use_gpu,
            validate=True
        )
        
        # Perform K-means clustering
        print("\nPerforming K-means clustering...")
        binary_mask = perform_kmeans_clustering(
            data,
            n_clusters=config.n_clusters,
            target_cluster=config.target_cluster,
            random_seed=config.random_seed,
            use_gpu=config.use_gpu
        )
        
        # Apply morphological operations
        print("\nApplying morphological operations...")
        processed_mask = apply_morphological_operations(
            binary_mask,
            kernel_size=config.kernel_size,
            use_gpu=config.use_gpu
        )
        
        # Calculate minimum distance for watershed
        min_distance = calculate_min_distance(
            config.particle_size_range,
            config.pixel_size
        )
        
        # Perform watershed segmentation
        print("\nPerforming watershed segmentation...")
        labels = perform_watershed_segmentation(
            processed_mask,
            min_distance=min_distance,
            pixel_size=config.pixel_size,
            use_gpu=config.use_gpu
        )
        
        # Save intermediate results
        print("\nSaving results...")
        save_tiff_stack(binary_mask, config.output_path, "binary_mask")
        save_tiff_stack(processed_mask, config.output_path, "processed_mask")
        save_tiff_stack(labels, config.output_path, "particle_labels")
        
        # Calculate and save statistics
        print("\nCalculating particle statistics...")
        stats = calculate_particle_statistics(labels, config.pixel_size)
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        plot_orthogonal_views(
            data, binary_mask, labels,
            config.output_path
        )
        plot_size_distribution(
            stats['equivalent_diameters'],
            config.pixel_size,
            config.output_path
        )
        
        print("\nProcessing complete! Results saved to:", config.output_path)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise


if __name__ == "__main__":
    main()
