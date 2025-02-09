"""Script to process multiple datasets in batch mode."""

import logging
from pathlib import Path
from typing import List
import sys

from src.config import Config
from main import setup_logging, main

def get_experiment_directories(base_path: Path) -> List[Path]:
    """Find all experiment directories in the sample_1 folder.
    
    Args:
        base_path: Path to the sample_1 directory
        
    Returns:
        List of paths to experiment directories containing volume.tiff
    """
    experiment_dirs = []
    for exp_dir in base_path.iterdir():
        if exp_dir.is_dir():
            volume_file = exp_dir / "trimmed_data" / "volume.tiff"
            if volume_file.exists():
                experiment_dirs.append(exp_dir)
    
    return sorted(experiment_dirs)

def process_dataset(exp_dir: Path, use_gpu: bool = True) -> None:
    """Process a single dataset using the main pipeline.
    
    Args:
        exp_dir: Path to the experiment directory
        use_gpu: Whether to use GPU acceleration
    """
    input_path = exp_dir / "trimmed_data" / "volume.tiff"
    output_path = exp_dir / "segmentation"
    
    # Create Config with paths for this experiment
    config = Config(
        input_path=input_path,
        output_path=output_path,
        pixel_size=0.54,  # microns
        particle_size_range=(5.0, 30.0),  # microns
        binning_factor=1,
        n_clusters=3,
        target_cluster=0,  # 0 = darkest
        use_gpu=use_gpu,
        kernel_size=3
    )
    
    # Override sys.argv to pass the config to main
    sys.argv = [sys.argv[0]]  # Clear any command line arguments
    
    # Process this dataset
    logging.info(f"Processing experiment directory: {exp_dir.name}")
    main(config=config)

def process_all_datasets(base_path: str, use_gpu: bool = True) -> None:
    """Process all datasets in the sample_1 directory.
    
    Args:
        base_path: Path to the sample_1 directory
        use_gpu: Whether to use GPU acceleration
    """
    setup_logging()
    base_path = Path(base_path)
    
    # Get all experiment directories
    experiment_dirs = get_experiment_directories(base_path)
    
    if not experiment_dirs:
        logging.error(f"No valid experiment directories found in {base_path}")
        return
    
    logging.info(f"Found {len(experiment_dirs)} experiment directories to process")
    
    # Process each directory
    for exp_dir in experiment_dirs:
        try:
            process_dataset(exp_dir, use_gpu)
        except Exception as e:
            logging.error(f"Error processing {exp_dir.name}: {str(e)}")
            continue

if __name__ == "__main__":
    # Example usage:
    sample_path = "/dls/science/users/qps56811/data/mg40414-1/sample_1"
    process_all_datasets(sample_path)
