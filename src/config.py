"""Configuration settings for the particle segmentation pipeline."""

from dataclasses import dataclass
from typing import Tuple, Optional
from pathlib import Path


@dataclass
class Config:
    # Paths
    input_path: Path
    output_path: Path

    # Data Parameters
    pixel_size: float  # Microns
    particle_size_range: Tuple[float, float]  # (min, max) in microns
    binning_factor: int = 1  # Optional, e.g., 1, 2, 4

    # K-means Parameters
    n_clusters: int = 3
    random_seed: Optional[int] = None
    target_cluster: int = 0  # 0 = darkest

    # Morphological Operations
    kernel_size: Optional[int] = None  # Will be calculated based on binning if not provided

    # Processing
    use_gpu: bool = True

    def __post_init__(self):
        # Convert string paths to Path objects
        self.input_path = Path(self.input_path)
        self.output_path = Path(self.output_path)

        # Validate paths
        if not self.input_path.exists():
            raise ValueError(f"Input path does not exist: {self.input_path}")

        # Create output directory if it doesn't exist
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Validate numerical parameters
        if self.pixel_size <= 0:
            raise ValueError(f"Pixel size must be positive, got {self.pixel_size}")

        if self.particle_size_range[0] <= 0 or self.particle_size_range[1] <= self.particle_size_range[0]:
            raise ValueError(f"Invalid particle size range: {self.particle_size_range}")

        if self.binning_factor < 1:
            raise ValueError(f"Binning factor must be >= 1, got {self.binning_factor}")

        # Calculate kernel size if not provided
        if self.kernel_size is None:
            # Default kernel size is 3 * binning_factor
            self.kernel_size = 3 * self.binning_factor
