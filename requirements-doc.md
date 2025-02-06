# X-ray Tomography Particle Segmentation Pipeline

## Project Overview
A Python-based pipeline for segmenting NMC cathode particles from X-ray micro-tomography data of solid-state batteries. The pipeline processes TIFF stacks using K-means clustering followed by watershed segmentation, with GPU acceleration where possible.

## System Requirements
- Python 3.8+
- CUDA-capable GPU (48GB VRAM available)
- Minimum 64GB RAM (system optimised for 1.5TB RAM)

## Dependencies
Primary dependencies to be managed via PDM:

### Core Processing
- numpy
- cupy (GPU-accelerated array operations)
- scikit-learn (or cuML if compatible)
- scipy
- tifffile (TIFF handling)

### Visualization
- matplotlib
- tqdm (progress bars)

## Pipeline Stages

### 1. Data Loading
- Input: Stack of grayscale TIFF images
- Validation:
  - Verify grayscale format
  - Check for binary mask (ones and zeros only)
  - Validate stack dimensions
  - Validate bit depth
- Output: 3D numpy/cupy array

### 2. Pre-processing
- Optional binning (2x, 4x, etc.)
- Automatic detection of binary masks
  - If binary mask detected:
    - Prompt user for morphological operations preference
    - Skip to watershed segmentation if chosen

### 3. K-means Clustering
- GPU-accelerated where possible
- Parameters:
  - Number of clusters (default: 3)
  - Random seed
  - Target cluster selection (0 = darkest)
- Output: Binary mask of selected cluster

### 4. Morphological Operations
- Erosion followed by dilation
- Ball kernel (size scales with binning factor)
- Applied to binary mask
- Parameters stored in pixels

### 5. Watershed Segmentation
- Input: Processed binary mask
- Parameters automatically calculated from:
  - Pixel size (μm)
  - Particle size range (μm)
- Features:
  - 3D Euclidean distance transform
  - Marker generation
  - Watershed segmentation
- Output: Label map (uint16)

### 6. Analysis and Visualization
- Particle statistics:
  - Equivalent spherical diameter distribution
  - Volume distribution
  - Count
- Visualization:
  - 3x3 grid of orthogonal slices (middle indices)
  - Columns: Raw data, Binary mask, Watershed labels
  - Rows: XY, XZ, YZ views
  - Particle size distribution histogram (linear scale)

### 7. Output
Save intermediates in output directory:
- K-means cluster results
- Binary mask pre-morphological operations
- Binary mask post-morphological operations
- Final watershed segmentation
- Statistics and visualizations

## Configuration Parameters

### Main Configuration (main.py)
```python
config = {
    # Paths
    'input_path': str,  # Directory containing TIFF stack
    'output_path': str, # Output directory

    # Data Parameters
    'pixel_size': float,  # Microns
    'particle_size_range': tuple,  # (min, max) in microns
    'binning_factor': int,  # Optional, e.g., 1, 2, 4

    # K-means Parameters
    'n_clusters': int,  # Default: 3
    'random_seed': int,  # Optional
    'target_cluster': int,  # 0 = darkest

    # Morphological Operations
    'kernel_size': int,  # Pixels (scales with binning)

    # Processing
    'use_gpu': bool,  # Enable/disable GPU acceleration
}
```

## Performance Considerations
- GPU acceleration prioritised for:
  - K-means clustering
  - Array operations
  - Distance transforms where possible
- Full 3D volume processing
- Loading full dataset into RAM
- Progress tracking and user feedback via tqdm

## Error Handling
- Input validation
- Graceful failure for GPU operations
- Clear error messages
- Memory usage warnings

## Output Format
- Segmented particles: TIFF stack (uint16)
- Visualizations: PNG
- Statistics: Text file
- All intermediate steps preserved

## Code Structure
```
project/
├── pyproject.toml
├── README.md
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── io/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   └── saver.py
│   ├── processing/
│   │   ├── __init__.py
│   │   ├── kmeans.py
│   │   ├── morphology.py
│   │   └── watershed.py
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── plotting.py
│   │   └── stats.py
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
├── tests/
│   └── __init__.py
└── main.py
```