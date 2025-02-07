# X-ray Tomography Particle Segmentation Pipeline

A Python-based pipeline for segmenting NMC cathode particles from X-ray micro-tomography data of solid-state batteries. The pipeline processes TIFF stacks using K-means clustering followed by watershed segmentation, with GPU acceleration where possible.

<p align="center">
  <img src="https://github.com/tristanmanchester/ParticleSegmentation/blob/main/example.png" width="700">
</p>

## Features
- GPU-accelerated processing using CUDA
- K-means clustering for initial segmentation
- Watershed segmentation for particle separation
- Comprehensive visualisation and analysis tools
- Support for large datasets

## Installation
This project uses Pixi for dependency management. To get started:

1. Make sure you have Pixi installed
2. Clone this repository
3. Run `pixi install` in the project directory

## Usage
Modify the config in `main.py`:
```python
    # Configuration
    input_path = Path("path/to/your/tiffs")
    output_path = Path("output")
    use_gpu = True
    
    config = Config(
        input_path=input_path,
        output_path=output_path,
        pixel_size=0.54,  # microns
        particle_size_range=(5.0, 30.0),  # microns
        binning_factor=8,
        n_clusters=3,
        target_cluster=0,  # 0 = darkest
        use_gpu=use_gpu,
        kernel_size=1
    )
```


