# X-ray Tomography Particle Segmentation Pipeline

A Python-based pipeline for segmenting NMC cathode particles from X-ray micro-tomography data of solid-state batteries. The pipeline processes TIFF stacks using K-means clustering followed by watershed segmentation, with GPU acceleration where possible.

## Features
- GPU-accelerated processing using CUDA
- K-means clustering for initial segmentation
- Watershed segmentation for particle separation
- Comprehensive visualization and analysis tools
- Support for large datasets (optimized for systems with 1.5TB RAM)

## Requirements
- Python 3.8+
- CUDA-capable GPU (48GB VRAM)
- 64GB+ RAM (optimized for 1.5TB)

## Installation
This project uses Pixi for dependency management. To get started:

1. Make sure you have Pixi installed
2. Clone this repository
3. Run `pixi install` in the project directory

## Usage
```bash
python main.py --input_path /path/to/tiff/stack --output_path /path/to/output --pixel_size 0.5
```

For more detailed configuration options, see the configuration section in the documentation.

## Project Structure
```
project/
├── src/
│   ├── io/          # Data loading and saving
│   ├── processing/  # Core processing algorithms
│   ├── visualization/ # Plotting and statistics
│   └── utils/       # Helper functions
├── tests/           # Unit tests
└── main.py         # Main entry point
```

## License
[Add your license here]
