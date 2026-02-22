# PyPlayground 🎮

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](./tests)

A professional playground for AI, data science, and visualization experiments.

## Features

- **3D Pose Visualization**: Interactive 3D cube rotation and translation with matplotlib
- **Animation Support**: Export to MP4/WebM using FFmpeg
- **CLI Tools**: Command-line interface for quick demos and video generation
- **Jupyter Integration**: Ready-to-use notebooks for experimentation
- **Tested**: Full pytest coverage

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/jclosure/py_playground.git
cd py_playground

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

### CLI Usage

```bash
# Show package info
pyplayground info

# Run interactive demo
pyplayground demo

# Generate a video
pyplayground video -o outputs/pose.mp4 --frames 120 --fps 30
```

### Python API

```python
from py_playground import PoseVisualizer
import numpy as np

# Create visualizer
viz = PoseVisualizer()

# Draw a single frame
viz.draw_frame(
    rotation=(0.5, 0.3, 0.2),
    translation=np.array([1, 0.5, 0])
)
viz.show()

# Save as video
viz.save_video("output.mp4", frames=60, fps=30)
```

### Jupyter Notebook

```bash
jupyter notebook notebooks/pose_visualization.ipynb
```

## Project Structure

```
py_playground/
├── src/py_playground/        # Main package source
│   ├── __init__.py
│   ├── cli/                  # Command-line interface
│   │   ├── __init__.py
│   │   └── main.py
│   └── visualization/        # Visualization tools
│       ├── __init__.py
│       └── pose.py
├── tests/                    # Test suite
│   ├── __init__.py
│   └── test_pose.py
├── notebooks/                # Jupyter notebooks
│   └── pose_visualization.ipynb
├── data/                     # Data directory (gitignored)
│   ├── raw/
│   └── processed/
├── outputs/                  # Generated outputs (gitignored)
├── pyproject.toml           # Project configuration
├── README.md
└── .gitignore
```

## Development

### Running Tests

```bash
pytest
```

### Code Quality

```bash
# Format code
black src tests

# Lint
ruff src tests

# Type check
mypy src
```

## Requirements

- Python 3.9+
- FFmpeg (for video export)

## License

MIT License - See [LICENSE](./LICENSE) for details.
