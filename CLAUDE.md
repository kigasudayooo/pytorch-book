# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PyTorch book repository containing educational materials organized by chapters. The project includes Jupyter notebooks, Python scripts, and web applications demonstrating various PyTorch concepts and AI/ML implementations.

## Development Environment

The project uses `uv` as the package manager with Python dependencies defined in `pyproject.toml`. PyTorch is configured to use CUDA 11.8 from the PyTorch index.

### Common Commands

**Package Management:**
```bash
uv sync                    # Install all dependencies
uv add <package>          # Add new dependency
uv run <command>          # Run command in virtual environment
```

**Development Tools:**
```bash
uv run black .            # Format Python code
uv run ruff check .       # Lint Python code
uv run ruff check . --fix # Auto-fix linting issues
uv run mypy .             # Type checking
uv run pytest            # Run tests
```

**Jupyter Notebooks:**
```bash
uv run jupyter notebook  # Start Jupyter server
uv run jupyter lab       # Start JupyterLab
```

## Code Architecture

### Chapter Organization
- **Contents Structure**: All educational content is in the `contents/` directory, organized by chapters (Chapter01-Chapter20)
- **Chapter Types**:
  - **Jupyter Notebooks**: Primary learning materials (`.ipynb` files)
  - **Python Scripts**: Standalone examples and utilities
  - **Web Applications**: Node.js/Express servers for AI integration demos
  - **Model Deployment**: TorchServe configurations and Flask applications

### Key Components

**Chapter 13 - Model Deployment**:
- Flask application with PyTorch model serving
- TorchServe setup with model archiving and serving commands
- Model handler and configuration files in `torchserve/` directory

**Chapter 17-18 - Web Applications**:
- Node.js Express servers for AI-powered applications
- Package.json configurations for novel analysis applications
- Dependencies: express, multer, node-fetch

**Chapter 20 - Advanced Topics**:
- Image generation and LoRA implementations
- Custom datasets and training scripts

### TorchServe Workflow
When working with Chapter 13 TorchServe examples:
1. Archive model: Use commands from `contents/Chapter13/torchserve/commands.txt`
2. Start server: TorchServe with model store and config
3. Test inference: Use provided test scripts

### Node.js Applications
For Chapters 17-18 web applications:
```bash
cd contents/Chapter17/ch17server  # or Chapter18/ch18server
npm install
npm start
```

## File Structure Patterns
- Jupyter notebooks follow naming: `PyTorch_Chapter_X.ipynb` or `ChapterX.ipynb`
- Multiple notebooks per chapter indicated by suffixes: `_Notebook_2`, `_2`, etc.
- Python scripts use descriptive names related to chapter content
- Model files stored as `.pth` files alongside training scripts
- Configuration files in dedicated `config/` directories where applicable