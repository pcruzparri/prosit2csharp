# Prosit2CSharp

This project converts Prosit TensorFlow models to ONNX format for use in C# applications.

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

### Prerequisites

- Python 3.11
- uv package manager

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/pcruzparri/prosit2csharp.git
   cd prosit2csharp
   ```

2. Install dependencies:
   ```bash
   uv sync
   ```

   **Note**: If you encounter TensorFlow installation issues, try force-reinstalling:
   ```bash
   uv pip install tensorflow==2.12.1 --force-reinstall
   ```

3. Run the conversion script:
   ```bash
   uv run ./tfmodel2onnx_compatible.py
   ```

## Troubleshooting

### TensorFlow Import Issues

If you get a `ModuleNotFoundError: No module named 'tensorflow'`, try:

1. Force reinstall TensorFlow:
   ```bash
   uv pip install tensorflow==2.12.1 --force-reinstall
   ```

2. Verify installation:
   ```bash
   uv run python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
   ```

### Windows-Specific Notes

- This project is configured to work with Windows and will automatically install `tensorflow-intel` for better performance.
- Ensure your Python version is exactly 3.11.x for compatibility.

## Files

- `tfmodel2onnx_compatible.py` - Main conversion script that creates ONNX-compatible models
- `tfmodel2onnx.py` - Original conversion script
- `models/prosit/` - Contains the original TensorFlow models
- `csharp/` - C# project for using the converted ONNX models