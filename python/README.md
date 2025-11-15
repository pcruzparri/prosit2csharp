# Prosit2CSharp

This project converts Prosit TensorFlow models to ONNX format for use in C# applications. It provides both Python tools for model conversion and a C# implementation for spectral prediction using the converted ONNX models.

## Project Structure

```
prosit2csharp/
├── python/                     # Python tools for model conversion
│   ├── tfmodel2onnx_compatible.py  # Main conversion script
│   ├── tfmodel2onnx.py            # Original conversion script  
│   ├── download_prosit_model.py   # Model download utility
│   ├── verify_installation.py    # Installation verification
│   ├── prosit_onnx_inference.ipynb # Jupyter notebook for testing
│   └── models/prosit/             # TensorFlow model files
└── csharp/                     # C# implementation
    └── PrositProject/
        ├── PrositProject.sln      # Visual Studio solution
        ├── SpectralPredictor.cs   # Main prediction class
        ├── Readers.cs             # CSV input file reader
        ├── Tensorizer.cs          # Feature encoding utilities
        └── Tests/                 # Unit tests and test data
```

## Features

- **Model Conversion**: Convert Prosit TensorFlow models to ONNX format
- **C# Integration**: Use converted models in .NET applications
- **Spectral Prediction**: Predict MS/MS spectra from peptide sequences
- **Input Processing**: Handle peptide sequences, charges, and collision energies
- **Testing**: Comprehensive test suite with sample data

## Installation

### Python Environment

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

#### Prerequisites

- Python 3.11.x (exactly, for TensorFlow compatibility)
- uv package manager
- Windows (recommended, project is optimized for Windows)

#### Setup

1. Clone the repository:
   ```powershell
   git clone https://github.com/pcruzparri/prosit2csharp.git
   cd prosit2csharp/python
   ```

2. Install dependencies:
   ```powershell
   uv sync
   ```

   **Note**: If you encounter TensorFlow installation issues, try force-reinstalling:
   ```powershell
   uv pip install tensorflow==2.12.1 --force-reinstall
   ```

3. Verify installation:
   ```powershell
   uv run python verify_installation.py
   ```

### C# Project

#### Prerequisites

- .NET 8.0 SDK
- Visual Studio 2022 or VS Code (recommended)

#### Setup

1. Navigate to the C# project:
   ```powershell
   cd csharp/PrositProject
   ```

2. Restore packages:
   ```powershell
   dotnet restore
   ```

3. Build the project:
   ```powershell
   dotnet build
   ```

4. Run tests:
   ```powershell
   dotnet test
   ```

## Usage

### Converting Models

Convert TensorFlow models to ONNX format:

```powershell
cd python
uv run python tfmodel2onnx_compatible.py
```

This will generate ONNX model files compatible with the C# implementation.

### C# Spectral Prediction

```csharp
using PrositProject;

// Initialize predictor with ONNX model
var predictor = new SpectralPredictor("Models/HLA_CID/weight_192_0.16253_compatible.onnx");

// Predict spectra from CSV input file
var predictions = predictor.Predict("path/to/input.csv");

// Process results...
predictor.Dispose();
```

### Input Format

The C# implementation expects CSV files with the following headers:
- `peptide_sequence`: Peptide sequence with modifications
- `precursor_charge`: Charge state
- `collision_energy`: Normalized collision energy

Example:
```csv
peptide_sequence,precursor_charge,collision_energy
PEPTIDER,2,0.35
M(ox)QIFVKTLTGK,3,0.30
```

## Dependencies

### Python
- TensorFlow 2.12.1
- ONNX 1.17.0
- tf2onnx 1.15.0
- ONNXRuntime 1.23.2
- NumPy, Pandas, PyYAML, H5Py

### C#
- .NET 8.0
- Microsoft.ML.OnnxRuntime 1.23.2
- NUnit 4.4.0 (for testing)
- Microsoft.NET.Test.Sdk 18.0.1
- NUnit3TestAdapter 5.2.0

## Troubleshooting

### TensorFlow Import Issues

If you get a `ModuleNotFoundError: No module named 'tensorflow'`:

1. Verify Python version:
   ```powershell
   python --version  # Should be 3.11.x
   ```

2. Force reinstall TensorFlow:
   ```powershell
   uv pip install tensorflow==2.12.1 --force-reinstall
   ```

3. Test installation:
   ```powershell
   uv run python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
   ```

### Windows-Specific Notes

- This project is optimized for Windows and uses `tensorflow-intel` for better performance
- Ensure your Python version is exactly 3.11.x for compatibility
- Use PowerShell for all commands         

### ONNX Model Issues

- Ensure models are converted with the `tfmodel2onnx_compatible.py` script
- Check that ONNX files are properly copied to the C# output directory
- Verify model paths in your C# code match the actual file locations