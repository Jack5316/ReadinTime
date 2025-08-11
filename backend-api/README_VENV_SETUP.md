# Virtual Environment Setup for arBooks Backend

## Overview
This document describes the setup of a clean virtual environment with all required packages installed from source.

## Environment Setup

### 1. Create New Virtual Environment
```bash
python -m venv venv-unified
venv-unified\Scripts\Activate.ps1
```

### 2. Upgrade Base Packages
```bash
python -m pip install --upgrade pip setuptools wheel
```

### 3. Install Packages from Source

#### Chatterbox TTS
```bash
pip install -e chatterbox/
```

#### WhisperX
```bash
pip install -e whisperX-3.4.2/
```

#### Markitdown
```bash
pip install -e markitdown-source/packages/markitdown/
```

### 4. Fix Dependency Conflicts
Due to version conflicts between chatterbox and whisperx:
```bash
pip install transformers==4.46.3
```

## Package Status

✅ **Successfully Installed:**
- chatterbox-tts (from source)
- whisperx (from source) 
- markitdown (from source)
- torch & torchaudio
- All supporting dependencies

✅ **Import Tests Pass:**
- All packages can be imported successfully
- No import errors in the virtual environment

⚠️ **Known Issues:**
- `pkg_resources` deprecation warning from `perth` package (dependency of chatterbox)
- This is a warning only and doesn't affect functionality
- The warning will be resolved when `perth` updates to use `importlib.resources`

## Testing

Run the import test script to verify all packages work:
```bash
python test_imports.py
```

Expected output:
```
Testing package imports...
✓ chatterbox imported successfully
✓ whisperx imported successfully
✓ markitdown imported successfully
✓ torch imported successfully
✓ torchaudio imported successfully

All import tests completed!
```

## Usage

The virtual environment is now ready for development. All packages can be imported directly:

```python
import chatterbox
import whisperx
import markitdown
import torch
import torchaudio
```

## Notes

- The virtual environment uses the correct paths for the current system
- All packages are installed in editable mode for development
- Dependencies are resolved with compatible versions
- The environment is isolated and reproducible 