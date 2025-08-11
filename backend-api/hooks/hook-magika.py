# backend-api/hooks/hook-magika.py
from PyInstaller.utils.hooks import collect_data_files, collect_submodules
import os
from pathlib import Path

# Collect all submodules
hiddenimports = collect_submodules('magika')

# Collect data files including models
datas = collect_data_files('magika')

# Add magika models directory specifically
try:
    import magika
    magika_path = Path(magika.__file__).parent
    models_path = magika_path / "models"
    
    if models_path.exists():
        # Add the entire models directory
        for model_dir in models_path.iterdir():
            if model_dir.is_dir():
                for file_path in model_dir.rglob("*"):
                    if file_path.is_file():
                        # Calculate relative path from models directory
                        rel_path = file_path.relative_to(models_path)
                        datas.append((str(file_path), f"magika/models/{model_dir.name}/{rel_path}"))
                        
        print(f"Added magika models from: {models_path}")
    else:
        print(f"Warning: magika models directory not found at {models_path}")
        
except ImportError as e:
    print(f"Warning: Could not import magika for hook: {e}")
except Exception as e:
    print(f"Warning: Error setting up magika hook: {e}") 