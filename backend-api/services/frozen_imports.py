import sys
import os
from pathlib import Path

def setup_frozen_imports():
    """Setup import paths for frozen PyInstaller executable."""
    if not getattr(sys, 'frozen', False):
        return  # Not running in frozen executable
    
    # Get the base directory where PyInstaller extracts files
    if hasattr(sys, '_MEIPASS'):
        base_path = Path(sys._MEIPASS)
    else:
        # Fallback for older PyInstaller versions
        base_path = Path(sys.executable).parent
    
    print(f"Setting up frozen imports from base path: {base_path}")
    
    # Add the base path itself FIRST to ensure local modules can be imported
    if str(base_path) not in sys.path:
        sys.path.insert(0, str(base_path))
        print(f"Added base path to Python path: {base_path}")
    else:
        print(f"Base path already in Python path: {base_path}")
    
    # Verify local application modules exist (but don't add them to path separately)
    # They should be importable from the base path
    local_modules = ['models', 'services']
    for module_dir in local_modules:
        module_path = base_path / module_dir
        if not module_path.exists():
            print(f"Warning: Local module directory not found: {module_path}")
        else:
            print(f"Local module directory found: {module_path}")
    
    # Add source-built package directories to Python path
    package_dirs = [
        'chatterbox',
        'whisperX-3.4.2', 
        'markitdown-source',
        'asteroid_filterbanks',
        'speechbrain',
        'torch_audiomentations',
        'pytorch_metric_learning',
        'torchmetrics',
        'omegaconf',
        'einops',
        'tensorboardX'
    ]
    
    for package_dir in package_dirs:
        package_path = base_path / package_dir
        if package_path.exists():
            # Add the package directory to Python path
            if str(package_path) not in sys.path:
                sys.path.insert(0, str(package_path))
                print(f"Added to Python path: {package_path}")
            
            # For packages with src structure, also add the src directory
            src_path = package_path / 'src'
            if src_path.exists() and str(src_path) not in sys.path:
                sys.path.insert(0, str(src_path))
                print(f"Added to Python path: {src_path}")
        else:
            print(f"Warning: Package directory not found: {package_path}")
    
    # Special handling for WhisperX - ensure it's importable
    whisperx_path = base_path / 'whisperX-3.4.2'
    if whisperx_path.exists():
        print(f"WhisperX source found at: {whisperx_path}")
        
        # Add the whisperx subdirectory specifically
        whisperx_subdir = whisperx_path / 'whisperx'
        if whisperx_subdir.exists():
            if str(whisperx_subdir) not in sys.path:
                sys.path.insert(0, str(whisperx_subdir))
                print(f"Added WhisperX subdirectory to Python path: {whisperx_subdir}")
        else:
            print(f"Warning: WhisperX subdirectory not found: {whisperx_subdir}")
        
        # Also add the parent directory so the package can be imported as 'whisperx'
        if str(whisperx_path) not in sys.path:
            sys.path.insert(0, str(whisperx_path))
            print(f"Added WhisperX parent directory to Python path: {whisperx_path}")
        
        # Try to import whisperx to verify it works
        try:
            import whisperx
            print("WhisperX import successful after path setup")
        except ImportError as e:
            print(f"Warning: WhisperX import failed after path setup: {e}")
            # Try alternative import approach
            try:
                # Add the path again and try importing
                sys.path.insert(0, str(whisperx_path))
                import whisperx
                print("WhisperX import successful with alternative path setup")
            except ImportError as e2:
                print(f"Warning: WhisperX import failed with alternative setup: {e2}")
        
        # Also test asteroid_filterbanks specifically
        try:
            import asteroid_filterbanks
            print("asteroid_filterbanks import successful after path setup")
        except ImportError as e:
            print(f"Warning: asteroid_filterbanks import failed after path setup: {e}")
    else:
        print(f"Warning: WhisperX source not found at: {whisperx_path}")
    
    print(f"Final Python path contains {len(sys.path)} entries")
    print(f"Python path entries containing packages: {[p for p in sys.path if any(pkg in p for pkg in package_dirs)]}")

def get_frozen_package_path(package_name: str) -> Path:
    """Get the path to a package in the frozen executable."""
    if not getattr(sys, 'frozen', False):
        return None
    
    if hasattr(sys, '_MEIPASS'):
        base_path = Path(sys._MEIPASS)
    else:
        base_path = Path(sys.executable).parent
    
    return base_path / package_name

def get_frozen_python_path() -> Path:
    """Get the Python executable path for frozen environment."""
    if not getattr(sys, 'frozen', False):
        return None
    
    return Path(sys.executable)

def create_frozen_script_wrapper(script_content: str) -> str:
    """Create a script wrapper that sets up imports for frozen environment."""
    if not getattr(sys, 'frozen', False):
        return script_content
    
    # Get the base path where packages are extracted
    if hasattr(sys, '_MEIPASS'):
        base_path = Path(sys._MEIPASS)
    else:
        base_path = Path(sys.executable).parent
    
    # Add import setup at the beginning of the script
    import_setup = f"""
import sys
import os
from pathlib import Path

# Setup paths for frozen environment
base_path = Path(r'{base_path}')
package_dirs = ['chatterbox', 'whisperX-3.4.2', 'markitdown-source']

print(f"Setting up frozen imports from base path: {{base_path}}")

for package_dir in package_dirs:
    package_path = base_path / package_dir
    if package_path.exists():
        if str(package_path) not in sys.path:
            sys.path.insert(0, str(package_path))
            print(f"Added to Python path: {{package_path}}")
        
        # For packages with src structure
        src_path = package_path / 'src'
        if src_path.exists() and str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
            print(f"Added to Python path: {{src_path}}")
    else:
        print(f"Warning: Package directory not found: {{package_path}}")

# Special handling for WhisperX
whisperx_path = base_path / 'whisperX-3.4.2'
if whisperx_path.exists():
    print(f"WhisperX source found at: {{whisperx_path}}")
    
    # Add the whisperx subdirectory specifically
    whisperx_subdir = whisperx_path / 'whisperx'
    if whisperx_subdir.exists():
        if str(whisperx_subdir) not in sys.path:
            sys.path.insert(0, str(whisperx_subdir))
            print(f"Added WhisperX subdirectory to Python path: {{whisperx_subdir}}")
    
    # Also add the parent directory so the package can be imported as 'whisperx'
    if str(whisperx_path) not in sys.path:
        sys.path.insert(0, str(whisperx_path))
        print(f"Added WhisperX parent directory to Python path: {{whisperx_path}}")
    
    # Try to import whisperx to verify it works
    try:
        import whisperx
        print("WhisperX import successful after path setup")
    except ImportError as e:
        print(f"Warning: WhisperX import failed after path setup: {{e}}")
        # Try alternative import approach
        try:
            import sys
            sys.path.insert(0, str(whisperx_path))
            import whisperx
            print("WhisperX import successful with alternative path setup")
        except ImportError as e2:
            print(f"Warning: WhisperX import failed with alternative setup: {{e2}}")
else:
    print(f"Warning: WhisperX source not found at: {{whisperx_path}}")

print(f"Final Python path contains {{len(sys.path)}} entries")
print(f"Python path entries containing packages: {{[p for p in sys.path if any(pkg in p for pkg in package_dirs)]}}")

"""
    
    return import_setup + script_content 