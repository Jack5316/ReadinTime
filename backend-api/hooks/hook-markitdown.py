# backend-api/hooks/hook-markitdown.py
from PyInstaller.utils.hooks import collect_submodules

hiddenimports = collect_submodules('markitdown') 