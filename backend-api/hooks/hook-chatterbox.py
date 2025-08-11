# backend-api/hooks/hook-chatterbox.py
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

hiddenimports = collect_submodules('chatterbox')
datas = collect_data_files('chatterbox', include_py_files=True) 