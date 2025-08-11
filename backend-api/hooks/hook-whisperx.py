# hook-whisperx.py
from PyInstaller.utils.hooks import collect_all, collect_data_files

datas, binaries, hiddenimports = collect_all('whisperx')
# Include WhisperX assets that are already installed
datas += collect_data_files('whisperx.assets')
# Include alignment models data if available
datas += collect_data_files('whisperx.align_models')
# add any other missing hiddenimports as needed 