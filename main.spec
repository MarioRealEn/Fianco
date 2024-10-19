# -*- mode: python ; coding: utf-8 -*-

import os
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

hidden_imports = collect_submodules('fianco_brain') + collect_submodules('numpy')
datas = collect_data_files('fianco_brain') + collect_data_files('numpy')
binaries = [
    (os.path.abspath(r'env\Lib\site-packages\fianco_brain\fianco_brain.cp311-win_amd64.pyd'), 'fianco_brain'),
]

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,     # Include binaries
    a.zipfiles,     # Include zip files
    a.datas,        # Include data files
    [],
    exclude_binaries=False,
    name='main',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
