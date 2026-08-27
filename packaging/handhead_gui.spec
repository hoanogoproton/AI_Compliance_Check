# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for the Hand-to-Head Detection GUI.
# Build (from project root):
#   .venv\Scripts\python.exe -m PyInstaller --clean --noconfirm packaging\handhead_gui.spec
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

spec_dir = Path(SPECPATH)
if not spec_dir.is_absolute():
    spec_dir = Path.cwd() / spec_dir
project_root = spec_dir.resolve().parent
if not (project_root / "gui" / "app.py").exists():
    # Fallback: build invoked from the project root directly
    project_root = Path.cwd().resolve()

datas = []
binaries = []
hiddenimports = [
    # imported lazily inside ultralytics trackers
    "lap",
]

for pkg in ("ultralytics", "torch", "torchvision", "mediapipe", "onnxruntime", "sklearn"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

a = Analysis(
    [str(project_root / "gui" / "app.py")],
    pathex=[str(project_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="HandHeadGUI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="HandHeadGUI",
)
