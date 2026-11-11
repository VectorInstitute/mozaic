"""Shared test setup.

The data-generation code reads its two folders from a paths file. When none is given, the
tests get an empty one in a temporary folder, so they never touch real data.
"""

import atexit
import os
import shutil
import tempfile
from pathlib import Path

if not os.environ.get("MOZAIC_GENERATION_PATHS"):
    _root = Path(tempfile.mkdtemp(prefix="mozaic_generation_test_"))
    atexit.register(shutil.rmtree, _root, ignore_errors=True)
    (_root / "voxceleb2").mkdir()
    (_root / "data").mkdir()
    (_root / "paths.yaml").write_text(f"voxceleb2: {_root / 'voxceleb2'}\ndata: {_root / 'data'}\n")
    os.environ["MOZAIC_GENERATION_PATHS"] = str(_root / "paths.yaml")
