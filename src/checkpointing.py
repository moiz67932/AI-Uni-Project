from __future__ import annotations

import os
import platform
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn


def _shape_of(obj: Any) -> Any:
    shape = getattr(obj, "shape", None)
    if shape is not None:
        try:
            return tuple(shape)
        except Exception:
            return str(shape)
    if isinstance(obj, (list, tuple, dict, set)):
        return len(obj)
    return None


def safe_save(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = Path(f"{path}.tmp")
    joblib.dump(obj, temp_path)
    os.replace(temp_path, path)


class CheckpointManager:
    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        return self.base_dir / name

    def exists(self, name: str) -> bool:
        return self._path(name).exists()

    def delete(self, name: str) -> None:
        path = self._path(name)
        if path.exists():
            path.unlink()

    def save(self, name: str, obj: Any, description: str | None = None) -> Path:
        payload = {
            "metadata": {
                "created_at": datetime.utcnow().isoformat() + "Z",
                "python_version": platform.python_version(),
                "sklearn_version": sklearn.__version__,
                "pandas_version": pd.__version__,
                "numpy_version": np.__version__,
                "shape": _shape_of(obj),
                "description": description or "",
            },
            "payload": obj,
        }
        path = self._path(name)
        safe_save(path, payload)
        return path

    def load(self, name: str, default: Any = None) -> Any:
        path = self._path(name)
        if not path.exists():
            return default
        try:
            data = joblib.load(path)
            if isinstance(data, dict) and "payload" in data:
                return data["payload"]
            return data
        except Exception as exc:
            corrupt_path = path.with_suffix(path.suffix + ".corrupt")
            try:
                os.replace(path, corrupt_path)
            except Exception:
                pass
            print(
                f"Warning: could not load checkpoint '{path.name}' because it looks corrupted. "
                f"Renamed to '{corrupt_path.name}' and returning default. Details: {exc}"
            )
            return default
