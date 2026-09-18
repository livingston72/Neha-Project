"""Shared paths, provenance and transparent error reporting."""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = {"block": "Block", "interjection": "Interjection", "soundrep": "SoundRep",
           "wordrep": "WordRep", "prolongation": "Prolongation"}
KEYS = ["Show", "EpId", "ClipId"]
PAPER_DOI = "10.1109/ICASSP49660.2025.10888625"
PACKAGES = ["numpy", "scipy", "pandas", "librosa", "soundfile", "scikit-learn",
            "imbalanced-learn", "matplotlib", "joblib", "torch"]


def initialize(root: Path) -> None:
    for folder in ("dataset", "data", "models", "results"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    errors = root / "results/processing_errors.csv"
    if not errors.exists():
        with errors.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["audio_path", "error_type", "error_message"])


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=str), encoding="utf-8")
    temporary.replace(path)


def log_error(root: Path, path, error_type: str, message: str) -> None:
    initialize(root)
    with (root / "results/processing_errors.csv").open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([str(path), error_type, message])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def environment() -> dict:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "NOT INSTALLED"
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": versions, "timestamp_utc": datetime.now(timezone.utc).isoformat()}


def stem(target: str, dataset: str = "sep28k") -> str:
    prefix = "" if dataset == "sep28k" else "fluencybank_"
    return f"{prefix}clean_vs_{target}"


def feature_path(root: Path, target: str, feature: str, dataset: str = "sep28k") -> Path:
    if (target, feature, dataset) == ("block", "CQT-LTAS", "sep28k"):
        return root / "data/ltas_features.csv"
    return root / "data" / f"{stem(target, dataset)}_{feature.lower()}_features.csv"
