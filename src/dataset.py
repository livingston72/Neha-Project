"""Audit every label and decode every WAV before selecting pure binary tasks."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from .common import KEYS, TARGETS, initialize, log_error, save_json, sha256, stem


class AudioError(ValueError):
    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


def read_audio(path: Path):
    """Decode actual samples, not just metadata. Empty is distinct from silence."""
    if not path.is_file():
        raise AudioError("missing_audio", "WAV file does not exist")
    try:
        info = sf.info(path)
        audio, sr = sf.read(path, dtype="float64", always_2d=True)
    except (RuntimeError, ValueError, OSError) as exc:
        raise AudioError("invalid_audio", str(exc)) from exc
    if len(audio) == 0:
        raise AudioError("empty_audio", "WAV decoded successfully but contains zero samples")
    if info.frames != len(audio):
        raise AudioError("invalid_audio", "Decoded frame count disagrees with WAV metadata")
    if sr <= 0 or audio.shape[1] == 0 or not np.isfinite(audio).all():
        raise AudioError("invalid_audio", "Invalid rate/channels or non-finite audio samples")
    mono = audio.mean(axis=1)  # [IMPLEMENTATION DECISION] arithmetic channel average
    meta = {"audio_path": str(path), "original_sample_rate": int(sr),
            "original_samples": len(mono), "original_duration": len(mono) / sr,
            "original_channels": audio.shape[1]}
    if len(mono) / sr < 0.020:
        raise AudioError("short_audio", "Less than one complete 20 ms frame; no padding or fabricated features")
    return mono, sr, meta


def read_labels(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Labels not found: {path}")
    df = pd.read_csv(path, dtype={key: str for key in KEYS}, keep_default_na=False)
    required = KEYS + list(TARGETS.values())
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing CSV columns: {missing}")
    for key in KEYS:
        df[key] = df[key].str.strip()
        if df[key].eq("").any() or df[key].str.contains(r"[/\\]").any():
            raise ValueError(f"Empty identifier or path separator in {key}")
    if df.duplicated(KEYS).any():
        raise ValueError("Duplicate Show/EpId/ClipId rows; resolve them before splitting to prevent leakage")
    for column in TARGETS.values():
        df[column] = pd.to_numeric(df[column], errors="raise")
        values = df[column].to_numpy(float)
        if not np.isfinite(values).all() or (values < 0).any() or (values % 1 != 0).any():
            raise ValueError(f"{column} must contain finite nonnegative integer annotation counts")
    return df


def resolve_audio(row, wav_dir: Path, template: str) -> Path:
    relative = Path(template.format(**{key: str(row[key]) for key in KEYS}))
    path = (wav_dir / relative).resolve()
    if not path.is_relative_to(wav_dir.resolve()):
        raise ValueError("WAV template escapes the WAV directory")
    return path


def prepare(root: Path, dataset="sep28k", targets=None, labels=None, wav_dir=None,
            template="{Show}{EpId}{ClipId}.wav") -> dict:
    initialize(root)
    labels = Path(labels) if labels else root / "dataset" / (
        "SEP-28k_labels.csv" if dataset == "sep28k" else "fluencybank_labels.csv")
    wav_dir = Path(wav_dir) if wav_dir else root / "dataset/clips/stuttering-clips/clips"
    if not wav_dir.is_dir():
        raise FileNotFoundError(f"WAV directory not found: {wav_dir}")
    df = read_labels(labels)
    positive = df[list(TARGETS.values())].gt(0)
    clean = ~positive.any(axis=1)
    overlap = positive.sum(axis=1).gt(1)
    patterns = positive.apply(lambda r: "+".join(r.index[r]) or "Clean", axis=1)
    paths = [resolve_audio(row, wav_dir, template) for _, row in df.iterrows()]
    if len(set(paths)) != len(paths):
        raise ValueError("WAV naming collision: different clip IDs resolve to the same file. Specify an unambiguous --wav-template")
    records = []
    for i, path in enumerate(paths, 1):
        record = {"audio_path": str(path), "status": "valid"}
        try:
            _, _, meta = read_audio(path)
            record.update(meta)
        except AudioError as exc:
            record["status"] = exc.kind
            log_error(root, path, exc.kind, str(exc))
        records.append(record)
        if i % 100 == 0 or i == len(paths):
            print(f"Audio verification: {i} / {len(paths)}", flush=True)
    audited = pd.concat([df.reset_index(drop=True), pd.DataFrame(records)], axis=1)
    audit_path = root / "data" / f"{dataset}_audio_audit.csv"
    audited.to_csv(audit_path, index=False)
    counts = Counter(audited.status)
    summary = {"dataset": dataset, "labels_path": str(labels.resolve()), "labels_sha256": sha256(labels),
               "wav_directory": str(wav_dir.resolve()), "wav_template": template,
               "total_samples": len(df), "clean_samples": int(clean.sum()),
               "stutter_counts_including_overlaps": positive.sum().astype(int).to_dict(),
               "overlapping_labels": int(overlap.sum()), "label_combinations": patterns.value_counts().to_dict(),
               "audio_status_counts": {key: counts.get(key, 0) for key in
                   ["valid", "missing_audio", "empty_audio", "invalid_audio", "short_audio"]},
               "selection_provenance": "[IMPLEMENTATION DECISION] Pure target vs all-five-zero clean; NOT SPECIFIED IN PAPER",
               "tasks": {}}
    for target in targets or TARGETS:
        column = TARGETS[target]
        pure = positive[column] & positive.drop(columns=column).sum(axis=1).eq(0)
        selected = audited.loc[clean | pure].copy()
        selected["label"] = pure.loc[selected.index].astype(int)
        valid = selected.loc[selected.status.eq("valid")].copy()
        selected.to_csv(root / "data" / f"{stem(target, dataset)}.csv", index=False)
        valid.to_csv(root / "data" / f"{stem(target, dataset)}_valid.csv", index=False)
        sc = Counter(selected.status)
        summary["tasks"][target] = {"original_clips": len(selected), "empty_clips": sc.get("empty_audio", 0),
                "missing_clips": sc.get("missing_audio", 0), "invalid_clips": sc.get("invalid_audio", 0),
                "short_clips": sc.get("short_audio", 0), "valid_clips": len(valid),
                "valid_clean": int(valid.label.eq(0).sum()), "valid_target": int(valid.label.eq(1).sum()),
                f"valid_{target}": int(valid.label.eq(1).sum())}
    save_json(root / "results" / f"{dataset}_dataset_summary.json", summary)
    print(__import__("json").dumps(summary, indent=2))
    return summary
